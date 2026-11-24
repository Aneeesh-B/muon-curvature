"""Run the paired, cumulative-matmul-FLOP experiment on one device."""

import argparse
from collections import Counter
import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import shutil
import subprocess

import torch

from .budget import BudgetBatches, Trial, evaluate, now
from .data import DATA, ROOT
from .diagnostics import append_json


ARMS = {
    "default_muon": "fixed",
    "same_batch_muon": "same_batch",
    "independent_batch_muon": "independent_batch",
    "muon_search": "fixed",
    "adamw_search": "adamw",
    "adago_search": "adago",
}
GRIDS = {
    "muon_search": [(lr, None) for lr in (0.003, 0.01, 0.02, 0.04, 0.08)],
    "adamw_search": [(lr, None) for lr in (0.0003, 0.001, 0.003, 0.006, 0.01)],
    "adago_search": [(0.02, 0.0001), (0.05, 0.0005), (0.10, 0.0005),
                     (0.30, 0.0005), (0.30, 0.003)],
}


def forward_flops(batch, config, vocab):
    t, d, layers = (config[k] for k in ("block_size", "n_embd", "n_layer"))
    return 2 * batch * t * (12 * layers * d * d + d * vocab) + 4 * batch * layers * t * t * d


def newton_schulz_flops(shape, iterations):
    rows, cols = sorted(shape)
    return iterations * (4 * rows * rows * cols + 2 * rows ** 3)


class Work:
    def __init__(self, config, vocab):
        batch = config["batch_size"]
        forward = forward_flops(batch, config, vocab)
        head = 2 * batch * config["block_size"] * config["n_embd"] * vocab
        self.costs = dict(training_forward=forward, training_backward=2 * forward,
                          probe_gradient=3 * forward - head)
        self.by_category = Counter()

    @property
    def total(self):
        return self.by_category.total()

    def record(self, kind):
        self.by_category[kind] += self.costs[kind]

    def orthogonalize(self, shape, iterations):
        self.by_category["newton_schulz"] += newton_schulz_flops(shape, iterations)


def update_cost(trial):
    work = trial.work
    cost = work.costs["training_forward"] + work.costs["training_backward"]
    optimizer = trial.muon or trial.adago
    if optimizer:
        cost += sum(newton_schulz_flops(p.shape, group["ns_steps"])
                    for group in optimizer.param_groups for p in group["params"])
    c = trial.config
    probe = trial.step >= c["probe_start"] and (trial.step - c["probe_start"]) % c["probe_every"] == 0
    if probe and trial.mode in ("same_batch", "independent_batch"):
        cost += work.costs["probe_gradient"] * (2 if trial.mode == "independent_batch" else 1)
    return cost


def validation_flops(trial, split):
    windows = (len(trial.batches.arrays[split]) - 1) // trial.config["block_size"]
    return forward_flops(windows, trial.config, trial.model_config.vocab_size)


def train_until(trial, cap, started, events, stop_at=None):
    work = trial.work
    while not trial.failed and (stop_at is None or trial.step < stop_at):
        if work.total + update_cost(trial) > cap:
            break
        before = work.total
        try:
            record = trial.train_step()
        except FloatingPointError as error:
            trial.failed = str(error)
            append_json(events, dict(event="failed", trial_id=trial.trial_id, step=trial.step,
                                     reason=trial.failed, charged_flops=work.total - before,
                                     cumulative_flops=work.total))
            break
        append_json(events, dict(event="step", trial_id=trial.trial_id,
                                 elapsed_seconds=now(trial.device) - started,
                                 charged_flops=work.total - before, cumulative_flops=work.total, **record))


def search(config, batches, seed, arm, cap, work, started, events):
    outcomes = []
    winner = None
    slot = cap * 2 // 25
    for index, (lr, epsilon) in enumerate(GRIDS[arm]):
        trial = Trial(config, batches, ARMS[arm], seed, lr, index, work, epsilon)
        before = work.total
        tuning_cost = validation_flops(trial, "tuning")
        train_until(trial, before + slot - tuning_cost, started, events)
        metrics = None
        if trial.step and not trial.failed:
            work.by_category["tuning_validation"] += tuning_cost
            metrics = evaluate(trial.model, batches, "tuning")
            if not math.isfinite(metrics["loss"]):
                trial.failed = "Nonfinite tuning loss"
                metrics = None
        loss = metrics["loss"] if metrics else None
        outcome = dict(trial_id=index, lr=lr, epsilon=epsilon, steps=trial.step, tuning_loss=loss,
                       tuning=metrics, charged_flops=work.total - before, slot_flops=slot,
                       failure=trial.failed or ("No eligible update" if not trial.step else None))
        outcomes.append(outcome)
        append_json(events, dict(event="candidate", cumulative_flops=work.total, **outcome))
        if metrics and (winner is None or loss < outcomes[winner.trial_id]["tuning_loss"]):
            winner = trial
    if winner is None:
        raise FloatingPointError("No search candidate completed tuning validation")
    picked = dict(trial_id=winner.trial_id, from_step=winner.step,
                  edge=winner.trial_id in (0, len(outcomes) - 1))
    append_json(events, dict(event="selection", cumulative_flops=work.total, **picked))
    train_until(winner, cap, started, events)
    picked["final_step"] = winner.step
    return winner, outcomes, picked


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def publish_checkpoint(trial, cap, work, started, out, candidates, selection):
    if trial.failed or not trial.step:
        raise FloatingPointError(trial.failed or "No eligible update")
    if work.total > cap:
        raise RuntimeError("FLOP cap exceeded")
    t = now(trial.device)
    metrics = evaluate(trial.model, trial.batches, "reporting")
    reporting_seconds = now(trial.device) - t
    if not math.isfinite(metrics["loss"]):
        raise FloatingPointError("Nonfinite reporting loss")
    checkpoint = out / f"cap-{cap}.pt"
    torch.save(dict(model=trial.model.state_dict(), config=trial.config, step=trial.step,
                    model_config=asdict(trial.model_config), charged_flops=work.total,
                    reporting=metrics), checkpoint)
    return dict(cap_flops=cap, charged_flops=work.total, unused_flops=cap - work.total,
                work=dict(work.by_category), reporting_flops=validation_flops(trial, "reporting"),
                reporting_seconds=reporting_seconds, elapsed_seconds=now(trial.device) - started,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if trial.device == "cuda" else None,
                selected_steps=trial.step, completed_steps=trial.step + sum(
                    c["steps"] for c in candidates if c["trial_id"] != trial.trial_id),
                reporting=metrics, probe_stats=trial.stats.copy(), checkpoint=checkpoint.name,
                candidates=candidates, selection=selection)


def run_campaign(config, batches, seed, arm, caps, out):
    events = out / "events.jsonl"
    vocab = json.loads((DATA / "meta.json").read_text())["vocab_size"]
    work = Work(config, vocab)
    if batches.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = now(batches.device)
    candidates, selection = [], None
    if arm in GRIDS:
        trial, candidates, selection = search(config, batches, seed, arm, caps[0], work, started, events)
    else:
        trial = Trial(config, batches, ARMS[arm], seed, config["lr"], work=work)
    results = []
    common = None
    for cap in caps:
        if arm not in GRIDS:
            if trial.step < 500:
                train_until(trial, cap, started, events, stop_at=500)
                if trial.step == 500:
                    t = now(trial.device)
                    reporting = evaluate(trial.model, batches, "reporting")
                    common = dict(step=trial.step, reporting=reporting,
                                  charged_flops=work.total,
                                  reporting_seconds=now(trial.device) - t,
                                  reporting_flops=validation_flops(trial, "reporting"))
            train_until(trial, cap, started, events)
        record = publish_checkpoint(trial, cap, work, started, out, candidates, selection)
        results.append(record)
        write_json(out / f"cap-{cap}.json", record)
        print(f"  {arm} seed={seed} cap={cap} used={record['charged_flops']} "
              f"steps={trial.step} reporting={record['reporting']['loss']:.4f} "
              f"elapsed={record['elapsed_seconds']:.1f}s", flush=True)
    return dict(arm=arm, seed=seed, measurements=results, common_update=common)


def summarize(out):
    rows = []
    for path in sorted(out.glob("seed-*/*/result.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        for cell in result["measurements"]:
            rows.append(dict(seed=result["seed"], arm=result["arm"], cap_flops=cell["cap_flops"],
                             charged_flops=cell["charged_flops"], unused_flops=cell["unused_flops"],
                             completed_steps=cell["completed_steps"], selected_steps=cell["selected_steps"],
                             elapsed_seconds=cell["elapsed_seconds"], peak_cuda_allocated_bytes=cell["peak_cuda_allocated_bytes"],
                             reporting_loss=cell["reporting"]["loss"], reporting_accuracy=cell["reporting"]["accuracy"],
                             probes=cell["probe_stats"]["probes"], fallbacks=cell["probe_stats"]["fallbacks"],
                             clamps=cell["probe_stats"]["clamps"], checkpoint=cell["checkpoint"]))
    if rows:
        with (out / "comparison.tmp").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        (out / "comparison.tmp").replace(out / "comparison.csv")
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/wide.json")
    parser.add_argument("--caps", type=int, nargs="+", required=True, help="Ordinary Muon update equivalents")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--arms", choices=tuple(ARMS), nargs="+", default=list(ARMS))
    args = parser.parse_args()
    if args.caps != sorted(set(args.caps)) or args.caps[0] < 1:
        parser.error("Caps must be positive and strictly increasing")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.arms)) != len(args.arms):
        parser.error("Seeds and arms must be distinct")
    config = json.loads(args.config.read_text())
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    batches = BudgetBatches(config["batch_size"], config["block_size"], args.device)
    data = json.loads((DATA / "meta.json").read_text())
    probe = Trial(config, batches, "fixed", args.seeds[0], config["lr"], work=Work(config, data["vocab_size"]))
    ordinary = update_cost(probe)
    del probe
    caps = [ordinary * x for x in args.caps]
    orders = {str(seed): random.Random(seed + 230).sample(args.arms, len(args.arms)) for seed in args.seeds}
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True)
    smi = shutil.which("nvidia-smi")
    driver = None
    if args.device == "cuda" and smi:
        driver = subprocess.run([smi, "--query-gpu=driver_version", "--format=csv,noheader"],
                                capture_output=True, text=True, check=True).stdout.strip()
    manifest = dict(config=config, data=data, seeds=args.seeds, update_equivalents=args.caps, caps_flops=caps,
                    ordinary_muon_flops=ordinary, orders=orders, grids={k: [list(x) for x in v] for k, v in GRIDS.items()},
                    revision=git.stdout.strip(), git_dirty=bool(status.stdout.strip()),
                    hardware=torch.cuda.get_device_name() if args.device == "cuda" else platform.processor(),
                    device=args.device, torch=str(torch.__version__), cuda=torch.version.cuda, driver=driver,
                    python=platform.python_version(),
                    source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for folder in ("curvature_muon", "third_party/nanogpt")
                                   for p in sorted((ROOT / folder).glob("*.py"))})
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise RuntimeError("Output directory has a different experiment manifest")
    write_json(manifest_path, manifest)
    failures = []
    total = 0
    for seed in args.seeds:
        for arm in orders[str(seed)]:
            for campaign_caps in ([[cap] for cap in caps] if arm in GRIDS else [caps]):
                total += 1
                name = f"{arm}-{campaign_caps[0]}" if arm in GRIDS else arm
                campaign = args.out / f"seed-{seed}" / name
                result_path = campaign / "result.json"
                if result_path.exists():
                    continue
                campaign.mkdir(parents=True, exist_ok=True)
                attempt = campaign / f"attempt-{len(list(campaign.glob('attempt-*'))) + 1}"
                attempt.mkdir()
                print(f"Campaign {total} seed={seed} {name}", flush=True)
                try:
                    result = run_campaign(config, batches, seed, arm, campaign_caps, attempt)
                except FloatingPointError as error:
                    write_json(attempt / "failure.json", dict(arm=arm, seed=seed, error=str(error)))
                    failures.append(str(attempt))
                    print(f"  Failed {arm}: {error}", flush=True)
                    continue
                for record in result["measurements"]:
                    record["checkpoint"] = f"{attempt.name}/{record['checkpoint']}"
                write_json(result_path, result)
                summarize(args.out)
    count = summarize(args.out)
    print(f"Completed {count} measurement cells across {total} scheduled campaigns", flush=True)
    if failures:
        raise SystemExit(f"Failed campaigns: {', '.join(failures)}")


if __name__ == "__main__":
    main()
