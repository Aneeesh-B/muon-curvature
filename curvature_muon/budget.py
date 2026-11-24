"""Compare models at equal wall time or at an exact update horizon.

Each search owns ONE budget across all trials. Exploration is followed by
continuation of the winning live Trial object, preserving all optimizer/RNG
state. Selection uses the first half of val.npy; reporting uses the second.
No test data is loaded. All model work, evaluation and checkpoint I/O is timed.
"""
import argparse
import copy
import csv
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

from third_party.nanogpt.model import GPT, GPTConfig
from .data import Batches, DATA, ROOT
from .diagnostics import append_json
from .optimizer import Muon, probe, independent_probe, backtrack, replay_rng, rng_state, select_lr, split_parameters

METHOD_MODES = {
    "guarded_base_muon": "guarded_base",
    "guarded_muon": "guarded",
    "adaptive_muon": "periodic",
    "muon_search": "fixed",
    "default_muon": "fixed",
    "adamw_search": "adamw",
    "scheduled_fixed_muon": "scheduled_fixed",
    "scheduled_same_batch_muon": "scheduled_same_batch",
    "scheduled_heldout_muon": "scheduled_heldout",
}
METHODS = tuple(METHOD_MODES)
SCHEDULED_PROBES = ("scheduled_same_batch", "scheduled_heldout")


def schedule_multiplier(step, updates, warmup=40):
    decay_start = updates - round(0.65 * updates)
    if step < warmup:
        return step / warmup
    if step <= decay_start:
        return 1.0
    return 1.0 - 0.95 * (step - decay_start) / (updates - 1 - decay_start)


def now(device):
    if device == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter()


class BudgetBatches(Batches):
    def __init__(self, batch_size, block_size, device):
        super().__init__(batch_size, block_size, device)
        validation = self.arrays.pop("val")
        middle = len(validation) // 2
        self.arrays.update(tuning=validation[:middle], reporting=validation[middle:])
        if min(middle, len(validation)-middle) <= block_size:
            raise ValueError("Validation halves must each exceed the context length")
        self.split_metadata = dict(validation_tokens=len(validation), split_at=middle,
                                   tuning_range=[0, middle], reporting_range=[middle, len(validation)])

    def evaluation_batches(self, split):
        if split not in ("tuning", "reporting"):
            raise ValueError("Only validation halves may be evaluated")
        data = self.arrays[split]
        count = (len(data)-1) // self.block_size
        for base in range(0, count, self.batch_size):
            starts = [i*self.block_size for i in range(base, min(base+self.batch_size, count))]
            x = torch.stack([data[i:i+self.block_size] for i in starts])
            y = torch.stack([data[i+1:i+1+self.block_size] for i in starts])
            yield x.to(self.device), y.to(self.device)


@torch.no_grad()
def evaluate(model, batches, split):
    previous_mode = model.training
    model.eval()
    nll, correct, tokens = 0.0, 0, 0
    try:
        for x, y in batches.evaluation_batches(split):
            logits, loss = model(x, y)
            n = y.numel()
            nll += loss.item()*n
            correct += (logits.argmax(dim=-1) == y).sum().item()
            tokens += n
    finally:
        model.train(previous_mode)
    if tokens == 0:
        raise ValueError("Empty evaluation split")
    return dict(loss=nll/tokens, accuracy=correct/tokens, tokens=tokens)


class Trial:
    """A resumable trajectory. Each trial owns its batch and dropout RNG streams."""
    def __init__(self, config, batches, mode, seed, lr, trial_id=0):
        self.config, self.batches, self.mode = config, batches, mode
        self.seed, self.initial_lr, self.trial_id = seed, lr, trial_id
        self.device = batches.device
        # Creating or training another trial cannot alter this trial's RNG.
        with replay_rng(rng_state()):
            torch.manual_seed(seed)
            if self.device == "cuda":
                torch.cuda.manual_seed_all(seed)
            vocab = json.loads((DATA / "meta.json").read_text())["vocab_size"]
            self.model_config = GPTConfig(
                **{k: config[k] for k in ("n_layer", "n_head", "n_embd", "block_size", "dropout")},
                vocab_size=vocab, bias=False)
            self.model = GPT(self.model_config).to(self.device)
            hidden, auxiliary = split_parameters(self.model)
            self.muon = (Muon(hidden, lr=lr, momentum=config["momentum"],
                              weight_decay=config["weight_decay"]) if mode != "adamw" else None)
            params = list(self.model.parameters()) if mode == "adamw" else auxiliary
            self.adam = torch.optim.AdamW([
                dict(params=[p for p in params if p.ndim >= 2], weight_decay=config["weight_decay"]),
                dict(params=[p for p in params if p.ndim < 2], weight_decay=0.0),
            ], lr=lr if mode == "adamw" else config["adam_lr"],
                betas=(0.9, 0.95), fused=False, foreach=False)
            self.rng = rng_state()
        self.generator = torch.Generator().manual_seed(1000+seed)
        self.calibration_generator = torch.Generator().manual_seed(2000+seed)
        self.schedule = dict(peak=lr, latest=lr) if mode.startswith("scheduled_") else None
        self.step = 0
        self.stats = dict(train_seconds=0.0, probe_seconds=0.0, evaluation_seconds=0.0,
                          snapshot_seconds=0.0, probes=0, fallbacks=0, clamps=0)
        self.best = None
        self.best_weights = None
        self.last_eval_seconds = 0.0
        self.max_step_seconds = 0.005
        self.failed = None

    @property
    def lr(self):
        return self.muon.param_groups[0]["lr"] if self.muon else self.adam.param_groups[0]["lr"]

    def set_muon_lr(self, rate):
        for group in self.muon.param_groups:
            group["lr"] = rate

    def scheduled_probe(self, direction, batch, forward_rng, state):
        distance = self.config["relative_distance"]
        if self.mode == "scheduled_same_batch":
            return probe(direction, lambda: self.model(*batch)[1], forward_rng, distance)
        # Peek at the next batch without consuming any training draws.
        peek = torch.Generator()
        peek.set_state(self.generator.get_state())
        next_batch = self.batches.get("train", peek)
        return independent_probe(direction, lambda: self.model(*next_batch)[1], state, distance)

    def train_step(self):
        started = now(self.device)
        probe_seconds = 0.0
        with replay_rng(self.rng):
            self.model.zero_grad(set_to_none=True)
            batch = self.batches.get("train", self.generator)
            state = rng_state()
            loss = self.model(*batch)[1]
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            direction = self.muon.prepare() if self.muon else None
            record = dict(step=self.step, train_loss=loss.item())
            c = self.config
            if self.schedule is not None:
                decay_start = c["steps"] - round(0.65 * c["steps"])
                if self.mode in SCHEDULED_PROBES and self.step > 0 and self.step % 25 == 0:
                    t = now(self.device)
                    measurement = self.scheduled_probe(direction, batch, state, rng_state())
                    rate, decision = select_lr(measurement, self.schedule["latest"],
                                               c["lr_min"], c["lr_max"], c["safety"])
                    self.schedule["latest"] = rate
                    if self.step <= decay_start:
                        self.schedule["peak"] = rate
                    probe_seconds = now(self.device) - t
                    record.update(probe=measurement, decision=decision, proposed_lr=rate)
                rate = self.schedule["peak"] * schedule_multiplier(self.step, c["steps"])
                if self.step > decay_start and self.mode in SCHEDULED_PROBES:
                    rate = min(rate, self.schedule["latest"])
                self.set_muon_lr(rate)
            elif (self.mode in ("periodic", "guarded", "guarded_base") and self.step >= c["probe_start"]
                    and (self.step-c["probe_start"]) % c["probe_every"] == 0):
                t = now(self.device)
                if self.mode in ("guarded", "guarded_base"):
                    calibration = self.batches.get("train", self.calibration_generator)
                    check = self.batches.get("train", self.calibration_generator)
                    probe_rng = rng_state()
                    measurement = independent_probe(direction, lambda: self.model(*calibration)[1], probe_rng, c["relative_distance"])
                    rate, decision = select_lr(measurement, c["lr"], c["lr_min"], c["lr_max"], c["safety"])
                    if self.mode == "guarded_base":
                        rate, decision = c["lr"], "base_control"
                    record["proposed_lr"] = rate
                    rate, guard = backtrack(direction, lambda: self.model(*check)[1], probe_rng, rate)
                    record["guard"] = guard
                else:
                    measurement = probe(direction, lambda: self.model(*batch)[1], state, c["relative_distance"])
                    rate, decision = select_lr(measurement, self.lr, c["lr_min"], c["lr_max"], c["safety"])
                self.set_muon_lr(rate)
                probe_seconds = now(self.device)-t
                record.update(probe=measurement, decision=decision)
            if "probe" in record:
                self.stats["probes"] += 1
                self.stats["fallbacks"] += record["decision"] == "fallback"
                self.stats["clamps"] += record["decision"] == "clamped"
            if self.muon:
                self.muon.step(direction)
            self.adam.step()
            self.rng = rng_state()
        self.step += 1
        elapsed = now(self.device)-started
        self.stats["train_seconds"] += elapsed-probe_seconds
        self.stats["probe_seconds"] += probe_seconds
        self.max_step_seconds = max(self.max_step_seconds, elapsed)
        return dict(record, lr=self.lr)

    def validate(self):
        t = now(self.device)
        metrics = evaluate(self.model, self.batches, "tuning")
        self.last_eval_seconds = now(self.device)-t
        self.stats["evaluation_seconds"] += self.last_eval_seconds
        if math.isfinite(metrics["loss"]) and (self.best is None or metrics["loss"] < self.best["loss"]):
            t = now(self.device)
            self.best_weights = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            self.best = dict(metrics, step=self.step, lr=self.lr, trial_id=self.trial_id)
            self.stats["snapshot_seconds"] += now(self.device)-t
        return metrics

    def state_dict(self):
        return dict(model=self.model.state_dict(), muon=self.muon.state_dict() if self.muon else None,
                    adam=self.adam.state_dict(), rng=self.rng, train_generator=self.generator.get_state(), calibration_generator=self.calibration_generator.get_state(),
                    step=self.step, stats=copy.deepcopy(self.stats), config=self.config,
                    model_config=asdict(self.model_config), mode=self.mode, seed=self.seed,
                    initial_lr=self.initial_lr, trial_id=self.trial_id, failed=self.failed,
                    best=self.best, best_weights=self.best_weights, schedule=copy.copy(self.schedule))

    def load_state_dict(self, state):
        self.model.load_state_dict(state["model"])
        self.adam.load_state_dict(state["adam"])
        if self.muon:
            self.muon.load_state_dict(state["muon"])
        self.rng = state["rng"]
        self.generator.set_state(state["train_generator"])
        if "calibration_generator" in state:
            self.calibration_generator.set_state(state["calibration_generator"])
        self.step, self.stats = state["step"], copy.deepcopy(state["stats"])
        self.best, self.best_weights, self.failed = state["best"], state["best_weights"], state["failed"]
        self.schedule = copy.copy(state.get("schedule"))


def candidate_rates(experiment, method, config):
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    adam = method == "adamw_search"
    default = config["adam_lr"] if adam else config["lr"]
    if not method.endswith("_search"):
        return [default]
    grid = experiment.get("search_grids", {}).get(method)
    if grid is not None:
        rates = [default, *grid]
        if len(rates) != experiment["search_candidates"] or len(set(rates)) != len(rates):
            raise ValueError(f"{method} requires distinct rates including the default")
        if any(not math.isfinite(rate) or rate <= 0 for rate in rates):
            raise ValueError(f"{method} rates must be positive and finite")
        return rates
    low, high = experiment["adamw_lr_range" if adam else "muon_lr_range"]
    if not 0 < low < high:
        raise ValueError("Learning-rate search range must be positive and ordered")
    generator = random.Random(experiment["seed"]+8100+(100 if adam else 0))
    return [default, *[math.exp(generator.uniform(math.log(low), math.log(high)))
                       for _ in range(experiment["search_candidates"]-1)]]


def run_window(trial, deadline, campaign_start, every_seconds, events):
    """Reserve time for an end-of-window selection evaluation; no free scoring."""
    device = trial.device

    def validation():
        metrics = trial.validate()
        append_json(events, dict(event="validation", trial_id=trial.trial_id, step=trial.step,
                                 elapsed_seconds=now(device)-campaign_start, tuning=metrics,
                                 best=trial.best))

    if trial.best is None:
        validation()
    next_validation = now(device)+every_seconds
    last_validated_step = trial.step
    while not trial.failed:
        # Guard both a complete (possibly probed) update and its final evaluation.
        reserve = 1.75*trial.last_eval_seconds + 2*trial.max_step_seconds + 0.005
        if now(device)+reserve >= deadline:
            break
        try:
            record = trial.train_step()
        except FloatingPointError as error:
            trial.failed = str(error)
            append_json(events, dict(event="failed", trial_id=trial.trial_id, step=trial.step,
                                     elapsed_seconds=now(device)-campaign_start, reason=trial.failed))
            break
        append_json(events, dict(event="step", trial_id=trial.trial_id,
                                 elapsed_seconds=now(device)-campaign_start, **record))
        current = now(device)
        if next_validation <= current and current + reserve < deadline:
            validation()
            last_validated_step = trial.step
            next_validation = now(device)+every_seconds
    if not trial.failed and trial.step != last_validated_step:
        validation()


def run_updates(trial, campaign_start, events):
    """Train one trajectory and score its final update."""
    for _ in range(trial.config["steps"]):
        record = trial.train_step()
        append_json(events, dict(event="step", trial_id=trial.trial_id,
                                 elapsed_seconds=now(trial.device)-campaign_start, **record))
    metrics = trial.validate()
    append_json(events, dict(event="validation", trial_id=trial.trial_id, step=trial.step,
                             elapsed_seconds=now(trial.device)-campaign_start, tuning=metrics,
                             best=trial.best))


class CampaignFailed(RuntimeError):
    pass


def run_campaign(config, experiment, method, out, device="cpu"):
    matched = experiment.get("matched_updates", False)
    mode = METHOD_MODES[method]
    search = method.endswith("_search")
    if matched and (search or config["steps"] < 1):
        raise ValueError("Matched updates require a single trajectory and a positive step count")
    if mode.startswith("scheduled_") and (not matched or config["steps"] < 120):
        raise ValueError("Scheduled methods require a fixed horizon of at least 120 updates")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {out}")
    rates = candidate_rates(experiment, method, config)
    # Protocol/rates are chosen before inspecting any losses. File/data/model
    # initialization below, every validation, and all model saves are charged.
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = now(device)
    budget = None if matched else experiment["seconds_per_method"]
    if budget is not None:
        final_reserve = min(experiment["final_reserve_seconds"], budget*0.25)
        training_deadline = started + budget - final_reserve
    batches = BudgetBatches(config["batch_size"], config["block_size"], device)
    trials = []
    events = out / "events.jsonl"
    append_json(events, dict(event="start", method=method, budget_seconds=budget, rates=rates,
                             elapsed_seconds=now(device)-started))
    for index, rate in enumerate(rates):
        trial = Trial(config, batches, mode, experiment["seed"], rate, index)
        trials.append(trial)
        if matched:
            try:
                run_updates(trial, started, events)
            except FloatingPointError as error:
                raise CampaignFailed(f"{method} diverged at update {trial.step}: {error}") from error
            break
        slot_deadline = (started+budget*experiment["exploration_fraction"]*(index+1)/len(rates)
                         if search else training_deadline)
        run_window(trial, min(slot_deadline, training_deadline), started,
                   experiment["validation_every_seconds"], events)
    viable = [trial for trial in trials if trial.best is not None and not trial.failed]
    if not viable:
        raise CampaignFailed(f"No viable trial for {method}")
    winner = min(viable, key=lambda trial: trial.best["loss"])
    continuation_from = winner.step
    if search:
        append_json(events, dict(event="selection", trial_id=winner.trial_id,
                                 elapsed_seconds=now(device)-started, continued_from_step=winner.step,
                                 tuning_loss=winner.best["loss"]))
        run_window(winner, training_deadline, started, experiment["validation_every_seconds"], events)
    # Every current trial remains restartable; the selected checkpoint below is
    # model-only because its weights can precede the live optimizer state.
    for trial in trials:
        torch.save(trial.state_dict(), out / f"trial_{trial.trial_id}_continuation.pt")
    # Select the best checkpoint available across ALL explored trajectories.
    # Reporting-validation values are never used by this choice or the search.
    selected = min((t for t in trials if t.best is not None), key=lambda t: t.best["loss"])
    selected.model.load_state_dict(selected.best_weights)
    t = now(device)
    reporting = evaluate(selected.model, batches, "reporting")
    reporting_seconds = now(device)-t
    if not math.isfinite(reporting["loss"]):
        raise CampaignFailed(f"Nonfinite reporting loss for {method}")
    torch.save(dict(model=selected.best_weights, model_config=asdict(selected.model_config),
                    training_config=config, selection=selected.best, reporting=reporting), out / "selected.pt")
    elapsed = now(device)-started
    result = dict(method=method, seed=experiment["seed"], budget_seconds=budget,
                  elapsed_seconds=elapsed, unused_seconds=max(0, budget-elapsed) if not matched else None,
                  budget_met=elapsed <= budget if not matched else None,
                  protocol="matched_updates" if matched else "equal_wall_time",
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else None,
                  reporting=reporting, selection=selected.best,
                  parameters=sum(p.numel() for p in selected.model.parameters()), rates=rates,
                  selected_trial_id=selected.trial_id, continued_trial_id=winner.trial_id if search else None,
                  continued_from_step=continuation_from if search else None,
                  completed_steps=sum(trial.step for trial in trials),
                  training_tokens=sum(trial.step for trial in trials)*config["batch_size"]*config["block_size"],
                  reporting_seconds=reporting_seconds, splits=batches.split_metadata,
                  trials=[dict(id=t.trial_id, mode=t.mode, initial_lr=t.initial_lr, final_lr=t.lr,
                               steps=t.step, best=t.best, failed=t.failed, **t.stats) for t in trials])
    # Publishing the accounting report is outside the model-work budget.
    (out / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    cap = f"/{budget}" if not matched else ""
    print(f"{method}: {elapsed:.2f}{cap}s, reporting loss={reporting['loss']:.4f}, "
          f"accuracy={100*reporting['accuracy']:.2f}%, steps={result['completed_steps']}", flush=True)
    if result["budget_met"] is False:
        print("Budget exceeded: retain this result for audit, but exclude it from a fair winner claim.", flush=True)
    return result


def write_report(out):
    out = Path(out)
    results = [json.loads(p.read_text()) for p in sorted(out.glob("*/result.json"))]
    failures = [json.loads(p.read_text()) for p in sorted(out.glob("*/failure.json"))]
    if not results and not failures:
        raise ValueError("No completed campaigns")
    matched = bool(results) and results[0].get("protocol") == "matched_updates"
    rows = [dict(method=r["method"], budget_seconds=r["budget_seconds"], elapsed_seconds=r["elapsed_seconds"],
                 budget_met=r["budget_met"], reporting_loss=r["reporting"]["loss"],
                 reporting_accuracy=r["reporting"]["accuracy"], selection_loss=r["selection"]["loss"],
                 selected_step=r["selection"]["step"], completed_steps=r["completed_steps"],
                 training_tokens=r["training_tokens"], probes=sum(t["probes"] for t in r["trials"]),
                 probe_seconds=sum(t["probe_seconds"] for t in r["trials"]),
                 peak_cuda_allocated_bytes=r.get("peak_cuda_allocated_bytes")) for r in results]
    with (out / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["method"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Matched-update optimizer comparison" if matched else "# Equal-budget optimizer comparison", "",
             "One paired seed on the same model, data and hardware.", "",
             "| Method | Budget (s) | Used (s) | Reporting loss | Accuracy | Total updates | Probes |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        mark = " (OVER BUDGET)" if row["budget_met"] is False else ""
        cap = "—" if matched else f"{row['budget_seconds']:g}"
        lines.append(f"| {row['method']}{mark} | {cap} | {row['elapsed_seconds']:.2f} | "
                     f"{row['reporting_loss']:.4f} | {100*row['reporting_accuracy']:.2f}% | "
                     f"{row['completed_steps']} | {row['probes']} |")
    if matched:
        lines += ["", "Reporting uses the final update of every completed trajectory."]
    else:
        lines += ["", "The first validation half selects checkpoints. The second half supplies reporting metrics.",
                  "Each search explores the default and additional logged rates, then continues the winning live trial. "
                  "Total updates and tokens include discarded trials.",
                  "Elapsed time includes initialization, compute, validation and checkpoint I/O. Over-budget methods are flagged."]
    if failures:
        lines += ["", "## Failed campaigns", ""]
        lines.extend(f"- {r['method']} — {r['error']}" for r in failures)
    (out / "comparison.md").write_text("\n".join(lines)+"\n")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/budget.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    horizon = parser.add_mutually_exclusive_group()
    horizon.add_argument("--seconds", type=float, help="Override each method's elapsed-time cap")
    horizon.add_argument("--updates", type=int, help="Train each trajectory for exactly this many updates")
    parser.add_argument("--training-config", help="Override the model/training JSON file")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--methods", nargs="+", choices=METHODS)
    args = parser.parse_args()
    experiment = json.loads(Path(args.config).read_text())
    for attr, key in (("seconds", "seconds_per_method"), ("seed", "seed"), ("methods", "methods")):
        if getattr(args, attr) is not None:
            experiment[key] = getattr(args, attr)
    if (experiment["seconds_per_method"] < 1 or experiment["search_candidates"] < 2
            or not 0 < experiment["exploration_fraction"] < 1
            or experiment["validation_every_seconds"] <= 0 or experiment["final_reserve_seconds"] <= 0):
        raise ValueError("Invalid budget experiment settings")
    if args.training_config:
        experiment["training_config"] = args.training_config
    config = json.loads((ROOT / experiment["training_config"]).read_text())
    experiment["matched_updates"] = args.updates is not None
    if args.updates is not None:
        config["steps"] = args.updates
        if args.updates < 1 or any(m.endswith("_search") for m in experiment["methods"]):
            parser.error("--updates requires a positive count and single-trajectory methods")
    if any(m.startswith("scheduled_") for m in experiment["methods"]) and (args.updates or 0) < 120:
        parser.error("Scheduled Muon methods require --updates of at least 120")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {out}")
    order = list(experiment["methods"])
    random.Random(experiment["seed"]+230).shuffle(order)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT)
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=ROOT)
    manifest = dict(experiment=experiment, training_config=config, order=order, device=args.device,
                    python=platform.python_version(), torch=str(torch.__version__), platform=platform.platform(),
                    revision=revision.stdout.strip(), git_dirty=bool(dirty.stdout.strip()),
                    hardware=torch.cuda.get_device_name() if args.device == "cuda" else platform.machine(),
                    torch_cuda=torch.version.cuda,
                    cuda_capability=torch.cuda.get_device_capability() if args.device == "cuda" else None,
                    data=json.loads((DATA / "meta.json").read_text()),
                    source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for folder in ("curvature_muon", "third_party/nanogpt")
                                   for p in sorted((ROOT / folder).glob("*.py"))})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    limit = f"{args.updates} updates" if args.updates is not None else f"{experiment['seconds_per_method']} seconds"
    for method in order:
        print(f"Starting {method} ({limit})", flush=True)
        try:
            run_campaign(config, experiment, method, out / method, args.device)
        except CampaignFailed as error:
            (out / method / "failure.json").write_text(json.dumps(dict(method=method, error=str(error)), indent=2)+"\n")
            print(f"Failed {method}: {error}", flush=True)
        write_report(out)


if __name__ == "__main__":
    main()
