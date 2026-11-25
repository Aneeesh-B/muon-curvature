"""5090 production protocol (docs/5090-overnight-experiment-plan.md).

Six arms under three cumulative estimated-matmul-FLOP caps, three or more paired
seeds, a frozen-direction mechanism check on fixed Muon, a common-update
isolation check, optional ablations (backtracking, scheduled probing, replay),
and reserved-test scoring after everything else is frozen.

Every campaign writes its own folder with result.json; a rerun skips finished
campaigns and moves an interrupted folder aside, so the queue is resumable.
"""
import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

from .budget import METHOD_MODES, BudgetBatches, Trial, evaluate, now
from .data import DATA, ROOT
from .diagnostics import append_json
from .optimizer import (displaced, dot, independent_probe, probe, replay_rng, rng_state, select_lr,
                        split_parameters)


# ----------------------------------------------------------------------------
# FLOP accounting (plan section 3)

def forward_flops(config, vocab, batch):
    """Dense-transformer matmul FLOPs of one forward on `batch` context windows."""
    B, T, L, d, V = batch, config["block_size"], config["n_layer"], config["n_embd"], vocab
    return 2*B*T*(12*L*d*d + d*V) + 4*B*L*T*T*d


def newton_schulz_flops(shapes, steps=5):
    total = 0
    for rows, cols in shapes:
        r, c = min(rows, cols), max(rows, cols)
        total += 4*r*r*c + 2*r*r*r
    return steps*total


@dataclass(frozen=True)
class Costs:
    forward: int
    adamw_update: int
    newton_schulz: int
    muon_update: int
    probe_pass: int
    tuning_validation: int
    reporting_validation: int
    test_evaluation: int
    windows: dict


def make_costs(config, vocab, shapes, windows):
    B, T, d = config["batch_size"], config["block_size"], config["n_embd"]
    f = forward_flops(config, vocab, B)
    ns = newton_schulz_flops(shapes)
    window = forward_flops(config, vocab, 1)
    return Costs(forward=f, adamw_update=3*f, newton_schulz=ns, muon_update=3*f+ns,
                 probe_pass=3*f - 2*B*T*d*vocab,
                 tuning_validation=window*windows["tuning"], reporting_validation=window*windows["reporting"],
                 test_evaluation=window*windows["test"], windows=dict(windows))


def step_cost(trial, costs):
    """Worst-case charge of the next update, including any due probe or guard."""
    passes, forwards = trial.probe_work()
    base = costs.adamw_update if trial.mode == "adamw" else costs.muon_update
    return base + passes*costs.probe_pass + forwards*costs.forward


def charged(trial, record, costs):
    """Actual charge of a completed update: only the passes that ran."""
    base = costs.adamw_update if trial.mode == "adamw" else costs.muon_update
    return base + record.get("probe_passes", 0)*costs.probe_pass + record.get("guard_forwards", 0)*costs.forward


# ----------------------------------------------------------------------------
# Data with a sealed test split

class ProductionBatches(BudgetBatches):
    def __init__(self, batch_size, block_size, device):
        super().__init__(batch_size, block_size, device)
        test = np.load(DATA / "test.npy").astype(np.int64)
        self.test_windows = (len(test)-1)//block_size
        self.test_loaded = False

    def windows(self):
        T = self.block_size
        return dict(tuning=(len(self.arrays["tuning"])-1)//T, reporting=(len(self.arrays["reporting"])-1)//T,
                    test=self.test_windows)

    def unseal_test(self):
        self.arrays["test"] = torch.from_numpy(np.load(DATA / "test.npy").astype(np.int64))
        self.test_loaded = True

    def evaluation_batches(self, split):
        if split == "test":
            if not self.test_loaded:
                raise RuntimeError("The test split is sealed until the test stage")
            data = self.arrays["test"]
            count = (len(data)-1)//self.block_size
            for base in range(0, count, self.batch_size):
                starts = [i*self.block_size for i in range(base, min(base+self.batch_size, count))]
                x = torch.stack([data[i:i+self.block_size] for i in starts])
                y = torch.stack([data[i+1:i+1+self.block_size] for i in starts])
                yield x.to(self.device), y.to(self.device)
            return
        yield from super().evaluation_batches(split)


def clean(x):
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    return x


def write_json(path, value):
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False)+"\n")


class Clock:
    """Synchronized timing split into charged work and off-budget measurement."""
    def __init__(self, device):
        self.device = device
        self.start = now(device)
        self.work = 0.0
        self.off_budget = 0.0

    def elapsed(self):
        return now(self.device)-self.start


def evaluate_split(trial, batches, split, clock, ledger, costs):
    t = now(trial.device)
    metrics = evaluate(trial.model, batches, split)
    seconds = now(trial.device)-t
    if split == "tuning":
        clock.work += seconds
    else:
        clock.off_budget += seconds
        ledger["reporting_flops"] += costs.reporting_validation
    return metrics


def model_weights(trial):
    return {k: v.detach().cpu().clone() for k, v in trial.model.state_dict().items()}


def save_weights(trial, path):
    torch.save(dict(model=model_weights(trial), step=trial.step, mode=trial.mode, seed=trial.seed,
                    initial_lr=trial.initial_lr, lr=trial.lr, model_config=asdict(trial.model_config)), path)


def trial_summary(trial):
    probes = trial.stats["probes"]
    return dict(id=trial.trial_id, mode=trial.mode, initial_lr=trial.initial_lr, final_lr=trial.lr,
                steps=trial.step, failed=trial.failed, invalid_rate=trial.stats["fallbacks"]/probes if probes else None,
                clamp_rate=trial.stats["clamps"]/probes if probes else None, **trial.stats)


def train_one(trial, costs, clock, events, extra=None):
    """One charged update. Returns its charge, or None if the trial diverged."""
    t = now(trial.device)
    try:
        record = trial.train_step()
    except FloatingPointError as error:
        trial.failed = f"{error} at update {trial.step}"
        append_json(events, dict(event="failed", trial_id=trial.trial_id, step=trial.step, reason=trial.failed))
        clock.work += now(trial.device)-t
        return None
    clock.work += now(trial.device)-t
    charge = charged(trial, record, costs)
    append_json(events, dict(event="step", trial_id=trial.trial_id, charge=charge, **(extra or {}), **record))
    return charge


def cap_point(trial, label, cap, spent, out, batches, clock, ledger, costs, chars):
    reporting = None
    if not trial.failed:
        reporting = evaluate_split(trial, batches, "reporting", clock, ledger, costs)
    path = out / f"weights_{label}.pt"
    t = now(trial.device)
    save_weights(trial, path)
    clock.off_budget += now(trial.device)-t
    return dict(cap=label, cap_flops=cap, flops=spent, unused_flops=cap-spent, eligible=spent <= cap and not trial.failed,
                selected_trajectory_steps=trial.step, training_characters=chars, lr=trial.lr,
                reporting=reporting, failed=trial.failed, weights=path.name,
                work_seconds=clock.work, elapsed_seconds=clock.elapsed())


# ----------------------------------------------------------------------------
# Mechanism check (plan section 4, step 5)

def mechanism_check(trial, batches, config, mechanism, seed, out):
    """Score proposals and a fixed grid at the CURRENT pre-update state, off-trajectory.

    The ordinary batch is peeked (not drawn), probes restore weights exactly,
    prepare() never advances momentum, and the trajectory's own RNG is replayed,
    so the fixed-Muon trajectory continues bit-for-bit as if unobserved.
    """
    step, model, c = trial.step, trial.model, config
    torch.save({k: v for k, v in trial.state_dict().items() if k != "best_weights"},
               out / f"mechanism_state_{step}.pt")
    peek = torch.Generator()
    peek.set_state(trial.generator.get_state())
    batch = batches.get("train", peek)
    diagnostic = torch.Generator().manual_seed(90_000 + 1000*seed + step)
    calibration = batches.get("train", diagnostic)
    fresh = [batches.get("train", diagnostic) for _ in range(mechanism["fresh_batches"])]
    with replay_rng(trial.rng):
        model.zero_grad(set_to_none=True)
        state = rng_state()
        model(*batch)[1].backward()
        direction = trial.muon.prepare()
        same = probe(direction, lambda: model(*batch)[1], state, c["relative_distance"])
        independent = independent_probe(direction, lambda: model(*calibration)[1], rng_state(),
                                        c["relative_distance"])
        model.zero_grad(set_to_none=True)

        @torch.no_grad()
        def fresh_loss(rate):
            with displaced(direction.params, direction.updates, -rate):
                return sum(model(*b)[1].item() for b in fresh)/len(fresh)

        grid = {rate: fresh_loss(rate) for rate in mechanism["grid"]}
        proposals = {}
        for name, measurement in (("same_batch", same), ("independent", independent)):
            rate, decision = select_lr(measurement, c["lr"], c["lr_min"], c["lr_max"], c["safety"])
            loss = grid[rate] if rate in grid else fresh_loss(rate)
            proposals[name] = dict(rate=rate, decision=decision, measurement=measurement, fresh_loss=loss)
    base, best_rate = grid[0.0], min(grid, key=grid.get)
    for value in proposals.values():
        value.update(delta=value["fresh_loss"]-base, helps_vs_zero=value["fresh_loss"] < base,
                     helps_vs_fixed=value["fresh_loss"] < grid[c["lr"]],
                     regret=value["fresh_loss"]-grid[best_rate])
    return dict(seed=seed, step=step, base_fresh_loss=base, fixed_rate=c["lr"],
                grid=[dict(rate=r, fresh_loss=l, delta=l-base) for r, l in grid.items()],
                best_grid_rate=best_rate, best_grid_loss=grid[best_rate], proposals=proposals)


# ----------------------------------------------------------------------------
# Campaigns

def prepare_folder(out):
    out = Path(out)
    if (out / "result.json").exists():
        return None
    if out.exists():
        out.rename(out.with_name(out.name + f".incomplete-{int(time.time())}"))
    out.mkdir(parents=True)
    return out


def start_campaign(device):
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    return Clock(device), dict(reporting_flops=0)


def finish(result, out, clock, ledger, device):
    result.update(elapsed_seconds=clock.elapsed(), work_seconds=clock.work, off_budget_seconds=clock.off_budget,
                  reporting_flops=ledger["reporting_flops"],
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else None)
    write_json(out / "result.json", result)
    return result


def run_trajectory(name, config, batches, costs, caps, seed, out, device, mechanism=None, isolation_step=None,
                   lr=None):
    """One live trajectory saved at the latest completed update at or below each cap."""
    out = prepare_folder(out)
    if out is None:
        return None
    clock, ledger = start_campaign(device)
    events = out / "events.jsonl"
    trial = Trial(config, batches, METHOD_MODES[name], seed, config["lr"] if lr is None else lr)
    chars_per_update = config["batch_size"]*config["block_size"]
    spent, points, checks, isolation, done = 0, [], [], None, set()
    for label, cap in caps:
        while not trial.failed:
            if mechanism and trial.step in mechanism["anchors"] and trial.step not in done:
                t = now(device)
                checks.append(mechanism_check(trial, batches, config, mechanism, seed, out))
                done.add(trial.step)
                clock.off_budget += now(device)-t
            if isolation_step is not None and trial.step == isolation_step and isolation is None:
                isolation = dict(step=trial.step, flops=spent, lr=trial.lr,
                                 reporting=evaluate_split(trial, batches, "reporting", clock, ledger, costs),
                                 stats=copy.deepcopy(trial.stats))
                save_weights(trial, out / f"weights_update{trial.step}.pt")
                isolation["weights"] = f"weights_update{trial.step}.pt"
            cost = step_cost(trial, costs)
            if spent + cost > cap:
                break
            charge = train_one(trial, costs, clock, events, dict(flops_before=spent))
            if charge is None:
                break
            assert charge <= cost, (charge, cost)
            spent += charge
        points.append(cap_point(trial, label, cap, spent, out, batches, clock, ledger, costs,
                                trial.step*chars_per_update))
        print(f"  {name} seed={seed} {label}: step={trial.step} flops={spent:.4e} "
              f"reporting={points[-1]['reporting']['loss'] if points[-1]['reporting'] else float('nan'):.4f}",
              flush=True)
    if checks:
        write_json(out / "mechanism.json", checks)
    result = dict(kind="trajectory", arm=name, mode=trial.mode, seed=seed, points=points, isolation=isolation,
                  trial=trial_summary(trial), mechanism=checks or None,
                  probe_over_train=trial.stats["probe_seconds"]/max(trial.stats["train_seconds"], 1e-30))
    return finish(result, out, clock, ledger, device)


def grid_edge(method, grid, index):
    keys = [rate[0] if isinstance(rate, (list, tuple)) else rate for rate in grid]
    key = keys[index]
    return key in (min(keys), max(keys))


def run_search(name, config, batches, costs, grid, slot_percent, seed, label, cap, out, device, tag=None):
    """Five candidates x slot_percent of the cap (incl. one tuning pass), then continue the winner."""
    out = prepare_folder(out)
    if out is None:
        return None
    clock, ledger = start_campaign(device)
    events = out / "events.jsonl"
    mode = METHOD_MODES[name]
    slot = cap*slot_percent//100
    spent, trials, candidates = 0, [], []
    for index, rate in enumerate(grid):
        trial = Trial(config, batches, mode, seed, tuple(rate) if isinstance(rate, list) else rate, index)
        used = 0
        while True:
            cost = step_cost(trial, costs)
            if used + cost + costs.tuning_validation > slot:
                break
            charge = train_one(trial, costs, clock, events, dict(phase="explore"))
            if charge is None:
                break
            used += charge
        tuning = None
        if not trial.failed:
            tuning = evaluate_split(trial, batches, "tuning", clock, ledger, costs)
            used += costs.tuning_validation
            append_json(events, dict(event="tuning", trial_id=index, step=trial.step, tuning=tuning))
        spent += used
        trials.append(trial)
        candidates.append(dict(id=index, rate=rate, steps=trial.step, flops=used, slot_flops=slot,
                               tuning=tuning, failed=trial.failed))
    viable = [c for c in candidates if c["tuning"] and math.isfinite(c["tuning"]["loss"])]
    if not viable:
        result = dict(kind="search", arm=tag or name, seed=seed, cap=label, cap_flops=cap, candidates=candidates,
                      failed="no viable candidate", points=[])
        return finish(result, out, clock, ledger, device)
    chosen = min(viable, key=lambda c: c["tuning"]["loss"])
    winner = trials[chosen["id"]]
    for trial in trials:
        if trial is not winner:
            trial.model.cpu()
    continued_from = winner.step
    append_json(events, dict(event="selection", trial_id=winner.trial_id, continued_from_step=continued_from,
                             flops=spent))
    while not winner.failed:
        cost = step_cost(winner, costs)
        if spent + cost > cap:
            break
        charge = train_one(winner, costs, clock, events, dict(phase="continue"))
        if charge is None:
            break
        spent += charge
    chars = sum(t.step for t in trials)*config["batch_size"]*config["block_size"]
    point = cap_point(winner, label, cap, spent, out, batches, clock, ledger, costs, chars)
    print(f"  {name} seed={seed} {label}: winner={chosen['rate']} steps={winner.step} flops={spent:.4e} "
          f"reporting={point['reporting']['loss'] if point['reporting'] else float('nan'):.4f}", flush=True)
    result = dict(kind="search", arm=tag or name, mode=mode, seed=seed, cap=label, cap_flops=cap, slot_flops=slot,
                  candidates=candidates, selected=chosen, continued_from_step=continued_from,
                  winner_on_grid_edge=grid_edge(name, grid, chosen["id"]), points=[point],
                  trials=[trial_summary(t) for t in trials])
    return finish(result, out, clock, ledger, device)


def run_matched(name, config, batches, costs, seed, horizon, out, device, lr=None, trace=None):
    """Fixed-horizon trajectory (scheduled ablations): final tuning + reporting scores."""
    out = prepare_folder(out)
    if out is None:
        return None
    clock, ledger = start_campaign(device)
    events = out / "events.jsonl"
    config = dict(config, steps=horizon)
    if trace is not None:
        config["lr_trace"] = list(trace)
    trial = Trial(config, batches, METHOD_MODES[name], seed, config["lr"] if lr is None else lr)
    spent, lrs = 0, []
    for _ in range(horizon):
        charge = train_one(trial, costs, clock, events)
        if charge is None:
            break
        spent += charge
        lrs.append(trial.lr)
    tuning = reporting = None
    if not trial.failed:
        tuning = evaluate_split(trial, batches, "tuning", clock, ledger, costs)
        reporting = evaluate_split(trial, batches, "reporting", clock, ledger, costs)
    save_weights(trial, out / "weights_final.pt")
    print(f"  {name} seed={seed} H={horizon} lr0={lr}: reporting="
          f"{reporting['loss'] if reporting else float('nan'):.4f}", flush=True)
    result = dict(kind="matched", arm=name, mode=trial.mode, seed=seed, horizon=horizon, initial_lr=trial.initial_lr,
                  flops=spent, steps=trial.step, tuning=tuning, reporting=reporting, failed=trial.failed,
                  lr_trace=lrs, trial=trial_summary(trial), weights="weights_final.pt",
                  probe_over_train=trial.stats["probe_seconds"]/max(trial.stats["train_seconds"], 1e-30))
    return finish(result, out, clock, ledger, device)


# ----------------------------------------------------------------------------
# Setup and stages

def setup(protocol, device, cap_set="caps"):
    config = json.loads((ROOT / protocol["training_config"]).read_text())
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    batches = ProductionBatches(config["batch_size"], config["block_size"], device)
    costs = costs_for(config, batches)
    caps = sorted(protocol[cap_set].items(), key=lambda kv: kv[1])
    return config, batches, costs, caps


def costs_for(config, batches):
    vocab = json.loads((DATA / "meta.json").read_text())["vocab_size"]
    with replay_rng(rng_state()):
        trial = Trial(config, batches, "fixed", 0, config["lr"])
    shapes = [tuple(p.shape) for p in split_parameters(trial.model)[0]]
    return make_costs(config, vocab, shapes, batches.windows())


def manifest(protocol, config, costs, device, out, extra=None):
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT)
    diff = subprocess.run(["git", "diff", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout
    status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=ROOT).stdout
    smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout if device == "cuda" else None
    value = dict(protocol=protocol, training_config=config, costs=asdict(costs), device=device,
                 python=platform.python_version(), torch=str(torch.__version__), torch_cuda=torch.version.cuda,
                 hardware=torch.cuda.get_device_name() if device == "cuda" else platform.machine(),
                 cuda_capability=torch.cuda.get_device_capability() if device == "cuda" else None,
                 tf32=dict(matmul=torch.backends.cuda.matmul.allow_tf32, cudnn=torch.backends.cudnn.allow_tf32),
                 deterministic=torch.are_deterministic_algorithms_enabled(), platform=platform.platform(),
                 revision=revision.stdout.strip(), git_status=status, git_diff_sha256=hashlib.sha256(diff.encode()).hexdigest(),
                 nvidia_smi=smi, free_disk_bytes=shutil.disk_usage(ROOT).free,
                 data=json.loads((DATA / "meta.json").read_text()),
                 source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for folder in ("curvature_muon", "third_party/nanogpt")
                                for p in sorted((ROOT / folder).glob("*.py"))},
                 started=time.strftime("%Y-%m-%dT%H:%M:%S%z"), **(extra or {}))
    out.mkdir(parents=True, exist_ok=True)
    (out / "git_diff.patch").write_text(diff)
    write_json(out / "manifest.json", value)


def core_campaigns(protocol, caps):
    items = [("trajectory", arm, None) for arm in protocol["single_arms"]]
    items += [("search", arm, label) for arm in protocol["search_arms"] for label, _ in caps]
    return items


def run_core(protocol, device, seeds, root, cap_set="caps"):
    config, batches, costs, caps = setup(protocol, device, cap_set)
    for seed in seeds:
        order = core_campaigns(protocol, caps)
        random.Random(seed+230).shuffle(order)
        folder = root / f"seed{seed}"
        folder.mkdir(parents=True, exist_ok=True)
        write_json(folder / "order.json", [list(o) for o in order])
        print(f"seed {seed}: order {[o[1] + ('@' + o[2] if o[2] else '') for o in order]}", flush=True)
        for kind, arm, label in order:
            if kind == "trajectory":
                mechanism = protocol["mechanism"] if arm == protocol["mechanism"]["arm"] else None
                isolation = protocol["isolation_step"] if arm in protocol["isolation_arms"] else None
                run_trajectory(arm, config, batches, costs, caps, seed, folder / arm, device, mechanism, isolation)
            else:
                cap = dict(caps)[label]
                run_search(arm, config, batches, costs, protocol["search_grids"][arm], protocol["slot_percent"],
                           seed, label, cap, folder / f"{arm}_{label}", device)


def run_ablations(protocol, device, seeds, root, cap_set="caps"):
    config, batches, costs, caps = setup(protocol, device, cap_set)
    ab = protocol["ablations"]
    for seed in seeds:
        folder = root / f"seed{seed}"
        for arm in ab["backtracking_arms"]:
            run_trajectory(arm, config, batches, costs, caps, seed, folder / arm, device,
                           isolation_step=protocol["isolation_step"])
        for horizon in ab["scheduled_horizons"]:
            for arm in ab["scheduled_arms"]:
                run_matched(arm, config, batches, costs, seed, horizon, folder / f"{arm}_H{horizon}", device)
            for peak in ab["scheduled_search_peaks"]:
                run_matched("scheduled_search", config, batches, costs, seed, horizon,
                            folder / f"scheduled_search_H{horizon}_lr{peak:g}", device, lr=peak)


def replay_trace(root, protocol, horizon):
    ab = protocol["ablations"]
    traces = []
    for seed in ab["replay_development_seeds"]:
        result = json.loads((root / f"seed{seed}" / f"{ab['replay_source_arm']}_H{horizon}" / "result.json").read_text())
        if result["failed"] or len(result["lr_trace"]) != horizon:
            raise RuntimeError(f"Replay source failed for seed {seed}")
        traces.append(np.log(np.array(result["lr_trace"])))
    return np.exp(np.mean(traces, axis=0)).tolist()


def run_replay(protocol, device, seeds, root, cap_set="caps"):
    """Follow-up: frozen geometric-mean LR curve from two development seeds, replayed on eval seeds."""
    config, batches, costs, caps = setup(protocol, device, cap_set)
    ab = protocol["ablations"]
    for horizon in ab["scheduled_horizons"]:
        for seed in ab["replay_development_seeds"]:
            run_matched(ab["replay_source_arm"], config, batches, costs, seed, horizon,
                        root / f"seed{seed}" / f"{ab['replay_source_arm']}_H{horizon}", device)
        trace = replay_trace(root, protocol, horizon)
        write_json(root / f"replay_trace_H{horizon}.json", trace)
        for seed in seeds:
            run_matched("scheduled_replay_muon", config, batches, costs, seed, horizon,
                        root / f"seed{seed}" / f"scheduled_replay_muon_H{horizon}", device, trace=trace)


def run_followup(protocol, device, seeds, root, cap_set="caps"):
    """Follow-up (not in the frozen plan): extended grids where a production search won on a grid edge."""
    config, batches, costs, caps = setup(protocol, device, cap_set)
    for seed in seeds:
        for arm, grid in protocol["followup_grids"].items():
            for label, cap in caps:
                run_search(arm, config, batches, costs, grid, protocol["slot_percent"], seed, label, cap,
                           root / f"seed{seed}" / f"{arm}_followup_{label}", device, tag=f"{arm}_followup")


def run_test(protocol, device, root, cap_set="caps"):
    """Score every saved reported checkpoint on the reserved test split (after freezing)."""
    config, batches, costs, caps = setup(protocol, device, cap_set)
    batches.unseal_test()
    path = root / "test_scores.json"
    scores = json.loads(path.read_text()) if path.exists() else {}
    for weights in sorted(root.glob("seed*/*/weights_*.pt")):
        key = str(weights.relative_to(root))
        if key in scores:
            continue
        saved = torch.load(weights, map_location="cpu", weights_only=False)
        with replay_rng(rng_state()):
            trial = Trial(config, batches, "fixed", 0, config["lr"])
        trial.model.load_state_dict(saved["model"])
        params = list(trial.model.parameters())
        finite = all(torch.isfinite(p).all() for p in params)
        scores[key] = evaluate(trial.model, batches, "test") if finite else None
        write_json(path, scores)
    print(f"Scored {len(scores)} checkpoints on test", flush=True)


def run_fidelity(protocol, device, seeds, root, repeats=10):
    """Q1 on the production model: finite-difference q versus an exact autodiff HVP.

    Uses the saved pre-update fixed-Muon states of the mechanism check. Both the
    independent calibration batch and the same (training) batch are measured, at
    three probe distances, so a negative same-batch estimate can be checked
    against exact curvature. HVP needs double backward, hence math attention;
    the finite difference is timed with both the default and the math backend.
    """
    from .audit import math_attention
    config, batches, costs, caps = setup(protocol, device)
    path = root / "fidelity.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    done = {(r["seed"], r["step"]) for r in rows}
    for seed in seeds:
        for state_path in sorted((root / f"seed{seed}" / protocol["mechanism"]["arm"]).glob("mechanism_state_*.pt"),
                                 key=lambda p: int(p.stem.rsplit("_", 1)[1])):
            step = int(state_path.stem.rsplit("_", 1)[1])
            if (seed, step) in done:
                continue
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            state["best_weights"] = None
            trial = Trial(config, batches, "fixed", seed, config["lr"])
            trial.load_state_dict(state)
            model = trial.model
            peek = torch.Generator()
            peek.set_state(trial.generator.get_state())
            batch = batches.get("train", peek)
            calibration = batches.get("train", torch.Generator().manual_seed(90_000 + 1000*seed + step))
            with replay_rng(trial.rng):
                model.zero_grad(set_to_none=True)
                model(*batch)[1].backward()
                direction = trial.muon.prepare()
                model.zero_grad(set_to_none=True)
                row = dict(seed=seed, step=step)
                for name, data in (("independent", calibration), ("same_batch", batch)):
                    closure = lambda data=data: model(*data)[1]

                    def fd(rho=config["relative_distance"]):
                        return independent_probe(direction, closure, rng_state(), rho)

                    def hvp():
                        with torch.enable_grad():
                            gradients = torch.autograd.grad(closure(), direction.params, create_graph=True)
                            slope = sum((g*d).sum() for g, d in zip(gradients, direction.updates))
                            product = torch.autograd.grad(slope, direction.params)
                        return dict(s=slope.item(), q=dot(direction.updates, product))

                    def timed(fn):
                        for _ in range(3):
                            fn()
                        times = []
                        for _ in range(repeats):
                            t = now(device)
                            fn()
                            times.append(now(device)-t)
                        return float(np.median(times))

                    seconds = dict(finite_difference_default=timed(fd))
                    with math_attention():
                        exact = hvp()
                        seconds.update(finite_difference_math=timed(fd), hvp_math=timed(hvp))
                        distances = {str(rho): fd(rho) for rho in (1e-4, 1e-3, 1e-2)}
                    default = fd()
                    row[name] = dict(exact=exact, finite_difference=distances, finite_difference_default_backend=default,
                                     relative_q_error={k: (abs(v["q"]-exact["q"])/max(abs(exact["q"]), 1e-30)
                                                           if v["q"] is not None else None) for k, v in distances.items()},
                                     exact_eta=exact["s"]/exact["q"] if exact["q"] > 0 and exact["s"] > 0 else None,
                                     median_seconds=seconds)
            rows.append(row)
            write_json(path, rows)
            print(f"  fidelity seed={seed} step={step}: independent q exact={row['independent']['exact']['q']:.4g} "
                  f"fd={row['independent']['finite_difference']['0.001']['q']:.4g}; same-batch q exact="
                  f"{row['same_batch']['exact']['q']:.4g} fd={row['same_batch']['finite_difference']['0.001']['q']:.4g}",
                  flush=True)
    return rows


def verify_paths(device, out):
    """Plan step 2 on the 0.81M config: probe arms to update 150+, streams, counters, search continuation."""
    out = out / time.strftime("%Y%m%dT%H%M%S")  # always a fresh check
    out.mkdir(parents=True)
    config = json.loads((ROOT / "configs/tiny.json").read_text())
    torch.use_deterministic_algorithms(True)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    batches = ProductionBatches(config["batch_size"], config["block_size"], device)
    costs = costs_for(config, batches)
    report = dict(costs=asdict(costs))
    trials = {}
    for name in ("fixed_muon", "same_batch_muon", "independent_muon"):
        trial = Trial(config, batches, METHOD_MODES[name], 0, config["lr"])
        spent, expected = 0, 0
        for _ in range(151):
            predicted = step_cost(trial, costs)
            record = trial.train_step()
            charge = charged(trial, record, costs)
            assert charge == predicted, (name, trial.step, charge, predicted)
            spent += charge
        passes = dict(fixed_muon=0, same_batch_muon=1, independent_muon=2)[name]
        expected = 151*costs.muon_update + (2 if passes else 0)*passes*costs.probe_pass
        assert spent == expected, (name, spent, expected)
        assert trial.stats["probes"] == (2 if passes else 0), trial.stats
        trials[name] = trial
        report[name] = dict(flops=spent, expected=expected, stats=trial.stats, final_lr=trial.lr)
    states = [t.generator.get_state() for t in trials.values()]
    assert all(torch.equal(states[0], s) for s in states), "ordinary batch streams differ"
    fresh = torch.Generator().manual_seed(2000)
    assert torch.equal(trials["same_batch_muon"].calibration_generator.get_state(), fresh.get_state())
    assert not torch.equal(trials["independent_muon"].calibration_generator.get_state(), fresh.get_state())
    report["ordinary_streams_match"] = True
    # Baseline optimizers and search continuation at a small cap.
    protocol = json.loads((ROOT / "configs/production.json").read_text())
    for name in protocol["search_arms"]:
        result = run_search(name, config, batches, costs, protocol["search_grids"][name], protocol["slot_percent"],
                            0, "verify", 60*costs.muon_update, out / name, device)
        assert all(c["flops"] <= result["slot_flops"] for c in result["candidates"])
        assert result["points"][0]["flops"] <= 60*costs.muon_update
        assert result["points"][0]["selected_trajectory_steps"] > result["continued_from_step"]
        report[name] = dict(selected=result["selected"]["rate"], steps=result["points"][0]["selected_trajectory_steps"],
                            continued_from=result["continued_from_step"], flops=result["points"][0]["flops"])
    write_json(out / "verify.json", report)
    print("Path verification passed", flush=True)
    return report


def preflight(protocol, device, root):
    """Plan step 3: time probe arms and short searches on the production model, seed 0."""
    config, batches, costs, caps = setup(protocol, device)
    seed = protocol["preflight_seed"]
    timing = {}
    short = [("short", 160*costs.muon_update)]
    for arm in ("fixed_muon", "same_batch_muon", "independent_muon"):
        result = run_trajectory(arm, config, batches, costs, short, seed, root / arm, device)
        result = result or json.loads((root / arm / "result.json").read_text())
        timing[arm] = result["work_seconds"]/result["points"][0]["flops"]
    for arm in protocol["search_arms"]:
        result = run_search(arm, config, batches, costs, protocol["search_grids"][arm], protocol["slot_percent"],
                            seed, "short", 200*costs.muon_update, root / arm, device)
        result = result or json.loads((root / arm / "result.json").read_text())
        timing[arm] = result["work_seconds"]/result["points"][0]["flops"]
    total_caps = sum(c for _, c in caps)
    high = caps[-1][1]
    per_seed = sum(timing[a]*high for a in protocol["single_arms"]) + sum(timing[a]*total_caps
                                                                           for a in protocol["search_arms"])
    plan = dict(seconds_per_flop=timing, projected_core_seconds_per_seed=per_seed,
                projected_core_seconds_3_seeds_with_headroom=1.3*3*per_seed,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else None,
                chosen_cap_set="caps" if 1.3*3*per_seed <= 6*3600 else "fallback_caps")
    write_json(root / "plan.json", plan)
    print(json.dumps(clean(plan), indent=2), flush=True)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["verify", "preflight", "core", "ablations", "replay", "fidelity", "followup", "test"])
    parser.add_argument("--protocol", default="configs/production.json")
    parser.add_argument("--root", default="runs/production")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--cap-set", choices=["caps", "fallback_caps"], default="caps")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    protocol = json.loads((ROOT / args.protocol).read_text())
    root = ROOT / args.root
    root.mkdir(parents=True, exist_ok=True)
    seeds = args.seeds or protocol["seeds"]
    if args.stage == "verify":
        verify_paths(args.device, root / "verify")
        return
    config, batches, costs, caps = setup(protocol, args.device, args.cap_set)
    manifest(protocol, config, costs, args.device, root / f"manifest_{args.stage}_{int(time.time())}",
             dict(stage=args.stage, seeds=seeds, cap_set=args.cap_set))
    if args.stage == "preflight":
        preflight(protocol, args.device, root / "preflight")
    elif args.stage == "core":
        run_core(protocol, args.device, seeds, root, args.cap_set)
    elif args.stage == "ablations":
        run_ablations(protocol, args.device, seeds, root, args.cap_set)
    elif args.stage == "replay":
        run_replay(protocol, args.device, seeds, root, args.cap_set)
    elif args.stage == "followup":
        run_followup(protocol, args.device, seeds, root, args.cap_set)
    elif args.stage == "fidelity":
        run_fidelity(protocol, args.device, seeds, root)
    elif args.stage == "test":
        run_test(protocol, args.device, root, args.cap_set)


if __name__ == "__main__":
    main()
