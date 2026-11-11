"""Targeted correctness audit of the first problematic adaptive-Muon probes.

Replays only the prefix needed for checkpoints (default: 251 updates), checks
against the recorded run, evaluates independent FP64 autodiff Hessian-vector
products, and scans frozen directions without altering the training policy.
This is diagnostic compute, not another optimizer-performance comparison.
"""
import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from .budget import BudgetBatches, Trial
from .data import ROOT
from .optimizer import Direction, displaced, dot, probe, replay_rng, rng_state, select_lr, split_parameters


def math_attention():
    # Math attention supports the double backward needed for an independent HVP.
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        return sdpa_kernel(SDPBackend.MATH)
    except ImportError:  # PyTorch 2.2 compatibility
        return torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True,
                                              enable_mem_efficient=False)


def same_state(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same_state(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(same_state(x, y) for x, y in zip(a, b))
    return a == b


def exact_curvature(model, batch, updates):
    """Independent FP64 reference; D is frozen, aux coordinates are held fixed."""
    shadow = copy.deepcopy(model).double()
    parameters, _ = split_parameters(shadow)
    directions = [d.double() for d in updates]
    with math_attention():
        loss = shadow(*batch)[1]
        gradients = torch.autograd.grad(loss, parameters, create_graph=True)
        slope = sum((g*d).sum() for g, d in zip(gradients, directions))
        hvp = torch.autograd.grad(slope, parameters)
        q = dot(directions, hvp)
        s = slope.item()
        detached = [g.detach().clone() for g in gradients]
        direction = Direction(parameters, detached, directions, [])
        sweep = {str(rho): probe(direction, lambda: shadow(*batch)[1], rng_state(), rho)
                 for rho in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)}
    return dict(s=s, q=q, eta_raw=s/q if q > 0 and s > 0 else None,
                finite_difference_sweep=sweep)


def auxiliary_delta(trial, auxiliary):
    """Compute the actual companion AdamW displacement, then restore its state."""
    originals = [p.detach().clone() for p in auxiliary]
    state = copy.deepcopy(trial.adam.state_dict())
    try:
        trial.adam.step()
        return [p.detach().clone()-old for p, old in zip(auxiliary, originals)]
    finally:
        with torch.no_grad():
            for p, old in zip(auxiliary, originals):
                p.copy_(old)
        trial.adam.load_state_dict(state)


def inspect(trial, fresh_count):
    c, model = trial.config, trial.model
    if c["dropout"] != 0 or c["weight_decay"] != 0:
        raise ValueError("This targeted audit isolates the existing zero-dropout/decay pilot")
    generator = torch.Generator()
    generator.set_state(trial.generator.get_state())
    batch = trial.batches.get("train", generator)
    parameters, auxiliary = split_parameters(model)
    muon_before = copy.deepcopy(trial.muon.state_dict())
    adam_before = copy.deepcopy(trial.adam.state_dict())
    generator_before = trial.generator.get_state().clone()
    rng_before = copy.deepcopy(trial.rng)
    with replay_rng(trial.rng):
        state = rng_state()
        model.zero_grad(set_to_none=True)
        base_loss = model(*batch)[1]
        base_loss.backward()
        direction = trial.muon.prepare()
        weights = [p.detach().clone() for p in model.parameters()]
        gradients = [p.grad.detach().clone() for p in model.parameters()]
        with replay_rng(state):
            repeated = torch.autograd.grad(model(*batch)[1], parameters)
        zero_delta = math.sqrt(dot([a-b for a, b in zip(repeated, direction.gradients)],
                                   [a-b for a, b in zip(repeated, direction.gradients)]))
        measurement = probe(direction, lambda: model(*batch)[1], state, c["relative_distance"])
        accepted, decision = select_lr(measurement, trial.lr, c["lr_min"], c["lr_max"], c["safety"])
        single_sweep = {str(rho): probe(direction, lambda: model(*batch)[1], state, rho)
                        for rho in (1e-4, 1e-3, 1e-2)}
        reference = exact_curvature(model, batch, direction.updates)
        delta_aux = auxiliary_delta(trial, auxiliary)
        for p, old, grad in zip(model.parameters(), weights, gradients):
            assert torch.equal(p, old), "Audit failed to restore parameter"
            assert torch.equal(p.grad, grad), "Audit changed a training gradient"
        fresh_generator = torch.Generator().manual_seed(83000+trial.step)
        fresh = [trial.batches.get("train", fresh_generator) for _ in range(fresh_count)]
        fresh_s = []
        for item in fresh:
            g = torch.autograd.grad(model(*item)[1], parameters)
            fresh_s.append(dot(g, direction.updates))

        @torch.no_grad()
        def losses():
            return dict(probe_loss=model(*batch)[1].item(),
                        fresh_loss=sum(model(*item)[1].item() for item in fresh)/len(fresh))

        base = losses()
        rates = sorted(set([0, .005, .01, .02, .04, .08, .1, .15, accepted,
                            *([measurement["eta_raw"]] if measurement["eta_raw"] is not None else [])]))
        scans = []
        for eta in rates:
            predicted = -eta*measurement["s"] + .5*eta*eta*measurement["q"]
            with displaced(parameters, direction.updates, -eta):
                matrices_only = losses()
                with displaced(auxiliary, delta_aux, 1):
                    joint = losses()
            scans.append(dict(lr=eta, predicted_probe_delta=predicted,
                              matrix_probe_delta=matrices_only["probe_loss"]-base["probe_loss"],
                              matrix_fresh_delta=matrices_only["fresh_loss"]-base["fresh_loss"],
                              joint_probe_delta=joint["probe_loss"]-base["probe_loss"],
                              joint_fresh_delta=joint["fresh_loss"]-base["fresh_loss"]))
        for p, old, grad in zip(model.parameters(), weights, gradients):
            assert torch.equal(p, old), "Scan failed to restore parameter"
            assert torch.equal(p.grad, grad), "Scan changed a training gradient"
    assert same_state(muon_before, trial.muon.state_dict()), "Audit changed Muon state"
    assert same_state(adam_before, trial.adam.state_dict()), "Audit changed companion AdamW state"
    assert torch.equal(generator_before, trial.generator.get_state()), "Audit consumed training data"
    assert same_state(rng_before, trial.rng), "Audit changed trial RNG"
    return dict(step=trial.step, original_loss=base_loss.item(), previous_lr=trial.lr,
                measurement=measurement, accepted_lr=accepted, decision=decision,
                zero_displacement_gradient_difference_norm=zero_delta,
                fp32_distance_sweep=single_sweep, fp64_hvp=reference,
                q_relative_error=abs(measurement["q"]-reference["q"])/max(abs(reference["q"]), 1e-30),
                accepted_step_over_probe_distance=accepted/measurement["epsilon"],
                mean_fresh_s=sum(fresh_s)/len(fresh_s), fresh_batches=fresh_count, base=base, scans=scans)


def write_report(rows, out, elapsed):
    flat = [dict(step=row["step"], **candidate) for row in rows for candidate in row["scans"]]
    with (out / "scans.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(flat)
    lines = ["# Adaptive Muon correctness audit", "",
             f"Targeted replay and frozen-direction diagnostics only; {elapsed:.1f} CPU seconds. No full equal-budget comparison was rerun.", "",
             "Positive loss delta means worse. All candidate steps begin at the same weights and use the same frozen Muon direction.", "",
             "| Step | Accepted LR | Probe q | FP64 HVP q | Relative q error | Predicted probe delta | Actual probe delta | Fresh delta | Joint fresh delta |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        chosen = next(c for c in row["scans"] if c["lr"] == row["accepted_lr"])
        lines.append(f"| {row['step']} | {row['accepted_lr']:.5g} | {row['measurement']['q']:.4f} | "
                     f"{row['fp64_hvp']['q']:.4f} | {100*row['q_relative_error']:.2f}% | "
                     f"{chosen['predicted_probe_delta']:+.4f} | {chosen['matrix_probe_delta']:+.4f} | "
                     f"{chosen['matrix_fresh_delta']:+.4f} | {chosen['joint_fresh_delta']:+.4f} |")
    lines += ["", "Probe q is the actual FP32 finite difference at rho=0.001; reference q is an independent FP64 second derivative using math attention. Full FP32/FP64 distance sweeps are in audit.json.",
              "The matrices-only columns isolate the curvature model. Joint columns also apply the exact companion AdamW update from the same original gradients; all weights and optimizer state are then restored.",
              "Fresh losses average independent training minibatches, not validation/test data. No probe distance, safety factor, fallback policy or training optimizer was changed by this audit.",
              "The replay assertions check recorded training loss, probe numerator/curvature, rates and decisions. Further assertions check original gradients/weights remain intact and probing does not advance the optimizer state."]
    (out / "audit.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recorded", default="results/budget/raw/adaptive_muon/events.jsonl")
    parser.add_argument("--config", default="configs/tiny.json")
    parser.add_argument("--out", default="results/audit")
    parser.add_argument("--steps", nargs="+", type=int, default=[100, 150, 200, 250])
    parser.add_argument("--fresh-batches", type=int, default=8)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {out}")
    config = json.loads(Path(args.config).read_text())
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    recorded = {row["step"]: row for row in [json.loads(line) for line in Path(args.recorded).read_text().splitlines()]
                if row["event"] == "step"}
    started = time.perf_counter()
    batches = BudgetBatches(config["batch_size"], config["block_size"], "cpu")
    trial = Trial(config, batches, "periodic", 0, config["lr"])
    rows = []
    for step in range(max(args.steps)+1):
        if step in args.steps:
            row = inspect(trial, args.fresh_batches)
            rows.append(row)
            print(f"Audited step {step}: q={row['measurement']['q']:.4f}, "
                  f"HVP={row['fp64_hvp']['q']:.4f}", flush=True)
        current = trial.train_step()
        expected = recorded[step]
        assert current["train_loss"] == expected["train_loss"], f"Replay loss mismatch at {step}"
        assert current["lr"] == expected["lr"], f"Replay rate mismatch at {step}"
        if "probe" in expected:
            assert current["probe"] == expected["probe"], f"Replay probe mismatch at {step}"
            assert current["decision"] == expected["decision"]
    elapsed = time.perf_counter()-started
    payload = dict(config=config, replayed_updates=trial.step, seed=0, seconds=elapsed,
                   original_optimizer_unchanged=True, checks=rows,
                   source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted((ROOT / "curvature_muon").glob("*.py"))})
    (out / "audit.json").write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")
    write_report(rows, out, elapsed)


if __name__ == "__main__":
    main()
