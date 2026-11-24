"""Replay the fixed-Muon trajectory for off-budget frozen-direction scans."""

import argparse
import json
from pathlib import Path

import torch

from .budget import BudgetBatches, Trial, now
from .diagnostics import append_json
from .flop_budget import Work, forward_flops, write_json
from .optimizer import displaced, independent_probe, probe, replay_rng, rng_state, select_lr


ANCHORS = (100, 500, 1000)
LR_GRID = (0.0, 0.001, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.16, 0.2)


def inspect_direction(trial):
    config, model = trial.config, trial.model
    started = now(trial.device)
    generator = torch.Generator()
    generator.set_state(trial.generator.get_state())
    batch = trial.batches.get("train", generator)
    calibration = torch.Generator().manual_seed(26000 + 2000 * trial.seed + trial.step)
    fresh_rng = torch.Generator().manual_seed(73000 + 2000 * trial.seed + trial.step)
    independent_batch = trial.batches.get("train", calibration)
    fresh = [trial.batches.get("train", fresh_rng) for _ in range(8)]
    work = Work(config, trial.model_config.vocab_size)

    def gradient_objective(data):
        work.record("probe_gradient")
        return model(*data)[1]

    with replay_rng(trial.rng):
        state = rng_state()
        model.zero_grad(set_to_none=True)
        work.record("training_forward")
        ordinary_loss = model(*batch)[1]
        work.record("training_backward")
        ordinary_loss.backward()
        direction = trial.muon.prepare(work.orthogonalize)
        same = probe(direction, lambda: gradient_objective(batch), state, config["relative_distance"])
        separate = independent_probe(direction, lambda: gradient_objective(independent_batch),
                                     rng_state(), config["relative_distance"])
        rates = dict(fixed=config["lr"])
        decisions = {}
        for name, measurement in (("same_batch", same), ("independent_batch", separate)):
            rates[name], decisions[name] = select_lr(measurement, config["lr"], config["lr_min"],
                                                    config["lr_max"], config["safety"])
        losses = {}
        for rate in sorted(set(LR_GRID) | set(rates.values())):
            with torch.no_grad(), displaced(direction.params, direction.updates, -rate):
                losses[rate] = [model(*data)[1].item() for data in fresh]
    scores = {rate: sum(values) / len(values) for rate, values in losses.items()}
    baseline = scores[0.0]
    best = min(scores[rate] for rate in LR_GRID)
    forward = forward_flops(config["batch_size"], config, trial.model_config.vocab_size)
    return dict(seed=trial.seed, step=trial.step, ordinary_loss=ordinary_loss.item(),
                elapsed_seconds=now(trial.device) - started,
                diagnostic_flops=work.total + len(fresh) * len(scores) * forward,
                rates=rates, decisions=decisions, measurements=dict(same_batch=same, independent_batch=separate),
                candidates=[dict(lr=rate, fresh_losses=losses[rate], mean_loss=mean,
                                 delta=mean - baseline, grid_regret=mean - best)
                            for rate, mean in scores.items()])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.run / "manifest.json").read_text(encoding="utf-8"))
    config, device = manifest["config"], manifest["device"]
    if config["dropout"] or config["weight_decay"]:
        parser.error("The frozen-direction protocol requires zero dropout and weight decay")
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    batches = BudgetBatches(config["batch_size"], config["block_size"], device)
    out = args.run / "frozen-direction"
    out.mkdir(exist_ok=True)
    for seed in manifest["seeds"]:
        fixed = args.run / f"seed-{seed}" / "default_muon"
        measurements = json.loads((fixed / "result.json").read_text(encoding="utf-8"))["measurements"]
        last = measurements[-1]["selected_steps"]
        anchors = [step for step in ANCHORS if step < last]
        if not anchors:
            continue
        event_path = fixed / Path(measurements[-1]["checkpoint"]).parent / "events.jsonl"
        with event_path.open(encoding="utf-8") as log:
            reference = {}
            for line in log:
                row = json.loads(line)
                if row["event"] == "step" and row["step"] in anchors:
                    reference[row["step"]] = row["train_loss"]
        trial = Trial(config, batches, "fixed", seed, config["lr"],
                      work=Work(config, manifest["data"]["vocab_size"]))
        started = now(device)
        for step in anchors:
            while trial.step < step:
                trial.train_step()
            row = inspect_direction(trial)
            if row["ordinary_loss"] != reference[step]:
                raise RuntimeError(f"Fixed trajectory replay disagreed at seed {seed}, step {step}")
            row.update(replay_flops=trial.work.total, replay_elapsed_seconds=now(device) - started)
            checkpoint = out / f"seed-{seed}-step-{step}.pt"
            torch.save(trial.state_dict(), checkpoint)
            # Keep the single row strict even for a nonfinite rejected probe.
            temporary = checkpoint.with_suffix(".tmp")
            temporary.unlink(missing_ok=True)
            append_json(temporary, row)
            temporary.replace(checkpoint.with_suffix(".json"))
            print(f"Frozen direction seed={seed} step={step}", flush=True)
    results = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(out.glob("seed-*.json"))]
    write_json(out / "diagnostics.json", results)


if __name__ == "__main__":
    main()
