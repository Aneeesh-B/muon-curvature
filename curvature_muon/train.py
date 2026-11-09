import argparse
from dataclasses import asdict
import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
import time

# Required by deterministic CUDA GEMMs; set before torch initializes CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

from third_party.nanogpt.model import GPT, GPTConfig
from .data import Batches, DATA, ROOT
from .diagnostics import append_json, scan
from .optimizer import Muon, probe, rng_state, select_lr, split_parameters

SOURCE_SHA256 = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for folder in ("curvature_muon", "third_party/nanogpt")
                 for p in sorted((ROOT / folder).glob("*.py"))}


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def clock(device):
    sync(device)
    return time.perf_counter()


@torch.no_grad()
def evaluate(model, batches, count):
    model.eval()
    # Fixed validation windows at every checkpoint and across all runs.
    generator = torch.Generator().manual_seed(404)
    loss = sum(model(*batches.get("val", generator))[1].item() for _ in range(count)) / count
    model.train()
    return loss


def run(config, out, mode="periodic", seed=0, device="cpu", diagnostics=False):
    if device not in ("cpu", "cuda"):
        raise ValueError("The FP32 pilot supports cpu and cuda only")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if diagnostics and mode != "fixed":
        raise ValueError("Use fixed Muon as the common diagnostic/staleness trajectory")
    if diagnostics and config["diagnostic_steps"] and config["stale_ages"]:
        required = max(config["diagnostic_steps"]) + max(config["stale_ages"]) + 1
        if config["steps"] < required:
            raise ValueError(f"Need at least {required} steps to cover every diagnostic anchor/age")
    if mode not in ("fixed", "once", "periodic", "adamw"):
        raise ValueError("Unknown mode")
    if config["steps"] < 1 or config["probe_every"] < 1 or config["relative_distance"] <= 0:
        raise ValueError("steps, probe_every and relative_distance must be positive")
    if not (DATA / "meta.json").exists():
        raise RuntimeError("Prepare data first: python -m curvature_muon.data")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a nonempty run directory: {out}")
    torch.set_num_threads(config["threads"])
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    meta = json.loads((DATA / "meta.json").read_text())
    model_config = GPTConfig(**{k: config[k] for k in ("n_layer", "n_head", "n_embd", "block_size", "dropout")},
                             vocab_size=meta["vocab_size"], bias=False)
    model = GPT(model_config).to(device)
    hidden, auxiliary = split_parameters(model)
    muon = Muon(hidden, lr=config["lr"], momentum=config["momentum"], weight_decay=config["weight_decay"])
    adam_params = list(model.parameters()) if mode == "adamw" else auxiliary
    # Decay only 2D matrices, never normalization vectors. Same companion config
    # and tied-embedding ownership across all Muon modes.
    adam = torch.optim.AdamW([
        dict(params=[p for p in adam_params if p.ndim >= 2], weight_decay=config["weight_decay"]),
        dict(params=[p for p in adam_params if p.ndim < 2], weight_decay=0.0),
    ], lr=config["adam_lr"], betas=(0.9, 0.95), fused=False, foreach=False)
    batches = Batches(config["batch_size"], config["block_size"], device)
    train_generator = torch.Generator().manual_seed(1000 + seed)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
    manifest = dict(config=config, mode=mode, seed=seed, device=device, model=asdict(model_config),
                    parameters=sum(p.numel() for p in model.parameters()), dtype="float32",
                    python=platform.python_version(), torch=str(torch.__version__), platform=platform.platform(),
                    hardware=torch.cuda.get_device_name() if device == "cuda" else platform.processor(),
                    git_revision=revision.stdout.strip() or None, git_dirty=bool(dirty.stdout.strip()),
                    source_sha256=SOURCE_SHA256,
                    model_sha256=hashlib.sha256((ROOT / "third_party/nanogpt/model.py").read_bytes()).hexdigest(),
                    data=meta, diagnostics=diagnostics, deterministic=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    cumulative = dict(train_seconds=0.0, probe_seconds=0.0, diagnostic_seconds=0.0, eval_seconds=0.0)
    probe_count, anchors = 0, {}
    start = clock(device)

    def validation(step):
        t = clock(device)
        value = evaluate(model, batches, config["eval_batches"])
        cumulative["eval_seconds"] += clock(device) - t
        append_json(out / "eval.jsonl", dict(step=step, tokens=step*config["batch_size"]*config["block_size"],
                    val_loss=value, lr=muon.param_groups[0]["lr"] if mode != "adamw" else config["adam_lr"],
                    probes=probe_count, **cumulative, wall_seconds=clock(device)-start))
        print(f"{mode} seed={seed} step={step} val={value:.4f}", flush=True)

    def checkpoint(name, step):
        torch.save(dict(model=model.state_dict(), muon=muon.state_dict(), adam=adam.state_dict(),
                        model_config=asdict(model_config), config=config, step=step, mode=mode, seed=seed,
                        rng=rng_state(), train_generator=train_generator.get_state()), out / name)

    validation(0)
    for step in range(config["steps"]):
        t = clock(device)
        model.zero_grad(set_to_none=True)
        batch = batches.get("train", train_generator)
        forward_rng = rng_state()
        loss = model(*batch)[1]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite training loss at step {step}")
        loss.backward()
        direction = muon.prepare() if mode != "adamw" else None
        cumulative["train_seconds"] += clock(device) - t
        record = dict(step=step, train_loss=loss.item())
        if diagnostics:
            t = clock(device)
            if step in config["diagnostic_steps"]:
                measurement = probe(direction, lambda: model(*batch)[1], forward_rng, config["relative_distance"])
                recommendation, _ = select_lr(measurement, config["lr"], config["lr_min"], config["lr_max"], config["safety"])
                anchors[step] = recommendation
                checkpoint(f"checkpoint_{step}.pt", step)
            for anchor, saved_lr in anchors.items():
                if step-anchor in config["stale_ages"]:
                    scan(model, direction, batch, forward_rng, batches, config, step, anchor, saved_lr, out / "scans.jsonl")
            cumulative["diagnostic_seconds"] += clock(device) - t
        should_probe = step >= config["probe_start"] and (
            (mode == "once" and step == config["probe_start"]) or
            (mode == "periodic" and (step-config["probe_start"]) % config["probe_every"] == 0))
        if should_probe:
            t = clock(device)
            measurement = probe(direction, lambda: model(*batch)[1], forward_rng, config["relative_distance"])
            lr, decision = select_lr(measurement, muon.param_groups[0]["lr"], config["lr_min"], config["lr_max"], config["safety"])
            for group in muon.param_groups:
                group["lr"] = lr
            cumulative["probe_seconds"] += clock(device) - t
            probe_count += 1
            record.update(probe=measurement, decision=decision)
        t = clock(device)
        if mode != "adamw":
            muon.step(direction)
        adam.step()
        cumulative["train_seconds"] += clock(device) - t
        record["lr"] = muon.param_groups[0]["lr"] if mode != "adamw" else config["adam_lr"]
        append_json(out / "train.jsonl", record)
        if (step+1) % config["eval_every"] == 0 or step+1 == config["steps"]:
            validation(step+1)
    checkpoint("final.pt", config["steps"])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=["fixed", "once", "periodic", "adamw"], default="periodic")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--probe-every", type=int)
    parser.add_argument("--diagnostics", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    for key in ("lr", "steps", "probe_every"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    run(config, args.out, args.mode, args.seed, args.device, args.diagnostics)


if __name__ == "__main__":
    main()
