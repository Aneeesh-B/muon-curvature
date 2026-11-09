"""Matched-token comparisons. Each run starts from the same seed/batch stream."""
import argparse
import json
from pathlib import Path

from .train import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--pilot", action="store_true", help="3 fixed rates + once + periodic k=10/50 + AdamW reference")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    variants = [("fixed", "fixed", {}), ("once", "once", {}), ("periodic", "periodic", {}), ("adamw", "adamw", {})]
    if args.pilot:
        variants = [(f"fixed_{lr}", "fixed", {"lr": lr}) for lr in (0.005, 0.02, 0.08)]
        variants += [("once", "once", {})]
        variants += [(f"periodic_{k}", "periodic", {"probe_every": k}) for k in (10, 50)]
        variants += [("adamw", "adamw", {})]
    for seed in args.seeds:
        for name, mode, changes in variants:
            run(dict(config, **changes), Path(args.out) / f"{name}_seed{seed}", mode, seed, args.device)


if __name__ == "__main__":
    main()
