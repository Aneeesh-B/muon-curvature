"""Generate portable Markdown and CSV reports without plotting dependencies."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(root, output):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    runs = []
    for manifest_path in sorted(root.rglob("manifest.json")):
        folder = manifest_path.parent
        if not (folder / "final.pt").exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        evaluations = read_rows(folder / "eval.jsonl")
        final = evaluations[-1]
        probes = [row for row in read_rows(folder / "train.jsonl") if "probe" in row]
        variant = folder.name.rsplit("_seed", 1)[0]
        runs.append(dict(run=str(folder), variant=variant, mode=manifest["mode"], seed=manifest["seed"],
                         parameters=manifest["parameters"], steps=final["step"], tokens=final["tokens"],
                         initial_val_loss=evaluations[0]["val_loss"], final_val_loss=final["val_loss"],
                         best_val_loss=min(row["val_loss"] for row in evaluations),
                         probes=len(probes), invalid=sum(row["decision"] == "fallback" for row in probes),
                         clamped=sum(row["decision"] == "clamped" for row in probes), final_lr=final["lr"],
                         train_seconds=final["train_seconds"], probe_seconds=final["probe_seconds"],
                         probe_over_train=final["probe_seconds"]/max(final["train_seconds"], 1e-30),
                         wall_seconds=final["wall_seconds"]))
    if not runs:
        raise RuntimeError(f"No completed runs in {root}")
    with (output / "runs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(runs[0]))
        writer.writeheader()
        writer.writerows(runs)
    groups = defaultdict(list)
    for row in runs:
        groups[row["variant"]].append(row)
    lines = ["# Experiment results", "", "Validation cross entropy in nats per character; lower is better.", "",
             "| Variant | Seeds | Final validation loss (mean ± sample SD) | Probe / train time | Invalid probes | Clamped probes |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, rows in sorted(groups.items()):
        values = [r["final_val_loss"] for r in rows]
        display = (f"{statistics.mean(values):.4f} ± {statistics.stdev(values):.4f}"
                   if len(values) > 1 else f"{values[0]:.4f} (SD n/a)")
        overhead = statistics.mean(r["probe_over_train"] for r in rows)
        n = sum(r["probes"] for r in rows)
        lines.append(f"| {name} | {len(rows)} | {display} | {100*overhead:.1f}% | {sum(r['invalid'] for r in rows)}/{n} | {sum(r['clamped'] for r in rows)}/{n} |")
    lines += ["", f"Each run: {runs[0]['parameters']:,} parameters, {runs[0]['steps']:,} updates, {runs[0]['tokens']:,} training tokens.",
              "The runner keeps training tokens equal across methods. Probe batches are additional compute and are not counted as new training tokens.",
              "Probe / train time is measured synchronized probe duration divided by ordinary training duration within the same run; it excludes diagnostics, evaluation, startup, logging and checkpoint I/O.",
              "These short sequential timings are noisy and are not GPU throughput estimates.", "",
              "Fixed-rate baselines cover only the configurations shown, not globally tuned Muon. AdamW is a single-rate reference, not a tuned baseline.",
              "One-time and periodic modes use the same initial rate, companion AdamW configuration, safety bounds, initialization and batch stream.",
              "Sample SD across seeds is not a confidence interval. Test data is untouched."]
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def summarize_scans(path, output):
    rows = read_rows(Path(path))
    records = []
    for row in rows:
        saved = next(c for c in row["candidates"] if c["lr"] == row["saved_lr"])
        best = min(row["candidates"], key=lambda c: c["fresh_train_loss"])
        records.append(dict(anchor=row["anchor_step"], age=row["age"], step=row["step"],
                            saved_lr=row["saved_lr"], current_lr=row["current_lr"],
                            curvature_status=row["measurement"]["status"],
                            saved_probe_delta=saved["probe_loss_delta"],
                            saved_fresh_delta=saved["fresh_train_loss_delta"],
                            saved_val_delta=saved["val_loss_delta"],
                            best_fresh_lr=best["lr"], best_fresh_delta=best["fresh_train_loss_delta"],
                            fresh_regret=saved["fresh_train_loss"]-best["fresh_train_loss"],
                            predicted_delta=saved["predicted_delta"]))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "scans.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    lines = ["# Directional scans", "", "Matrix-only counterfactuals; companion AdamW and decay are frozen.", "",
             "| Anchor | Age | Saved LR | Probe Δ loss | Fresh train Δ loss | Best scanned fresh LR | Fresh regret |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in records:
        lines.append(f"| {row['anchor']} | {row['age']} | {row['saved_lr']:.5g} | {row['saved_probe_delta']:+.4f} | {row['saved_fresh_delta']:+.4f} | {row['best_fresh_lr']:.5g} | {row['fresh_regret']:.4f} |")
    lines += ["", "Negative delta is an improvement. Regret is saved-step loss minus the best loss among scanned candidates, including zero.",
              "At each age, the direction is rebuilt from the current gradient and momentum on a fixed-rate Muon trajectory; only the saved scalar is stale.",
              "The fresh minibatches are independently sampled training windows; validation windows are separate. Within a scan all candidates share windows and random masks.",
              "The raw JSONL includes local quadratic predictions, current estimates, validation deltas, and probe-distance sweeps at anchors.",
              "These one-step counterfactuals do not measure branching multi-step training continuations."]
    (output / "scans.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", help="Directory of completed training runs")
    parser.add_argument("--scans", help="Optional scans.jsonl path")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.root:
        summarize(args.root, args.out)
    if args.scans:
        summarize_scans(args.scans, args.out)
