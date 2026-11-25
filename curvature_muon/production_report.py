"""Tables and the main figure for the 5090 production protocol.

Reads runs/production (result.json, events.jsonl, test_scores.json) and writes
results/production/{REPORT.md, *.csv, fig_loss_vs_flops.png/pdf, raw/}.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import statistics

from .data import ROOT

ARM_LABELS = {
    "fixed_muon": "Fixed Muon (0.02)",
    "muon_search": "Searched Muon",
    "same_batch_muon": "Same-batch curvature",
    "independent_muon": "Independent-batch curvature",
    "adamw_search": "Searched AdamW",
    "adago_search": "Searched AdaGO",
    "guarded_muon": "Independent + backtracking",
    "guarded_base_muon": "Backtracking, fixed 0.02 proposal",
    "adamw_search_followup": "Searched AdamW, extended grid (follow-up)",
    "adago_search_followup": "Searched AdaGO, extended grid (follow-up)",
}
# Reference categorical palette, fixed order (validated: adjacent CVD dE >= 9.1).
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
CAPS = ("low", "middle", "high")


def load(root):
    results = []
    for path in sorted(root.glob("seed*/*/result.json")):
        value = json.loads(path.read_text())
        value["_folder"] = path.parent
        results.append(value)
    return results


def fmt(x, digits=4):
    return "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{digits}f}"


def mean_sd(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def paired(a, b):
    """a - b over seeds present in both (negative favors a)."""
    seeds = sorted(set(a) & set(b))
    diffs = [a[s] - b[s] for s in seeds if a[s] is not None and b[s] is not None]
    if not diffs:
        return None
    m, sd = mean_sd(diffs)
    p = None
    if len(diffs) > 1 and sd > 0:
        from scipy import stats
        p = float(stats.ttest_1samp(diffs, 0.0).pvalue)
    return dict(n=len(diffs), mean=m, sd=sd, wins=sum(d < 0 for d in diffs), p=p, diffs=diffs)


def paired_text(d):
    if d is None:
        return "—"
    p = "—" if d["p"] is None else f"{d['p']:.3f}"
    return f"{d['mean']:+.4f} ± {d['sd']:.4f} ({d['wins']}/{d['n']} seeds lower, p={p})"


def core_rows(results, tests, protocol):
    rows = []
    arms = (protocol["single_arms"] + protocol["search_arms"] + protocol["ablations"]["backtracking_arms"]
            + [f"{a}_followup" for a in protocol.get("followup_grids", {})])
    for r in results:
        if r.get("arm") not in arms or r["kind"] not in ("trajectory", "search"):
            continue
        for point in r["points"]:
            key = f"{r['_folder'].relative_to(r['_folder'].parents[1])}/{point['weights']}"
            test = tests.get(key)
            rows.append(dict(
                seed=r["seed"], arm=r["arm"], cap=point["cap"], cap_flops=point["cap_flops"], flops=point["flops"],
                unused_flops=point["unused_flops"], eligible=point["eligible"],
                reporting_loss=point["reporting"]["loss"] if point["reporting"] else None,
                reporting_accuracy=point["reporting"]["accuracy"] if point["reporting"] else None,
                test_loss=test["loss"] if test else None, test_accuracy=test["accuracy"] if test else None,
                selected_trajectory_steps=point["selected_trajectory_steps"],
                training_characters=point["training_characters"],
                selected_rate=json.dumps(r["selected"]["rate"]) if r["kind"] == "search" else None,
                winner_on_grid_edge=r.get("winner_on_grid_edge"),
                elapsed_seconds=r["elapsed_seconds"] if r["kind"] == "search" else point["elapsed_seconds"],
                work_seconds=r["work_seconds"] if r["kind"] == "search" else point["work_seconds"],
                peak_cuda_allocated_bytes=r["peak_cuda_allocated_bytes"],
                probes=r["trial"]["probes"] if r["kind"] == "trajectory" else None,
                invalid_rate=r["trial"]["invalid_rate"] if r["kind"] == "trajectory" else None,
                clamp_rate=r["trial"]["clamp_rate"] if r["kind"] == "trajectory" else None,
                failed=point["failed"]))
    return rows


def by(rows, arm, cap, field):
    return {r["seed"]: r[field] for r in rows if r["arm"] == arm and r["cap"] == cap}


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def figure(rows, arms, out, field="reporting_loss", name="fig_loss_vs_flops"):
    """(a) loss vs cumulative FLOPs, all arms and caps; (b) the high cap per seed, zoomed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    muted, grid_color, spine = "#52514e", "#ecebe8", "#c8c7c2"
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 3.4), dpi=200, gridspec_kw=dict(width_ratios=[1.55, 1]))
    dodge = {arm: 1 + 0.035*(i - (len(arms)-1)/2) for i, arm in enumerate(arms)}
    cap_x = {}
    for i, arm in enumerate(arms):
        color, marker = COLORS[i], MARKERS[i]
        xs, ys = [], []
        for cap in CAPS:
            pts = [(r["flops"], r[field]) for r in rows if r["arm"] == arm and r["cap"] == cap and r[field] is not None]
            if not pts:
                continue
            x = statistics.fmean(p[0] for p in pts)
            cap_x.setdefault(cap, []).append(x)
            ax.scatter([x*dodge[arm]]*len(pts), [p[1] for p in pts], s=10, color=color, alpha=0.35, linewidths=0,
                       marker=marker, zorder=2)
            xs.append(x*dodge[arm])
            ys.append(statistics.fmean(p[1] for p in pts))
        if xs:
            ax.plot(xs, ys, color=color, lw=2, marker=marker, ms=5.5, mec="#fcfcfb", mew=1.2, zorder=3,
                    label=ARM_LABELS[arm])
        high = sorted((r["seed"], r[field]) for r in rows if r["arm"] == arm and r["cap"] == "high" and r[field] is not None)
        if high:
            bx.scatter([i]*len(high), [v for _, v in high], s=14, color=color, alpha=0.45, marker=marker, linewidths=0, zorder=2)
            m = statistics.fmean(v for _, v in high)
            bx.plot([i-0.28, i+0.28], [m, m], color=color, lw=2.2, zorder=3, solid_capstyle="round")
    ticks = [statistics.fmean(cap_x[c]) for c in CAPS if c in cap_x]
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t/1e12:.1f}T\n({c})" for t, c in zip(ticks, [c for c in CAPS if c in cap_x])])
    ax.minorticks_off()
    ax.set_xlim(ticks[0]/1.25, ticks[-1]*1.25)
    ax.set_xlabel("Cumulative estimated matmul FLOPs (all work charged)", fontsize=7.5, color=muted)
    ax.set_ylabel(("Reporting" if field.startswith("reporting") else "Test") + " cross-entropy (nats/char)",
                  fontsize=7.5, color=muted)
    ax.set_title("(a) Loss at each FLOP cap", fontsize=8, loc="left")
    bx.set_title("(b) High cap, each seed (bar = mean)", fontsize=8, loc="left")
    bx.set_xticks(range(len(arms)))
    short = dict(fixed_muon="Fixed\nMuon", muon_search="Muon\nsearch", same_batch_muon="Same-\nbatch",
                 independent_muon="Indep.-\nbatch", adamw_search="AdamW\nsearch", adago_search="AdaGO\nsearch")
    bx.set_xticklabels([short.get(a, a) for a in arms], fontsize=6)
    bx.set_xlim(-0.6, len(arms)-0.4)
    vals = [r[field] for r in rows if r["cap"] == "high" and r["arm"] in arms and r[field] is not None]
    lo = min(vals)
    focus = sorted(vals)[int(0.8*len(vals))]  # keep the cluster readable; outliers are clipped and marked
    bx.set_ylim(lo-0.01, max(focus+0.02, lo+0.08))
    top = bx.get_ylim()[1]
    for i, arm in enumerate(arms):
        above = [r for r in rows if r["arm"] == arm and r["cap"] == "high" and r[field] is not None and r[field] > top]
        if above:
            bx.annotate(f"{len(above)} above\n(max {max(r[field] for r in above):.2f})", (i, top), xytext=(0, -2),
                        textcoords="offset points", ha="center", va="top", fontsize=5, color=muted)
    for a in (ax, bx):
        a.tick_params(labelsize=6.5, colors=muted)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color(spine)
        a.grid(True, axis="y", color=grid_color, lw=0.6)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out / f"{name}.png")
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)


def guard_counts(folder):
    accepted = rejected = trials = 0
    path = folder / "events.jsonl"
    if not path.exists():
        return None
    for line in path.open():
        if '"guard"' not in line:
            continue
        row = json.loads(line)
        if "guard" in row:
            accepted += row["guard"]["status"] == "accepted"
            rejected += row["guard"]["status"] == "rejected"
            trials += row["guard"]["trials"]
    return dict(accepted=accepted, rejected=rejected, trials=trials)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="runs/production")
    parser.add_argument("--out", default="results/production")
    parser.add_argument("--protocol", default="configs/production.json")
    args = parser.parse_args()
    root, out = ROOT / args.root, ROOT / args.out
    protocol = json.loads((ROOT / args.protocol).read_text())
    out.mkdir(parents=True, exist_ok=True)
    results = load(root)
    tests = json.loads((root / "test_scores.json").read_text()) if (root / "test_scores.json").exists() else {}
    rows = core_rows(results, tests, protocol)
    write_csv(out / "comparison.csv", rows)
    core_arms = ["fixed_muon", "muon_search", "same_batch_muon", "independent_muon", "adamw_search", "adago_search"]
    figure(rows, core_arms, out)
    if tests:
        figure(rows, core_arms, out, "test_loss", "fig_test_loss_vs_flops")
    seeds = sorted({r["seed"] for r in rows if r["arm"] in core_arms})
    L = ["# 5090 production results", "",
         f"Protocol: [docs/5090-overnight-experiment-plan.md](../../docs/5090-overnight-experiment-plan.md), frozen in "
         f"[configs/production.json](../../configs/production.json). Model: 9,724,288-parameter NanoGPT "
         f"([configs/wide.json](../../configs/wide.json)), Tiny Shakespeare characters, FP32 (TF32 off), RTX 5090. "
         f"Paired seeds: {', '.join(map(str, seeds))}. Loss = cross-entropy in nats per character on the reporting "
         "half of validation (the tuning half only selects search candidates). Paired differences are "
         "method minus comparator (negative favours the method), mean ± SD over seeds, two-sided paired t-test.", ""]

    def table(arms, field, title):
        L.extend([f"## {title}", "", "| Arm | " + " | ".join(f"{c} cap" for c in CAPS) + " |",
                  "|---|" + "---:|"*len(CAPS)])
        for arm in arms:
            cells = []
            for cap in CAPS:
                m, sd = mean_sd(list(by(rows, arm, cap, field).values()))
                n = len([v for v in by(rows, arm, cap, field).values() if v is not None])
                cells.append("—" if m is None else f"{m:.4f} ± {sd:.4f} (n={n})")
            L.append(f"| {ARM_LABELS[arm]} | " + " | ".join(cells) + " |")
        L.append("")

    table(core_arms, "reporting_loss", "Main comparison: reporting loss at each FLOP cap")
    L.extend(["![Reporting loss vs FLOPs](fig_loss_vs_flops.png)", "",
              "(a) Faint points are individual seeds; lines join per-cap means, placed at each cap's mean actual FLOPs and "
              "dodged horizontally by up to ±9% for legibility (actual FLOPs differ by under 0.1% between arms; see the "
              "compute table). (b) The high cap only, zoomed; values above the panel are counted at its top edge.", ""])
    comparisons = [("independent_muon", "muon_search"), ("independent_muon", "fixed_muon"),
                   ("independent_muon", "same_batch_muon"), ("same_batch_muon", "muon_search"),
                   ("same_batch_muon", "fixed_muon"), ("muon_search", "fixed_muon"),
                   ("independent_muon", "adamw_search"), ("independent_muon", "adago_search")]

    delta_rows = []

    def deltas(field, title, pairs):
        for a, b in pairs:
            for c in CAPS:
                d = paired(by(rows, a, c, field), by(rows, b, c, field))
                if d:
                    seeds_here = sorted(set(by(rows, a, c, field)) & set(by(rows, b, c, field)))
                    delta_rows.append(dict(metric=field, method=a, comparator=b, cap=c, n=d["n"], mean=d["mean"], sd=d["sd"],
                                           seeds_lower=d["wins"], p=d["p"],
                                           per_seed=" ".join(f"{s}:{x:+.4f}" for s, x in zip(seeds_here, d["diffs"]))))
        L.extend([f"### {title}", "", "| Method − comparator | " + " | ".join(f"{c} cap" for c in CAPS) + " |",
                  "|---|" + "---|"*len(CAPS)])
        for a, b in pairs:
            L.append(f"| {ARM_LABELS[a]} − {ARM_LABELS[b]} | " +
                     " | ".join(paired_text(paired(by(rows, a, c, field), by(rows, b, c, field))) for c in CAPS) + " |")
        L.append("")

    deltas("reporting_loss", "Paired reporting-loss differences", comparisons)
    # Companion table.
    L.extend(["### Compute and runtime (means over seeds)", "",
              "| Arm | Cap | Actual FLOPs | Unused FLOPs | Selected-trajectory updates | Training characters (incl. discarded trials) | Work s | Elapsed s | Peak alloc MB |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"])
    for arm in core_arms:
        for cap in CAPS:
            sub = [r for r in rows if r["arm"] == arm and r["cap"] == cap]
            if not sub:
                continue
            f = lambda k: statistics.fmean(r[k] for r in sub)
            L.append(f"| {ARM_LABELS[arm]} | {cap} | {f('flops'):.4e} | {f('unused_flops'):.3e} | "
                     f"{f('selected_trajectory_steps'):.0f} | {f('training_characters'):,.0f} | {f('work_seconds'):.1f} | "
                     f"{f('elapsed_seconds'):.1f} | {statistics.fmean(r['peak_cuda_allocated_bytes'] for r in sub)/2**20:.0f} |")
    L.extend(["", "Single-trajectory arms report work/elapsed time up to each cap; search arms report the whole "
              "campaign for that cap. Reporting evaluation, checkpoint saves, the mechanism check and the "
              "isolation check are off-budget (not in FLOPs or work time).", ""])
    failures = [r for r in rows if r["failed"] or not r["eligible"]]
    L.extend(["### Failures and ineligible points", ""] +
             ([f"- seed {r['seed']} {r['arm']} {r['cap']}: {r['failed']}" for r in failures] or ["- none"]) + [""])

    # Search outcomes.
    L.extend(["## Search outcomes", "", "| Arm | Seed | Cap | Tuning loss per candidate (rate: loss @ updates) | Selected | Edge |",
              "|---|---|---|---|---|---|"])
    for r in sorted((r for r in results if r["kind"] == "search"), key=lambda r: (r["arm"], r["seed"], CAPS.index(r["cap"]))):
        cand = "; ".join(f"{json.dumps(c['rate'])}: {fmt(c['tuning']['loss'] if c['tuning'] else None, 3)} @ {c['steps']}"
                         + (" FAILED" if c["failed"] else "") for c in r["candidates"])
        L.append(f"| {ARM_LABELS[r['arm']]} | {r['seed']} | {r['cap']} | {cand} | "
                 f"{json.dumps(r['selected']['rate']) if r.get('selected') else '—'} | {'yes' if r.get('winner_on_grid_edge') else 'no'} |")
    L.append("")

    # Probe statistics.
    L.extend(["## Probe activity on the single trajectories (whole high-cap run)", "",
              "| Arm | Seed | Probes | Invalid | Clamped | Probe time / ordinary training time | Final LR |", "|---|---|---:|---:|---:|---:|---:|"])
    for r in sorted((r for r in results if r["kind"] == "trajectory"), key=lambda r: (r["arm"], r["seed"])):
        t = r["trial"]
        if not t["probes"]:
            continue
        L.append(f"| {ARM_LABELS[r['arm']]} | {r['seed']} | {t['probes']} | {t['fallbacks']} | {t['clamps']} | "
                 f"{100*r['probe_over_train']:.1f}% | {t['final_lr']:.4g} |")
    L.append("")

    # Isolation check.
    iso = {(r["arm"], r["seed"]): r["isolation"] for r in results if r["kind"] == "trajectory" and r.get("isolation")}
    L.extend([f"## Isolation check at update {protocol['isolation_step']}", "",
              "Same number of ordinary updates for every arm; probe work is extra.", "",
              "| Arm | Reporting loss (mean ± SD) | Probe time / training time | Invalid rate | Clamp rate |", "|---|---:|---:|---:|---:|"])
    iso_loss = {}
    for arm in protocol["isolation_arms"] + protocol["ablations"]["backtracking_arms"]:
        vals = {s: v["reporting"]["loss"] for (a, s), v in iso.items() if a == arm}
        if not vals:
            continue
        iso_loss[arm] = vals
        stats = [v["stats"] for (a, s), v in iso.items() if a == arm]
        probes = sum(s["probes"] for s in stats)
        over = statistics.fmean(s["probe_seconds"]/max(s["train_seconds"], 1e-30) for s in stats)
        m, sd = mean_sd(list(vals.values()))
        L.append(f"| {ARM_LABELS[arm]} | {m:.4f} ± {sd:.4f} | {100*over:.1f}% | "
                 f"{fmt(sum(s['fallbacks'] for s in stats)/probes if probes else None, 3)} | "
                 f"{fmt(sum(s['clamps'] for s in stats)/probes if probes else None, 3)} |")
    L.append("")
    for a, b in (("independent_muon", "fixed_muon"), ("same_batch_muon", "fixed_muon"), ("independent_muon", "same_batch_muon")):
        if a in iso_loss and b in iso_loss:
            L.append(f"- {ARM_LABELS[a]} − {ARM_LABELS[b]}: {paired_text(paired(iso_loss[a], iso_loss[b]))}")
    L.append("")

    # Mechanism check.
    mech = [m for r in results if r.get("mechanism") for m in r["mechanism"]]
    mech_rows = []
    if mech:
        L.extend(["## Mechanism check (frozen weights and direction on fixed Muon)", "",
                  f"Eight fresh training batches that supplied neither estimate; grid {protocol['mechanism']['grid']}. "
                  "Delta = fresh loss after the matrix step minus fresh loss before it. Regret = loss at the proposal "
                  "minus the best grid loss.", "",
                  "| Update | Proposal | Mean rate | Valid | Mean delta | Helps vs no step | Helps vs fixed 0.02 | Mean regret |",
                  "|---:|---|---:|---:|---:|---:|---:|---:|"])
        for step in protocol["mechanism"]["anchors"]:
            here = [m for m in mech if m["step"] == step]
            for name in ("same_batch", "independent"):
                ps = [m["proposals"][name] for m in here]
                L.append(f"| {step} | {name.replace('_', '-')} | {statistics.fmean(p['rate'] for p in ps):.4f} | "
                         f"{sum(p['decision'] != 'fallback' for p in ps)}/{len(ps)} | {statistics.fmean(p['delta'] for p in ps):+.4f} | "
                         f"{sum(p['helps_vs_zero'] for p in ps)}/{len(ps)} | {sum(p['helps_vs_fixed'] for p in ps)}/{len(ps)} | "
                         f"{statistics.fmean(p['regret'] for p in ps):.4f} |")
            fixed = [next(g for g in m["grid"] if g["rate"] == m["fixed_rate"]) for m in here]
            L.append(f"| {step} | fixed 0.02 | 0.0200 | — | {statistics.fmean(g['delta'] for g in fixed):+.4f} | "
                     f"{sum(g['delta'] < 0 for g in fixed)}/{len(fixed)} | — | "
                     f"{statistics.fmean(g['fresh_loss']-m['best_grid_loss'] for g, m in zip(fixed, here)):.4f} |")
            L.append(f"| {step} | best grid rate | {statistics.fmean(m['best_grid_rate'] for m in here):.4f} "
                     f"(per seed: {', '.join(f'{m['best_grid_rate']:g}' for m in sorted(here, key=lambda m: m['seed']))}) | — | "
                     f"{statistics.fmean(m['best_grid_loss']-m['base_fresh_loss'] for m in here):+.4f} | — | — | 0 |")
        L.append("")
        for m in mech:
            for name, p in m["proposals"].items():
                mech_rows.append(dict(seed=m["seed"], step=m["step"], proposal=name, rate=p["rate"], decision=p["decision"],
                                      eta_raw=p["measurement"]["eta_raw"], s=p["measurement"]["s"], q=p["measurement"]["q"],
                                      delta=p["delta"], regret=p["regret"], best_grid_rate=m["best_grid_rate"],
                                      fixed_delta=next(g for g in m["grid"] if g["rate"] == m["fixed_rate"])["delta"]))
        write_csv(out / "mechanism.csv", mech_rows)

    # Ablation: backtracking.
    bt = protocol["ablations"]["backtracking_arms"]
    if any(r["arm"] in bt for r in rows):
        table(bt + ["independent_muon"], "reporting_loss", "Ablation: third-batch backtracking (same FLOP caps)")
        deltas("reporting_loss", "Paired differences", [("guarded_muon", "guarded_base_muon"),
                                                        ("guarded_muon", "independent_muon"),
                                                        ("guarded_muon", "muon_search")])
        L.extend(["| Arm | Seed | Guard checks accepted | rejected (rate 0) | candidate forwards |", "|---|---|---:|---:|---:|"])
        for r in sorted((r for r in results if r.get("arm") in bt), key=lambda r: (r["arm"], r["seed"])):
            g = guard_counts(r["_folder"])
            L.append(f"| {ARM_LABELS[r['arm']]} | {r['seed']} | {g['accepted']} | {g['rejected']} | {g['trials']} |")
        L.append("")

    # Follow-up: extended search grids.
    followups = [f"{a}_followup" for a in protocol.get("followup_grids", {})]
    if any(r["arm"] in followups for r in rows):
        L.extend(["## Follow-up: extended search grids (added after production)", "",
                  protocol.get("followup_note", ""), "",
                  "Grids: " + "; ".join(f"{a}: {json.dumps(g)}" for a, g in protocol["followup_grids"].items()), ""])
        table([a for pair in zip(protocol["followup_grids"], followups) for a in pair] + ["independent_muon"],
              "reporting_loss", "Follow-up results (reporting loss)")
        deltas("reporting_loss", "Follow-up paired differences",
               [(f, a) for a, f in zip(protocol["followup_grids"], followups)] +
               [("independent_muon", f) for f in followups])
        L.extend(["| Arm | Cap | Selected per seed | Edge wins |", "|---|---|---|---:|"])
        for f in followups:
            for cap in CAPS:
                rs = sorted((r for r in results if r.get("arm") == f and r.get("cap") == cap), key=lambda r: r["seed"])
                if rs:
                    L.append(f"| {ARM_LABELS[f]} | {cap} | {', '.join(json.dumps(r['selected']['rate']) for r in rs)} | "
                             f"{sum(bool(r['winner_on_grid_edge']) for r in rs)}/{len(rs)} |")
        L.append("")

    # Ablation: scheduled probing, tuned scheduled Muon, replay.
    matched = [r for r in results if r["kind"] == "matched"]
    sched_rows = []
    if matched:
        L.extend(["## Ablation: scheduled probing at a common schedule and horizon", "",
                  "40-update warmup, linear decay from 35% of the horizon to 5% of the peak; probes every 25 updates from "
                  "update 25. Tuned scheduled Muon = best of five peaks (0.003-0.08) by tuning-half loss, per seed; its "
                  "search costs five full runs. Replay = geometric mean of the next-batch LR curves of development seeds "
                  f"{protocol['ablations']['replay_development_seeds']} (follow-up; construction cost = those two runs).", ""])
        eval_seeds = set(seeds)
        for horizon in protocol["ablations"]["scheduled_horizons"]:
            group = [r for r in matched if r["horizon"] == horizon]
            arms = {}
            for name in ("scheduled_fixed_muon", "scheduled_same_batch_muon", "scheduled_heldout_muon", "scheduled_replay_muon"):
                arms[name] = {r["seed"]: r for r in group if r["arm"] == name and r["seed"] in eval_seeds}
            tuned, search_flops, chosen = {}, {}, {}
            for seed in eval_seeds:
                cands = [r for r in group if r["arm"] == "scheduled_search" and r["seed"] == seed and r["tuning"]]
                if len(cands) == len(protocol["ablations"]["scheduled_search_peaks"]):
                    best = min(cands, key=lambda r: r["tuning"]["loss"])
                    tuned[seed], chosen[seed] = best, best["initial_lr"]
                    search_flops[seed] = sum(r["flops"] for r in cands)
            arms["tuned"] = tuned
            label = dict(scheduled_fixed_muon="Scheduled Muon, peak 0.02", scheduled_same_batch_muon="Scheduled same-batch probe",
                         scheduled_heldout_muon="Scheduled next-batch probe", scheduled_replay_muon="Replay (follow-up)",
                         tuned="Tuned scheduled Muon")
            L.extend([f"### Horizon {horizon} updates", "", "| Arm | Reporting loss | Test loss | FLOPs per run | Work s per run | Probe time / training |",
                      "|---|---:|---:|---:|---:|---:|"])
            loss = {}
            for name, runs in arms.items():
                if not runs:
                    continue
                loss[name] = {s: (r["reporting"]["loss"] if r["reporting"] else None) for s, r in runs.items()}
                tl = [tests.get(f"seed{s}/{r['_folder'].name}/weights_final.pt") for s, r in runs.items()]
                m, sd = mean_sd(list(loss[name].values()))
                tm, tsd = mean_sd([t["loss"] for t in tl if t])
                L.append(f"| {label[name]} | {fmt(m)} ± {fmt(sd)} | {fmt(tm)} ± {fmt(tsd)} | "
                         f"{statistics.fmean(r['flops'] for r in runs.values()):.4e} | "
                         f"{statistics.fmean(r['work_seconds'] for r in runs.values()):.1f} | "
                         f"{100*statistics.fmean(r['probe_over_train'] for r in runs.values()):.1f}% |")
                for s, r in runs.items():
                    sched_rows.append(dict(horizon=horizon, arm=name, seed=s, reporting_loss=loss[name][s],
                                           test_loss=tests.get(f"seed{s}/{r['_folder'].name}/weights_final.pt", {}) and
                                           tests[f"seed{s}/{r['_folder'].name}/weights_final.pt"]["loss"],
                                           flops=r["flops"], initial_lr=json.dumps(r["initial_lr"])))
            if chosen:
                L.append(f"\nTuned peaks per seed: {', '.join(f'{s}: {chosen[s]:g}' for s in sorted(chosen))}; "
                         f"search cost {statistics.fmean(search_flops.values()):.4e} FLOPs per seed.")
            dev = [r for r in group if r["arm"] == "scheduled_heldout_muon" and r["seed"] in protocol["ablations"]["replay_development_seeds"]]
            if dev:
                L.append(f"Replay curve construction cost: {sum(r['flops'] for r in dev):.4e} FLOPs (two development runs).")
            L.append("")
            for a, b in (("scheduled_heldout_muon", "tuned"), ("scheduled_heldout_muon", "scheduled_fixed_muon"),
                         ("scheduled_heldout_muon", "scheduled_same_batch_muon"), ("scheduled_same_batch_muon", "tuned"),
                         ("scheduled_replay_muon", "tuned"), ("scheduled_replay_muon", "scheduled_heldout_muon")):
                if a in loss and b in loss:
                    d = paired(loss[a], loss[b])
                    per_seed = ", ".join(f"{x:+.4f}" for x in d["diffs"]) if d else ""
                    L.append(f"- {label[a]} − {label[b]}: {paired_text(d)}; per seed {per_seed}")
            L.append("")
        write_csv(out / "scheduled.csv", sched_rows)

    # Fidelity: finite difference versus exact HVP.
    if (root / "fidelity.json").exists():
        fid = json.loads((root / "fidelity.json").read_text())
        L.extend(["## Curvature fidelity on the production model (finite difference vs exact HVP)", "",
                  "Saved pre-update fixed-Muon states (mechanism anchors). Exact q = D^T H D by double backward "
                  "(math attention). Relative error = |q_fd - q_exact| / |q_exact|. Times are synchronized medians of "
                  "ten repeats; the production probe uses the default attention backend.", "",
                  "| Update | Batch | Exact q (mean) | Exact eta = s/q valid | Rel. error rho=1e-4 | rho=1e-3 | rho=1e-2 | "
                  "FD sign matches exact (rho=1e-3) | FD s (default) | FD s (math) | HVP s (math) |",
                  "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for step in sorted({r["step"] for r in fid}):
            here = [r for r in fid if r["step"] == step]
            for name in ("independent", "same_batch"):
                xs = [r[name] for r in here]
                err = lambda rho: statistics.median(x["relative_q_error"][rho] for x in xs if x["relative_q_error"][rho] is not None)
                sign = sum((x["finite_difference"]["0.001"]["q"] > 0) == (x["exact"]["q"] > 0) for x in xs)
                L.append(f"| {step} | {name.replace('_', '-')} | {statistics.fmean(x['exact']['q'] for x in xs):.4g} | "
                         f"{sum(x['exact_eta'] is not None for x in xs)}/{len(xs)} | {err('0.0001'):.3g} | {err('0.001'):.3g} | "
                         f"{err('0.01'):.3g} | {sign}/{len(xs)} | "
                         f"{statistics.fmean(x['median_seconds']['finite_difference_default'] for x in xs)*1e3:.1f} ms | "
                         f"{statistics.fmean(x['median_seconds']['finite_difference_math'] for x in xs)*1e3:.1f} ms | "
                         f"{statistics.fmean(x['median_seconds']['hvp_math'] for x in xs)*1e3:.1f} ms |")
        L.extend(["", "Relative errors are medians over seeds.", ""])

    # Test split.
    if tests:
        table(core_arms, "test_loss", "Reserved test split (scored after the protocol and all runs were frozen)")
        deltas("test_loss", "Paired test-loss differences", comparisons)
        L.extend(["![Test loss vs FLOPs](fig_test_loss_vs_flops.png)", ""])
    write_csv(out / "paired_deltas.csv", delta_rows)
    (out / "REPORT.md").write_text("\n".join(L) + "\n")
    # Versionable raw records (small files only; checkpoints stay in runs/).
    raw = out / "raw"
    for r in results:
        dest = raw / r["_folder"].relative_to(root)
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(r["_folder"] / "result.json", dest / "result.json")
        if (r["_folder"] / "mechanism.json").exists():
            shutil.copy2(r["_folder"] / "mechanism.json", dest / "mechanism.json")
    for extra in ["test_scores.json", *[p.name for p in root.glob("replay_trace_H*.json")]]:
        if (root / extra).exists():
            shutil.copy2(root / extra, raw / extra)
    for p in root.glob("seed*/order.json"):
        (raw / p.parent.name).mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, raw / p.parent.name / "order.json")
    manifests = sorted(root.glob("manifest_*"))
    for m in manifests:
        shutil.copytree(m, raw / m.name, dirs_exist_ok=True)
    print(f"Wrote {out / 'REPORT.md'}")


if __name__ == "__main__":
    main()
