"""Analysis for the curvature-probe Muon project. Reads results/*.json, writes report/ tables + figures."""
import glob, json, math, os, sys, itertools, statistics as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.expanduser("~/cs182_curvature_muon")
OUT = f"{ROOT}/report"
os.makedirs(f"{OUT}/fig", exist_ok=True)

def load(pattern):
    rs = {}
    for f in sorted(glob.glob(f"{ROOT}/results/{pattern}.json")):
        r = json.load(open(f))
        rs[os.path.basename(f)[:-5]] = r
    return rs

def fmt(x, d=4):
    return "nan" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{d}f}"

def mean_sd(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    if not xs: return float("nan"), float("nan")
    return st.mean(xs), (st.stdev(xs) if len(xs) > 1 else 0.0)

lines = []
def P(s=""):
    lines.append(s); print(s)

def hp_of(r):
    a = r["args"]
    if a["optim"] in ("muon", "adamw"):
        return a["matrix_lr"] if a["matrix_lr"] > 0 else (r["nanochat_default_muon_lr"] if a["optim"] == "muon" else 3e-3)
    if a["optim"] == "adago":
        return a["adago_eta"]
    return a["probe_every"]

# ---------------------------------------------------------------- pilots / searches
def search_table(prefix, label):
    rs = load(f"{prefix}_*")
    if not rs: return None
    P(f"\n### {label}\n")
    P("| run | hyperparameter | final val loss | final val bpb | train min | diverged |")
    P("|---|---|---|---|---|---|")
    best = None
    for k, r in sorted(rs.items(), key=lambda kv: hp_of(kv[1])):
        v = r["final_val"]["loss"]
        P(f"| {k} | {hp_of(r):.4g} | {fmt(v)} | {fmt(r['final_val']['bpb'])} | {r['timing']['train_time_s']/60:.2f} | {r['nan_abort']} |")
        if math.isfinite(v) and (best is None or v < best[1]):
            best = (k, v, hp_of(r))
    if best: P(f"\nSelected (lowest val loss): **{best[0]}**, hyperparameter {best[2]:.4g}, val loss {best[1]:.4f}")
    return best, rs

def expected_best_of_n(vals, n_max):
    """Expected min val loss after n random-search trials, averaged over all orderings of the observed trials."""
    vals = [v if math.isfinite(v) else 1e9 for v in vals]
    out = []
    m = len(vals)
    s = sorted(vals)
    for n in range(1, min(n_max, m) + 1):
        # E[min of n draws without replacement] = sum_i s_i * P(min = s_i)
        tot = 0.0
        denom = math.comb(m, n)
        for i, v in enumerate(s):
            tot += v * math.comb(m - i - 1, n - 1) / denom
        out.append(tot)
    return out

# ---------------------------------------------------------------- diagnostics
def grid_items(line):
    return sorted([(v["eta"], v) for k, v in line.items() if k.startswith("grid_")], key=lambda t: t[0])

def fmean(v):
    return sum(v["fresh"]) / len(v["fresh"])

def diag_section(r):
    recs = r["diag"]
    base_steps = sorted({b for rec in recs for (b, d) in rec["bases"]})
    eps_default = r["args"]["probe_eps_rel"]
    P("\n### D1: predicted step vs a log-spaced grid (same checkpoint, same direction D)\n")
    P("Loss changes are relative to the unchanged weights. Fresh = mean over 4 fresh minibatches of 131,072 tokens.\n")
    P("| step | scheduled Muon LR | eta_hat | best grid eta (probe batch) | best grid eta (fresh) | dL probe @eta_hat | dL probe best | dL fresh @eta_hat | dL fresh best | dL fresh @sched | predicted dL @eta_hat |")
    P("|---|---|---|---|---|---|---|---|---|---|---|")
    rows = []
    for rec in recs:
        line = rec["line"]; g = grid_items(line)
        hp = min(g, key=lambda t: t[1]["probe"]); hf = min(g, key=lambda t: fmean(t[1]))
        L0f = sum(rec["L0_fresh"]) / len(rec["L0_fresh"]); L0p = rec["L0_probe"]
        key = f"hat_eps{eps_default:.0e}"
        eh = rec["eta_hat"]
        hat = line.get(key)
        ref = next(p for p in rec["probes"] if abs(p["eps_rel"] - eps_default) < 1e-15)
        pred = (-eh * rec["gD"] + 0.5 * eh * eh * ref["q_hat"]) if eh else None
        row = dict(step=rec["step"], sched=rec["sched_lr"], eta_hat=eh, best_p=hp[0], best_f=hf[0],
                   dLp_hat=(hat["probe"] - L0p) if hat else None, dLp_best=hp[1]["probe"] - L0p,
                   dLf_hat=(fmean(hat) - L0f) if hat else None, dLf_best=fmean(hf[1]) - L0f,
                   dLf_sched=fmean(line["sched"]) - L0f, pred=pred)
        rows.append(row)
        P(f"| {row['step']} | {row['sched']:.4g} | {fmt(eh,4) if eh else 'invalid'} | {row['best_p']:.4g} | {row['best_f']:.4g} | {fmt(row['dLp_hat'])} | {fmt(row['dLp_best'])} | {fmt(row['dLf_hat'])} | {fmt(row['dLf_best'])} | {fmt(row['dLf_sched'])} | {fmt(pred)} |")
    # quadratic-model accuracy over the grid
    P("\n**Quadratic model accuracy.** For grid points with eta <= 2*eta_hat, predicted dL = -eta<g,D> + eta^2 q_hat/2 vs actual dL on the probe batch and on fresh batches.\n")
    P("| step | n points | corr(pred, probe) | median rel. error (probe) | corr(pred, fresh) | median rel. error (fresh) |")
    P("|---|---|---|---|---|---|")
    for rec in recs:
        eh = rec["eta_hat"]
        if not eh: continue
        ref = next(p for p in rec["probes"] if abs(p["eps_rel"] - eps_default) < 1e-15)
        L0f = sum(rec["L0_fresh"]) / len(rec["L0_fresh"]); L0p = rec["L0_probe"]
        pts = [(e, v) for e, v in grid_items(rec["line"]) if e <= 2 * eh]
        if len(pts) < 3: continue
        pr = np.array([-e * rec["gD"] + 0.5 * e * e * ref["q_hat"] for e, _ in pts])
        ap = np.array([v["probe"] - L0p for _, v in pts]); af = np.array([fmean(v) - L0f for _, v in pts])
        rel = lambda a: float(np.median(np.abs(pr - a) / np.maximum(np.abs(a), 1e-4)))
        P(f"| {rec['step']} | {len(pts)} | {np.corrcoef(pr, ap)[0,1]:.3f} | {rel(ap):.2f} | {np.corrcoef(pr, af)[0,1]:.3f} | {rel(af):.2f} |")
    # line plots at base steps
    for b in base_steps:
        rec = next(x for x in recs if x["step"] == b)
        g = grid_items(rec["line"]); L0f = sum(rec["L0_fresh"]) / len(rec["L0_fresh"]); L0p = rec["L0_probe"]
        e = np.array([t[0] for t in g])
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        ax.plot(e, [t[1]["probe"] - L0p for t in g], "o-", label="probe batch")
        ax.plot(e, [fmean(t[1]) - L0f for t in g], "s-", label="fresh batches (mean)")
        ref = next(p for p in rec["probes"] if abs(p["eps_rel"] - eps_default) < 1e-15)
        ee = np.logspace(-4, 0, 200)
        ax.plot(ee, -ee * rec["gD"] + 0.5 * ee ** 2 * ref["q_hat"], "--", color="gray", label="quadratic model")
        if rec["eta_hat"]: ax.axvline(rec["eta_hat"], color="C3", label="eta_hat")
        ax.axvline(rec["sched_lr"], color="C2", ls=":", label="tuned Muon LR (sched)")
        ax.set_xscale("log"); ax.set_xlabel("step size eta along -D"); ax.set_ylabel("loss change (nats)")
        lo = min(min(t[1]["probe"] - L0p for t in g), min(fmean(t[1]) - L0f for t in g))
        ax.set_ylim(lo - 0.05 * abs(lo) - 0.01, max(0.3, -lo)); ax.set_title(f"step {b}"); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(f"{OUT}/fig/d1_line_step{b}.png", dpi=130); plt.close(fig)
    # continuations
    if r["continuation"]:
        P("\n### Q1: short training continuations (constant Muon LR from the checkpoint; fresh-batch loss)\n")
        P("| step | eta_hat | candidate | eta | loss after 1 | after 10 | after 50 |")
        P("|---|---|---|---|---|---|---|")
        for c in r["continuation"]:
            for name, v in c["cands"].items():
                tr = v["traj"]
                P(f"| {c['step']} | {fmt(c['eta_hat'],4) if c['eta_hat'] else 'invalid'} | {name} | {('schedule' if v['eta'] is None else f'{v[chr(101)+chr(116)+chr(97)]:.4g}')} | {fmt(tr.get('1'))} | {fmt(tr.get('10'))} | {fmt(tr.get(str(r['args']['diag_cont_steps'])))} |")
    # staleness
    P("\n### D2: staleness of saved recommendations (fresh-batch regret vs best grid step at the later step)\n")
    P("| base step | delta | eta_hat(base) | eta_hat(base+delta) | ratio | regret stale | regret fresh eta_hat | regret sched |")
    P("|---|---|---|---|---|---|---|---|")
    stale_rows = []
    by_step = {rec["step"]: rec for rec in recs}
    for b in base_steps:
        eb = by_step[b]["eta_hat"]
        for d in r["args"]["diag_deltas"].split(","):
            d = int(d); s = b + d
            if s not in by_step: continue
            rec = by_step[s]; g = grid_items(rec["line"]); best = min(fmean(t[1]) for t in g)
            stale = rec["line"].get(f"stale_b{b}_d{d}") if d > 0 else rec["line"].get(f"hat_eps{eps_default:.0e}")
            fresh = rec["line"].get(f"hat_eps{eps_default:.0e}")
            rs_ = (fmean(stale) - best) if stale else None
            rf_ = (fmean(fresh) - best) if fresh else None
            rsch = fmean(rec["line"]["sched"]) - best
            ratio = (rec["eta_hat"] / eb) if (eb and rec["eta_hat"]) else None
            stale_rows.append((b, d, rs_, rf_, rsch))
            P(f"| {b} | {d} | {fmt(eb,4) if eb else 'invalid'} | {fmt(rec['eta_hat'],4) if rec['eta_hat'] else 'invalid'} | {fmt(ratio,2)} | {fmt(rs_)} | {fmt(rf_)} | {fmt(rsch)} |")
    # probe distance
    P("\n### D3: probe-distance sensitivity and numerical noise floor\n")
    P("noise curvature = <D, g_repeat - g>/eps, from a second gradient at the same weights. SNR = |q_hat| / |noise curvature|. fd q = central finite difference of the probe-batch line loss.\n")
    P("| step | fd q (line) | " + " | ".join(f"eps {p['eps_rel']:.0e}: q_hat / eta_hat / SNR" for p in recs[0]["probes"]) + " |")
    P("|---|---|" + "---|" * len(recs[0]["probes"]))
    for rec in recs:
        cells = []
        for p in rec["probes"]:
            noise_q = abs(rec["noise_dot"]) / p["eps"] if p["eps"] > 0 else float("inf")
            snr = abs(p["q_hat"]) / noise_q if noise_q > 0 else float("inf")
            cells.append(f"{p['q_hat']:.3g} / {('%.3g' % p['eta_hat']) if p['eta_hat'] else 'inv'} / {snr:.2g}")
        P(f"| {rec['step']} | {rec['fd_q']:.3g} | " + " | ".join(cells) + " |")
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for b in base_steps:
        rec = by_step[b]
        xs = [p["eps_rel"] for p in rec["probes"] if p["eta_hat"]]; ys = [p["eta_hat"] for p in rec["probes"] if p["eta_hat"]]
        ax.plot(xs, ys, "o-", label=f"step {b}")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("probe distance eps_rel (||eps D|| / ||W||)"); ax.set_ylabel("eta_hat")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(f"{OUT}/fig/d3_eps.png", dpi=130); plt.close(fig)
    return rows, stale_rows

def eta_traj_plot(rs, fname, title):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for k, r in rs.items():
        ax.plot([s["step"] for s in r["steps"]], [s.get("muon_lr", s.get("adago_alpha_mean")) for s in r["steps"]], label=k)
    ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("Muon multiplier eta (AdaGO: mean alpha)"); ax.set_title(title)
    ax.legend(fontsize=6); fig.tight_layout(); fig.savefig(f"{OUT}/fig/{fname}", dpi=130); plt.close(fig)

def curve_plot(rs, fname, title):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for k, r in rs.items():
        ax.plot([e["step"] for e in r["evals"]] + [r["num_iterations"]], [e["loss"] for e in r["evals"]] + [r["final_val"]["loss"]], label=k)
    ax.set_xlabel("step"); ax.set_ylabel("val loss (nats/token)"); ax.set_title(title); ax.set_ylim(None, 5.5)
    ax.legend(fontsize=6); fig.tight_layout(); fig.savefig(f"{OUT}/fig/{fname}", dpi=130); plt.close(fig)

def overhead(r):
    return r["timing"]["probe_time_s"] / max(1e-9, r["timing"]["train_time_s"] - r["timing"]["probe_time_s"])

def finals_table(prefix_map, label):
    P(f"\n### {label}\n")
    P("| method | runs | diverged | val loss mean ± sd | val bpb mean | test loss mean ± sd | test bpb mean | train min / run | probe overhead |")
    P("|---|---|---|---|---|---|---|---|---|")
    out = {}
    for name, pat in prefix_map.items():
        rs = load(pat) if isinstance(pat, str) else {k: v for p in pat for k, v in load(p).items()}
        if not rs: continue
        vm, vs = mean_sd([r["final_val"]["loss"] for r in rs.values()])
        tm, ts = mean_sd([r["final_test"]["loss"] for r in rs.values() if r["final_test"]])
        vb, _ = mean_sd([r["final_val"]["bpb"] for r in rs.values()])
        tb, _ = mean_sd([r["final_test"]["bpb"] for r in rs.values() if r["final_test"]])
        ok = [r for r in rs.values() if not r["nan_abort"]]
        ndiv = len(rs) - len(ok)
        mins, _ = mean_sd([r["timing"]["train_time_s"] / 60 for r in ok])
        ov, _ = mean_sd([overhead(r) for r in ok])
        out[name] = dict(val=(vm, vs), test=(tm, ts), mins=mins, runs=rs)
        P(f"| {name} | {len(rs)} | {ndiv} | {vm:.4f} ± {vs:.4f} | {vb:.4f} | {tm:.4f} ± {ts:.4f} | {tb:.4f} | {mins:.2f} | {100*ov:.1f}% |")
    return out

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    pil = load("p_wd_*")
    if pil:
        P("### Pilot: Muon weight decay (default Muon LR, seed 0)\n")
        P("| run | Muon WD | final val loss | val bpb | train min | mean step s |")
        P("|---|---|---|---|---|---|")
        for k, r in pil.items():
            P(f"| {k} | {r['muon_wd']:.3f} | {fmt(r['final_val']['loss'])} | {fmt(r['final_val']['bpb'])} | {r['timing']['train_time_s']/60:.2f} | {r['timing']['mean_step_s']:.3f} |")
    searches = {}
    for pre, lab in [("s_muon", "Random search: Muon peak LR (small model, seed 0)"),
                     ("s_adamw", "Random search: AdamW peak LR, epsilon 1e-8 (used; best of both AdamW searches)"),
                     ("s_adamwE", "Random search: AdamW peak LR, epsilon 1e-10 (worse at every LR, not used)"),
                     ("s_adago", "Random search: AdaGO eta (small model, seed 0)"),
                     ("w_s_muon", "Random search: Muon peak LR (wide model, seed 0)"),
                     ("w_s_adamw", "Random search: AdamW peak LR, epsilon 1e-8 (wide model, seed 0)")]:
        res = search_table(pre, lab)
        if res: searches[pre] = res
    diag = load("d_muon_diag")
    if diag:
        P("\n## Diagnostic experiments (tuned Muon run, seed 0)")
        diag_section(diag["d_muon_diag"])
    kab = {**load("k_*"), **load("c_*")}
    if kab:
        P("\n### Probe interval ablation (small model, seed 0)\n")
        P("| run | k | scale c | final val loss | probes | accepted | mean probe s | probe overhead | final eta |")
        P("|---|---|---|---|---|---|---|---|---|")
        for k, r in sorted(kab.items(), key=lambda kv: (kv[1]["args"]["probe_every"] or 10**6, kv[1]["args"]["probe_eps_rel"])):
            pr = r["probes"]; acc = sum(p["accepted"] for p in pr)
            mp = st.mean([p["seconds"] for p in pr[1:]]) if len(pr) > 1 else float("nan")
            P(f"| {k} | {r['args']['probe_every'] or 'once'} | {r['args']['probe_scale']:g} | {fmt(r['final_val']['loss'])} | {len(pr)} | {acc} | {mp:.3f} | {100*overhead(r):.1f}% | {r['steps'][-1]['muon_lr']:.4g} |")
        eta_traj_plot(kab, "k_eta.png", "probe-selected eta by probe interval")
        curve_plot(kab, "k_val.png", "val loss by probe interval")
    fin = finals_table({"Tuned Muon": "f_muon_s*", "Tuned AdamW": "f_adamw_s*", "AdaGO": "f_adago_s*", "Probe Muon, calibrated c=0.5, first probe at step 25": "f_curvS_s*", "Probe Muon, literal c=1, first probe at step 25": "f_curvc1S_s*",
                        "Probe Muon, calibrated c=0.5, probe at step 0": "f_curv_s[0-9]", "Probe Muon, literal c=1, probe at step 0": "f_curvc1_s[0-9]"},
                       "Main comparison: small model (depth 6, d_model 384), 50M tokens, seeds 1-3")
    if fin:
        reps = {k: next((r for r in v["runs"].values() if not r["nan_abort"]), None) for k, v in fin.items()}
        reps = {k: r for k, r in reps.items() if r is not None and "step 0" not in k}
        curve_plot(reps, "finals_val.png", "val loss, seed 1 of each method")
        eta_traj_plot({k: v for k, v in reps.items() if k != "Tuned AdamW"}, "finals_eta.png", "Muon multiplier / AdaGO alpha (seed 1)")
    # Q2 with three seeds (gap fill): probe interval and probe-once, first probe at step 25
    q2 = finals_table({"c=1, k=10": "q2_k10_c1_s*", "c=1, k=25": "f_curvc1S_s*", "c=1, k=50": "q2_k50_c1_s*",
                       "c=1, k=100": "q2_k100_c1_s*", "c=1, once": "q2_once_c1_s*",
                       "c=0.5, k=25": "f_curvS_s*", "c=0.5, k=100": "q2_k100_c0.5_s*", "c=0.5, once": "q2_once_c0.5_s*"},
                      "Q2 with three seeds: probe interval k and probe-once (small model, first probe at step 25, seeds 1-3)")
    if q2:
        P("\nFinal accepted eta per run (mean over seeds) and number of accepted probes:\n")
        P("| setting | final eta mean | accepted probes (mean) | first accepted probe step(s) |")
        P("|---|---|---|---|")
        for name, v in q2.items():
            rs = list(v["runs"].values())
            fe = st.mean([r["steps"][-1]["muon_lr"] for r in rs])
            acc = st.mean([sum(p["accepted"] for p in r["probes"]) for r in rs])
            first = [next((p["step"] for p in r["probes"] if p["accepted"]), None) for r in rs]
            P(f"| {name} | {fe:.4g} | {acc:.1f} | {first} |")
        from scipy import stats
        def welch(a, b):
            xa = [r["final_test"]["loss"] for r in q2[a]["runs"].values()]; xb = [r["final_test"]["loss"] for r in q2[b]["runs"].values()]
            t = stats.ttest_ind(xa, xb, equal_var=False)
            P(f"| {a} vs {b} | {st.mean(xa) - st.mean(xb):+.4f} | {t.pvalue:.3g} |")
        P("\nWelch t-tests on test loss (difference = first minus second; negative favours the first):\n")
        P("| comparison | difference | p |")
        P("|---|---|---|")
        for a, b in [("c=1, k=10", "c=1, k=100"), ("c=1, k=25", "c=1, k=100"), ("c=1, k=50", "c=1, k=100"),
                     ("c=1, k=100", "c=1, once"), ("c=1, k=25", "c=1, once"),
                     ("c=0.5, k=25", "c=0.5, once"), ("c=0.5, k=100", "c=0.5, once"), ("c=0.5, k=25", "c=0.5, k=100")]:
            if a in q2 and b in q2: welch(a, b)
        fig, ax = plt.subplots(figsize=(5.6, 3.6))
        for c, col in (("1", "C3"), ("0.5", "C0")):
            xs, ms, sds = [], [], []
            for i, k in enumerate(("10", "25", "50", "100", "once")):
                key = f"c={c}, k={k}" if k != "once" else f"c={c}, once"
                if key in q2:
                    xs.append(i); ms.append(q2[key]["val"][0]); sds.append(q2[key]["val"][1])
            ax.errorbar(xs, ms, yerr=sds, fmt="o-", capsize=3, color=col, label=f"scale c = {c}")
        if fin and "Tuned Muon" in fin:
            ax.axhline(fin["Tuned Muon"]["val"][0], color="gray", ls="--", label="tuned Muon")
        ax.set_xticks(range(5)); ax.set_xticklabels(["k=10", "k=25", "k=50", "k=100", "once"])
        ax.set_ylabel("final val loss (mean ± sd, 3 seeds)"); ax.set_title("probe interval vs probe-once (first probe at step 25)")
        ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(f"{OUT}/fig/q2_seeds.png", dpi=130); plt.close(fig)
    wfin = finals_table({"Tuned Muon (wide search)": "w_f_muon_s*", "Muon, small-model LR transferred": "w_muonT_s*",
                         "Tuned AdamW (wide search)": "w_f_adamw_s*", "AdamW, small-model LR transferred": "w_adamwT_s*",
                         "AdaGO, small-model eta transferred": "w_adago_s*", "Probe Muon, small-model rule, first probe at step 25": "w_curvS_s*", "Probe Muon, small-model rule, probe at step 0": "w_curv_s[0-9]"},
                        "Width transfer: wide model (depth 6, d_model 768), 50M tokens, seeds 1-3")
    # ------------------------------------------------ modified method (v2 -> v4), development log and finals
    scr = {**load("v2_*"), **load("v3h_*"), **load("w_v3h_*"), **load("v4_*"), **load("ref_muon_*")}
    if scr:
        P("\n## Modified method: development runs (screening seeds 0 and 10; selection on val only)\n")
        P("| run | width | seed | final val loss | final Muon multiplier | probes |")
        P("|---|---|---|---|---|---|")
        for k, r in sorted(scr.items()):
            a = r["args"]
            P(f"| {k} | {'768' if a['aspect_ratio'] == 128 else '384'} | {a['seed']} | {fmt(r['final_val']['loss'])} | {r['steps'][-1]['muon_lr']:.4g} | {len(r.get('probes', []))} |")
        P("\nReference, tuned Muon seed 0 (search run s_muon_03): 3.8597.")
    mod = finals_table({"Tuned Muon (seeds 1-3)": "f_muon_s*", "v2 probe-set peak, c=0.5 (seeds 1-3)": "f_v2_s*", "v3 held-out probe (seeds 1-3)": "f_v3_s*",
                        "v4 held-out probe + warmdown cap (seeds 1-3)": "f_v4_s*",
                        "Tuned Muon (confirmation seeds 4-6)": "cf_muon_s[456]", "v2 (confirmation seeds 4-6)": "cf_v2_s*", "v3 (confirmation seeds 4-6)": "cf_v3_s*",
                        "Tuned Muon (confirmation seeds 7-9)": "cf_muon_s[789]", "v4 (confirmation seeds 7-9)": "cf_v4_s*"},
                       "Modified method: small model finals and fresh-seed confirmations")
    wmod = finals_table({"Tuned Muon (wide search)": "w_f_muon_s*", "Muon, small-model LR transferred": "w_muonT_s*",
                         "v2 (small-model rule)": "w_v2_s*", "v3 (small-model rule)": "w_v3_s*", "v4 (small-model rule)": "w_v4_s[123]"},
                        "Modified method: wide model, no wide tuning")
    if mod:
        from scipy import stats
        def tl(names): return [load(n)[n]["final_test"]["loss"] for n in names]
        P("\nWelch t-tests on test loss (modified method minus tuned Muon):\n")
        P("| comparison | difference | p |")
        P("|---|---|---|")
        for lab, A, B in [("v2 vs Muon, seeds 1-3", "f_v2_s{}", "f_muon_s{}", ), ("v3 vs Muon, seeds 1-3", "f_v3_s{}", "f_muon_s{}"), ("v4 vs Muon, seeds 1-3", "f_v4_s{}", "f_muon_s{}")]:
            a = tl([A.format(s) for s in (1, 2, 3)]); b = tl([B.format(s) for s in (1, 2, 3)])
            P(f"| {lab} | {st.mean(a) - st.mean(b):+.4f} | {stats.ttest_ind(a, b, equal_var=False).pvalue:.3g} |")
        for lab, A, ss in [("v2 vs Muon, seeds 4-6", "cf_v2_s{}", (4, 5, 6)), ("v3 vs Muon, seeds 4-6", "cf_v3_s{}", (4, 5, 6)), ("v4 vs Muon, seeds 7-9", "cf_v4_s{}", (7, 8, 9))]:
            a = tl([A.format(s) for s in ss]); b = tl([f"cf_muon_s{s}" for s in ss])
            P(f"| {lab} | {st.mean(a) - st.mean(b):+.4f} | {stats.ttest_ind(a, b, equal_var=False).pvalue:.3g} |")
        S = (1, 2, 3, 7, 8, 9)
        a = tl([f"f_v4_s{s}" if s < 4 else f"cf_v4_s{s}" for s in S]); b = tl([f"f_muon_s{s}" if s < 4 else f"cf_muon_s{s}" for s in S])
        P(f"| v4 vs Muon, all 6 seeds (paired by seed) | {st.mean(a) - st.mean(b):+.4f} | {stats.ttest_rel(a, b).pvalue:.3g} |")
        for lab, A, B in [("v4 wide vs wide-tuned Muon", "w_v4_s{}", "w_f_muon_s{}"), ("v4 wide vs transferred-LR Muon", "w_v4_s{}", "w_muonT_s{}")]:
            a = tl([A.format(s) for s in (1, 2, 3)]); b = tl([B.format(s) for s in (1, 2, 3)])
            P(f"| {lab} | {st.mean(a) - st.mean(b):+.4f} | {stats.ttest_ind(a, b, equal_var=False).pvalue:.3g} |")
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
        for ax, (m, ssn, v, rp, title) in zip(axs, [("f_muon_s1", "f_ssmuon_s1", "f_v4_s1", "rp_v4trace_s1", "small model, seed 1"),
                                                     ("w_f_muon_s1", "w_f_ssmuon_s1", "w_v4_s1", "w_rp_s1", "wide model, seed 1")]):
            for n, lab, sty in ((m, "tuned Muon (searched peak, linear warmdown)", "-"), (ssn, "shape-tuned Muon (4-D schedule search)", ":"),
                                (v, "v4 probe rule (no LR search)", "-"), (rp, "replay of v4 screening-seed curve", "--"),
                                ("w_rpT_s1" if title.startswith("wide") else "", "replay of the small-model curve", "-.")):
                r = load(n).get(n)
                if r: ax.plot([s["step"] for s in r["steps"]], [s["muon_lr"] for s in r["steps"]], sty, label=lab)
            ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("Muon learning rate"); ax.set_title(title); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(f"{OUT}/fig/v4_lr.png", dpi=130); plt.close(fig)
    # ------------------------------------------------ shape-tuned Muon baseline and replay control
    for pre, lab in (("ss_muon", "Shape-tuned Muon search (small, seed 0): peak LR x warmdown shape x warmdown fraction x final fraction"),
                     ("w_ss_muon", "Shape-tuned Muon search (wide, seed 0)")):
        rs = load(f"{pre}_[0-9]*")
        if not rs: continue
        P(f"\n### {lab}\n")
        P("| run | peak LR | shape | warmdown fraction | final fraction | val loss |")
        P("|---|---|---|---|---|---|")
        for k, r in sorted(rs.items(), key=lambda kv: kv[1]["final_val"]["loss"]):
            a = r["args"]
            P(f"| {k} | {a['matrix_lr']:.4g} | {a['lr_shape']} | {a['lr_warmdown_ratio']:.3g} | {a['lr_final_frac']:.3g} | {fmt(r['final_val']['loss'])} |")
    ss = finals_table({"Tuned Muon, default shape (seeds 1-3, 7-9)": ["f_muon_s[123]", "cf_muon_s[789]"], "Shape-tuned Muon (seeds 1-3, 7-9)": "f_ssmuon_s*",
                       "v4 probe rule (seeds 1-3, 7-9)": ["f_v4_s[123]", "cf_v4_s[789]"], "Replay: Muon on the v4 screening-seed LR curve (seeds 1-3, 7-9)": "rp_v4trace_s*"},
                      "Schedule controls: small model, 6 seeds")
    wss = finals_table({"Tuned Muon, default shape (wide search)": "w_f_muon_s*", "Shape-tuned Muon, small-model setting transferred": "w_ssmuonT_s*",
                        "Shape-tuned Muon, wide search": "w_f_ssmuon_s*", "v4 probe rule (small-model rule)": "w_v4_s[123]",
                        "Replay: Muon on the v4 wide screening-seed LR curve": "w_rp_s*", "Replay: small-model v4 screening curve, re-used at wide": "w_rpT_s*"},
                       "Schedule controls: wide model, seeds 1-3")
    if ss:
        from scipy import stats
        def tl(names): return [load(n)[n]["final_test"]["loss"] for n in names]
        S6 = (1, 2, 3, 7, 8, 9)
        nm = {"muon": lambda s: f"f_muon_s{s}" if s < 4 else f"cf_muon_s{s}", "ss": lambda s: f"f_ssmuon_s{s}",
              "v4": lambda s: f"f_v4_s{s}" if s < 4 else f"cf_v4_s{s}", "rp": lambda s: f"rp_v4trace_s{s}"}
        P("\nPaired t-tests on test loss over seeds 1, 2, 3, 7, 8, 9 (first minus second):\n")
        P("| comparison | difference | p | first better on |")
        P("|---|---|---|---|")
        for a, b in (("v4", "muon"), ("v4", "ss"), ("ss", "muon"), ("rp", "muon"), ("v4", "rp")):
            try:
                xa = tl([nm[a](s) for s in S6]); xb = tl([nm[b](s) for s in S6])
            except KeyError:
                continue
            P(f"| {a} vs {b} | {st.mean(xa) - st.mean(xb):+.4f} | {stats.ttest_rel(xa, xb).pvalue:.3g} | {sum(x < y for x, y in zip(xa, xb))}/6 |")
        if wss:
            P("\nWide model, tests on test loss (first minus second; seeds 1-3):\n")
            P("| comparison | difference | Welch p | paired p | first better on |")
            P("|---|---|---|---|---|")
            for lab, pa, pat in (("v4 vs shape-tuned Muon, wide search", "w_v4_s{}", "w_f_ssmuon_s{}"), ("v4 vs shape-tuned Muon, transferred", "w_v4_s{}", "w_ssmuonT_s{}"),
                                 ("v4 vs default-shape Muon, wide search", "w_v4_s{}", "w_f_muon_s{}"), ("v4 vs transferred-LR Muon", "w_v4_s{}", "w_muonT_s{}"),
                                 ("wide replay vs default-shape Muon, wide search", "w_rp_s{}", "w_f_muon_s{}"), ("wide replay vs v4", "w_rp_s{}", "w_v4_s{}"),
                                 ("small-curve replay vs default-shape Muon, wide search", "w_rpT_s{}", "w_f_muon_s{}"), ("small-curve replay vs transferred-LR Muon", "w_rpT_s{}", "w_muonT_s{}")):
                try:
                    xa = tl([pa.format(s) for s in (1, 2, 3)]); xb = tl([pat.format(s) for s in (1, 2, 3)])
                except KeyError:
                    continue
                P(f"| {lab} | {st.mean(xa) - st.mean(xb):+.4f} | {stats.ttest_ind(xa, xb, equal_var=False).pvalue:.3g} | {stats.ttest_rel(xa, xb).pvalue:.3g} | {sum(x < y for x, y in zip(xa, xb))}/3 |")
        # search-cost figure for the paper: expected best val loss vs tuning runs, v4 at zero runs
        fig, ax = plt.subplots(figsize=(6, 3.8))
        for pre, lab, col in (("s_muon", "Muon: peak-LR search", "C0"), ("ss_muon", "Muon: peak LR + schedule-shape search", "C1"), ("s_adago", "AdaGO: eta search", "C2")):
            rs = load(f"{pre}_[0-9]*")
            if not rs: continue
            ev = expected_best_of_n([r["final_val"]["loss"] for r in rs.values()], len(rs))
            ax.plot(range(1, len(ev) + 1), ev, "o-", ms=3, color=col, label=lab)
        v4s0 = load("v4_cap_sd0").get("v4_cap_sd0")
        if v4s0:
            ax.axhline(v4s0["final_val"]["loss"], color="C3", ls="--", label="v4 probe rule, zero tuning runs")
        ax.set_xlabel("tuning runs (seed 0)"); ax.set_ylabel("expected best val loss"); ax.set_ylim(3.8, 4.3)
        ax.set_title("search cost, small model"); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(f"{OUT}/fig/paper_search_cost.png", dpi=130); plt.close(fig)
    # tuning cost
    if searches:
        P("\n### Search cost: expected best val loss after n random-search trials (averaged over trial orderings)\n")
        P("| method | " + " | ".join(f"n={n}" for n in range(1, 9)) + " |")
        P("|---|" + "---|" * 8)
        for pre, (best, rs) in searches.items():
            if pre == "s_adamw": continue
            ev = expected_best_of_n([r["final_val"]["loss"] for r in rs.values()], 8)
            P(f"| {pre} | " + " | ".join(f"{v:.4f}" if v < 1e8 else "div" for v in ev) + " |")
        # figure: best val loss vs number of tuning runs
        fig, ax = plt.subplots(figsize=(6, 3.8))
        labels = {"s_muon": "Muon (random LR search)", "s_adamwE": "AdamW eps 1e-10", "s_adago": "AdaGO (random eta search)"}
        for pre, (best, rs) in searches.items():
            if pre not in labels: continue
            vals = [r["final_val"]["loss"] for r in rs.values()]
            if pre == "s_adamwE":
                vals = vals + [r["final_val"]["loss"] for r in load("s_adamw_[0-9]*").values()]
                labels[pre] = "AdamW (both searches)"
            ev = expected_best_of_n(vals, len(vals))
            ax.plot(range(1, len(ev) + 1), ev, "o-", label=labels[pre])
        order = ["k_10", "k_25", "k_50", "k_100", "k_once", "c_k25_c0.1", "c_k25_c0.25", "c_k25_c0.5", "c_k100_c0.1", "c_k100_c0.25", "c_k100_c0.5", "k_once_valid"]
        allc = {**load("k_*"), **load("c_*")}
        seq = [allc[o]["final_val"]["loss"] for o in order if o in allc]
        if seq:
            bsf = [min(seq[:i + 1]) for i in range(len(seq))]
            ax.plot(range(1, len(bsf) + 1), bsf, "s-", color="C3", label="curvature probe (calibration runs, in run order)")
        ax.set_xlabel("full training runs spent on tuning (small model)"); ax.set_ylabel("best val loss so far")
        ax.set_ylim(3.8, 4.8); ax.legend(fontsize=7); ax.set_title("search cost (seed 0)")
        fig.tight_layout(); fig.savefig(f"{OUT}/fig/search_cost.png", dpi=130); plt.close(fig)
    open(f"{OUT}/analysis_{which}.md", "w").write("\n".join(lines) + "\n")
