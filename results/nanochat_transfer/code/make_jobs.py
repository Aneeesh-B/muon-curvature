"""Generates job files. Random-search draws are log-uniform with fixed RNG seeds (reproducible)."""
import json, math, random, sys, os
ROOT = os.path.expanduser("~/cs182_curvature_muon")
def loguni(rng, lo, hi, n):
    return [math.exp(rng.uniform(math.log(lo), math.log(hi))) for _ in range(n)]
def write(name, jobs):
    with open(f"{ROOT}/jobs/{name}.jsonl", "w") as f:
        for j in jobs: f.write(json.dumps(j) + "\n")
    print(name, len(jobs))
def search(prefix, optim, flag, lo, hi, n, seed, extra=()):
    rng = random.Random(seed)
    return [dict(name=f"{prefix}_{i:02d}", args=["--optim", optim, flag, f"{v:.6g}", "--seed", "0", *extra])
            for i, v in enumerate(loguni(rng, lo, hi, n))]
def best(prefix, flag):
    import glob
    rs = []
    for f in glob.glob(f"{ROOT}/results/{prefix}_[0-9]*.json"):
        r = json.load(open(f)); v = r["final_val"]["loss"]
        if math.isfinite(v): rs.append((v, r["args"][flag], os.path.basename(f)))
    v, hp, name = min(rs)
    print(f"best {prefix}: {name} {flag}={hp} val={v:.4f}")
    return hp

if __name__ == "__main__":
    phase = sys.argv[1]
    if phase == "diag_k":
        lr = best("s_muon", "matrix_lr")
        write("d_diag", [dict(name="d_muon_diag", args=["--optim", "muon", "--matrix-lr", f"{lr:.6g}", "--seed", "0",
                                                         "--diag-base-steps", "20,190,280"])])
        eps = sys.argv[2] if len(sys.argv) > 2 else "1e-3"
        write("k_abl", [dict(name=f"k_{k if k else 'once'}", args=["--optim", "curv", "--probe-every", str(k), "--probe-eps-rel", eps, "--seed", "0"])
                        for k in (10, 25, 50, 100, 0)])
    if phase == "c_abl":
        k = sys.argv[2]
        write("c_abl", [dict(name=f"c_k{k}_c{c}", args=["--optim", "curv", "--probe-every", k, "--probe-eps-rel", "1e-3",
                                                        "--probe-scale", c, "--seed", "0"]) for c in ("0.1", "0.25", "0.5")])
    if phase == "finals":
        k, eps = sys.argv[2], sys.argv[3]
        c = sys.argv[4] if len(sys.argv) > 4 else "1.0"
        lr_m = best("s_muon", "matrix_lr"); # tuned AdamW = best over both AdamW searches (epsilon 1e-8 and 1e-10)
        import glob as _g
        cand = []
        for pre in ("s_adamw", "s_adamwE"):
            for f in _g.glob(f"{ROOT}/results/{pre}_[0-9]*.json"):
                r = json.load(open(f)); v = r["final_val"]["loss"]
                if math.isfinite(v): cand.append((v, r["args"]["matrix_lr"], r["args"].get("adamw_eps", 1e-8), os.path.basename(f)))
        v_a, lr_a, eps_a, name_a = min(cand)
        print(f"best AdamW: {name_a} lr={lr_a} eps={eps_a} val={v_a:.4f}"); eta_g = best("s_adago", "adago_eta")
        jobs = []
        for sd in (1, 2, 3):
            jobs += [dict(name=f"f_muon_s{sd}", args=["--optim", "muon", "--matrix-lr", f"{lr_m:.6g}", "--seed", str(sd)]),
                     dict(name=f"f_adamw_s{sd}", args=["--optim", "adamw", "--matrix-lr", f"{lr_a:.6g}", "--adamw-eps", f"{eps_a:g}", "--seed", str(sd)]),
                     dict(name=f"f_adago_s{sd}", args=["--optim", "adago", "--adago-eta", f"{eta_g:.6g}", "--seed", str(sd)]),
                     dict(name=f"f_curv_s{sd}", args=["--optim", "curv", "--probe-every", k, "--probe-eps-rel", eps, "--probe-scale", c, "--seed", str(sd)])]
            if c != "1.0":  # also run the literal (uncalibrated) rule for transparency
                jobs.append(dict(name=f"f_curvc1_s{sd}", args=["--optim", "curv", "--probe-every", k, "--probe-eps-rel", eps, "--seed", str(sd)]))
        write("finals", jobs)
        # width transfer: everything except the wide Muon search/finals uses small-model settings unchanged
        W = ["--aspect-ratio", "128"]
        wj = []
        for sd in (1, 2, 3):
            wj += [dict(name=f"w_curv_s{sd}", args=["--optim", "curv", "--probe-every", k, "--probe-eps-rel", eps, "--probe-scale", c, "--seed", str(sd), *W]),
                   dict(name=f"w_muonT_s{sd}", args=["--optim", "muon", "--matrix-lr", f"{lr_m:.6g}", "--seed", str(sd), *W]),
                   dict(name=f"w_adago_s{sd}", args=["--optim", "adago", "--adago-eta", f"{eta_g:.6g}", "--seed", str(sd), *W])]
        write("wide_transfer", wj)
    if phase == "wide_finals":
        lr = best("w_s_muon", "matrix_lr")
        write("wide_finals", [dict(name=f"w_f_muon_s{sd}", args=["--optim", "muon", "--matrix-lr", f"{lr:.6g}", "--seed", str(sd), "--aspect-ratio", "128"]) for sd in (1, 2, 3)])
    if phase == "search_small":
        write("s_muon", search("s_muon", "muon", "--matrix-lr", 0.002, 0.08, 8, 1001))
        write("s_adamw", search("s_adamw", "adamw", "--matrix-lr", 3e-4, 3e-2, 8, 1002))
        write("s_adamwE", search("s_adamwE", "adamw", "--matrix-lr", 3e-4, 3e-2, 8, 1002, ("--adamw-eps", "1e-10")))
        write("s_adago", search("s_adago", "adago", "--adago-eta", 0.01, 1.0, 6, 1003))
    if phase == "q2_seeds":  # gap fill: 3 seeds for the Q2 probe-interval / probe-once ablation, first probe at step 25
        S = ["--probe-start", "25", "--probe-eps-rel", "1e-3"]
        jobs = []
        for sd in (1, 2, 3):
            for k in ("10", "50", "100"):  # literal rule c = 1 (k = 25 is f_curvc1S)
                jobs.append(dict(name=f"q2_k{k}_c1_s{sd}", args=["--optim", "curv", "--probe-every", k, *S, "--seed", str(sd)]))
            jobs.append(dict(name=f"q2_once_c1_s{sd}", args=["--optim", "curv", "--probe-every", "0", "--probe-once-retry", *S, "--seed", str(sd)]))
            # calibrated rule c = 0.5 (k = 25 is f_curvS)
            jobs.append(dict(name=f"q2_k100_c0.5_s{sd}", args=["--optim", "curv", "--probe-every", "100", *S, "--probe-scale", "0.5", "--seed", str(sd)]))
            jobs.append(dict(name=f"q2_once_c0.5_s{sd}", args=["--optim", "curv", "--probe-every", "0", "--probe-once-retry", *S, "--probe-scale", "0.5", "--seed", str(sd)]))
        write("q2_seeds", jobs)
    if phase == "search_wide_adamw":  # gap fill: tuned AdamW at the wide width (epsilon 1e-8, the small-model winner)
        write("w_s_adamw", search("w_s_adamw", "adamw", "--matrix-lr", 3e-4, 3e-2, 8, 1005, ("--aspect-ratio", "128")))
    if phase == "wide_adamw_finals":
        lr = best("w_s_adamw", "matrix_lr")
        W = ["--aspect-ratio", "128"]
        jobs = []
        for sd in (1, 2, 3):
            jobs.append(dict(name=f"w_adamwT_s{sd}", args=["--optim", "adamw", "--matrix-lr", "0.00330383", "--seed", str(sd), *W]))
            jobs.append(dict(name=f"w_f_adamw_s{sd}", args=["--optim", "adamw", "--matrix-lr", f"{lr:.6g}", "--seed", str(sd), *W]))
        write("wide_adamw_finals", jobs)
    if phase in ("search_shape", "search_shape_wide"):
        # Shape-tuned Muon baseline: random search over peak LR, warmdown shape, warmdown fraction, final LR fraction
        wide = phase == "search_shape_wide"
        rng = random.Random(1007 if wide else 1006); n = 8 if wide else 16
        pre = "w_ss_muon" if wide else "ss_muon"
        jobs = []
        for i in range(n):
            lr = math.exp(rng.uniform(math.log(0.005 if wide else 0.01), math.log(0.15)))
            shape = rng.choice(["linear", "cosine", "exp"])
            wd = rng.uniform(0.3, 0.95)
            fin = math.exp(rng.uniform(math.log(0.005), math.log(0.2)))
            a = ["--optim", "muon", "--matrix-lr", f"{lr:.6g}", "--lr-shape", shape, "--lr-warmdown-ratio", f"{wd:.4g}",
                 "--lr-final-frac", f"{fin:.4g}", "--seed", "0"] + (["--aspect-ratio", "128"] if wide else [])
            jobs.append(dict(name=f"{pre}_{i:02d}", args=a))
        half = (n + 1) // 2
        write(f"{pre}_a", jobs[:half]); write(f"{pre}_b", jobs[half:])
    if phase == "shape_finals":
        import glob as _g
        def bestshape(pre):
            rs = []
            for f in _g.glob(f"{ROOT}/results/{pre}_[0-9]*.json"):
                r = json.load(open(f)); v = r["final_val"]["loss"]
                if math.isfinite(v): rs.append((v, os.path.basename(f), r["args"]))
            v, name, a = min(rs, key=lambda t: t[0]); print(f"best {pre}: {name} val={v:.4f}")
            return ["--optim", "muon", "--matrix-lr", f"{a['matrix_lr']:.6g}", "--lr-shape", a["lr_shape"],
                    "--lr-warmdown-ratio", f"{a['lr_warmdown_ratio']:.4g}", "--lr-final-frac", f"{a['lr_final_frac']:.4g}"]
        B = bestshape("ss_muon")
        jobs = [dict(name=f"f_ssmuon_s{s}", args=B + ["--seed", str(s)]) for s in (1, 2, 3, 7, 8, 9)]
        jobs += [dict(name=f"w_ssmuonT_s{s}", args=B + ["--seed", str(s), "--aspect-ratio", "128"]) for s in (1, 2, 3)]
        write("shape_finals", jobs)
    if phase == "shape_finals_wide":
        import glob as _g
        rs = []
        for f in _g.glob(f"{ROOT}/results/w_ss_muon_[0-9]*.json"):
            r = json.load(open(f)); v = r["final_val"]["loss"]
            if math.isfinite(v): rs.append((v, os.path.basename(f), r["args"]))
        v, name, a = min(rs, key=lambda t: t[0]); print(f"best w_ss_muon: {name} val={v:.4f}")
        B = ["--optim", "muon", "--matrix-lr", f"{a['matrix_lr']:.6g}", "--lr-shape", a["lr_shape"],
             "--lr-warmdown-ratio", f"{a['lr_warmdown_ratio']:.4g}", "--lr-final-frac", f"{a['lr_final_frac']:.4g}", "--aspect-ratio", "128"]
        write("shape_finals_wide", [dict(name=f"w_f_ssmuon_s{s}", args=B + ["--seed", str(s)]) for s in (1, 2, 3)])
    if phase == "search_wide":
        write("w_s_muon", search("w_s_muon", "muon", "--matrix-lr", 0.002, 0.08, 6, 1004, ("--aspect-ratio", "128")))
