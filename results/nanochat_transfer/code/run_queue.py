"""Sequential, resumable job queue. Usage: python run_queue.py jobs/phaseX.jsonl
Each line: {"name": "...", "args": ["--optim", "muon", ...]}. Skips jobs whose results/<name>.json exists."""
import json, os, subprocess, sys, time
ROOT = os.path.expanduser("~/cs182_curvature_muon")
PY = f"{ROOT}/.venv/bin/python"
env = dict(os.environ, NANOCHAT_BASE_DIR=f"{ROOT}/base", OMP_NUM_THREADS="1", PYTHONUNBUFFERED="1", WSL_INTEROP="/run/WSL/2_interop")
for jf in sys.argv[1:]:
    for line in open(jf):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        job = json.loads(line)
        out = f"{ROOT}/results/{job['name']}.json"
        if os.path.exists(out):
            print(f"[queue] skip {job['name']}", flush=True); continue
        log = f"{ROOT}/logs/{job['name']}.log"
        cmd = [PY, "-m", "scripts.curv_train", "--out", out, "--tag", job['name']] + job["args"]
        print(f"[queue] {time.strftime('%F %T')} start {job['name']}: {' '.join(job['args'])}", flush=True)
        t = time.time()
        with open(log, "w") as f:
            rc = subprocess.call(cmd, cwd=f"{ROOT}/nanochat", env=env, stdout=f, stderr=subprocess.STDOUT)
        print(f"[queue] {time.strftime('%F %T')} done {job['name']} rc={rc} {(time.time()-t)/60:.1f} min", flush=True)
print("[queue] ALL DONE", flush=True)
