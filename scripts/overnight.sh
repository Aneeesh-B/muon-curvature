#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

profile="${1:-}"
case "$profile" in
  4070|5090|5090-wide) ;;
  *) echo "Usage: bash scripts/overnight.sh {4070|5090|5090-wide}" >&2; exit 2 ;;
esac

python="${PYTHON:-.venv/bin/python}"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
root="runs/overnight-$profile-$stamp"
mkdir -p runs
mkdir "$root"
exec > >(tee "$root/launcher.log") 2>&1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
echo "Writing to $root"
nvidia-smi
"$python" - <<'PY'
import torch

assert torch.cuda.is_available(), "Install a CUDA-enabled PyTorch build"
x = torch.ones((16, 16), device="cuda")
assert (x @ x).sum().item() == 4096
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())
PY
"$python" -m curvature_muon.data
"$python" -m unittest discover -s tests

matched=(guarded_muon guarded_base_muon scheduled_fixed_muon scheduled_same_batch_muon scheduled_heldout_muon)

run_model() {
  local size="$1" config="$2" seconds="$3" matched_allowance="$4"
  local steps seed
  local common=(-m curvature_muon.budget --config configs/overnight.json --device cuda --training-config "$config")

  "$python" "${common[@]}" --out "$root/$size-smoke-updates" --updates 151 --methods "${matched[@]}"
  "$python" "${common[@]}" --out "$root/$size-smoke-search" --seconds 90 --methods muon_search adamw_search

  # Pick the matched horizon using runtime, before production losses exist.
  steps="$("$python" - "$root/$size" "$matched_allowance" <<'PY'
import json
from pathlib import Path
import sys

base, allowance = sys.argv[1], float(sys.argv[2])
matched = list(Path(base + "-smoke-updates").glob("*/result.json"))
search = list(Path(base + "-smoke-search").glob("*/result.json"))
assert len(matched) == 5 and len(search) == 2, "Incomplete preflight"
results = [json.loads(path.read_text()) for path in matched]
for result in results:
    assert result["completed_steps"] == 151, result["method"]
    if result["method"] != "scheduled_fixed_muon":
        assert result["trials"][0]["probes"] > 0, result["method"]
for path in search:
    result = json.loads(path.read_text())
    assert result["budget_met"], f"Search preflight over budget: {path}"
    assert all(t["steps"] > 0 and t["best"] for t in result["trials"]), f"Empty search trial: {path}"
    winner = result["trials"][result["continued_trial_id"]]
    assert winner["steps"] > result["continued_from_step"], f"No continuation: {path}"

seconds_per_update = sum(r["elapsed_seconds"] for r in results) / 151
for horizon in (2000, 1000):
    if 1.3 * 3 * horizon * seconds_per_update <= allowance:
        print(horizon)
        break
else:
    print(0)
PY
)"
  echo "$size production plan: $seconds seconds/method for 5 methods × 3 seeds; matched horizon $steps (0 means deferred)."

  for seed in 0 1 2; do
    "$python" "${common[@]}" --out "$root/$size-wall-seed$seed" --seed "$seed" --seconds "$seconds"
  done
  if [[ "$steps" -gt 0 ]]; then
    for seed in 0 1 2; do
      "$python" "${common[@]}" --out "$root/$size-matched-seed$seed" --seed "$seed" \
        --updates "$steps" --methods "${matched[@]}"
    done
  fi
}

if [[ "$profile" != 5090-wide ]]; then
  run_model small configs/tiny.json 480 5400
fi
if [[ "$profile" != 4070 ]]; then
  run_model wide configs/wide.json 720 3600
fi
echo "Queue finished. Review campaign reports and any failures in $root."
