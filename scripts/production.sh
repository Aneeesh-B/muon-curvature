#!/usr/bin/env bash
# 5090 production queue (docs/5090-overnight-experiment-plan.md). Resumable:
# finished campaigns are skipped, interrupted folders are moved aside.
set -euo pipefail
cd "$(dirname "$0")/.."
python="${PYTHON:-.venv/bin/python}"
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
run() { echo "=== $(date -Is) $*"; "$python" -m curvature_muon.production "$@" --device cuda; }
"$python" -m unittest discover -s tests
run verify
run preflight
run core --seeds 3 4 5
run core --seeds 6 7                      # plan: add paired seeds 6 and 7 if time remains
run ablations --seeds 3 4 5 6 7           # backtracking + scheduled probing (optional ablations)
run replay --seeds 3 4 5 6 7              # follow-up: frozen replay curve from dev seeds 10, 11
run fidelity --seeds 3 4 5 6 7            # Q1: finite difference vs exact HVP at the mechanism anchors
run followup --seeds 3 4 5 6 7            # follow-up: extended grids where searches won on a grid edge
run test                                  # reserved test split, after everything above is frozen
"$python" -m curvature_muon.production_report
echo "=== $(date -Is) queue finished"
