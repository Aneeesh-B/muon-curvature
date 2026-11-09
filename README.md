# Curvature-calibrated Muon on tiny NanoGPT

A local CS 182 research pilot for **Can Occasional Curvature Probes Reduce
Learning-Rate Tuning for Muon?** Uses the actual pinned NanoGPT model with a small
FP32, single-device training harness. Start with the measured results in
[`results/INITIAL_FINDINGS.md`](results/INITIAL_FINDINGS.md) and the experiment
scope in [`docs/experiment.md`](docs/experiment.md).

## Setup

From this repository directory:

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -e .
.venv/bin/python -m curvature_muon.data
.venv/bin/python -m unittest discover -s tests -v
```

Install a CUDA-compatible PyTorch build for your GPU machine using the
[official PyTorch instructions](https://pytorch.org/get-started/locally/) before
the editable install if needed. No transformers, tokenizer package, W&B account,
or datasets package is needed. Python >=3.10, PyTorch >=2.2, NumPy >=1.24.
Locally the experiments were run with the existing parent environment at
`../.venv/bin/python` (Python 3.14.5, PyTorch 2.15.0, CPU).

## Reproduce the initial experiment

```bash
# 21 very short matched-token runs: 7 configurations x 3 seeds.
.venv/bin/python -m curvature_muon.suite \
  --config configs/smoke.json --out runs/smoke_pilot --pilot --seeds 0 1 2

# Frozen-direction scans and all stale ages (0, 1, 10, 50, 100).
# 201 steps covers the age-100 scan for the anchor at step 100.
.venv/bin/python -m curvature_muon.train \
  --config configs/smoke.json --mode fixed --steps 201 --diagnostics \
  --out runs/diagnostics_complete

.venv/bin/python -m curvature_muon.report runs/smoke_pilot \
  --scans runs/diagnostics_complete/scans.jsonl --out results/smoke
```

All output directories must be new/empty; existing runs are never overwritten.
An experiment uses 106,880 parameters and 102,400 training tokens per smoke run.
The diagnostic trajectory has one additional update to cover the final scan.

## Larger tiny-model pilot

```bash
# Four methods, one seed, ~0.81M parameters, 4.096M tokens per run.
# Probe start is step 100 here, unlike the initialization stress test above.
.venv/bin/python -m curvature_muon.suite \
  --config configs/tiny.json --out runs/tiny_seed0 --device cuda

# Recommended next GPU comparison: 7 configurations x 3 seeds.
.venv/bin/python -m curvature_muon.suite \
  --config configs/tiny.json --out runs/tiny_pilot --pilot \
  --device cuda --seeds 0 1 2

.venv/bin/python -m curvature_muon.train \
  --config configs/tiny.json --mode fixed --diagnostics \
  --out runs/tiny_diagnostics --device cuda

.venv/bin/python -m curvature_muon.report runs/tiny_pilot \
  --scans runs/tiny_diagnostics/scans.jsonl --out results/tiny_pilot
```

Use `--device cpu` for local runs. MPS, AMP, compilation and distributed
training are deliberately outside this initial implementation. GPU code is
provided but has not been validated on a physical GPU in this workspace.

## Optimizer rule

For hidden matrices only, build the momentum-based, layer-scaled Muon direction
`D` once. The matrix update is `W <- W - eta*D` before decoupled decay.

```text
epsilon = rho * ||W_matrix||_F / ||D||_F
s       = sum <g_i, D_i>
q       = sum <D_i, g_i(W + epsilon D) - g_i(W)> / epsilon
eta_raw = s / q
```

The perturbed gradient uses the same minibatch and random state as the original
forward. Probe parameters restore exactly using saved copies even on exceptions.
`autograd.grad` does not overwrite training gradients. Momentum advances once,
when the actual update commits; probing never advances either optimizer.

- `fixed`: fixed Muon rate, default 0.02.
- `once`: calibrate at `probe_start`, then freeze the accepted rate.
- `periodic`: calibrate every `probe_every` updates starting at `probe_start`;
  freeze the entire accepted rate between probes.
- `adamw`: all parameters with AdamW, a reference at one rate (not tuned).

The default accepted rate is `clip(eta_raw, 1e-5, 0.2)` with safety factor 1.
Nonpositive curvature, non-descent direction, nonfinite estimates, a zero norm,
or a numerically unresolved gradient difference reuse the previous rate. Bounds,
initial rate, companion AdamW rate and probe distance **are still hyperparameters**.
No extra line search, LR schedule, gradient clipping or hidden smoothing is used.
Reusing a previous rate does not guarantee descent when momentum makes the
current direction non-descent.

Embeddings (including positional embeddings), tied output head and normalization
vectors use companion AdamW at 0.001, betas `(0.9, 0.95)`. Zero weight decay and
dropout are defaults to isolate calibration; nonzero decay is implemented as
`W*(1-eta*wd)-eta*D` and is excluded from the curvature model. Muon uses the
aspect-ratio convention in KellerJordan/Muon, with FP32 Newton-Schulz; see
[`third_party/README.md`](third_party/README.md) for precise provenance/differences.

## Outputs and interpretation

Each run records resolved config, software/hardware, dataset hash, source hashes,
seed and revision information in `manifest.json`; all step losses/probes/rates
in `train.jsonl`; fixed-window held-out validation losses and separated timings
in `eval.jsonl`; and model/optimizer/RNG states in `final.pt`. Diagnostic runs
also save anchor checkpoints and `scans.jsonl` with the entire candidate grid,
quadratic predictions, same/fresh-batch deltas, staleness and distance sweeps.
Diagnostic checkpoints are pre-update snapshots for investigation; a general
resume CLI is not implemented.

`report` emits CSV and Markdown tables. `results/` is versionable; large raw
`runs/`, downloaded text and prepared arrays are Git-ignored. The test split is
prepared separately and never loaded by the training/diagnostic code.

This experiment tests implementation correctness and whether curvature-selected
steps generalize to fresh batches. It does not establish reduced tuning cost at
10M parameters, equal-compute superiority, or transfer to nanochat. AdaGO,
forward-only probes and branching multi-step continuations remain follow-ups.
