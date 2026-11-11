# Initial findings

This repo contains a working implementation and an executed CPU pilot. The
engineering approach is feasible. The first results do **not** support a
performance or learning-rate-tuning advantage for the current rule.

The short equal-compute comparison is now complete: four methods, one seed,
120 seconds per method. Default Muon scored 1.5290 reporting loss / 54.21%
next-character accuracy; random-search Muon 1.5468 / 53.72%; random-search AdamW
1.6406 / 50.89%; and adaptive Muon 2.1583 / 36.80%. All methods stayed under
their caps. See [`budget/comparison.md`](budget/comparison.md) and
[`budget/NOTES.md`](budget/NOTES.md). Its reporting split differs from the older
whole-validation measurements below, so do not compare their absolute values
as if they were the same evaluation.

## What was actually run

- Reported smoke comparison: 21 paired-seed runs, seven configurations, three seeds, 106,880
  parameters, 200 updates / 102,400 training tokens per run.
- A separate 201-update fixed-Muon diagnostic trajectory with three anchors,
  all five requested ages, candidate-rate scans and three probe distances.
- Four delayed-calibration comparison runs at 812,288 parameters, one seed,
  2,000 updates / 4,096,000 training tokens each. Treat this as exploratory.
- CPU FP32, Python 3.14.5, PyTorch 2.15.0, NumPy 2.5.4, four torch threads.
  Hardware/runtime details are recorded per run. No GPU results are claimed.

See [`smoke/summary.md`](smoke/summary.md), [`smoke/runs.csv`](smoke/runs.csv),
[`smoke/scans.md`](smoke/scans.md) and [`smoke/scans.csv`](smoke/scans.csv).
The small raw JSONL logs and manifests are retained in `smoke/raw/`; full model
checkpoints remain in the local ignored `runs/` directory.

## Three-seed smoke result

Final validation cross entropy, nats per character (mean ± sample SD):

| Method | Loss |
|---|---:|
| Fixed Muon, LR 0.005 | 2.5117 ± 0.0114 |
| Fixed Muon, LR 0.02 | 2.5236 ± 0.0198 |
| Fixed Muon, LR 0.08 | 2.6482 ± 0.0178 |
| AdamW, LR 0.001 reference | 2.5696 ± 0.0158 |
| One-time calibration at initialization | 2.7843 ± 0.0243 |
| Periodic calibration, k=10 | 2.7536 ± 0.0626 |
| Periodic calibration, k=50 | 2.7679 ± 0.0350 |

Measured within-run probe time / ordinary training time in the final verification
run averaged 8.0% for k=10 and 1.3% for k=50. These short CPU timings exclude evaluation, diagnostics,
startup and logging; measure GPU overhead afresh. Periodic k=10 fell back on
43 of 60 probes. Every initialization-only estimate hit the upper LR cap of 0.2.

## What the scans reveal

At the step-50 fixed-Muon checkpoint, the predicted rate is 0.05325:

- Probe-batch loss changes by **-0.0522** (improves).
- Mean loss on three fresh training minibatches changes by **+0.0294** (worsens).
- The best scanned fresh-batch rate is 0.003; saved-rate regret is 0.0296 nats.

At step 100, the predicted rate 0.06268 also improves the probe batch (-0.0897)
and worsens fresh batches (+0.0474). This is direct evidence of the distinction
the proposal aims to test: local same-batch agreement is insufficient for useful
training steps. The sample of fresh batches is small; this is a pilot observation.

The step-0 recommendation is capped at 0.2 and initially improves fresh loss.
Reusing that scalar after ten fixed-Muon updates raises fresh loss by +0.4774;
the best scanned rate then is 0.03. A stale initial recommendation can become
poor quickly even when the direction is rebuilt from current gradients.

At step 50, changing rho from 1e-4 to 1e-3 changes raw eta from 0.05305 to
0.05325; at step 100 from 0.06226 to 0.06268. These observations suggest that
the fresh-batch issue is not solely tiny-distance cancellation at these anchors.
Initialization is more distance-sensitive: raw eta rises from 0.418 at rho=1e-4
to 1.062 at rho=1e-2. Distance still needs explicit sensitivity analysis.

## Delayed calibration at 812,288 parameters

All four methods use the same initialization, 16-sequence batches, context 128,
and held-out windows. Calibration begins at step 100 and repeats every 50
steps for the periodic variant. Final validation loss (single seed):

| Method | Loss |
|---|---:|
| Fixed Muon, LR 0.02 | 1.5253 |
| One-time calibration | 1.7098 |
| Periodic calibration | 2.5271 |
| AdamW, LR 0.001 reference | 1.7042 |

See [`tiny/summary.md`](tiny/summary.md) and [`tiny/runs.csv`](tiny/runs.csv).
Both calibration methods initially accept 0.04110 at step 100. Periodic
calibration accepts 0.10175 at step 150, then falls back on every subsequent
probe (36/38 total probes invalid), retaining that large rate. This exposes a
specific weakness of the proposed fallback: keeping a previous rate can lock
training into a bad scale after an aggressive estimate. No rate in this run
was clamped, so hitting the upper cap is not required for the failure.

The delayed-start result weakens an initialization-only explanation. It does
not establish that every curvature-based variant fails; safeguards, batch size,
geometry and trajectory length remain experimental choices.

## What to do next

Keep the current implementation as the unmodified experimental reference. Use
the delayed-start comparison as evidence that initialization alone does not
explain this failure. Next investigate larger calibration batches, a conservative
rate-reduction policy on invalid probes, and explicit safety
factors as separate, paired-seed ablations. For larger calibration batches,
compute BOTH unperturbed and perturbed gradients on that batch and replay all
microbatch random states; mixing a small-batch g with a large-batch shifted g
would not estimate the intended curvature.

Do not move straight to the full 10M / 50M-token study, or call this tuning-free,
based on the present evidence. The available suite can run the next three-seed
GPU comparison when desired. AdaGO, tuned AdamW and equal-total-search-compute
curves are still required for the proposal's eventual comparative claim.

## Validation

Nine tests pass, including the analytic quadratic, fallback/bounds,
dropout/RNG restoration, tied parameter ownership, serialization, exception
restoration and a complete-trajectory comparison with/without diagnostics.
The scan-enabled smoke trajectory also reproduces the original fixed-Muon
validation losses through step 200. The final code reproduced all 21 smoke
training JSONL logs and all 15 directional scans byte for byte; timing records
vary, as expected. No final test-set results were inspected.
