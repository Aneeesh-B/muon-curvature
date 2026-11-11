# Why adaptive Muon is losing: implementation and method audit

The estimator and training-state handling passed the targeted checks. The
controller does **not** ensure a better training learning rate: it optimizes a
local model of the current gradient batch, then reuses that scalar on other
batches/directions without an acceptance test. The observed failure is an
algorithm-design issue in the checked paths, not a discovered sign or state
corruption bug. Correct finite differences alone cannot guarantee lower
held-out or eventual training loss.

## What was verified

- Replayed updates 0 through 250 of the actual 812K-parameter budget experiment.
  All recorded training losses, probe quantities, accepted learning rates and
  fallback decisions reproduced exactly.
- Checked that repeating the unperturbed gradient evaluation through
  `autograd.grad` gave zero difference from the saved training gradient at all
  four audited checkpoints.
- Checked exact restoration of weights, gradients, both optimizers' states,
  the training batch generator, and the trial's random state after the audit.
- Compared `q = <D, (g(W+epsilon D)-g(W))/epsilon>` against independent FP64
  autodifferentiation of `D^T H D`, using math attention for double backward.
- Swept relative finite-difference distances in FP32 and FP64. Near the first
  two accepted updates, smaller distances converge to the autodiff reference.
  FP32 and FP64 finite differences at the same distance agree closely.
- Ran all 17 tests. New tests cover a coupled quadratic with cross-matrix
  Hessian terms, independent NanoGPT Hessian-vector products, and a convex
  quartic example in which an accurate local quadratic estimate still proposes
  a harmful step.

Muon momentum, Nesterov combination, Newton-Schulz coefficients and aspect-ratio
scaling were also compared with the pinned KellerJordan reference. The existing
documented FP32-versus-BF16 difference remains; all our fixed/adaptive Muon
variants use the same FP32 geometry. No benchmark policy was changed.

## Direct cause: a better probe-batch step can hurt fresh batches

At each point below, both candidate updates start at the SAME weights, use the
SAME frozen Muon direction and the SAME companion AdamW displacement. Fresh
loss averages eight independently sampled training minibatches. Negative delta
means improvement; positive delta means deterioration.

| Checkpoint | Candidate LR | Probe-batch loss delta, full joint update | Fresh-batch loss delta, full joint update |
|---|---:|---:|---:|
| Step 100 | Fixed 0.02 | -0.0681 | **-0.0024** |
| Step 100 | Estimated 0.04110 | **-0.0854** | +0.0275 |
| Step 150 | Fixed 0.02 | -0.0763 | **-0.0065** |
| Step 150 | Estimated 0.10175 | **-0.1228** | +0.0758 |

Thus the accepted updates really ARE better on their own minibatch, while
being worse on the fresh sample. There is no contradiction: these are different
loss functions. At step 150 the original directional slope `<g,D>` is 4.0648,
but the mean slope on fresh batches is only 0.5897. A direction partly fitted
to the gradient batch appears much more profitable on that batch.

Holding companion AdamW fixed gives the same conclusion: the step-150 adaptive
matrix update raises fresh loss by +0.0748, versus +0.0758 with AdamW included.
Companion-update interactions therefore do not explain away this failure.
The experiment uses zero weight decay, so omitted weight-decay curvature is not
the cause here either. These are short diagnostic counterfactuals, not proof
that any smaller rate will be best over a complete training trajectory.

## The curvature is approximate, but early numerical error is not the main cause

| Step | Original finite-difference q | Independent FP64 HVP q | Estimated eta | Eta using HVP |
|---|---:|---:|---:|---:|
| 100 | 105.1083 | 108.6335 | 0.04110 | 0.03976 |
| 150 | 39.9498 | 42.4960 | 0.10175 | 0.09565 |

The relative curvature error is 3.25% and 5.99%, respectively. Correcting it
does not bring the rates close to the fresh-batch grid optima (0.01 and 0.02).
At step 150 even the best scanned probe-batch rate, 0.08, worsens fresh loss.

The actual accepted displacement is 39 times and 61 times the probe distance,
respectively. The quadratic at step 150 predicts a matrices-only reduction of
0.2068, while the actual reduction is 0.1202. The best scanned same-batch rate
is 0.08 rather than 0.10175. The local parabola is useful but is not an exact
line search over the proposed displacement.

After the aggressive changes, distance sensitivity becomes substantially worse.
At steps 200 and 250 the independent HVP curvature is genuinely negative; the
invalid-curvature flags are not merely FP32 cancellation artifacts. Their
finite-distance curvature magnitudes are inaccurate, showing that the local
model also becomes less reliable along this trajectory.

## Fallback and holding the rate amplify the problem

The actual LR history is 0.02 -> 0.04110 -> 0.10175, followed by 42 fallback
decisions that retain 0.10175. Reusing a previous scalar does not establish
that it remains safe when the current model no longer has a valid positive
quadratic minimum along D.

Furthermore, each recommendation is applied to 50 successive updates, even
though the gradient batch and momentum-derived direction change on every
update. Local optimality for a single frozen direction/batch does not imply
optimality for those other updates. The instantaneous fresh-batch failures
already occur at the initial accepted steps, so staleness alone is not the
whole explanation.

Only 2.04 of the adaptive campaign's 118.99 seconds were spent inside probes.
Its best tuning checkpoint occurred at update 212 although it completed 2,284
updates. Reduced throughput alone cannot explain why the subsequent trajectory
failed to improve on that early checkpoint.

## Implications for the next method revision

The research hypothesis is still testable, but the current controller should
not be described as choosing an optimal or guaranteed-better learning rate.
Before another budget comparison, a defensible revision would explicitly test:

1. A calibration batch independent of the batch used to construct D, or a
   larger calibration pool. BOTH base and shifted gradients must use that
   calibration objective; mixing gradients from different batches breaks the
   finite-difference calculation.
2. An acceptance/backtracking safeguard for proposed steps, with held-out
   training-batch checks or a clearly justified conservative policy. Checking
   only the gradient batch would not detect the demonstrated fresh-batch issue.
3. A rate reduction or return to a conservative reference when the local model
   is invalid, instead of indefinitely retaining an aggressive prior rate.

These are algorithm changes with added compute/hyperparameters, not fixes to
the sign of the existing formula. Their cost must be charged in the next
equal-budget comparison. This audit preserves the original optimizer, settings
and published pilot results as the reference.

Reproduce with a NEW output directory:

```bash
../.venv/bin/python -m curvature_muon.audit --out runs/my-audit
```

Machine-readable evidence is in `audit.json` and `scans.csv`; the concise table
is in `audit.md`. Both targeted audit passes together used approximately 47
CPU seconds, plus the short unit tests; no full comparison was rerun.
