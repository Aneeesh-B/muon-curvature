# Initial experiment scope

## Question and boundary

Is a finite-difference directional-curvature estimate numerically meaningful for
Muon, and does its proposed matrix step improve loss beyond the measured batch?
Implement that mechanism before spending the proposal's larger GPU budget.
NanoGPT is suitable for this contained experiment; moving to nanochat changes
the architecture, batching, optimizer implementation and update scaling and needs
its own checks. The proposal's intended 10M / 50M-token study is a later phase.

## Phase 0: correctness

Use an analytic quadratic to verify `q = D^T H D` and `eta = <g,D>/q`.
Check exact parameter/gradient/RNG restoration, no momentum advance in prepare
or probe, one state advance on commit, optimizer serialization, negative
curvature and non-descent fallback, explicit clamping and tied-weight ownership.
With a dropout-enabled NanoGPT, compare the entire model, both optimizer states
and RNG states after training with and without diagnostics. They must match.

The model has no mutable BatchNorm-like buffers. A generic optimizer port must
account for any model buffers that change during forward.

## Phase 1: tiny-model diagnostics

- Dataset: Tiny Shakespeare character modeling, contiguous 90/5/5 splits.
- Model: 2 layers, 2 heads, width 64, context 64; 106,880 parameters.
- Batch: 8 sequences (512 tokens/update), FP32, no dropout or decay.
- Fixed-Muon trajectory: LR 0.02. Anchors at steps 0, 50 and 100.
- Perturbation: relative aggregate matrix norm `rho=1e-3`; sweep `1e-4,1e-3,1e-2`.
- At each anchor: freeze D, scan zero, a log-like LR grid and accepted predicted
  rates; score the original probe batch, three fresh training minibatches and
  three validation minibatches with common windows/randomness per scan.
- Repeat with the saved scalar at ages 1, 10, 50 and 100. Rebuild the CURRENT D
  from that checkpoint's gradient/momentum; do not reuse an old direction.
- Freeze auxiliary parameters and decay during scans. This isolates the matrix
  curvature claim; the end-to-end comparison includes companion AdamW updates.

Log raw and accepted estimates, curvature sign, fallback/clamp frequency,
gradient difference / gradient norm, predicted loss delta, actual probe/fresh
delta, best scanned rate, and regret to that finite grid. This is an empirical
grid comparison, not proof of the global one-dimensional optimum.

## Phase 2: end-to-end smoke comparison

200 updates / 102,400 training tokens per run. Three paired seeds, same initial
weights, batch stream, companion optimizer and held-out validation windows.

| Family | Configurations | Runs |
|---|---|---:|
| Fixed Muon | LR 0.005, 0.02, 0.08 | 9 |
| One-time calibration | step 0 | 3 |
| Periodic calibration | k=10, k=50, from step 0 | 6 |
| AdamW reference | LR 0.001 | 3 |

The initialization probe is an intentional stress test. Initial LR 0.02,
absolute bounds `[1e-5,0.2]`, safety factor 1. No tuning based on test loss.

Measure synchronized ordinary-training, probe, diagnostic and evaluation time
separately. Equal tokens and within-run probe overhead are available now. A
best-of-three fixed-rate comparison spends three times the training search
compute; this pilot does not yet construct equal-search-compute performance
curves. Report every fixed-rate trial, including failed configurations.

## Phase 3: slightly larger model, delayed calibration

4 layers, 4 heads, width 128, context 128; 812,288 parameters. Batch 16, 2,000
updates / 4,096,000 training tokens. Probe starts at step 100; default interval
50. This removes initialization-only calibration as the sole explanation.
The four-method one-seed run is a follow-up diagnostic, not a tuned benchmark.
The next comparison is the same three-rate, three-seed pilot grid at this size,
with anchors at steps 100, 900 and 1800 and eight fresh batches per scan.

## Decision gates

Proceed to ~10M parameters only if:

1. Correctness tests pass and finite differences remain stable over a reasonable
   probe-distance range. Validate GPU FP32 independently before using AMP.
2. Fresh-batch regret is small across checkpoints, not only probe-batch loss.
3. Repeated calibration helps beyond one-time calibration enough to justify
   measured probe overhead. Ideally below 10% in the actual GPU setting.
4. Three-seed validation results approach the best fixed-rate grid point, with
   all calibration/search costs disclosed. A provisional target is within 0.02
   nats/character on this pilot, chosen as an engineering gate, not a theorem.

If probes fit their own batch but hurt fresh batches, first investigate a larger
probe batch (both base and perturbed gradients must use that batch), smaller
safety factors, and delayed starts as explicit ablations. Do not silently tune
the safeguards and claim a tuning-free optimizer. If curvature varies strongly
with distance, investigate finite precision and local-model error first.

## Deferred work

AdaGO and a forward-only probe baseline; proper LR search for AdamW; nonzero
decay and schedules; gradient accumulation; AMP/BF16 and GPU cost benchmarks;
multi-step branch continuations; ~10M models; architecture/width transfer;
equal-total-search-compute curves; and final held-out test evaluation.

## Proposal review notes

The main design is implementable. Clarify that descent depends on `<g,D>`, not
the sign of a matrix D. Require BOTH positive numerator and positive curvature;
their ratio being positive alone is insufficient. Specify the exact Muon layer
scaling, perturbation norm, LR bounds and warmup/start policy. Include companion
AdamW/decay interactions as a limitation of the matrix-only quadratic. The
Methods section still has a citation TODO for Roulet et al.; its stated autodiff
costs and the bibliography should be independently checked before submission.
This implementation uses the document as a proposed research design, not as
instructions to carry out its entire 50–60-run study.
