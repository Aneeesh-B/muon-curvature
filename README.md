# Curvature-calibrated Muon

Test whether occasional directional-curvature probes choose useful Muon learning rates and earn back their compute compared with learning-rate search. The method keeps Muon’s update direction and layer scaling, and uses a finite-difference estimate of slope divided by curvature to choose one shared scalar for the matrix updates.

## Models and data

| Role | Model | Data |
|---|---|---|
| Smoke checks and diagnostics | NanoGPT, **0.81M parameters** ([config](configs/tiny.json)) | Tiny Shakespeare, character-level next-token prediction |
| Main GPU comparison | NanoGPT, **9.72M parameters**, four layers, width 448, context 128, batch size 16 ([config](configs/wide.json)) | Same Tiny Shakespeare data and fixed splits |
| Optional transfer study | Larger nanochat model, if the groupmate’s setup is available | Its original corpus and tokenizer. Report this as a separate benchmark. |

Use FP32 and the same model, initialization, ordinary training batches, and evaluation splits within each paired seed. Keep the Muon direction and companion AdamW settings fixed across Muon variants. Use tuning validation to select search candidates, reporting validation to compare methods, and the reserved test split after the protocol and endpoints are frozen.

## Primary comparison

| Arm | What to test |
|---|---|
| **Fixed Muon** | Default matrix LR of 0.02. |
| **Searched Muon** | Tune a constant matrix LR. Charge the search and the selected trial’s continuation to the same compute cap. |
| **Same-batch curvature Muon** | Estimate the step size on the minibatch that produced the Muon direction, reusing the ordinary gradient. |
| **Independent-batch curvature Muon** | Estimate the step size on a separately sampled training minibatch. Compute both probe gradients on that batch and preserve the ordinary training stream. |
| **Searched AdamW** | Apply AdamW to all parameters and tune its LR under the same compute cap. |
| **Searched AdaGO** | Use the reference adaptive matrix update and tune its step-size settings under the same compute cap. Use the same companion AdamW as Muon. [AdaGO, Algorithm 2](https://arxiv.org/html/2509.02981v1#S2) |

The two curvature arms should differ only in the batch used for measurement. Match their probe cadence, distance, bounds, and invalid-estimate fallback. Hold the accepted LR between probes. The primary arms use constant LRs between changes and run without backtracking or a decay schedule.

## Measurement standards

- **Compute efficiency.** Use at least three paired seeds. Measure all six arms at the same low, middle, and high cumulative compute caps. The 5090 targets are approximately **112.54, 225.08, and 450.16 trillion estimated matmul FLOPs**. Charge ordinary training, Newton–Schulz, extra probe passes, all search trials, and tuning validation. Report the actual charged FLOPs and synchronized wall time.
- **Effect of the estimate.** Compare fixed Muon and both curvature arms at a common update count. At shared early, middle, and late checkpoints, freeze the weights and Muon direction. Score both proposed rates, the fixed rate, and a small LR grid on fresh batches that supplied neither estimate.
- **Deliverables.** Plot held-out cross-entropy in nats per character versus cumulative FLOPs. Show per-seed differences and means. Save resolved configs, actual compute and updates, search outcomes, probe validity and clamp rates, and failures. Better performance than the same-batch arm supports the batch change. Better performance than searched Muon at equal total compute supports the tuning-cost claim.

## Optional ablations

- **Backtracking.** Compare independent-batch curvature with and without third-batch backtracking. Include a fixed-proposal backtracking control if testing whether curvature adds value beyond the check.
- **Scheduled next-batch probing.** Test the groupmate’s schedule against schedule-only Muon and scheduled same-batch probing. Use a common schedule and horizon. Treat replay as a separate follow-up and include the cost of constructing the curve.

## Research integrity and sanity checks

Evaluate whether the method helps. Worse loss, a tie, or gains that disappear after charging probe compute are all valid outcomes. Freeze the methods, seeds, tuning rules, and compute caps before production. Report every planned seed and cap, including failures and exclusions with their reasons. Use tuning validation for tuning decisions and keep reporting and test results out of those decisions.

If a result looks surprising, check for concrete problems such as mismatched initialization or batch streams, incorrect LR application or probe-state restoration, invalid losses, and missing FLOP charges. Apply the same checks to every arm. If you confirm a bug, record it, retain the original logs, and rerun all affected comparisons under the corrected protocol. If the checks pass, report the result and its uncertainty. Label any changes inspired by production results as follow-up experiments.

## Handoff status

The [RTX 4070 overnight results](results/rtx-4070-2026-11-24/README.md) cover 162 measurements across nine paired seeds. The results directory includes the combined CSV, loss plot, per-update FLOP logs, frozen-direction diagnostics, and the exact runner source used for the comparison.

The [5090 experiment plan](docs/5090-overnight-experiment-plan.md) gives the shared protocol. The [4070 experiment plan](docs/4070-overnight-experiment-plan.md) gives coordinated seeds, smaller cap options, and a timing-based preflight. The current runner supports wall-time and matched-update pilots. The production comparison still needs the unguarded independent-batch arm, a shared fallback for the two probe arms, AdaGO, and FLOP cap enforcement. The [earlier overnight runbook](docs/overnight.md) describes the existing pilot queue.

For prior evidence, see the [correctness audit](results/audit_verified/FINDINGS.md), [guarded pilot](results/guarded_pilot/README.md), and [guarded-control ablation](results/guarded_ablation/README.md).
