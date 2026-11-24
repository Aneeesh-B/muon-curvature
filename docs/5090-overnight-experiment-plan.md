# RTX 5090 overnight experiment plan

**Goal.** Test whether occasional directional-curvature estimates improve Muon’s choice of learning rate and whether the improvement is worth the compute spent on probes. Use one RTX 5090, one 9.72M-parameter NanoGPT, three paired seeds, and three cumulative FLOP caps. Run the six arms below before adding extensions.

**Launch status.** This document defines the production protocol. The current [budget runner](../curvature_muon/budget.py) supports wall-time and matched-update runs. Production requires an unguarded independent-batch arm, a common fallback for both probe arms, the AdaGO baseline, and the FLOP counter and cap scheduler below. The [existing overnight launcher](../scripts/overnight.sh) runs the earlier protocol. Use its setup checks if helpful and treat its results as wall-time or matched-update pilots.

## 1. Freeze the experimental setup

- **Model.** Use [configs/wide.json](../configs/wide.json). Four layers, width 448, seven heads, context 128, batch size 16, vocabulary 65, and 9,724,288 parameters. Each ordinary update processes 2,048 characters. Use FP32 with TF32 disabled, zero dropout, and zero weight decay for all arms.
- **Data.** Use the repository’s checksum-verified Tiny Shakespeare character data and contiguous train, validation, and test splits. Split validation in half. The first half selects search candidates. Use the second half for reporting. Keep test data closed until the protocol and all reported checkpoints are frozen.
- **Pairing.** Production seeds are **3, 4, and 5**. Use seed 0 for preflight. Within each production seed, start each arm and search candidate from the same initialization and ordinary training-batch stream. Give independent calibration its own seeded training-batch stream. Randomize method order within seed and record it.
- **Muon configuration.** Use the repo’s Muon direction, layer scaling, momentum 0.95, and five Newton–Schulz iterations. Use the same companion AdamW for embeddings, normalization, and the tied output head in all Muon and AdaGO arms. Its LR is 0.001 and its betas are (0.9, 0.95).

## 2. Run six arms

| Arm | Policy |
|---|---|
| Fixed Muon | Matrix LR 0.02 throughout. |
| Searched Muon | Search constant matrix LRs 0.003, 0.01, 0.02, 0.04, and 0.08. |
| Same-batch curvature Muon | Estimate the LR on the batch that built the Muon direction. Reuse that batch’s base gradient and compute one extra shifted gradient. |
| Independent-batch curvature Muon | Estimate the LR on a separate draw from the training split. Use this same draw for both gradients. Keep the ordinary training-batch stream intact. |
| Searched AdamW | Train all parameters with AdamW, betas (0.9, 0.95), epsilon 1e-8, and a constant LR. Search 0.0003, 0.001, 0.003, 0.006, and 0.01. |
| Searched AdaGO | Use the matrix update in Algorithm 2 of the AdaGO paper and tune five (eta, epsilon) pairs. Use the same matrix parameter set and companion AdamW as Muon. |

**Shared probe policy.** Start at LR 0.02. Probe at zero-based update 100 and every 50 updates thereafter. Freeze the Muon direction for the probe, perturb by 0.001 times the Frobenius norm of the matrix weights, and use the repo’s directional finite difference. Apply the clipped ratio of slope to curvature in [1e-5, 0.2] when the estimate is valid. On an invalid estimate, use 0.02. Hold the chosen scalar until the next probe. Both arms use the same validity check and perturbation. Third-batch backtracking, LR schedules, and replay are separate experiments.

**AdaGO port.** Use the reference’s orthogonalized exponential moving average, with momentum 0.95. Use each matrix’s Frobenius gradient norm, clipping constant gamma = 1, initial accumulator value v0 = 1e-6, and step-size floor epsilon. Accumulate squared clipped norms and apply the reference rule `max(epsilon, eta * clipped_norm / accumulator)`. Use the repo’s five Newton–Schulz iterations to approximate orthogonalization. Apply the Algorithm 2 direction and step rule without importing the repo Muon-specific Nesterov update or extra aspect-ratio multiplier. Use the following proposed NanoGPT tuning grid, frozen before production. [(AdaGO paper, Algorithm 2 and §4)](https://arxiv.org/html/2509.02981v1#S2)

| Candidate | eta | epsilon |
|---|---:|---:|
| 1 | 0.02 | 0.0001 |
| 2 | 0.05 | 0.0005 |
| 3 | 0.10 | 0.0005 |
| 4 | 0.30 | 0.0005 |
| 5 | 0.30 | 0.003 |

**Search protocol.** For each searched method, seed, and cap, run a fresh search from the paired initialization. Give each of five candidates **8% of the total cap**, including one end-of-slot pass on the tuning validation half. Choose the completed candidate with the lowest finite tuning loss. Continue that candidate’s live weights, optimizer, and RNG state with the rest of the cap. Charge work already performed by failed candidates and log their failures. Use the final eligible checkpoint of the continued candidate for reporting. Record all candidates and flag a search whose winner sits on an edge of its grid. Keep the grids frozen after preflight.

## 3. FLOP caps and accounting

Use the exact integer caps below under the stated **estimated matmul FLOP** convention. TFLOPs in this table are cumulative operation counts, with one TFLOP equal to 1e12 FLOPs.

| Measurement | Ordinary Muon update equivalents | Cap in FLOPs | Approximate TFLOPs |
|---|---:|---:|---:|
| Low | 500 | 112540123136000 | 112.54 |
| Middle | 1,000 | 225080246272000 | 225.08 |
| High | 2,000 | 450160492544000 | 450.16 |

If preflight projects that the primary matrix will exceed the time allowance, freeze all arms at the fallback caps **90032098508800**, **180064197017600**, and **360128394035200** FLOPs, corresponding to 400, 800, and 1,600 ordinary Muon updates. The decision uses timing and memory only, before production losses are viewed.

Count one multiply-add as two FLOPs. With batch size B, context T, width d, layer count L, and vocabulary V, charge the dense-transformer matmuls in a forward as `F(B) = 2*B*T*(12*L*d*d + d*V) + 4*B*L*T*T*d`. The attention term uses the same dense convention across all methods. Charge a full forward/backward as `3*F(B)`. For a probe gradient that targets the hidden matrices only, subtract the output-head weight-gradient matmul, `2*B*T*d*V`, from that amount. Charge each Newton–Schulz iteration on an r-by-c matrix with r <= c as `4*r*r*c + 2*r*r*r` for its three matrix multiplications. Sum over the actual matrix shapes and five iterations for Muon and AdaGO. This convention omits elementwise arithmetic, normalization, softmax, data movement, and checkpoint I/O. Log actual synchronized wall time alongside the estimate. [Transformer FLOP accounting reference, Appendix F](https://arxiv.org/pdf/2203.15556)

| Work item on the 9.72M configuration | Charge |
|---|---:|
| Ordinary AdamW update | 124375007232 FLOPs |
| Newton–Schulz per ordinary Muon or AdaGO update | 100705239040 FLOPs |
| Ordinary Muon or AdaGO update including Newton–Schulz | 225080246272 FLOPs |
| Each extra hidden-matrix forward/backward probe pass | 124255731712 FLOPs |
| One full pass on the first validation half, 217 complete context windows | 562278678528 FLOPs |

Count each attempted training or probe pass that actually runs, including work on discarded search trials. Charge tuning validation using the number of actual windows. Reporting-only validation, reserved-test evaluation, and off-trajectory diagnostics have their own logged compute totals.

For each arm, save the latest completed update at or below each cap. Reserve enough for the next whole update, including any due probe, before attempting it. Preserve the lower-cap checkpoint before continuing a live trajectory. Log both target and actual FLOPs and any unused budget. Report points at actual cumulative FLOPs, with the target cap identified in the table. An over-cap checkpoint is ineligible.

## 4. Overnight run order

Assume an **eight-hour reservation**. Run one campaign at a time and keep the GPU free of unrelated jobs. Aim for a projected **six-hour production matrix** after adding 30% runtime headroom. Use the rest for setup, preflight, evaluation, and diagnostics.

1. **Prepare the machine.** Record the Git revision, diff if dirty, `nvidia-smi`, Python/PyTorch/CUDA versions, precision settings, and free disk space. Use a Blackwell-compatible PyTorch build. Prepare data with `python -m curvature_muon.data` and run `python -m unittest discover -s tests -v`. [PyTorch CUDA installation](https://pytorch.org/get-started/locally/)
2. **Verify the new paths.** On the 0.81M config, exercise both probe arms through at least update 150, verify their ordinary batch streams match, and check exact cumulative counter increments. Exercise the baseline optimizer and search continuation paths. Confirm validation accounting handles a partial final evaluation batch and that the runner preserves checkpoints below each cap. Check AdaGO’s accumulator and LR against a hand-computed one-matrix update.
3. **Preflight the production model.** On seed 0, time at least 151 updates of the two probe variants and a short search for each of Muon, AdamW, and AdaGO with five candidates and continuation. Record peak allocated memory. Estimate the full queue from observed elapsed time per charged FLOP for each arm plus checkpoint and evaluation overhead. Choose one cap set and save it in the manifest. If even the fallback set cannot fit, record that and retain complete paired comparisons as space permits. Treat an incomplete matrix as a pilot.
4. **Run production.** Complete the matrix one paired seed at a time. Fixed Muon and both probe arms can save all three measurements in one trajectory per seed because their policies are independent of the final horizon. Each of the three searched arms runs a separate campaign per cap and seed. This is **36 campaigns** and at most **49,500 ordinary-Muon-update equivalents** under the primary caps, excluding preflight, reporting, and diagnostics.
5. **Run the mechanism check.** Preserve pre-update weights, optimizer state, and RNG at updates 100, 500, and 1,000 on each fixed-Muon trajectory. Freeze the weights, direction, and companion parameters during the check. Score each proposed rate and the fixed grid 0, 0.001, 0.003, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, and 0.20 on the same eight fresh batches that supplied neither estimate. Report actual loss changes, how often proposals help, and regret to the best rate in the finite grid. Leave these off-trajectory evaluations outside the main campaign caps. If time remains after the core matrix and diagnostic, add paired seeds 6 and 7 before extending the method list.

## 5. Handoff results for the draft

- **Main figure.** Reporting cross-entropy in nats per character versus cumulative FLOPs on a log x-axis. Show all six arms at all three caps, with individual paired seed points and means. Include the actual FLOPs, total training characters including discarded trials, selected-trajectory steps, synchronized elapsed time, and peak allocated GPU memory in a companion table.
- **Isolation check.** Save update 1,000 from the default and two probe trajectories when reached, or use the latest lower update shared by those trajectories under the frozen cap set. Report losses at that common update, together with probe overhead, invalidity rate, and clamp rate. Report the frozen-direction diagnostic alongside it.
- **Confirmation and reproducibility.** Once methods, settings, and endpoints are frozen, score each reported checkpoint on the reserved test split. Return resolved configs, source revision, driver/software details, method order, seed-level metrics, all search outcomes, per-step LR and FLOP logs, checkpoints or their paths, and any failures.
- **Interpretation.** Beating the same-batch arm supports the change in measurement batch. Beating fixed Muon at matched updates supports the usefulness of the estimated LR. Beating searched Muon at matched total FLOPs supports the compute-budget claim. If the independent-batch arm loses to tuned Muon, report where it helped and how much its probes cost. Scope the result to this character-modeling setup and the tested search grids.
