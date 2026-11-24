# RTX 4070 overnight experiment plan

Use the 4070 to run a second set of paired seeds for the curvature-Muon comparison. Keep the model, dataset, probe rules, baseline tuning, and FLOP accounting the same as in the [RTX 5090 protocol](5090-overnight-experiment-plan.md). Pick the 4070 budget from its own preflight timings.

**Implementation status.** Before production, verify that the runner has the unguarded independent-batch arm, the same invalid-probe fallback in both curvature arms, AdaGO, and cumulative FLOP cap enforcement. The original repository launcher runs wall-time and matched-update pilots. A successful 151-update run exercises a training path and does not verify the FLOP-capped protocol by itself.

## Model, data, and optimizer arms

- **Smoke model:** the [0.81M NanoGPT configuration](../configs/tiny.json). Use seed 0 for preflight.
- **Production model:** the [9.72M NanoGPT configuration](../configs/wide.json), with four layers, width 448, seven heads, context 128, and batch size 16. Use FP32 with TF32 disabled, zero dropout, and zero weight decay.
- **Data:** checksum-verified Tiny Shakespeare character data with the same train, tuning-validation, reporting-validation, and reserved test splits as the 5090 run. Pair initialization and ordinary training-batch order across arms within each seed.
- **Production seeds:** **6, 7, and 8** on the 4070. Reserve **3, 4, and 5** for the 5090. These are additional paired seeds. To measure hardware repeatability instead, explicitly rerun a shared seed and label it as a repeat.

| Arm | Policy |
|---|---|
| Fixed Muon | Matrix LR 0.02. |
| Searched Muon | Tune the constant matrix LR using the shared five-rate grid. |
| Same-batch curvature Muon | Reuse the gradient from the minibatch that produced the direction and compute one shifted gradient. |
| Independent-batch curvature Muon | Draw a separate training minibatch for both probe gradients. Keep the ordinary training-batch stream intact. |
| Searched AdamW | Apply AdamW to all parameters and use the shared five-rate LR grid. |
| Searched AdaGO | Use the reference matrix update, the shared five-pair grid for its step-size parameters, and the same companion AdamW as Muon. |

Use the exact probe settings and search grids in the [shared protocol](5090-overnight-experiment-plan.md#2-run-six-arms). Probe at update 100 and every 50 updates thereafter. Give both curvature arms the same bounds and the same 0.02 fallback. Exclude backtracking, schedule changes, and replay from the core comparison. Each searched arm uses 40% of its cap for five exploration slots, including tuning validation, then continues the winning trial with the remaining available compute. Charge every trial’s work, including failures.

## Choose the FLOP caps before production

Use the same estimated matmul accounting as the 5090 plan, where one ordinary 9.72M Muon update is **225080246272 FLOPs**. Each row lists the three measurement caps for all six arms.

| Cap set | Ordinary Muon update equivalents | Exact cumulative FLOP caps | Total production work in update equivalents |
|---|---|---|---:|
| Standard shared caps | 500 / 1,000 / 2,000 | 112540123136000 / 225080246272000 / 450160492544000 | 49,500 |
| Shorter shared caps | 400 / 800 / 1,600 | 90032098508800 / 180064197017600 / 360128394035200 | 39,600 |
| Reduced 4070, if the 5090 uses standard caps | 250 / 500 / 1,000 | 56270061568000 / 112540123136000 / 225080246272000 | 24,750 |
| Reduced 4070, if the 5090 uses shorter caps | 200 / 400 / 800 | 45016049254400 / 90032098508800 / 180064197017600 | 19,800 |

Prefer the same cap set as the 5090 if the 4070 can complete it. Otherwise use the corresponding reduced row so that two measurement caps still overlap. If running the 4070 alone with a short window, use the 250 / 500 / 1,000 set. Pick caps from timing, memory, and the available window before looking at production losses. Record the choice in the manifest.

Use **36 campaigns** for any complete cap set. The three direct arms each save all three cap checkpoints in one trajectory per seed. Each of the three searched arms starts a fresh campaign at every cap and seed. Save the last completed update at or below each cap and record its actual charged FLOPs.

## Run order and preflight

1. **Verify the small smoke.** The 4070 operator reported 4.5–4.8 seconds per 151-update run on the 0.81M model, roughly 30–32 ms per update including run overhead. Check the logs for the intended probe events, paired batch streams, AdaGO settings, search continuation, and FLOP charges where applicable. Use the small run as a smoke check.
2. **Time the production model.** On the 9.72M model, time at least 151 updates for both probe arms, including the probes at updates 100 and 150. Run short searches for Muon, AdamW, and AdaGO. Record their elapsed seconds per charged FLOP, search overhead, and peak allocated CUDA memory. Measure the actual 4070 rather than extrapolating from the small model or the 5090.
3. **Freeze the queue.** For an eight-hour reservation, target at most six hours for production after adding 30% timing headroom. Keep the rest for preflight, evaluation, and diagnostics. If no complete cap set fits, run a paired pilot at shared caps and report the missing cells. Run one campaign at a time on this card.
4. **Collect evidence.** Save reporting loss at all three caps, the search outcomes, probe status, and synchronized time. For the fixed and both probe arms, also save loss at a common update count, preferably update 500 or the latest shared update if the largest cap is too short. If time remains, run the shared frozen-direction diagnostic at early, middle, and late checkpoints within the trajectory.

## Reporting across cards

Report held-out cross-entropy in nats per character against **actual cumulative FLOPs**, with paired seed differences. Keep wall-time and memory results separated by device. Use only shared caps for a combined view, show which device produced each seed, and compute paired optimizer differences within each device. Keep test data closed until endpoints are frozen. Report losses, ties, failures, and any unused budget. Apply the same sanity checks to every arm and preserve the original logs if a confirmed bug requires a rerun.
