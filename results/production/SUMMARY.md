# RTX 5090 production run

**Setup.**
- Protocol: [docs/5090-overnight-experiment-plan.md](../../docs/5090-overnight-experiment-plan.md), with the settings
  frozen in [configs/production.json](../../configs/production.json).
- Model and data: the 9.72M NanoGPT on Tiny Shakespeare characters, FP32 with TF32 off.
- Seeds: 3–7, each pairing the initialization with the training-batch stream.
- Compute caps: 112.5T, 225.1T and 450.2T estimated matmul FLOPs (500, 1,000 and 2,000 Muon-update equivalents). Probe
  passes, search trials and tuning evaluations count against the cap.
- Loss: cross-entropy in nats per character on the reporting half of validation.
- Differences: paired by seed, as mean ± SD with the number of seeds below zero and a two-sided paired t-test.

Full tables are in [REPORT.md](REPORT.md).

| Arm | 112.5T | 225.1T | 450.2T |
|---|---:|---:|---:|
| Fixed Muon (LR 0.02) | 1.697 | 1.604 | 1.553 |
| Searched Muon | 1.884 | 2.102 | 1.570 |
| Same-batch probe | 2.036 | 1.925 | 1.899 |
| Independent-batch probe | 1.635 | 1.591 | 1.552 |
| Searched AdamW | 1.955 | 1.685 | 1.516 |
| Searched AdaGO | 1.930 | 1.636 | 1.510 |

Independent-batch probe minus the comparator:

| Comparator | 112.5T | 225.1T | 450.2T |
|---|---|---|---|
| Fixed Muon | -0.063 ± 0.038 (5/5, p=0.021) | -0.013 ± 0.066 (3/5, p=0.69) | -0.001 ± 0.031 (3/5, p=0.96) |
| Searched Muon | -0.249 ± 0.141 (5/5, p=0.017) | -0.512 ± 0.096 (5/5, p<0.001) | -0.018 ± 0.026 (4/5, p=0.20) |
| Same-batch probe | -0.401 ± 0.245 (5/5, p=0.022) | -0.334 ± 0.178 (5/5, p=0.014) | -0.347 ± 0.207 (5/5, p=0.020) |
| Searched AdamW | -0.320 ± 0.026 (5/5, p<0.001) | -0.095 ± 0.071 (5/5, p=0.041) | +0.036 ± 0.039 (1/5, p=0.11) |
| Searched AdaGO | -0.295 ± 0.028 (5/5, p<0.001) | -0.046 ± 0.064 (3/5, p=0.18) | +0.042 ± 0.032 (0/5, p=0.040) |

## Results

### Same-batch probing

The same-batch arm trails the independent-batch arm by 0.33–0.40 and fixed Muon by 0.32–0.35 at every cap, on 5/5
seeds in both cases. Two problems show up.

**The finite-difference estimate of q breaks down.** The check below uses the fixed-Muon checkpoints and compares the
same-batch estimate with the exact HVP (double backward):

| Step | Same-batch estimate | Independent-batch estimate |
|---:|---|---|
| 100 | 0.9% median relative error | 0.2% |
| 500 | 75% | 0.4% |
| 1000 | Wrong sign on 4/5 seeds (exact q positive on 4/5) | 4.9%, correct sign on every seed |

This fits the probe log of the same-batch arm, where 36–37 of 38 probes per run were invalid and fell back to 0.02.

**The step overfits the probe batch.** At step 100 the estimate is still accurate, yet the frozen-direction check gives
the same-batch step a fresh-batch regret of 0.09. The independent-batch step and fixed 0.02 both have regret
3–5 × 10⁻⁴.

### Independent-batch probe vs. fixed Muon

- **112.5T:** the difference is significant. It is -0.063 on validation, and -0.047 on the test split (5/5, p = 0.053).
- **225.1T and 450.2T:** the differences are small and not significant. The cross-device replicate below shows that the
  probe arm's run-to-run variation at these caps exceeds its gap to fixed Muon, so we report them as unresolved rather
  than as ties.
- **Fixed 0.02 is close to a tuned constant.** It is also the rate the search selects at 450.2T.

### Searched Muon

The 8%-slot search is a weak baseline:
- It loses to fixed 0.02 at all three caps: +0.19, +0.50 and +0.02, with fixed lower on all five seeds each time.
- Its slots last 37, 77 and 157 updates at the three caps.
- At 112.5T and 225.1T the slots select 0.04, on 3/5 and 5/5 seeds respectively, and 0.04 underperforms over the
  continuation.
- At 450.2T the search does select 0.02. But the winning trajectory is only 1,359 updates long, against 2,000 for fixed
  Muon, because the other four slots use up the rest of the budget.
- The probe's margin over searched Muon mostly reflects this.

### Adaptive baselines

- **450.2T:** searched AdaGO (1.510) and AdamW (1.516) beat every Muon variant. AdaGO is below the independent-batch
  probe on 5/5 seeds.
- **112.5T:** the ordering reverses, with the probe about 0.3 ahead of both, 5/5 seeds.

### Ablations

- **Third-batch backtracking:** no consistent effect. Every |Δ| ≤ 0.05, with all p ≥ 0.078.
- **Scheduled probing.** The schedule is the one from the nanochat runs: a 40-step warmup, linear decay from 35% of the
  horizon down to 5% of the peak, and a probe every 25 steps.
  - The next-batch probe trails peak-tuned scheduled Muon at H = 500 and H = 1000 (+0.150 and +0.033, 0/5 seeds). It
    ties at H = 2000 (+0.006, p = 0.29).
  - Replaying a frozen probe LR curve does not differ significantly from the live probe (p ≥ 0.2).
  - The nanochat gain does not transfer to this model.
- **Extended grids.**
  - AdamW selected its lower edge (3e-4) in every search, and AdaGO usually selected its upper edge at 450.2T.
  - Grids extended past those edges found no better AdamW setting. They improved AdaGO only at 112.5T (-0.086).
  - The conclusions above are unchanged.
- **Cost.**
  - Probing takes 2.0% of training time for the independent-batch arm and 1.5% for the same-batch arm, probing every 50
    steps. It takes 3–4% when probing every 25 steps.
  - On this GPU a finite-difference probe (19 ms) is no cheaper than an exact HVP (20 ms).

## Cross-device replicate (seeds 6 and 7)

**Why the overlap exists.** This queue ran seeds 6 and 7 as the plan's optional extra seeds. The plan was then revised
to assign seeds 6–8 to the 4070. The 4070 run (seeds 6–14, separate implementation) therefore replicates these two
seeds. The 36 shared cells are in [cross_card_seeds_6_7.csv](cross_card_seeds_6_7.csv).

**What matches.**
- FLOP charges, update counts and all 18 search selections are identical.
- Fixed Muon, searched AdamW and searched AdaGO agree within 0.011, and AdamW within 10⁻⁴. Searched Muon differs by up
  to 0.079 (seed 6, 225.1T, where both runs continued at LR 0.04).

**What diverges.**
- The probe arms agree at 112.5T, then diverge: by up to 0.15 at 225.1T and 0.23 at 450.2T (seed 7, independent
  batch: 1.513 against 1.746). No probe-by-probe comparison was made, so the cause is not established.
- The low-cap effect replicates: -0.063 here and -0.064 on the 4070 over seeds 6–14.

**Pooling.** For pooled analyses, treat 5090 seeds 3–5 as new seeds and 6 and 7 as replicates.

**Scope.** One model and one dataset. Seeds vary the initialization and the batch order jointly.
