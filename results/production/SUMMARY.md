# 5090 production run: findings for the draft

**Question.** Do occasional curvature probes choose useful Muon learning rates, and do they pay back their compute
against learning-rate search?

**Setup.**
- **Model and data.** The 9.72M-parameter NanoGPT on Tiny Shakespeare characters, in FP32 on one RTX 5090.
- **Seeds.** Five paired seeds (3–7): the plan's three, plus seeds 6 and 7, which it allows if time remains. The plan
  later reserved seeds 6–8 for the 4070, which also ran 6 and 7. For a combined view, treat 5090 seeds 6 and 7 as
  labelled repeats ([CROSS_CARD_CHECK.md](CROSS_CARD_CHECK.md)).
- **Compute caps.** Three caps on cumulative estimated matmul FLOPs (112.5T, 225.1T and 450.2T, i.e. 500, 1000 and
  2000 Muon updates).
- **Accounting.** Every probe pass, search trial and tuning evaluation is charged against the cap.
- **Loss.** Cross-entropy in nats per character on the reporting half of validation. The test split was scored only
  after all runs had finished.
- **Statistics.** All differences are paired by seed, as mean ± SD with a two-sided paired t-test.
- **Where the numbers live.** Every number below comes from [REPORT.md](REPORT.md) or
  [paired_deltas.csv](paired_deltas.csv).

## Findings

1. **Measuring on an independent batch is essential.**
   - The independent-batch arm beats the same-batch arm at every cap: −0.40, −0.33 and −0.35 nats, 5/5 seeds,
     p ≤ 0.022 (REPORT, "Paired reporting-loss differences").
   - After its first accepted step, the same-batch probe was invalid on 36–37 of 38 probes ("Probe activity").
   - The same-batch arm also loses to fixed Muon at every cap: +0.32 to +0.35, 0/5 seeds.
2. **The finite-difference probe fails on the same batch.**
   - Against an exact Hessian–vector product, the probe is accurate on an independent batch: median relative error
     0.2% at update 100, 0.4% at update 500 and 4.9% at update 1000, with the sign always right.
   - On the same batch it is accurate only at update 100 (0.9%). The error grows to 75% at update 500. At update 1000
     the sign is wrong on 4/5 seeds, even though the exact curvature is positive on 4/5 ("Curvature fidelity").
   - So the same-batch "negative curvature" is a failure of the finite difference, not of the landscape: along D, the
     same-batch loss is far from quadratic at the probe distance.
   - On the GPU the probe is not cheaper than the exact HVP: 19 ms against 20 ms. The CPU pilot measured 72 ms
     against 105 ms.
3. **Against fixed Muon at 0.02, the estimated rate helps early and ties later.**

   | Comparison (independent-batch − fixed Muon) | Difference | Seeds lower | p |
   |---|---:|---:|---:|
   | Low cap | −0.063 | 5/5 | 0.021 |
   | Middle cap | −0.013 | 3/5 | 0.69 |
   | High cap | −0.001 | 3/5 | 0.96 |
   | Matched update 1000 ("Isolation check") | −0.024 | 3/5 | 0.53 |

   - The test split agrees: −0.047 at the low cap (p = 0.053).
   - In the frozen-direction check, the independent proposal's regret to the best grid rate is 0.0003–0.0005. That
     matches fixed 0.02. The same-batch proposal's regret is 0.09 at update 100 and 0.26 at update 500, and
     it was invalid at update 1000 ("Mechanism check").
4. **Against searched Muon at equal total compute, the probe wins mostly because the search is a poor use of compute.**
   - The independent-batch arm beats searched Muon by −0.25 at the low cap and −0.51 at the middle cap (5/5 seeds).
     At the high cap the difference is −0.018 (4/5 seeds, p = 0.20).
   - But searched Muon loses to plain fixed Muon at all three caps (+0.19, +0.50 and +0.02, 0/5 seeds).
   - The 8%-of-cap slots are 37–157 updates long. Slots that short favour LR 0.04, which then degrades over the long
     constant-rate continuation ("Search outcomes").
   - So the result supports "a probe is cheaper than this search", not "the probe beats a well-chosen constant rate".
5. **At the high cap the adaptive baselines win.**
   - Searched AdaGO reaches 1.510 and searched AdamW 1.516, against 1.552 for the independent-batch arm and 1.553 for
     fixed Muon.
   - The independent-batch arm trails AdaGO by +0.042 (0/5 seeds, p = 0.040) and AdamW by +0.036 (1/5 seeds,
     p = 0.11).
   - At the low cap the ordering reverses: the independent-batch arm beats both by about −0.3, 5/5 seeds.
6. **Backtracking adds nothing consistent.**
   - Against the fixed-proposal guard: −0.020, −0.045 and +0.035 across the caps (all p ≥ 0.078).
   - Against the unguarded independent-batch arm: not significant at any cap.
   - The guard rejected 1–5 of the 37 checks per run ("Ablation: third-batch backtracking").
7. **Scheduled probing does not beat a tuned schedule here.** The schedule is the groupmate's nanochat schedule, with
   probes every 25 updates.

   | Comparison | Horizon 500 | Horizon 1000 | Horizon 2000 |
   |---|---:|---:|---:|
   | Next-batch probe − tuned scheduled Muon | +0.150 (0/5) | +0.033 (0/5) | +0.006 (2/5, p = 0.29) |
   | Same-batch probe − tuned scheduled Muon | +0.94 | +1.13 | +1.15 |
   | Replay − live next-batch probe | p ≥ 0.2 | p ≥ 0.2 | p ≥ 0.2 |

   - The same-batch scheduled probe collapses at every horizon.
   - The replay curve (from development seeds 10 and 11) ties the live next-batch probe and is still worse than tuned
     scheduled Muon.
   - The nanochat gain of the scheduled next-batch probe (see [../nanochat_transfer](../nanochat_transfer/README.md))
     does **not** replicate on this character model.
   - The default peak of 0.02 was already the tuned choice at horizons 500 and 2000 (0.01 at 1000).
8. **Probe overhead is small.** Probes take 1.5% of training time for the same-batch arm and 2.0% for the
   independent-batch arm, probing every 50 updates. With scheduled probing every 25 updates it is 3–4%.
9. **Follow-up: extended search grids.**
   - Every AdamW search won on its lowest rate, 0.0003, and high-cap AdaGO usually won on its largest step.
   - A labelled follow-up with grids extended past those edges found no better optimum. AdamW at smaller rates is
     worse (+0.10 to +0.12 at the low and middle caps). AdaGO with larger steps helps only at the low cap (−0.086).
   - Findings 4–5 are unchanged ("Follow-up").

## How to read this in the paper

- **Supported.** Measuring curvature on an independent batch is necessary: the same-batch rule fails, and its finite
  difference is wrong.
- **Supported only early.** At matched compute, the independent-batch rule beats fixed-rate Muon at the smallest
  budget (replicated on the 4070). Later the two arms are within run-to-run variation. It also beats this particular
  short-slot LR search.
- **Not supported.**
  - The rule does not beat tuned or scheduled Muon, AdaGO or AdamW at the largest budget.
  - Scheduled probing does not beat a peak-tuned schedule on this model.
- **Scope.** One character-level model and dataset, five seeds, and the frozen search grids. Seeds vary the
  initialization and the batch stream together.

## Caveats

- **Run-to-run sensitivity of the probe arms.** The 4070 run repeated seeds 6 and 7 with an independently written
  runner ([CROSS_CARD_CHECK.md](CROSS_CARD_CHECK.md)):
  - FLOPs, update counts and search winners match in every cell, and the fixed-rate arms agree within 0.011.
  - The probe arms agree at the low cap but differ by up to 0.15 at the middle cap and 0.23 at the high cap.
  - Finding 3's middle- and high-cap ties are therefore within run-to-run variation. Report them as unresolved rather
    than as ties. The low-cap gain replicates on the 4070 (−0.064 over nine seeds).
- **Fixed-Muon reference.** Fixed Muon uses 0.02, which turned out to be the search's own best constant rate at the
  high cap. The "fixed" arm is therefore close to a tuned constant LR.
- **AdaGO grid.** Its two η = 0.3 candidates are nearly identical within a short search slot, because the ε floor
  rarely binds early.
- **Fidelity times.** These use the math attention backend, which the exact HVP needs for its double backward. The
  production probe runs on the default backend at a similar speed (19 ms).
