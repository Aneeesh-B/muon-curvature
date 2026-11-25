# Cross-card check: seeds 6 and 7 on both GPUs

**Why the overlap exists.** The 5090 queue ran seeds 6 and 7 as its optional extra seeds and finished at 04:01 on
2026-11-24. At 04:03 the plan was revised to reserve seeds 6–8 for the 4070. The 4070 run
(branch `results/rtx-4070-2026-11-24`) then used seeds 6–14. So seeds 6 and 7 were run twice: by two separately written
runners, on two GPUs, under the same protocol.

**Treat them as labelled repeats.** For a combined cross-card view:
- the 5090's own seeds are 3–5;
- 5090 seeds 6 and 7 are repeats of the 4070's seeds 6 and 7.

The five-seed 5090 tables in [REPORT.md](REPORT.md) and [SUMMARY.md](SUMMARY.md) stay as computed.

## What agrees

All 36 shared cells (6 arms × 3 caps × 2 seeds) are listed in [cross_card_seeds_6_7.csv](cross_card_seeds_6_7.csv).
The 4070 values come from that branch's `comparison.csv`, and the search winners from its per-campaign `result.json`
files.

- **Accounting.** Charged FLOPs and selected update counts are identical in all 36 cells. Both runners implement the
  plan's FLOP counter and cap scheduler the same way.
- **Search winners.** All 18 searches (Muon, AdamW and AdaGO at 3 caps × 2 seeds) picked the same grid point on both
  cards.
- **Arms without a probe.** The largest reporting-loss gap between the cards in each arm:

  | Arm | Low | Middle | High |
  |---|---:|---:|---:|
  | Searched AdamW | 0.0000 | 0.0000 | 0.0000 |
  | Searched AdaGO | 0.0000 | 0.0009 | 0.0091 |
  | Fixed Muon | 0.0107 | 0.0086 | 0.0067 |
  | Searched Muon | 0.0154 | 0.0788 | 0.0124 |

  Searched Muon's 0.079 gap is seed 6 at the middle cap, where both cards continued LR 0.04. That is the rate that
  degrades over a long constant-rate run (SUMMARY, finding 4).

## What does not agree

The two probe arms agree at the low cap but then drift apart:

| Arm | Low | Middle | High |
|---|---:|---:|---:|
| Independent-batch probe | 0.0065 | 0.1484 | 0.2333 |
| Same-batch probe | 0.2054 | 0.2235 | 0.2517 |

The largest gaps are on seed 7, for the independent-batch arm:

| Cap | 5090 | 4070 |
|---|---:|---:|
| Middle | 1.6545 | 1.5062 |
| High | 1.5128 | 1.7461 |

The fixed-rate arms start from the same weights and see the same batches on both cards, yet differ by about 0.01. The
probe arms turn the same small floating-point differences into different learning rates, and over hundreds of updates
those rates send the runs to visibly different losses. The probe never set a rate above 0.037 on any 5090 seed, so this
is not a runaway learning rate.

## Consequence for the draft

The probe arms' middle- and high-cap results vary from run to run far more than the fixed-rate arms do. Rerunning the
same seed can move the independent-batch arm by more than its gap to fixed Muon.

- The 5090's "ties fixed Muon at the high cap" (−0.001, five seeds) and the 4070's "worse than fixed Muon at the high
  cap" (+0.091 mean over nine seeds, driven by seeds 7 and 13) are both within this variation.
- Report the high-cap comparison as unresolved, with per-seed values, rather than as a tie or a loss.
- The low-cap result is stable across cards. Independent-batch minus fixed Muon is −0.063 (5090, five seeds) and
  −0.064 (4070, nine seeds).
