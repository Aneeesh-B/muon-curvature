# RTX 4070 curvature-Muon comparison

Completed 162 of 162 planned measurements. Paired seeds 6, 7, 8, 9, 10, 11, 12, 13, 14. Loss is measured on the reporting-validation split in nats per character. The reserved test split stayed closed.

PyTorch 2.14.1+cu130 with CUDA 13.0 and driver 591.86. The run manifests record base revision `7a8b6d6905c449fe91dadd0964f6f38a62561cd3` and the hashes of the local runner sources.

## Mean reporting loss

| Method | 112.54 TFLOPs | 225.08 TFLOPs | 450.16 TFLOPs |
|---|---:|---:|---:|
| Fixed Muon | 1.6948 (n=9) | 1.6047 (n=9) | 1.5514 (n=9) |
| Same-batch curvature Muon | 1.9346 (n=9) | 1.8387 (n=9) | 1.8066 (n=9) |
| Independent-batch curvature Muon | 1.6305 (n=9) | 1.5930 (n=9) | 1.6425 (n=9) |
| Searched Muon | 1.8579 (n=9) | 2.1482 (n=9) | 1.5804 (n=9) |
| Searched AdamW | 1.9586 (n=9) | 1.6857 (n=9) | 1.5200 (n=9) |
| Searched AdaGO | 1.9441 (n=9) | 1.6280 (n=9) | 1.5158 (n=9) |

## Paired differences

Each difference is independent-batch curvature Muon minus the named comparison. Negative favors the independent-batch arm.

| Cap (TFLOPs) | Comparison | Seed differences | Core mean | All-seed mean |
|---:|---|---|---:|---:|
| 112.54 | Same-batch curvature Muon | 6 (-0.6030), 7 (-0.7581), 8 (-0.2142), 9 (-0.1559), 10 (-0.3078), 11 (-0.1821), 12 (-0.1359), 13 (-0.1841), 14 (-0.1958) | -0.5251 | -0.3041 |
| 112.54 | Fixed Muon | 6 (-0.0264), 7 (-0.0800), 8 (-0.1260), 9 (-0.0592), 10 (-0.0878), 11 (-0.0546), 12 (+0.0315), 13 (-0.0942), 14 (-0.0826) | -0.0775 | -0.0644 |
| 112.54 | Searched Muon | 6 (-0.0645), 7 (-0.3283), 8 (-0.3805), 9 (-0.1119), 10 (-0.4299), 11 (-0.1309), 12 (-0.2838), 13 (-0.1507), 14 (-0.1667) | -0.2578 | -0.2275 |
| 112.54 | Searched AdamW | 6 (-0.2999), 7 (-0.3271), 8 (-0.3882), 9 (-0.3268), 10 (-0.3390), 11 (-0.3501), 12 (-0.2354), 13 (-0.3444), 14 (-0.3424) | -0.3384 | -0.3282 |
| 112.54 | Searched AdaGO | 6 (-0.2511), 7 (-0.3138), 8 (-0.4709), 9 (-0.2918), 10 (-0.4708), 11 (-0.3074), 12 (-0.1801), 13 (-0.2577), 14 (-0.2792) | -0.3453 | -0.3137 |
| 225.08 | Same-batch curvature Muon | 6 (-0.6761), 7 (-0.7572), 8 (-0.0754), 9 (-0.0561), 10 (-0.1525), 11 (-0.2022), 12 (-0.1438), 13 (-0.0734), 14 (-0.0748) | -0.5029 | -0.2457 |
| 225.08 | Fixed Muon | 6 (-0.0649), 7 (-0.0867), 8 (+0.0333), 9 (+0.0329), 10 (+0.0144), 11 (-0.0877), 12 (+0.0264), 13 (+0.0222), 14 (+0.0046) | -0.0394 | -0.0117 |
| 225.08 | Searched Muon | 6 (-0.5551), 7 (-0.4946), 8 (-0.3976), 9 (-0.6231), 10 (-0.5742), 11 (-0.5621), 12 (-0.5247), 13 (-0.6659), 14 (-0.5999) | -0.4824 | -0.5552 |
| 225.08 | Searched AdamW | 6 (-0.1574), 7 (-0.1884), 8 (-0.0709), 9 (-0.0265), 10 (-0.0340), 11 (-0.1591), 12 (-0.0731), 13 (-0.0617), 14 (-0.0627) | -0.1389 | -0.0927 |
| 225.08 | Searched AdaGO | 6 (-0.0792), 7 (-0.1131), 8 (+0.0179), 9 (+0.0018), 10 (+0.0074), 11 (-0.1202), 12 (-0.0097), 13 (+0.0014), 14 (-0.0209) | -0.0581 | -0.0350 |
| 450.16 | Same-batch curvature Muon | 6 (-0.6883), 7 (-0.4629), 8 (-0.1370), 9 (-0.0153), 10 (-0.2394), 11 (-0.1255), 12 (-0.1735), 13 (+0.4927), 14 (-0.1270) | -0.4294 | -0.1640 |
| 450.16 | Fixed Muon | 6 (+0.0009), 7 (+0.1856), 8 (-0.0080), 9 (+0.0882), 10 (-0.0348), 11 (+0.0170), 12 (+0.0268), 13 (+0.5666), 14 (-0.0216) | +0.0595 | +0.0912 |
| 450.16 | Searched Muon | 6 (-0.0236), 7 (+0.1981), 8 (-0.0441), 9 (+0.0347), 10 (-0.0581), 11 (-0.0060), 12 (-0.0062), 13 (+0.5193), 14 (-0.0550) | +0.0435 | +0.0621 |
| 450.16 | Searched AdamW | 6 (+0.0452), 7 (+0.2118), 8 (-0.0037), 9 (+0.1192), 10 (+0.0254), 11 (+0.0490), 12 (+0.0598), 13 (+0.5849), 14 (+0.0110) | +0.0844 | +0.1225 |
| 450.16 | Searched AdaGO | 6 (+0.0522), 7 (+0.2428), 8 (+0.0066), 9 (+0.1020), 10 (+0.0202), 11 (+0.0483), 12 (+0.0586), 13 (+0.5940), 14 (+0.0161) | +0.1005 | +0.1268 |

## Common update count

| Seed | Method | Step | Reporting loss |
|---:|---|---:|---:|
| 6 | Fixed Muon | 500 | 1.6968 |
| 6 | Independent-batch curvature Muon | 500 | 1.6765 |
| 6 | Same-batch curvature Muon | 500 | 2.2762 |
| 7 | Fixed Muon | 500 | 1.6961 |
| 7 | Independent-batch curvature Muon | 500 | 1.6173 |
| 7 | Same-batch curvature Muon | 500 | 2.3759 |
| 8 | Fixed Muon | 500 | 1.7002 |
| 8 | Independent-batch curvature Muon | 500 | 1.5722 |
| 8 | Same-batch curvature Muon | 500 | 1.7934 |
| 9 | Fixed Muon | 500 | 1.7005 |
| 9 | Independent-batch curvature Muon | 500 | 1.6342 |
| 9 | Same-batch curvature Muon | 500 | 1.8036 |
| 10 | Fixed Muon | 500 | 1.6958 |
| 10 | Independent-batch curvature Muon | 500 | 1.6018 |
| 10 | Same-batch curvature Muon | 500 | 1.9149 |
| 11 | Fixed Muon | 500 | 1.6726 |
| 11 | Independent-batch curvature Muon | 500 | 1.6184 |
| 11 | Same-batch curvature Muon | 500 | 1.7951 |
| 12 | Fixed Muon | 500 | 1.6970 |
| 12 | Independent-batch curvature Muon | 500 | 1.7461 |
| 12 | Same-batch curvature Muon | 500 | 1.8641 |
| 13 | Fixed Muon | 500 | 1.7066 |
| 13 | Independent-batch curvature Muon | 500 | 1.6096 |
| 13 | Same-batch curvature Muon | 500 | 1.7951 |
| 14 | Fixed Muon | 500 | 1.6880 |
| 14 | Independent-batch curvature Muon | 500 | 1.6032 |
| 14 | Same-batch curvature Muon | 500 | 1.7942 |

## Search selections

Counts show how many seeds selected each setting on tuning validation.

| Cap (TFLOPs) | Method | Winning settings | Candidate failures |
|---:|---|---|---:|
| 112.54 | Searched Muon | 0.02 (5), 0.04 (4) | 0 |
| 112.54 | Searched AdamW | 0.0003 (9) | 0 |
| 112.54 | Searched AdaGO | η=0.02, ε=0.0001 (2), η=0.05, ε=0.0005 (5), η=0.1, ε=0.0005 (2) | 0 |
| 225.08 | Searched Muon | 0.04 (9) | 0 |
| 225.08 | Searched AdamW | 0.0003 (9) | 0 |
| 225.08 | Searched AdaGO | η=0.1, ε=0.0005 (6), η=0.3, ε=0.0005 (3) | 0 |
| 450.16 | Searched Muon | 0.02 (9) | 0 |
| 450.16 | Searched AdamW | 0.0003 (9) | 0 |
| 450.16 | Searched AdaGO | η=0.1, ε=0.0005 (4), η=0.3, ε=0.0005 (1), η=0.3, ε=0.003 (4) | 0 |

## Frozen-direction check

27 scans on eight fresh batches per checkpoint. Negative mean loss change indicates improvement.

| Policy | Improved scans | Mean loss change | Mean grid regret |
|---|---:|---:|---:|
| Fixed Muon | 21/27 | -0.0023 | 0.0005 |
| Same-batch probe | 8/27 | +0.0794 | 0.0822 |
| Independent-batch probe | 24/27 | -0.0025 | 0.0003 |

## Files

- [Combined comparison CSV](comparison.csv), [loss plot](loss.png), and [wall-time pilot summary](wall-time-pilot.md)
- [Exact source snapshot](source/) and protocol configuration
- Per-block manifests, per-campaign results, per-update FLOP and loss logs, and frozen-direction diagnostics beneath the three seed-block directories
- [Checkpoint inventory](checkpoints.json) with archive names, checksums, and checkpoint paths

The checkpoint ZIPs are intended for the [RTX 4070 results release](https://github.com/Aneeesh-B/muon-curvature/releases/tag/rtx-4070-results-2026-11-24). Extract the ZIPs into this results directory to put the `.pt` files beside their JSON records.
