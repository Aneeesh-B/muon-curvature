# Directional scans

Matrix-only counterfactuals; companion AdamW and decay are frozen.

| Anchor | Age | Saved LR | Probe Δ loss | Fresh train Δ loss | Best scanned fresh LR | Fresh regret |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0.2 | -0.2930 | -0.1671 | 0.1 | 0.0167 |
| 0 | 1 | 0.2 | -0.2006 | -0.1418 | 0.1 | 0.0103 |
| 0 | 10 | 0.2 | +0.3541 | +0.4774 | 0.03 | 0.4901 |
| 0 | 50 | 0.2 | +0.2183 | +0.2863 | 0.003 | 0.2865 |
| 50 | 0 | 0.053246 | -0.0522 | +0.0294 | 0.003 | 0.0296 |
| 50 | 1 | 0.053246 | -0.0417 | +0.0097 | 0.02 | 0.0167 |
| 50 | 10 | 0.053246 | -0.0522 | +0.0447 | 0 | 0.0447 |
| 0 | 100 | 0.2 | +0.2993 | +0.5460 | 0.02 | 0.5550 |
| 50 | 50 | 0.053246 | -0.0930 | +0.0271 | 0.02 | 0.0361 |
| 100 | 0 | 0.062684 | -0.0897 | +0.0474 | 0.02 | 0.0564 |
| 100 | 1 | 0.062684 | -0.0621 | +0.0473 | 0.01 | 0.0513 |
| 100 | 10 | 0.062684 | -0.0440 | +0.0279 | 0.02 | 0.0393 |
| 50 | 100 | 0.053246 | -0.1012 | +0.0217 | 0.01 | 0.0229 |
| 100 | 50 | 0.062684 | -0.1052 | +0.0318 | 0.01 | 0.0330 |
| 100 | 100 | 0.062684 | -0.0563 | +0.0950 | 0.001 | 0.0950 |

Negative delta is an improvement. Regret is saved-step loss minus the best loss among scanned candidates, including zero.
At each age, the direction is rebuilt from the current gradient and momentum on a fixed-rate Muon trajectory; only the saved scalar is stale.
The fresh minibatches are independently sampled training windows; validation windows are separate. Within a scan all candidates share windows and random masks.
The raw JSONL includes local quadratic predictions, current estimates, validation deltas, and probe-distance sweeps at anchors.
These one-step counterfactuals do not measure branching multi-step training continuations.
