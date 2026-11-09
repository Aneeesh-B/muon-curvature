# Experiment results

Validation cross entropy in nats per character; lower is better.

| Variant | Seeds | Final validation loss (mean ± sample SD) | Probe / train time | Invalid probes | Clamped probes |
|---|---:|---:|---:|---:|---:|
| adamw | 3 | 2.5696 ± 0.0158 | 0.0% | 0/0 | 0/0 |
| fixed_0.005 | 3 | 2.5117 ± 0.0114 | 0.0% | 0/0 | 0/0 |
| fixed_0.02 | 3 | 2.5236 ± 0.0198 | 0.0% | 0/0 | 0/0 |
| fixed_0.08 | 3 | 2.6482 ± 0.0178 | 0.0% | 0/0 | 0/0 |
| once | 3 | 2.7843 ± 0.0243 | 0.4% | 0/3 | 3/3 |
| periodic_10 | 3 | 2.7536 ± 0.0626 | 8.0% | 43/60 | 11/60 |
| periodic_50 | 3 | 2.7679 ± 0.0350 | 1.3% | 7/12 | 4/12 |

Each run: 106,880 parameters, 200 updates, 102,400 training tokens.
The runner keeps training tokens equal across methods. Probe batches are additional compute and are not counted as new training tokens.
Probe / train time is measured synchronized probe duration divided by ordinary training duration within the same run; it excludes diagnostics, evaluation, startup, logging and checkpoint I/O.
These short sequential timings are noisy and are not GPU throughput estimates.

Fixed-rate baselines cover only the configurations shown, not globally tuned Muon. AdamW is a single-rate reference, not a tuned baseline.
One-time and periodic modes use the same initial rate, companion AdamW configuration, safety bounds, initialization and batch stream.
Sample SD across seeds is not a confidence interval. Test data is untouched.
