# Experiment results

Validation cross entropy in nats per character; lower is better.

| Variant | Seeds | Final validation loss (mean ± sample SD) | Probe / train time | Invalid probes | Clamped probes |
|---|---:|---:|---:|---:|---:|
| adamw | 1 | 1.7042 (SD n/a) | 0.0% | 0/0 | 0/0 |
| fixed | 1 | 1.5253 (SD n/a) | 0.0% | 0/0 | 0/0 |
| once | 1 | 1.7098 (SD n/a) | 0.0% | 0/1 | 0/1 |
| periodic | 1 | 2.5271 (SD n/a) | 2.0% | 36/38 | 0/38 |

Each run: 812,288 parameters, 2,000 updates, 4,096,000 training tokens.
The runner keeps training tokens equal across methods. Probe batches are additional compute and are not counted as new training tokens.
Probe / train time is measured synchronized probe duration divided by ordinary training duration within the same run; it excludes diagnostics, evaluation, startup, logging and checkpoint I/O.
These short sequential timings are noisy and are not GPU throughput estimates.

Fixed-rate baselines cover only the configurations shown, not globally tuned Muon. AdamW is a single-rate reference, not a tuned baseline.
One-time and periodic modes use the same initial rate, companion AdamW configuration, safety bounds, initialization and batch stream.
Sample SD across seeds is not a confidence interval. Test data is untouched.
