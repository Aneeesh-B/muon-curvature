# Equal-budget optimizer comparison

One seed, same model/data and hardware, one elapsed-time cap per method. Lower loss and higher next-character accuracy are better.

| Method | Budget (s) | Used (s) | Reporting loss | Accuracy | Total updates | Probes |
|---|---:|---:|---:|---:|---:|---:|
| adamw_search | 120 | 119.10 | 1.6406 | 50.89% | 3716 | 0 |
| adaptive_muon | 120 | 118.99 | 2.1583 | 36.80% | 2284 | 44 |
| default_muon | 120 | 118.80 | 1.5290 | 54.21% | 2500 | 0 |
| muon_search | 120 | 118.92 | 1.5468 | 53.72% | 2406 | 0 |

Selection uses only the first half of the validation text. The selected checkpoint is evaluated once on the second half, over nonoverlapping target positions. The test split is never loaded.
Each search includes the default plus two log-uniform random learning rates, spending 37.5% of its cap on equal exploration slots and continuing the best trial with the remaining budget. Trial continuation preserves model, optimizer and RNG state.
All data/model initialization, training, probes, validation, in-memory snapshots, selection and model checkpoint I/O are charged. Shared Python import/environment setup and publication of result tables are excluded. No other work runs concurrently in the suite.
A small final reserve and conservative update/evaluation guards can leave part of the cap unused; actual time is shown. Any over-budget campaign is flagged and cannot support a fair winner claim.
Total updates/tokens include discarded trials; selected-checkpoint steps and probe timings are in the CSV. Methods may complete different numbers of updates under the same cap.
This is a single-seed CPU pilot on a previously studied dataset, not an unseen-task or GPU benchmark. Timing depends on system load. Search ranges/controller settings were fixed before this run; no controller tuning is charged to this pilot.
