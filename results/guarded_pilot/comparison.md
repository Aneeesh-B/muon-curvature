# Equal-budget optimizer comparison

One seed, same model/data and hardware, one elapsed-time cap per method. Lower loss and higher next-character accuracy are better.

| Method | Budget (s) | Used (s) | Reporting loss | Accuracy | Total updates | Probes |
|---|---:|---:|---:|---:|---:|---:|
| adaptive_muon | 60 | 58.91 | 2.1539 | 36.88% | 1101 | 21 |
| default_muon | 60 | 58.58 | 1.6044 | 52.12% | 1190 | 0 |
| guarded_muon | 60 | 58.95 | 1.5733 | 52.94% | 1201 | 23 |

Selection uses only the first half of the validation text. The selected checkpoint is evaluated once on the second half, over nonoverlapping target positions. The test split is never loaded.
Each search includes the default plus two log-uniform random learning rates, spending 37.5% of its cap on equal exploration slots and continuing the best trial with the remaining budget. Trial continuation preserves model, optimizer and RNG state.
All data/model initialization, training, probes, validation, in-memory snapshots, selection and model checkpoint I/O are charged. Shared Python import/environment setup and publication of result tables are excluded. Small unit tests ran concurrently during part of the original adaptive campaign; timing comparability is therefore imperfect.
A small final reserve and conservative update/evaluation guards can leave part of the cap unused; actual time is shown. Any over-budget campaign is flagged and cannot support a fair winner claim.
Total updates/tokens include discarded trials; selected-checkpoint steps and probe timings are in the CSV. Methods may complete different numbers of updates under the same cap.
This is a single-seed CPU pilot on a previously studied dataset, not an unseen-task or GPU benchmark. Timing depends on system load. The guarded controller was designed after inspecting earlier pilot results; this is an exploratory comparison, not an independent confirmation.
