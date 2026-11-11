# Interpretation of the short budget comparison

The four methods each had one 120-second cap, seed 0, on the same CPU with four
PyTorch threads. Actual charged time was 118.80-119.10 seconds each (475.82
seconds across all methods). Reserved evaluation/saving time was not filled
with artificial waits. All four remained inside their caps.

| Method | Reporting cross entropy | Next-character accuracy |
|---|---:|---:|
| Default Muon, LR 0.02 | **1.5290** | **54.21%** |
| Random-search Muon | 1.5468 | 53.72% |
| Random-search AdamW | 1.6406 | 50.89% |
| Current adaptive Muon | 2.1583 | 36.80% |

These values are for the checkpoint selected by tuning-validation loss, not
necessarily the final training weights. Both halves of validation contain 27,776
evaluated next-character targets. Reporting values were computed only after
selection and were never used by the search/controller.

The current adaptive rule did **not** save tuning compute in this pilot. It made
44 probes and completed 2,284 updates, but its best tuning checkpoint was at
update 212. Its first two accepted rates were approximately 0.04110 and 0.10175;
later probes reused the previous rate. This is consistent with the earlier
aggressive-step/fallback failure. The failure is not explained simply by the
time spent probing: the trajectory stopped improving well before its cap.

Muon search tested rates 0.02, 0.0075642 and 0.0171072. AdamW search tested
0.001, 0.000123856 and 0.00692368. Both selected their default-rate trial to
continue. Default Muon therefore benefited from spending its full budget on an
already good rate, while Muon search spent part of the same budget exploring
alternatives. This result applies to these predeclared ranges and this seen
task, rather than to every possible tuning strategy.

Implementation checks passed: 14 tests, including exact agreement with the old
adaptive trainer and interrupted/resumed training with dropout. In the actual
comparison all four methods had identical first-batch loss, default/adaptive
Muon agreed through the first probe's original forward pass, and the resumed
default-rate Muon search trial matched the standalone default trajectory through
all 1,781 of that trial's updates. Source hashes were unchanged during execution.

This is a one-seed development pilot, not a statistical conclusion about all
adaptive optimizers or GPU training. The dataset and default rates had already
been examined in earlier development. No safeguards were retuned or experiments
repeated in response to these reporting losses.

Protocol: `docs/compute-budget.md` at the repository root. Full tables and
machine-readable metrics are in `comparison.md` / `comparison.csv`. Frozen
settings are in `manifest.json`; small per-method logs are in `raw/`. Full
continuation and selected-model checkpoints stay in `runs/budget_seed0/`.
