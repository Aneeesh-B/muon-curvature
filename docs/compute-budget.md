# Short compute-budget experiment

Question: with one shared elapsed-time allowance per method, does spending
compute on online Muon calibration yield a better selected model than spending
it on learning-rate search?

## Protocol

The default is **four methods, one seed, 120 seconds per method**: approximately
eight minutes of model work in total. All campaigns run sequentially on the
same device. The 812,288-parameter model, training split, batch size 16, context
128, FP32, zero dropout/decay, and companion AdamW configuration follow
`configs/tiny.json`. The old 2,000-update limit is replaced by the clock.

| Method | Policy |
|---|---|
| `adaptive_muon` | One trajectory, initial LR 0.02, probe at step 100 then every 50 updates. |
| `muon_search` | Three candidates: LR 0.02 plus two log-uniform samples from [0.001, 0.1]. |
| `default_muon` | One trajectory at LR 0.02. |
| `adamw_search` | Three candidates: LR 0.001 plus two log-uniform samples from [0.0001, 0.01]. |

The adaptive rule is unchanged: relative probe distance 0.001, bounds
[0.00001, 0.2], safety factor 1, and reuse the previous LR on invalid estimates.
This is an evaluation of that existing rule, not an additional tuning sweep.
All rates remain constant between probes; the search baselines tune a constant
rate, not a schedule. Other hyperparameters are fixed.

Each search spends 37.5% of its total cap on equal exploration slots. At 120
seconds that is approximately 15 seconds per candidate. It selects the trial
with the lowest observed tuning-validation loss and continues that SAME live
trial for the remaining approximately 75 seconds. Weights, Muon momentum,
AdamW moments/step counters, training-batch generator, and dropout random state
are preserved. There is no fresh restart after selecting a rate. Poor/nonfinite
trials are stopped, their consumed time remains charged, and their earlier valid
saved checkpoints remain eligible for final model selection.

All candidates within a method, and all methods, use the same initial model
seed and independent copies of the same training-batch stream. Search sampling
uses a separate seeded generator. Method order is shuffled once with a fixed
seed and recorded in the manifest. This is a single-machine wall-time comparison,
not a FLOP-count comparison or an extrapolation to GPU training.

## Validation and accuracy

Only the prepared training and validation arrays are loaded. The validation
array is divided contiguously into equal tuning/reporting halves. Every
evaluation covers all complete context windows in its half, with nonoverlapping
target positions. Batches do not cross the split boundary. Partial final batches
are weighted by their actual token counts.

- **Tuning half:** used for trial selection and selecting the best checkpoint
  seen within each method. Evaluated about every five seconds of active trial
  time, and at exploration/continuation boundaries.
- **Reporting half:** evaluated once per method, after checkpoint selection. Its
  loss and accuracy are never used to select trials or checkpoints.
- **Test split:** never loaded.

The primary metric is reporting cross entropy (nats per character, lower is
better) of the tuning-selected checkpoint. The secondary metric is top-1
next-character prediction accuracy (higher is better), computed from the same
forward passes. Accuracy measures teacher-forced next-character predictions;
it is not a human assessment of generated samples.

The reporting half is independent of selection within THIS experiment. Both
halves were part of the validation split used in earlier development, so this
is not a previously unseen-data confirmation study. One seed cannot establish
statistical reliability or generality across architectures/tasks.

## Time accounting

The clock starts before method-specific data loading and model/optimizer
initialization. It includes training, probes, validation, copying best weights,
selection, trial switching, step/event logging, reporting evaluation, and model
checkpoint serialization. The final reporting pass and saves must fit inside
the same cap. The search does not get a separate allowance per trial.

Shared Python import/environment setup, tests, and writing the final accounting
tables are outside the model-work cap. Source hashes, resolved settings, method
order, software/device information and dataset metadata are recorded.

A 0.75-second final reserve plus measured update/evaluation guards prevents
giving methods a free last update or final evaluation. Some time can remain
unused, so both cap and actual consumed time are reported. No artificial idle
padding is counted as computation. Any campaign exceeding its cap is explicitly
flagged and cannot support a fair winner claim. The four-second debug version
uses the same conservative reserve and therefore leaves a larger fraction of
its cap unused; it is only an accounting/integration check.

## Run

From the repository root, using the existing local parent environment:

```bash
../.venv/bin/python -m unittest discover -s tests -v
../.venv/bin/python -m curvature_muon.budget \
  --config configs/budget.json --out runs/budget_seed0 --device cpu
```

For a repository-local environment, replace `../.venv/bin/python` with
`.venv/bin/python`. Existing output directories are never overwritten.
`--seconds`, `--seed`, and `--methods` can override the configuration. A new seed
means a new comparison campaign; it is not another trial in the existing cap.
The CUDA path is available but has not been checked on a GPU in this workspace.

## Outputs

- Suite `manifest.json`: frozen protocol, source hashes, resolved config and order.
- Per-method `events.jsonl`: updates, curvature decisions, selection evaluations,
  elapsed time, and continuation selection.
- Per-method `result.json`: reporting metrics, actual time/cap, selected checkpoint,
  all candidate rates, steps, total tokens and component timings.
- `trial_*_continuation.pt`: complete live training states for resuming a trial.
- `selected.pt`: selected model weights/config and evaluation metrics. This is
  model-only because the best earlier weights may precede the final live
  optimizer state. Do not pair it with that later optimizer state to resume.
- `comparison.md` and `comparison.csv`: side-by-side final results.

Large checkpoints stay in ignored `runs/`. The final report, suite manifest,
results and event logs are copied into versionable `results/budget/` after the
run. No training configuration is changed in response to the reporting losses.

The tests cover exact agreement with the existing adaptive trainer, interrupted
and resumed trajectories with dropout, optimizer and RNG preservation, split
boundaries, token-weighted accuracy including partial batches, reproducible
random candidate selection, and the shared search cap/continuation behavior.
