# Guarded Muon: independent-batch pilot

Guarded Muon is an experimental version of the Muon optimizer that checks how big a step to take using additional training batches. Think of it as choosing a direction, estimating a step size using a second sample, and checking that step on a third sample before taking it.

## Why add a guard?

Muon changes a model's weights to reduce prediction error. Its learning rate controls how large a change it makes. In the earlier experiment, adaptive Muon chose a learning rate that looked good on the same batch used to choose the update direction. But that step often made the loss worse on fresh batches. It could also keep an overly large rate after later curvature estimates failed.

## How it works

Every 50 updates, starting at update 100 in this pilot:

1. **Choose a direction.** Use the normal training batch to work out which way Muon will move the model's matrix weights.
2. **Estimate a step size.** Sample a separate calibration batch and measure how its loss curves along that direction. Both gradient measurements for this estimate use the same calibration batch. If the estimate is invalid, start with the configured reference rate, 0.02.
3. **Check the step.** Sample another training batch and measure its loss before and after the proposed matrix update. If loss increases or is nonfinite, halve the rate and try again, for at most six candidate rates. If none passes, set the matrix learning rate to zero until the next probe.
4. **Train.** Apply the selected matrix rate and hold it for the next 50 updates, while the ordinary training batches and directions continue to change. The companion AdamW update still runs, including when the matrix rate is zero.

The check is like trying a step on a different practice problem before committing to its size. It is not a guarantee: the check uses only one sampled batch, checks only the matrix update (not the companion AdamW update), and does not recheck each of the following updates. The separately sampled batches may contain overlapping examples; they are not disjoint dataset splits.

## What happened?

This was one seed on CPU, with a 60-second budget for each method. Lower loss and higher accuracy are better.

| Method | Reporting loss | Next-character accuracy |
|---|---:|---:|
| Guarded Muon | 1.5733 | 52.94% |
| Fixed-rate Muon | 1.6044 | 52.12% |
| Original adaptive Muon | 2.1539 | 36.88% |

The result is encouraging, but preliminary. The guarded method was designed after inspecting earlier results, and small unit tests ran concurrently during part of the original adaptive run, making the timing comparison imperfect. We need more seeds and separate tests of calibration and backtracking to understand which change helped. See [NOTES.md](NOTES.md) for probe counts and [comparison.md](comparison.md) for the full accounting.

## Reproduce

From the project root, with the parent environment and prepared dataset described in the [project README](../../README.md):

```bash
../.venv/bin/python -m curvature_muon.budget \
  --config configs/budget.json --out runs/guarded_pilot \
  --device cpu --seconds 60 \
  --methods guarded_muon adaptive_muon default_muon
```

Use a new or empty output directory. The run charges calibration, checks, training, evaluation, and checkpoint work to each method's time budget.
