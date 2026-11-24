# Adaptive-rate ablation: longer pilot

Does the adaptive proposal help, or is the guarded result simply explained by falling back to the base rate? Both methods build the same type of Muon direction, sample separate calibration and check batches, compute the same curvature probe, and use the same backtracking safeguard. The adaptive method proposes the estimated rate when valid and 0.02 otherwise. The control always proposes 0.02. Both pay for the probe, even though the control does not use its result. Once trajectories differ, the batches are drawn with matching random seeds but directions and weights differ.

| Method | Time | Updates | Selected update | Reporting loss | Accuracy |
|---|---:|---:|---:|---:|---:|
| Guarded adaptive | 118.90 s | 2,288 | 2,288 | 1.5084 | 54.67% |
| Guarded base control | 118.74 s | 2,324 | 1,950 | 1.5161 | 54.31% |

Each method had a 120-second cap. Selection used tuning validation; the reporting half was evaluated after selection. The adaptive method performed 44 probes: 39 valid estimates and 5 fallback proposals. Its median proposed rate was 0.00609 and median post-check rate was 0.00478; five checks rejected every candidate. The control used a 0.02 proposal for all 45 probes; five checks rejected every candidate. The adaptive result therefore cannot be explained simply by repeatedly proposing the base rate, but this single-seed difference is small and cannot establish a reliable advantage. The check and adaptation interact, and this does not isolate their separate effects without further ablations.

This is a development experiment on previously inspected data, not an independent final evaluation. Timing and checkpoint selection influence the result; run multiple seeds and a matched-update comparison to investigate reliability. The test split was not loaded.

Reproduce from the project root with a new output directory:

```bash
../.venv/bin/python -m curvature_muon.budget --config configs/budget.json \
  --out runs/guarded_ablation --device cpu --seconds 120 \
  --methods guarded_muon guarded_base_muon
```
