# Adaptive Muon correctness audit

Targeted replay and frozen-direction diagnostics only; 24.1 CPU seconds. No full equal-budget comparison was rerun.

Positive loss delta means worse. All candidate steps begin at the same weights and use the same frozen Muon direction.

| Step | Accepted LR | Probe q | FP64 HVP q | Relative q error | Predicted probe delta | Actual probe delta | Fresh delta | Joint fresh delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 0.041096 | 105.1083 | 108.6335 | 3.25% | -0.0888 | -0.0827 | +0.0257 | +0.0275 |
| 150 | 0.10175 | 39.9498 | 42.4960 | 5.99% | -0.2068 | -0.1202 | +0.0748 | +0.0758 |
| 200 | 0.10175 | -39.9627 | -24.3386 | 64.19% | -0.4207 | -0.1061 | -0.0049 | -0.0048 |
| 250 | 0.10175 | -107.5430 | -80.1861 | 34.12% | -0.7796 | -0.0771 | -0.0010 | -0.0015 |

Probe q is the actual FP32 finite difference at rho=0.001; reference q is an independent FP64 second derivative using math attention. Full FP32/FP64 distance sweeps are in audit.json.
The matrices-only columns isolate the curvature model. Joint columns also apply the exact companion AdamW update from the same original gradients; all weights and optimizer state are then restored.
Fresh losses average independent training minibatches, not validation/test data. No probe distance, safety factor, fallback policy or training optimizer was changed by this audit.
The replay assertions check recorded training loss, probe numerator/curvature, rates and decisions. Further assertions check original gradients/weights remain intact and probing does not advance the optimizer state.
