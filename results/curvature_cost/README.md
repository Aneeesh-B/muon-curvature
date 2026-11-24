# Curvature probe computation cost

At the selected guarded-Muon checkpoint from the longer ablation (update 2,288), we timed the independent-batch finite-difference calculation against an exact automatic-differentiation Hessian-vector product (HVP). Both use the same frozen Muon direction, calibration batch, FP32 model, and math attention backend. Timings include the base and shifted gradients for finite differences and both derivative stages for the HVP, including scalar reductions and probe restoration. Neither includes direction construction or the guarded method's separate acceptance/backtracking check.

| Calculation | Median time per estimate | Directional curvature q |
|---|---:|---:|
| Finite difference | 72.46 ms | 74.8822 |
| Exact autodiff HVP | 105.42 ms | 75.0129 |

The finite difference took about 31% less time (the HVP was about 1.45 times as slow). The finite-difference curvature was 0.17% below the HVP at this checkpoint, direction, batch, and probe distance; this is not a bound on its error elsewhere. Three warmups preceded 12 alternating-order paired measurements. The machine was a four-thread CPU running an 812,288-parameter model with batch size 16 and context 128. Both calculations use math attention because the HVP requires double-backward support; this controlled backend comparison does not measure optimized GPU kernels.

An HVP computes the directional quantity without explicitly materializing the full Hessian. Constructing a full Hessian is not necessary to obtain the proposal's scalar and was not benchmarked. These measurements do not establish a general GPU speedup, memory savings, or total-training savings: guarded Muon also performs a fresh-batch acceptance check and possibly multiple backtracking forwards. Peak memory was not measured. Both methods need independent base gradients in this comparison; the original same-training-batch probe can reuse the training gradient, which is a different accounting. The earlier audit found larger numerical errors at other checkpoints and invalid curvature later in that trajectory.

The checkpoint is generated locally by the guarded ablation and is not tracked in Git. With that checkpoint available, reproduce from the project root:

```bash
../.venv/bin/python -m curvature_muon.benchmark_curvature \
  --checkpoint results/guarded_ablation/guarded_muon/selected.pt \
  --out runs/curvature_cost.json
```

The output directory must exist. Raw individual timings and configuration are in [benchmark.json](benchmark.json).
