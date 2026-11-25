# nanochat transfer study: raw records

The groupmate's nanochat benchmark, exported so the joint paper can check every number. It answers the request in
[docs/overnight.md](../../docs/overnight.md) ("exact revision, configs, split definitions, seed-level metrics, raw
timings and LR traces"). Report it as a **separate benchmark** from the Tiny Shakespeare study.

## Setup

- **Code.** nanochat (Karpathy) at revision `92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`, plus the changes in [code/](code/):
  - `nanochat_92d63d4_tracked.diff`: the four-way data split (`dataset.py`, `dataloader.py`).
  - `curv_optim.py`: split Muon step, AdaGO (Algorithm 2) and the probe.
  - `curv_train.py`: single-GPU trainer for all optimizers.
  - `test_curv_optim.py`: the split Muon step is bit-identical to nanochat's fused kernel.
  - Queue and analysis tools: `run_queue.py`, `make_jobs.py`, `make_trace.py`, `analyze.py`.
- **Data.** ClimbMix-400B shards 0-9. The split:
  - train = shards 0-6;
  - fresh = shard 7, used only for diagnostics;
  - test = shard 8, reserved and scored only at the end of each run;
  - val = shard 9, used for every selection decision.

  BPE vocabulary of 32,768, retrained on train.
- **Models.**
  - d384: depth 6, d_model 384, 10.6M transformer-matrix parameters, 73.5M in total.
  - d768: depth 6, d_model 768, i.e. 2× the width and 4× the matrix parameters (42.5M), 168M in total.
- **Training.** 381 steps × 131,072 tokens = 49.9M tokens per run.
  - Schedule: nanochat's 40-step warmup, then a linear warmdown over the last 65% to 5% of the peak.
  - Muon weight decay: 3.70, from nanochat's rule.
  - Seeds change only the initialization; nanochat's data order is fixed.
- **Hardware.** One RTX 5090.

## Files

| File | Contents |
|---|---|
| `runs.csv` | One row per run (228 runs): optimizer, seed, width, flags, final val/test loss and bpb, tokens, probe count, train/probe/wall seconds, peak memory |
| `run_args.jsonl` | Full resolved command-line arguments of every run |
| `lr_traces.csv` | Per-step train loss and applied Muon LR of every run |
| `probes.csv` | Every curvature probe: slope `gD`, curvature `q_hat`, `eta_hat`, distance, validity, applied rate, seconds |
| `traces/` | The two frozen replay curves (geometric mean of the v4 development runs at seeds 0 and 10) |
| `jobs/` | The exact job queues, in run order (`$PROJECT` = project root) |

Run tags:

- `s_*`: LR searches.
- `f_*`: finals on seeds 1-3.
- `cf_*`: confirmation seeds 4-9.
- `w_*`: d768 runs.
- `rp_*`: replay.
- `ss_*`: schedule-shape search.
- `*T` (d768): the d384 hyperparameter or curve, reused.

## Key comparisons (test loss, nats/token, recomputed from `runs.csv`)

| Comparison | Seeds | Paired difference | Paired p |
|---|---|---:|---:|
| v4 (next-batch probe + schedule) − tuned Muon, d384 | 1-3 | −0.0287 | 0.007 |
| same | 7-9 (never used for selection) | −0.0145 | 0.157 |
| same | all six | −0.0216 | 0.005 |
| Replay of the frozen v4 curve − tuned Muon, d384 | six | −0.0293 | 0.002 |
| Replay − live v4, d384 | six | −0.0077 | 0.032 |
| Shape-tuned Muon (16-trial schedule search) − tuned Muon, d384 | six | +0.0073 | 0.069 |
| v4 − re-tuned Muon, d768 | 1-3 | −0.0055 | 0.750 |
| d384 replay curve at d768 − re-tuned d768 Muon | 1-3 | −0.0191 | 0.011 |

Live v4 spends 7.5% of training time on probes at d384.

## Caveats

These must travel with the numbers.

- **Development before selection.** Reaching v4 took 12 calibration runs and 16 screening runs at d384. Seeds 1-3
  were used by three candidates (v2, v3, v4). v4 was picked on seeds 0 and 10. Seeds 7-9, the replay curve and every
  d768 result played no part in selection, apart from two d768 seed-0 screening runs (`w_v3h_c1`, `w_v3h_c0.5`).
- **"Zero tuning runs" means no per-run search after development.**
- **The proposal's original same-batch rule lost to tuned Muon.** Report it.
- **What the probe actually contributes.** The gain comes from the schedule the probe discovers: a high early peak,
  then roughly 4× decay per 100 steps. Online adaptation is not the source, because replaying the curve does at least
  as well.
- **Scope.** One short horizon, batch size, dataset and architecture.
