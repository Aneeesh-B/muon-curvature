# nanochat benchmark records

Raw records for the nanochat benchmark, exported so the paper's numbers can be audited. Report it as a benchmark
separate from the Tiny Shakespeare study. The export covers what [docs/overnight.md](../../docs/overnight.md) asks for:
- the revision and configs;
- the data splits;
- per-seed metrics, timings and LR traces.

## Setup

**Code.** nanochat at revision `92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`, plus these changes in [code/](code/):

| File | Contents |
|---|---|
| `nanochat_92d63d4_tracked.diff` | The data split (`dataset.py`, `dataloader.py`) |
| `curv_optim.py` | The split Muon step, AdaGO (Algorithm 2) and the probe |
| `curv_train.py` | A single-GPU trainer for all the optimizers |
| `test_curv_optim.py` | Checks that the split Muon step matches nanochat's fused kernel bit for bit |
| `run_queue.py`, `make_jobs.py`, `make_trace.py`, `analyze.py` | The queue and analysis scripts |

**Data.** ClimbMix-400B shards 0–9, with a 32,768-token BPE vocabulary retrained on the training shards.

| Shards | Role |
|---|---|
| 0–6 | Training |
| 7 | Diagnostics only |
| 8 | Test, scored once at the end of each run |
| 9 | Validation, used for every selection decision |

**Models.** Both have depth 6.

| Model | Width | Matrix parameters | Total parameters |
|---|---:|---:|---:|
| d384 | 384 | 10.6M | 73.5M |
| d768 | 768 | 42.5M | 168M |

**Training.**
- Each run is 381 steps of 131,072 tokens, 49.9M tokens in total.
- nanochat's 40-step warmup, then a linear decay over the last 65% of training down to 5% of the peak.
- Muon weight decay is 3.70, from nanochat's rule.
- Seeds change only the initialization. The data order is fixed.
- One RTX 5090.

## Files

| File | Contents |
|---|---|
| `runs.csv` | One row per run (228 runs): optimizer, seed, width, flags, final val/test loss and bpb, tokens, probe count, train/probe/wall seconds, peak memory |
| `run_args.jsonl` | The full command-line arguments of every run |
| `lr_traces.csv` | Train loss and applied Muon LR at every step of every run |
| `probes.csv` | Every probe: slope, curvature estimate, proposed LR, distance, validity, applied rate, seconds |
| `traces/` | The two frozen replay curves (the geometric mean of the v4 development runs at seeds 0 and 10) |
| `jobs/` | The job queues, in run order (`$PROJECT` is the project root) |

Run tags:

| Tag | Runs |
|---|---|
| `s_*` | LR searches |
| `f_*` | Finals on seeds 1–3 |
| `cf_*` | Confirmation seeds 4–9 |
| `w_*` | d768 runs |
| `rp_*` | Replay |
| `ss_*` | Schedule-shape search |
| `*T` (on d768 runs) | Reuses the d384 setting or curve |

## Main comparisons

Test loss in nats per token, recomputed from `runs.csv`.

| Comparison | Seeds | Paired difference | p |
|---|---|---:|---:|
| v4 (next-batch probe + schedule) − tuned Muon, d384 | 1–3 | −0.0287 | 0.007 |
| same | 7–9 (never used for selection) | −0.0145 | 0.157 |
| same | all six | −0.0216 | 0.005 |
| Replay of the frozen v4 curve − tuned Muon, d384 | six | −0.0293 | 0.002 |
| Replay − live v4, d384 | six | −0.0077 | 0.032 |
| Shape-tuned Muon (16-trial schedule search) − tuned Muon, d384 | six | +0.0073 | 0.069 |
| v4 − re-tuned Muon, d768 | 1–3 | −0.0055 | 0.750 |
| d384 replay curve at d768 − re-tuned d768 Muon | 1–3 | −0.0191 | 0.011 |

Live v4 spends 7.5% of training time on probes at d384.

## Caveats

These should accompany any use of the numbers above.

- **Development.**
  - v4 came out of 12 calibration runs and 16 screening runs at d384.
  - Seeds 1–3 were used for three candidates (v2, v3 and v4), and v4 was picked on seeds 0 and 10.
  - Seeds 7–9, the replay curve and the d768 results played no part in choosing it. The one exception is two d768
    seed-0 screening runs, `w_v3h_c1` and `w_v3h_c0.5`.
- **"Zero tuning runs"** means no per-run search after development.
- **The original rule.** The proposal's same-batch rule lost to tuned Muon.
- **Where the gain comes from.** It comes from the schedule the probe finds: a high early peak, then roughly 4× decay per
  100 steps. It does not come from adapting during the run, because replaying the curve does at least as well.
- **Scope.** One short horizon, one batch size, one dataset and one architecture.
