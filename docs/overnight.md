# Overnight Muon experiments

Start on the 4070 if it is available now. Use one GPU per comparison and run its
methods sequentially. If the 5090 becomes available after the small-model queue
finishes, run the wide queue there. Keep wall-clock comparisons within one
card, model, and seed.

## Launch

On the GPU host, use Bash on Linux or WSL and a Python environment with a
CUDA-enabled PyTorch build and the editable project installed. The 5090 needs
a Blackwell-compatible wheel and driver. Use the same PyTorch build on both
hosts when practical. [PyTorch CUDA installation](https://pytorch.org/get-started/locally/)

```bash
# One-time setup if the host does not already have the project installed.
uv venv --python 3.12
# Install a CUDA-enabled torch wheel into .venv using the PyTorch selector.
uv pip install --python .venv/bin/python -e .

# Launch exactly one of these on a given GPU, from a persistent shell.
bash scripts/overnight.sh 4070
bash scripts/overnight.sh 5090
bash scripts/overnight.sh 5090-wide   # Small queue already finished elsewhere.
```

Set `PYTHON` to use an existing environment. Set `CUDA_VISIBLE_DEVICES` to
select a GPU if needed. Check the launcher's device name before leaving. It
records the driver, checks an actual CUDA matrix multiply, prepares verified
Tiny Shakespeare data, and runs the tests. Each size then gets a 151-update
check of all matched-update methods and a 90-second campaign for each search
baseline. The launcher checks that each required probe ran, each search
candidate trained, the winner continued, and both timed smokes fit their
caps. An unsuccessful preflight stops the queue before production on that
size. Keep other GPU workloads off the selected device. Check free disk
space before launch and allow several GB for checkpoints and event logs.

## Run order and budget

| Profile | Run order | Equal-wall-time caps | Matched-update planning allowance |
|---|---|---:|---:|
| 4070 | 0.81M NanoGPT only | 5 methods × 3 seeds × 8 min = 2 h | Up to 90 min estimated |
| 5090 | 0.81M, then 9.72M NanoGPT | 2 h small + 5 × 3 × 12 min = 3 h wide | Up to 90 min small + 60 min wide estimated |
| 5090-wide | 9.72M NanoGPT only | 3 h | Up to 60 min estimated |

The wall-time column sums campaign caps. Add preflight, process startup and
report writing. The launcher estimates matched-update runtime from the
151-update timings with 30% headroom. It chooses 2,000 updates per run
(4.096M training characters), then 1,000 (2.048M) if needed, or defers
that stage if neither fits the planning allowance. The horizon is chosen
before any production losses are available. Matched-update runs finish their
full horizon, so their allowance is an estimate. The 5090 both-size plan is
5 h of wall-time campaigns plus preflight and up to 2.5 h of estimated
matched-update work. For a shorter window, finish the small queue first.

Both sizes use four layers, context 128, 16 sequences per update and FP32.
The small model uses width 128 and the wide model uses width 448. Both use the
same character corpus, batch policy, and optimizer settings. This is a width
check at fixed data and architecture family. The configs are
[small](../configs/tiny.json), [wide](../configs/wide.json) and
[overnight protocol](../configs/overnight.json).

## 1. Equal wall time — the main result

Use seeds 0, 1, 2 with paired initialization and training batch draws within
each seed. Method order is shuffled per seed. Keep FP32 and the configured
guard fixed throughout the run.

| Method | What it establishes |
|---|---|
| Guarded Muon | The proposed optimizer, with independent curvature and a separate backtracking batch. |
| Guarded base Muon | Always proposes 0.02 and pays for the same curvature work and backtracking. This controls for the guard while testing the curvature proposal. |
| Default Muon | Ordinary constant-rate Muon at 0.02. |
| Muon rate search | Five fixed rates, 0.003, 0.01, 0.02, 0.04, 0.08. |
| AdamW rate search | Five fixed rates, 0.0003, 0.001, 0.003, 0.006, 0.01. |

For each search, 40% of the cap is divided across the five rates. The remaining
time continues the tuning-validation winner with its live weights, optimizer
and RNG state. Every candidate, probe, validation pass and checkpoint save
is charged. Both Muon search and the adaptive methods use the same companion
AdamW configuration. This answers whether the **current guarded controller**
is a better use of wall time than this particular fixed-LR search budget.
Log all candidate results, including any divergence.

## 2. Matched updates — identify what helped

Run these five trajectories for each seed, score the final update, and report
training time and probe overhead alongside loss.

- Guarded Muon and guarded base Muon. Same budget in ordinary updates and
  the same guard. Their difference measures the value of the curvature
  proposal within that guard.
- Schedule-only Muon, scheduled same-batch Muon, and scheduled next-batch
  Muon. All share a 40-update warmup, a decay beginning around 35% of the
  horizon, and a linear envelope down to 5% of its peak. The two probe
  variants probe every 25 updates starting at 25. Comparing the probe
  variants isolates which batch supplied curvature. Comparing each with
  schedule-only Muon tests whether calibration improves on the fixed
  schedule at the default peak of 0.02.

The next-batch arm peeks at the next training draw and leaves the real batch
stream intact. Both gradients use that same next batch before the current
update. A valid estimate sets the peak before decay and caps the frozen
schedule envelope during decay. An invalid estimate keeps the previous one.
Both probe arms use the repo's rate bounds and the same initial peak as the
schedule-only arm. The adaptation is a local transfer of the PDF's rule to
NanoGPT. A claim about beating *tuned scheduled Muon* would require a separate
peak search under that same schedule. Matched runs give every method the same
number of ordinary training updates. The additional probe compute is visible
in elapsed time.

## 3. Read the results in the morning

1. Inspect `runs/overnight-*/launcher.log`, per-seed `comparison.csv`,
   `manifest.json`, and any `failure.json`. Check for all three paired
   seeds, cap compliance, successful search continuation, complete matched
   horizons, and nontrivial probe activity. Keep failures visible and
   exclude over-cap pairs from fair wall-time claims.
2. For the main result report all three paired reporting-loss deltas
   (Guarded minus Muon search, Guarded minus Default, Guarded minus
   Guarded Base), their mean and seed-to-seed spread. Negative is better.
   Give loss in nats per character. Accuracy, selected step, total ordinary
   tokens, actual elapsed time and peak allocated CUDA memory are secondary.
   First-half validation alone selects the timed checkpoints. The second
   half is for reporting. Keep the reserved test split closed while making
   design decisions.
3. From the matched runs plot LR against updates and summarize probe status,
   clamp/fallback rate, guard rejection count, time in probes relative to
   ordinary training, and final reporting deltas. Look for agreement across
   seeds and widths. Treat three pairs as descriptive. A 0.02 nats/character
   gap is the repo's existing practical pilot threshold. A smaller or mixed
   gap is a close result at this sample size.
4. This is enough for a course-project conclusion about this optimizer on
   Tiny Shakespeare and, with the 5090 extension, sensitivity to width.
   For a confirmatory claim, freeze the methods and comparisons, then score
   selected checkpoints on the reserved test split and add paired seeds.
   Transfer to another dataset, tokenizer or training regime needs its own
   experiment. Avoid spending the first night on a broad sweep or a rewrite
   of the trainer.

## Groupmate's nanochat PDF

The local and PDF evidence agree that a direction built on a batch can
produce an overly optimistic step estimate on that same batch. The
next-batch slope is unbiased for the expected slope conditional on the
current weights and direction if training draws are independent. This
argument does not make the noisy ratio of slope to curvature unbiased.
The reported next-batch method also changes the schedule and was chosen
after screening 16 runs. “Zero tuning runs” is best read as zero per-run
search after development. The reported narrow-model gain over peak-tuned
Muon is about 0.022 nats/token over six paired seeds. The three held-back
seeds average about 0.015 nats/token improvement with p=0.16. The wider
live result is inconclusive with three seeds (p=0.75). The replay result
is promising and uses two curve-building runs. Treat these as distinct
nanochat results (ClimbMix, 50M tokens, 73.5M and 168M total parameters).
Request the exact revision, configs, split definitions, seed-level metrics,
raw timings and LR traces before relying on the numerical claims in the
joint paper. The PDF reports 17 GPU-hours on a 5090 for 230 runs.

The immediate local test is the next-batch arm and its two schedule
controls. If it consistently helps, the next experiment is a frozen replay
curve from two separate development seeds. Charge both curve-building runs,
replay on paired evaluation seeds at the same horizon, and compare with
live probing and peak-tuned scheduled Muon. Use the same frozen curve at
the second width to test transfer. Existing per-step LR logs provide
the source data for that follow-up.
