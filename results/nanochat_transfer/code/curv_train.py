"""
CS182 project: "Can Occasional Curvature Probes Reduce Learning-Rate Tuning for Muon?"

Single-GPU pretraining script derived from scripts/base_train.py (same model, data pipeline, scaling rules
for batch size / LR / weight decay, LR + momentum + weight-decay schedules), with four matrix optimizers:

  --optim muon   tuned Muon (nanochat Muon; peak LR via --matrix-lr, nanochat warmup/warmdown schedule)
  --optim adamw  tuned AdamW on the matrices (decoupled WD), same schedule
  --optim adago  AdaGO (Algorithm 2 of arXiv:2509.02981), fixed eta, no schedule, no weight decay
  --optim curv   proposed rule: Muon direction, one shared multiplier eta set by a gradient-difference
                 curvature probe every k steps (eta frozen between probes; previous eta kept if invalid)

In every case the embeddings, lm_head and scalar parameters use nanochat's AdamW companion configuration
(identical across optimizers). Optional in-run diagnostics (--diag-steps) for the proposal's diagnostic
experiments. Results are written as one JSON file (--out).

Run from the nanochat root:  python -m scripts.curv_train --optim curv --out results/x.json
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import json
import math
import time
import argparse

import torch
import torch._dynamo
# nanochat's fused AdamW kernel is compiled per parameter shape; the AdamW baseline adds the matrix shapes
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.recompile_limit = 64

from nanochat.gpt import GPT, GPTConfig
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit, tokenizing_distributed_data_loader_with_state_bos_bestfit
from nanochat.common import compute_init, compute_cleanup, print0, autodetect_device_type, COMPUTE_DTYPE
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.curv_optim import (CurvMuonAdamW, curvature_probe, curvature_probe_heldout, stacked_params, stacked_grads,
                                 add_to_params, set_params, _dot)
from nanochat.flash_attention import USE_FA3

parser = argparse.ArgumentParser()
# model
parser.add_argument("--depth", type=int, default=6)
parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
parser.add_argument("--head-dim", type=int, default=128)
parser.add_argument("--max-seq-len", type=int, default=2048)
parser.add_argument("--window-pattern", type=str, default="L", help="L = full attention (SDPA has no sliding window)")
# horizon / batch
parser.add_argument("--tokens", type=int, default=50_000_000, help="training tokens per run")
parser.add_argument("--device-batch-size", type=int, default=16)
parser.add_argument("--total-batch-size", type=int, default=-1, help="-1 = nanochat auto rule")
# optimizer choice
parser.add_argument("--optim", type=str, required=True, choices=["muon", "adamw", "adago", "curv"])
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--matrix-lr", type=float, default=-1.0, help="effective peak matrix LR (muon/adamw). -1 = nanochat default 0.02*batch_scale (muon) / 3e-3 (adamw)")
parser.add_argument("--muon-wd", type=float, default=-1.0, help="Muon weight decay; -1 = nanochat scaling rule")
parser.add_argument("--adamw-matrix-wd", type=float, default=0.1)
parser.add_argument("--adamw-matrix-betas", type=str, default="0.9,0.95")
parser.add_argument("--adamw-eps", type=float, default=1e-8, help="AdamW epsilon for the matrix params of the AdamW baseline (nanochat's own AdamW groups use 1e-10)")
parser.add_argument("--adamw-all", action="store_true", help="AdamW baseline: one AdamW LR (--matrix-lr) for matrices, embeddings, value embeds and lm_head (standard GPT recipe); scalars keep nanochat settings")
parser.add_argument("--embed-lr-scale", type=float, default=1.0, help="diagnostic: scale the companion embedding/value-embedding/lm_head LRs")
parser.add_argument("--adago-eta", type=float, default=0.1)
parser.add_argument("--adago-gamma", type=float, default=1.0)
parser.add_argument("--adago-eps-frac", type=float, default=0.1, help="AdaGO epsilon = frac * eta^2 (paper: eps < eta^2)")
parser.add_argument("--adago-mu", type=float, default=0.95)
parser.add_argument("--adago-v0", type=float, default=1e-6)
# curvature probe
parser.add_argument("--probe-every", type=int, default=25, help="k; 0 = probe once at step 0 only")
parser.add_argument("--probe-start", type=int, default=0, help="first probe step (probes at start, start+k, ...); before it eta = fallback LR. 0 = probe at initialization")
parser.add_argument("--probe-once-retry", action="store_true", help="with --probe-every 0: probe every step until the first valid estimate, then freeze eta for the rest of training")
parser.add_argument("--probe-eps-rel", type=float, default=1e-3, help="||eps*D|| = eps_rel * ||W|| over Muon matrices")
parser.add_argument("--probe-scale", type=float, default=1.0, help="eta = scale * eta_hat (1.0 = the proposal's rule)")
parser.add_argument("--probe-init-lr", type=float, default=-1.0, help="fallback eta if the very first probe is invalid; -1 = nanochat default")
parser.add_argument("--probe-sched", action="store_true", help="v2: the probe sets the PEAK multiplier; Muon LR = eta * nanochat warmup/warmdown multiplier (same fixed shape as tuned Muon)")
parser.add_argument("--probe-heldout", action="store_true", help="v2: measure <g,D> and q on the NEXT training minibatch (not the one that built D) to remove the noise bias")
parser.add_argument("--muon-lr-trace", default="", help="replay control: JSON list of per-step Muon LRs (overrides the Muon LR schedule; AdamW companion keeps nanochat's schedule)")
parser.add_argument("--lr-shape", default="linear", choices=["linear", "cosine", "exp"], help="baseline: warmdown shape of the LR multiplier")
parser.add_argument("--lr-warmdown-ratio", type=float, default=-1.0, help="baseline: fraction of training in warmdown; -1 = nanochat 0.65")
parser.add_argument("--lr-final-frac", type=float, default=-1.0, help="baseline: final LR as a fraction of peak; -1 = nanochat 0.05")
parser.add_argument("--probe-smooth", action="store_true", help="v4: eta <- sqrt(eta * scale*eta_hat) (geometric damping of the closed loop)")
parser.add_argument("--probe-cap", action="store_true", help="v4: after warmdown start, probes keep running but only set a cap: LR = min(eta*sched, eta_hat)")
parser.add_argument("--probe-until", type=int, default=-1, help="v2: no probes after this step; -1 = no limit, -2 = stop at warmdown start")
# evaluation
parser.add_argument("--val-every", type=int, default=50)
parser.add_argument("--val-tokens", type=int, default=1_048_576)
parser.add_argument("--final-tokens", type=int, default=10_485_760)
parser.add_argument("--no-test", action="store_true")
# diagnostics (muon only)
parser.add_argument("--diag-base-steps", type=str, default="", help="comma list, e.g. 20,190,280")
parser.add_argument("--diag-deltas", type=str, default="0,1,10,50,100")
parser.add_argument("--diag-eps-rels", type=str, default="1e-6,1e-5,1e-4,1e-3,1e-2,1e-1")
parser.add_argument("--diag-cont-steps", type=int, default=50)
parser.add_argument("--diag-fresh-batches", type=int, default=4)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--tag", type=str, default="")
args = parser.parse_args()
t_script = time.time()

# -----------------------------------------------------------------------------
device_type = autodetect_device_type()
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
assert not ddp, "single GPU only"
sync = torch.cuda.synchronize
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
print0(f"COMPUTE_DTYPE={COMPUTE_DTYPE} USE_FA3={USE_FA3} seed={args.seed} optim={args.optim}")

tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()

def build_model_meta(depth, aspect_ratio):
    base_dim = depth * aspect_ratio
    model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    num_heads = model_dim // args.head_dim
    config = GPTConfig(sequence_len=args.max_seq_len, vocab_size=vocab_size, n_layer=depth, n_head=num_heads,
                       n_kv_head=num_heads, n_embd=model_dim, window_pattern=args.window_pattern)
    with torch.device("meta"):
        return GPT(config)

model = build_model_meta(args.depth, args.aspect_ratio)
model_config = model.config
model.to_empty(device=device)
model.init_weights()  # uses the global RNG -> seed controls the initialization
param_counts = model.num_scaling_params()
num_scaling_params = param_counts['transformer_matrices'] + param_counts['lm_head']

# nanochat scaling rules (base_train.py), with the horizon set to --tokens.
# The d12 reference keeps nanochat's default aspect ratio (64) so both widths share the same batch rule.
target_tokens = args.tokens
d12_ref = build_model_meta(12, 64)
rc = d12_ref.num_scaling_params()
D_REF = 12 * (rc['transformer_matrices'] + rc['lm_head'])
B_REF = 2 ** 19
if args.total_batch_size == -1:
    predicted = B_REF * (target_tokens / D_REF) ** 0.383
    total_batch_size = 2 ** round(math.log2(predicted))
else:
    total_batch_size = args.total_batch_size
batch_lr_scale = (total_batch_size / B_REF) ** 0.5
nanochat_muon_wd = 0.28 * math.sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)
muon_wd = nanochat_muon_wd if args.muon_wd < 0 else args.muon_wd
num_iterations = target_tokens // total_batch_size
tokens_per_micro = args.device_batch_size * args.max_seq_len
assert total_batch_size % tokens_per_micro == 0
grad_accum = total_batch_size // tokens_per_micro
print0(f"params {param_counts} | scaling params {num_scaling_params:,}")
print0(f"batch {total_batch_size:,} tokens ({grad_accum} x {tokens_per_micro}) | iters {num_iterations} | lr scale {batch_lr_scale:.4f} | muon wd {muon_wd:.4f} (nanochat rule {nanochat_muon_wd:.4f})")

# -----------------------------------------------------------------------------
# Optimizer param groups (mirrors GPT.setup_optimizer; AdamW companion identical across optimizers)
model_dim = model_config.n_embd
dmodel_lr_scale = (model_dim / 768) ** -0.5
unembedding_lr, embedding_lr, scalar_lr = 0.008 * batch_lr_scale * args.embed_lr_scale, 0.3 * batch_lr_scale * args.embed_lr_scale, 0.5 * batch_lr_scale
matrix_params = list(model.transformer.h.parameters())
groups = [
    dict(kind='adamw', params=list(model.lm_head.parameters()), lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
    dict(kind='adamw', params=list(model.transformer.wte.parameters()), lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
    dict(kind='adamw', params=list(model.value_embeds.parameters()), lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
    dict(kind='adamw', params=[model.resid_lambdas], lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
    dict(kind='adamw', params=[model.x0_lambdas], lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
    dict(kind='adamw', params=[model.smear_gate.weight, model.smear_lambda, model.backout_lambda], lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
]
nanochat_default_muon_lr = 0.02 * batch_lr_scale
shapes = sorted({p.shape for p in matrix_params})
if args.optim in ("muon", "curv"):
    muon_lr = (args.matrix_lr if args.matrix_lr > 0 else nanochat_default_muon_lr)
    if args.optim == "curv":
        muon_lr = args.probe_init_lr if args.probe_init_lr > 0 else nanochat_default_muon_lr
    for shape in shapes:
        groups.append(dict(kind='muon', params=[p for p in matrix_params if p.shape == shape], lr=muon_lr,
                           momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=muon_wd))
elif args.optim == "adamw":
    lr = args.matrix_lr if args.matrix_lr > 0 else 3e-3
    b1, b2 = (float(x) for x in args.adamw_matrix_betas.split(","))
    if args.adamw_all:
        # standard GPT AdamW recipe: same LR for embeddings / value embeds / lm_head (no weight decay on them)
        for gi in range(3):
            groups[gi].update(lr=lr, betas=(b1, b2), eps=1e-8, weight_decay=0.0)
    for shape in shapes:
        groups.append(dict(kind='adamw', params=[p for p in matrix_params if p.shape == shape], lr=lr,
                           betas=(b1, b2), eps=args.adamw_eps, weight_decay=args.adamw_matrix_wd, matrix=True))
elif args.optim == "adago":
    for shape in shapes:
        groups.append(dict(kind='adago', params=[p for p in matrix_params if p.shape == shape], lr=args.adago_eta,
                           mu=args.adago_mu, gamma=args.adago_gamma, eps=args.adago_eps_frac * args.adago_eta ** 2,
                           v0=args.adago_v0, ns_steps=5))
optimizer = CurvMuonAdamW(groups)
for g in optimizer.param_groups:
    g["initial_lr"] = g["lr"]
    g.setdefault("matrix", g["kind"] != 'adamw')

orig_model = model
model = torch.compile(model, dynamic=False)

# -----------------------------------------------------------------------------
# Schedules (base_train.py)
WARMUP, WARMDOWN_RATIO, FINAL_LR_FRAC = 40, 0.65, 0.05
# Shape-tuned baseline: the LR multiplier's warmdown can be re-shaped (momentum schedule and the probe's warmdown-start
# keep nanochat's defaults). Defaults reproduce nanochat exactly.
MUON_LR_TRACE = json.load(open(args.muon_lr_trace)) if args.muon_lr_trace else None
LR_WD_RATIO = args.lr_warmdown_ratio if args.lr_warmdown_ratio > 0 else WARMDOWN_RATIO
LR_FINAL = args.lr_final_frac if args.lr_final_frac > 0 else FINAL_LR_FRAC
def get_lr_multiplier(it):
    if args.lr_shape != "linear" or args.lr_warmdown_ratio > 0 or args.lr_final_frac > 0:
        if it < WARMUP:
            return (it + 1) / WARMUP
        wd = round(LR_WD_RATIO * num_iterations); start = num_iterations - wd
        if it <= start:
            return 1.0
        x = (it - start) / wd   # 0 -> 1 over the warmdown
        if args.lr_shape == "linear":
            return 1.0 - x * (1.0 - LR_FINAL)
        if args.lr_shape == "cosine":
            return LR_FINAL + (1.0 - LR_FINAL) * 0.5 * (1.0 + math.cos(math.pi * x))
        return LR_FINAL ** x    # exponential (geometric) decay
    warmdown_iters = round(WARMDOWN_RATIO * num_iterations)
    if it < WARMUP:
        return (it + 1) / WARMUP
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    progress = (num_iterations - it) / warmdown_iters
    return progress * 1.0 + (1 - progress) * FINAL_LR_FRAC

def get_muon_momentum(it):
    warmdown_iters = round(WARMDOWN_RATIO * num_iterations)
    warmdown_start = num_iterations - warmdown_iters
    if it < 400:
        frac = it / 400
        return (1 - frac) * 0.85 + frac * 0.97
    elif it >= warmdown_start:
        progress = (it - warmdown_start) / warmdown_iters
        return 0.97 * (1 - progress) + 0.90 * progress
    return 0.97

def get_weight_decay(it):
    return muon_wd * 0.5 * (1 + math.cos(math.pi * it / num_iterations))

curv_eta = None  # current accepted multiplier for --optim curv
curv_cap = None  # v4 --probe-cap: held-out optimum measured during warmdown caps the scheduled LR
WARMDOWN_START = num_iterations - round(WARMDOWN_RATIO * num_iterations)
if args.optim == "curv" and args.probe_start > 0:
    curv_eta = args.probe_init_lr if args.probe_init_lr > 0 else nanochat_default_muon_lr
def apply_schedules(it, muon_lr_override=None):
    lrm = get_lr_multiplier(it)
    for g in optimizer.param_groups:
        if g['kind'] == 'adamw':
            g['lr'] = g['initial_lr'] * lrm           # companion + AdamW-baseline matrices follow the schedule
        elif g['kind'] == 'muon':
            g['momentum'] = get_muon_momentum(it)
            g['weight_decay'] = get_weight_decay(it)
            if muon_lr_override is not None:
                g['lr'] = muon_lr_override
            elif MUON_LR_TRACE is not None:
                g['lr'] = MUON_LR_TRACE[min(it, len(MUON_LR_TRACE) - 1)]   # replay control: fixed per-step Muon LR
            elif args.optim == 'curv':
                g['lr'] = curv_eta * (lrm if args.probe_sched else 1.0)
                if curv_cap is not None:
                    g['lr'] = min(g['lr'], curv_cap)
            else:
                g['lr'] = g['initial_lr'] * lrm
        elif g['kind'] == 'adago':
            g['lr'] = g['initial_lr']                 # AdaGO: fixed eta, adaptivity from the accumulator
    return lrm

# -----------------------------------------------------------------------------
# Data
B, T = args.device_batch_size, args.max_seq_len
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(tokenizer, B, T, split="train", device=device)
def fetch_train():
    x, y, _ = next(train_loader)
    return (x.clone(), y.clone())

@torch.no_grad()
def evaluate(split, tokens):
    loader = tokenizing_distributed_data_loader_bos_bestfit(tokenizer, B, T, split=split, device=device)
    steps = max(1, tokens // (B * T))
    nats = torch.zeros((), dtype=torch.float64, device=device)
    nats_b = torch.zeros((), dtype=torch.float64, device=device)
    nbytes = torch.zeros((), dtype=torch.int64, device=device)
    n = 0
    for _ in range(steps):
        x, y = next(loader)
        l = model(x, y, loss_reduction='none').view(-1).double()
        tb = token_bytes[y.view(-1)]
        nats += l.sum(); n += l.numel()
        nats_b += (l * (tb > 0)).sum(); nbytes += tb.sum()
    return dict(loss=float(nats / n), bpb=float(nats_b / (math.log(2) * nbytes.double())), tokens=steps * B * T)

def forward_backward(batches, fetch_next=False):
    """Accumulate gradients over the micro-batches of one optimizer step (mean loss)."""
    tot = torch.zeros((), device=device)
    nxt = []
    for x, y in batches:
        loss = model(x, y)
        (loss / len(batches)).backward()
        tot += loss.detach()
        if fetch_next:
            nxt.append(fetch_train())
    return tot / len(batches), nxt

@torch.no_grad()
def batch_loss(batches):
    tot = torch.zeros((), dtype=torch.float64, device=device)
    for x, y in batches:
        tot += model(x, y).double()
    return float(tot / len(batches))

# -----------------------------------------------------------------------------
# Diagnostics setup (proposal "Diagnostic experiments")
diag_base = [int(s) for s in args.diag_base_steps.split(",") if s.strip()]
diag_deltas = [int(s) for s in args.diag_deltas.split(",") if s.strip()]
diag_eps = [float(s) for s in args.diag_eps_rels.split(",") if s.strip()]
diag_steps = {}
for b in diag_base:
    for d in diag_deltas:
        if b + d < num_iterations:
            diag_steps.setdefault(b + d, []).append((b, d))
DIAG_GRID = [10 ** (e / 4) for e in range(-16, 1)]  # 1e-4 ... 1, 4 points per decade
diag_records, cont_records = [], []
saved_recs = {}  # base step -> eta_hat (default eps) for staleness checks
if diag_steps:
    assert args.optim == "muon", "diagnostics are defined on a Muon training run"
    fl = tokenizing_distributed_data_loader_bos_bestfit(tokenizer, B, T, split="fresh", device=device)
    fresh_batches = []
    for _ in range(args.diag_fresh_batches):
        fresh_batches.append([tuple(t.clone() for t in next(fl)) for _ in range(grad_accum)])
    del fl
    # continuation stream: fresh shard, starting at row group 40 (disjoint from the fresh eval batches)
    cl = tokenizing_distributed_data_loader_with_state_bos_bestfit(tokenizer, B, T, split="fresh", device=device,
                                                                   resume_state_dict={"pq_idx": 0, "rg_idx": 39, "epoch": 1})
    cont_batches = []
    for _ in range(args.diag_cont_steps):
        cont_batches.append([tuple(t.clone() for t in next(cl)[:2]) for _ in range(grad_accum)])
    del cl
    fresh_eval_all = [mb for fb in fresh_batches for mb in fb]

def snapshot_state():
    params = {n: p.detach().clone() for n, p in orig_model.named_parameters()}
    opt = {}
    for p, st in optimizer.state.items():
        opt[p] = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in st.items()}
    return params, opt

def restore_state(snap):
    params, opt = snap
    with torch.no_grad():
        for n, p in orig_model.named_parameters():
            p.copy_(params[n])
    optimizer.state.clear()
    for p, st in opt.items():
        optimizer.state[p] = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in st.items()}
    optimizer._pending = None

def run_diagnostics(step, cur_batches, sched_lr):
    t0 = time.time()
    rng = torch.cuda.get_rng_state()
    def regrad():
        torch.cuda.set_rng_state(rng)
        forward_backward(cur_batches)
    g0 = stacked_grads(optimizer)
    W0 = stacked_params(optimizer)
    Ds = optimizer.compute_muon_directions(mutate=False)
    gD = float(_dot(g0, Ds))
    g_norm = math.sqrt(sum(float(g.double().square().sum()) for g in g0))
    # (D3) probe-distance sweep
    probes = [curvature_probe(model, optimizer, Ds, g0, regrad, e, W0) for e in diag_eps]
    # noise floor of the gradient difference: recompute the gradient at the same point
    saved = {p: p.grad for p in model.parameters()}
    for p in model.parameters():
        p.grad = None
    regrad()
    g_rep = stacked_grads(optimizer)
    noise_dot = float(_dot(Ds, [a - b for a, b in zip(g_rep, g0)]))
    noise_norm = math.sqrt(sum(float((a - b).double().square().sum()) for a, b in zip(g_rep, g0)))
    for p in model.parameters():
        p.grad = saved[p]
    del g_rep
    ref = next((p for p in probes if abs(p['eps_rel'] - args.probe_eps_rel) < 1e-15), probes[len(probes) // 2])
    eta_hat = ref['eta_hat']
    # (D1) loss along -D for a log grid + predicted step + scheduled LR (+ stale recommendations for D2)
    cands = {f"grid_{e:.3e}": e for e in DIAG_GRID}
    cands["sched"] = sched_lr
    for p in probes:
        if p['eta_hat'] is not None:
            cands[f"hat_eps{p['eps_rel']:.0e}"] = p['eta_hat']
    for (b, d) in diag_steps[step]:
        if d > 0 and b in saved_recs and saved_recs[b] is not None:
            cands[f"stale_b{b}_d{d}"] = saved_recs[b]
    # line curvature on the probe batch by central differences (reference for q_hat)
    h = (eta_hat if eta_hat else sched_lr) * 0.25
    cands["fd_plus"] = h; cands["fd_minus"] = -h
    with torch.no_grad():
        L0_probe = batch_loss(cur_batches)
        L0_fresh = [batch_loss(fb) for fb in fresh_batches]
        line = {}
        for name, eta in cands.items():
            set_params(optimizer, W0)
            add_to_params(optimizer, Ds, -eta)
            line[name] = dict(eta=eta, probe=batch_loss(cur_batches), fresh=[batch_loss(fb) for fb in fresh_batches])
        set_params(optimizer, W0)
    fd_q = (line["fd_plus"]["probe"] - 2 * L0_probe + line["fd_minus"]["probe"]) / h ** 2
    fd_slope = (line["fd_plus"]["probe"] - line["fd_minus"]["probe"]) / (2 * h)  # = -<g,D> to first order
    rec = dict(step=step, bases=diag_steps[step], gD=gD, g_norm=g_norm, probes=probes, noise_dot=noise_dot,
               noise_norm=noise_norm, eta_hat=eta_hat, sched_lr=sched_lr, L0_probe=L0_probe, L0_fresh=L0_fresh,
               line=line, fd_q=fd_q, fd_slope=fd_slope, fd_h=h)
    if step in diag_base:
        saved_recs[step] = eta_hat
        # (Q1) short training continuations from this checkpoint with constant Muon LR candidates
        snap = snapshot_state()
        base_eta = eta_hat if eta_hat else sched_lr
        cand_c = {f"hatx{m:g}": base_eta * m for m in (0.25, 0.5, 1.0, 2.0, 4.0)}
        cand_c["sched"] = None
        cres = {}
        for name, eta in cand_c.items():
            restore_state(snap)
            traj = {}
            for i in range(args.diag_cont_steps):
                it = min(step + i, num_iterations - 1)
                apply_schedules(it, muon_lr_override=eta)  # eta None -> tuned-Muon schedule
                for p in model.parameters():
                    p.grad = None
                forward_backward(cont_batches[i])
                optimizer.step()
                if (i + 1) in (1, 10, args.diag_cont_steps):
                    traj[i + 1] = batch_loss(fresh_eval_all)
            cres[name] = dict(eta=eta, traj=traj)
        restore_state(snap)
        for p in model.parameters():
            p.grad = saved[p]
        apply_schedules(step)
        cont_records.append(dict(step=step, eta_hat=eta_hat, sched_lr=sched_lr, L0=sum(L0_fresh) / len(L0_fresh), cands=cres))
    rec["seconds"] = time.time() - t0
    diag_records.append(rec)
    print0(f"[diag] step {step} eta_hat {eta_hat} sched {sched_lr:.4g} gD {gD:.4g} fd_q {fd_q:.4g} q_hat {ref['q_hat']:.4g} ({rec['seconds']:.1f}s)")

# -----------------------------------------------------------------------------
# Training loop
log_steps, evals, probe_log = [], [], []
train_time = 0.0       # optimizer steps incl. probes, excluding eval/diag and the first 10 steps (compile)
probe_time = 0.0
probe_time_all = 0.0
cur_batches = [fetch_train() for _ in range(grad_accum)]
step = 0
nan_abort = False
gc.collect()
while True:
    last = step == num_iterations
    if args.val_every > 0 and (step % args.val_every == 0) and not last:
        model.eval(); ev = evaluate("val", args.val_tokens); model.train()
        ev["step"] = step; evals.append(ev)
        print0(f"step {step} val loss {ev['loss']:.4f} bpb {ev['bpb']:.4f}")
    if last or nan_abort:
        break
    sync(); t0 = time.time()
    once_pending = args.probe_once_retry and not any(p.get('accepted') for p in probe_log)
    ps = args.probe_start
    probe_now = args.optim == "curv" and step >= ps and ((args.probe_every > 0 and (step - ps) % args.probe_every == 0) or (args.probe_every == 0 and (step == ps or once_pending)))
    if probe_now and args.probe_until != -1:
        until = (num_iterations - round(WARMDOWN_RATIO * num_iterations)) if args.probe_until == -2 else args.probe_until
        probe_now = step <= until
    rng = torch.cuda.get_rng_state() if probe_now else None
    loss, next_batches = forward_backward(cur_batches, fetch_next=True)
    lrm = apply_schedules(step)
    sched_lr = optimizer.muon_groups()[0]['lr'] if optimizer.muon_groups() else None
    pinfo = None
    if probe_now:
        sync(); tp = time.time()
        g0 = stacked_grads(optimizer)
        W0 = stacked_params(optimizer)
        Ds = optimizer.compute_muon_directions(mutate=True)  # this step's direction (state advanced once)
        def regrad():
            torch.cuda.set_rng_state(rng)
            forward_backward(cur_batches)
        if args.probe_heldout:
            rng_h = torch.cuda.get_rng_state()
            def regrad_heldout():
                torch.cuda.set_rng_state(rng_h)
                forward_backward(next_batches)
            pinfo = curvature_probe_heldout(model, optimizer, Ds, regrad_heldout, args.probe_eps_rel, W0)
        else:
            pinfo = curvature_probe(model, optimizer, Ds, g0, regrad, args.probe_eps_rel, W0)
        prev = curv_eta
        if pinfo['valid'] and args.probe_cap and step > WARMDOWN_START:
            curv_cap = args.probe_scale * pinfo['eta_hat']   # v4 cap: peak stays frozen, cap limits the scheduled LR
            pinfo['accepted'] = True; pinfo['cap'] = curv_cap
        elif pinfo['valid']:
            new = args.probe_scale * pinfo['eta_hat']
            curv_eta = math.sqrt(curv_eta * new) if (args.probe_smooth and curv_eta is not None) else new
            pinfo['accepted'] = True
        else:
            pinfo['accepted'] = False
            if curv_eta is None:
                curv_eta = args.probe_init_lr if args.probe_init_lr > 0 else nanochat_default_muon_lr
        pinfo.update(step=step, eta_used=curv_eta, eta_prev=prev)
        apply_schedules(step)
        del g0, W0, Ds
        sync(); dtp = time.time() - tp
        pinfo['seconds'] = dtp
        probe_log.append(pinfo)
        probe_time_all += dtp
        if step > 10:
            probe_time += dtp
        print0(f"[probe] step {step} gD {pinfo['gD']:.4g} q {pinfo['q_hat']:.4g} eta_hat {pinfo['eta_hat']} -> eta {curv_eta:.4g} ({dtp:.2f}s)")
    if step in diag_steps:
        sync()
        run_diagnostics(step, cur_batches, sched_lr)
        sync(); t0 += 0  # diagnostics time excluded below
    td0 = time.time()
    optimizer.step()
    alpha = optimizer.last_adago_alpha
    model.zero_grad(set_to_none=True)
    loss_f = loss.item()
    sync(); t1 = time.time()
    diag_sec = diag_records[-1]['seconds'] if (step in diag_steps) else 0.0
    dt = (t1 - t0) - diag_sec
    if step > 10:
        train_time += dt
    rec = dict(step=step, loss=loss_f, dt=dt, lrm=lrm)
    if optimizer.muon_groups():
        rec['muon_lr'] = optimizer.muon_groups()[0]['lr']
    if alpha is not None:
        rec['adago_alpha_mean'] = float(alpha.mean()); rec['adago_alpha_min'] = float(alpha.min()); rec['adago_alpha_max'] = float(alpha.max())
    log_steps.append(rec)
    if not math.isfinite(loss_f):
        print0(f"non-finite loss at step {step}; aborting run")
        nan_abort = True
    if step % 25 == 0 or step < 3:
        extra = f" muon_lr {rec.get('muon_lr', float('nan')):.4g}" if 'muon_lr' in rec else ""
        extra += f" alpha {rec['adago_alpha_mean']:.4g}" if 'adago_alpha_mean' in rec else ""
        print0(f"step {step:04d}/{num_iterations} loss {loss_f:.4f} dt {dt*1000:.0f}ms tok/s {int(total_batch_size/dt):,}{extra}")
    cur_batches = next_batches
    step += 1
    if step == 1:
        gc.collect(); gc.freeze(); gc.disable()

# -----------------------------------------------------------------------------
# Final evaluation
model.eval()
final_val = evaluate("val", args.final_tokens) if not nan_abort else dict(loss=float('nan'), bpb=float('nan'))
final_test = evaluate("test", args.final_tokens) if (not args.no_test and not nan_abort) else None
print0(f"FINAL val loss {final_val['loss']:.4f} bpb {final_val['bpb']:.4f}" + (f" | test loss {final_test['loss']:.4f} bpb {final_test['bpb']:.4f}" if final_test else ""))
steps_timed = max(1, num_iterations - 11)
result = dict(
    args=vars(args), tag=args.tag, model=dict(config=model_config.__dict__, param_counts=param_counts, scaling_params=num_scaling_params),
    total_batch_size=total_batch_size, grad_accum=grad_accum, num_iterations=num_iterations, tokens_trained=num_iterations * total_batch_size,
    muon_wd=muon_wd, nanochat_muon_wd=nanochat_muon_wd, batch_lr_scale=batch_lr_scale, nanochat_default_muon_lr=nanochat_default_muon_lr,
    final_val=final_val, final_test=final_test, evals=evals, steps=log_steps, probes=probe_log,
    diag=diag_records, continuation=cont_records, nan_abort=nan_abort,
    timing=dict(train_time_s=train_time, mean_step_s=train_time / steps_timed, probe_time_s=probe_time, probe_time_all_s=probe_time_all,
                wall_s=time.time() - t_script, peak_mem_gib=torch.cuda.max_memory_allocated() / 2**30),
)
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
tmp = args.out + ".tmp"
with open(tmp, "w") as f:
    json.dump(result, f, default=lambda o: o if not isinstance(o, torch.Tensor) else o.tolist())
os.replace(tmp, args.out)
print0(f"wrote {args.out} | train {train_time/60:.2f} min, probes {probe_time_all:.1f}s, wall {(time.time()-t_script)/60:.2f} min")
compute_cleanup()
