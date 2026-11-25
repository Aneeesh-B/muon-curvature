"""
CS182 project: "Can Occasional Curvature Probes Reduce Learning-Rate Tuning for Muon?"

Single-GPU optimizer used by scripts/curv_train.py. It extends nanochat's MuonAdamW with:

1) A split Muon step. nanochat's fused kernel (momentum -> MuonEq -> Polar Express -> Muon+ ->
   NorMuon variance reduction -> cautious-WD update) is split into
      muon_direction(): everything up to the final matrix direction (same math, same order)
      muon_apply():     cautious weight decay + parameter update W <- W - lr*g - lr*wd*W*mask
   so that the curvature probe can use exactly the direction the step will apply.
   D_t (the proposal's "Muon matrix direction with the usual layer scaling") is
      D_t = s * g,   s = max(1, rows/cols)**0.5  (nanochat's per-shape LR scale)
   and the update before weight decay is W_{t+1} = W_t - eta * D_t with one shared eta.

2) AdaGO (Zhang, Liu, Schaeffer 2025, arXiv:2509.02981, Algorithm 2), per matrix:
      M_t   = mu M_{t-1} + (1 - mu) G_t
      v_t^2 = v_{t-1}^2 + min(||G_t||_F^2, gamma^2)
      O_t   = Orth(M_t)                       (Polar Express, as in nanochat)
      W_t   = W_{t-1} - max(eps, eta * min(||G_t||_F, gamma) / v_t) * O_t
   No weight decay (as in the paper).

3) Curvature probe (proposal Section 4, "Step-size rule"):
      q_hat   = <D_t, g(W_t + eps D_t) - g_t> / eps
      eta_hat = <g_t, D_t> / q_hat
   with eps chosen relative to the parameter norm: ||eps D_t|| = eps_rel * ||W_t|| (Muon matrices).
   The second gradient uses the same minibatch and RNG state; weights are restored exactly and the
   optimizer state is not advanced by the probe.
"""

import math
import torch
from torch import Tensor

from nanochat.common import COMPUTE_DTYPE
from nanochat.optim import MuonAdamW, polar_express_coeffs


# -----------------------------------------------------------------------------
# Split Muon step (identical math to nanochat.optim.muon_step_fused)

@torch.compile(dynamic=False, fullgraph=True)
def muon_direction(
    stacked_grads: Tensor,          # (K, m, n) raw minibatch gradients (not modified)
    momentum_buffer: Tensor,        # (K, m, n) updated in place
    second_momentum_buffer: Tensor, # (K, m, 1) or (K, 1, n) updated in place
    momentum_t: Tensor,             # () CPU
    beta2_t: Tensor,                # () CPU
    ns_steps: int,
    red_dim: int,
) -> Tensor:
    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp(momentum_buffer, momentum)

    X = g.bfloat16() if COMPUTE_DTYPE == torch.bfloat16 else g

    # MuonEq row equilibration
    target = X.float().norm(dim=(-2, -1), keepdim=True) / (X.size(-2) ** 0.5)
    row_norm = X.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
    X = X * (target / row_norm).to(X.dtype)

    # Polar Express orthogonalization
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X.to(stacked_grads.dtype)

    # Muon+ renormalization
    target_norm = min(g.size(-2), g.size(-1)) ** 0.5
    current_norm = g.float().norm(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    g = g * (target_norm / current_norm).to(g.dtype)

    # NorMuon variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    return g


@torch.compile(dynamic=False, fullgraph=True)
def muon_apply(stacked_params: Tensor, g: Tensor, lr_t: Tensor, wd_t: Tensor) -> None:
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


@torch.compile(dynamic=False, fullgraph=True)
def polar_express_orth(M: Tensor, ns_steps: int) -> Tensor:
    """Orth(M) via nanochat's Polar Express iteration (no MuonEq / Muon+ / NorMuon extras)."""
    X = M.bfloat16() if COMPUTE_DTYPE == torch.bfloat16 else M
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    if M.size(-2) > M.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    return X.to(M.dtype)


def shape_scale(shape) -> float:
    return max(1.0, shape[-2] / shape[-1]) ** 0.5


# -----------------------------------------------------------------------------

class CurvMuonAdamW(MuonAdamW):
    """
    Single-GPU MuonAdamW with a split Muon step, AdaGO groups, and hooks for curvature probes.

    Group kinds:
      'adamw' : nanochat fused AdamW (embeddings, lm_head, scalars; or matrices for the AdamW baseline)
      'muon'  : nanochat Muon (split direction/apply). group['lr'] is the shared multiplier eta.
      'adago' : AdaGO (Algorithm 2), group['lr'] is AdaGO's eta.
    """

    def __init__(self, param_groups):
        super().__init__(param_groups)
        self._pending = None  # list of direction tensors g (without shape scale) for muon groups, consumed by step()
        self.last_adago_alpha = None

    # ---------------- Muon helpers ----------------
    def muon_groups(self):
        return [g for g in self.param_groups if g['kind'] == 'muon']

    def _muon_state(self, group):
        p = group['params'][0]
        shape = p.shape
        state = self.state[p]
        K = len(group['params'])
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(K, *shape, dtype=p.dtype, device=p.device)
        if "second_momentum_buffer" not in state:
            state_shape = (K, shape[-2], 1) if shape[-2] >= shape[-1] else (K, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=p.dtype, device=p.device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        return state, red_dim

    @torch.no_grad()
    def compute_muon_directions(self, mutate: bool = True):
        """
        Compute g for every Muon group from the current .grad. If mutate=True the momentum/second-moment
        buffers advance (this is the real step's direction; step() will consume it without recomputing).
        If mutate=False, buffers are cloned and left untouched (used by diagnostics).
        Returns list of D = s*g stacked per Muon group.
        """
        gs, Ds = [], []
        for group in self.muon_groups():
            state, red_dim = self._muon_state(group)
            grads = torch.stack([p.grad for p in group['params']])
            mb, sb = state["momentum_buffer"], state["second_momentum_buffer"]
            if not mutate:
                mb, sb = mb.clone(), sb.clone()
            self._muon_momentum_t.fill_(group["momentum"])
            self._muon_beta2_t.fill_(group["beta2"])
            g = muon_direction(grads, mb, sb, self._muon_momentum_t, self._muon_beta2_t, group["ns_steps"], red_dim)
            gs.append(g)
            Ds.append(g * shape_scale(group['params'][0].shape))
        if mutate:
            self._pending = gs
        return Ds

    # ---------------- step ----------------
    @torch.no_grad()
    def step(self):
        gather_list = []
        muon_idx = 0
        pending = self._pending
        if pending is None and any(g['kind'] == 'muon' for g in self.param_groups):
            self.compute_muon_directions(mutate=True)
            pending = self._pending
        alphas = []
        for group in self.param_groups:
            kind = group['kind']
            if kind == 'adamw':
                info = self._reduce_adamw(group, 1)
                self._compute_adamw(group, info, gather_list, 0, 1)
            elif kind == 'muon':
                params = group['params']
                stacked = torch.stack(params)
                g = pending[muon_idx]
                muon_idx += 1
                self._muon_lr_t.fill_(group["lr"] * shape_scale(params[0].shape))
                self._muon_wd_t.fill_(group["weight_decay"])
                muon_apply(stacked, g, self._muon_lr_t, self._muon_wd_t)
                torch._foreach_copy_(params, list(stacked.unbind(0)))
            elif kind == 'adago':
                alphas.append(self._adago_step(group))
            else:
                raise ValueError(kind)
        self._pending = None
        if alphas:
            self.last_adago_alpha = torch.cat(alphas)

    @torch.no_grad()
    def _adago_step(self, group):
        params = group['params']
        p0 = params[0]
        state = self.state[p0]
        K = len(params)
        if "M" not in state:
            state["M"] = torch.zeros(K, *p0.shape, dtype=torch.float32, device=p0.device)
            state["v2"] = torch.full((K,), group["v0"] ** 2, dtype=torch.float32, device=p0.device)
        G = torch.stack([p.grad for p in params]).float()
        gnorm = G.flatten(1).norm(dim=1)                                  # ||G_t||_F per matrix
        mu = group["mu"]
        state["M"].mul_(mu).add_(G, alpha=1 - mu)                         # M_t = mu M + (1-mu) G
        gamma = group["gamma"]
        state["v2"].add_(torch.minimum(gnorm.square(), torch.full_like(gnorm, gamma ** 2)))
        O = polar_express_orth(state["M"], group["ns_steps"])
        alpha = torch.clamp_min(group["lr"] * torch.clamp_max(gnorm, gamma) / state["v2"].sqrt(), group["eps"])
        stacked = torch.stack(params)
        stacked.sub_(alpha.view(K, 1, 1).to(stacked.dtype) * O.to(stacked.dtype))
        torch._foreach_copy_(params, list(stacked.unbind(0)))
        return alpha


# -----------------------------------------------------------------------------
# Probe utilities (operate on the Muon matrices only)

def _dot(a_list, b_list):
    return sum((a.double() * b.double()).sum() for a, b in zip(a_list, b_list))


@torch.no_grad()
def stacked_params(optimizer):
    return [torch.stack(g['params']) for g in optimizer.muon_groups()]


@torch.no_grad()
def stacked_grads(optimizer):
    return [torch.stack([p.grad for p in g['params']]) for g in optimizer.muon_groups()]


@torch.no_grad()
def add_to_params(optimizer, Ds, alpha):
    """W <- W + alpha * D for Muon matrices."""
    for group, D in zip(optimizer.muon_groups(), Ds):
        torch._foreach_add_(group['params'], list(D.unbind(0)), alpha=alpha)


@torch.no_grad()
def set_params(optimizer, Ws):
    for group, W in zip(optimizer.muon_groups(), Ws):
        torch._foreach_copy_(group['params'], list(W.unbind(0)))


def curvature_probe(model, optimizer, Ds, g0, compute_grads, eps_rel, W0=None):
    """
    Gradient-difference directional curvature along D (proposal eq. for q_hat).
      compute_grads(): recomputes all .grad on the SAME minibatch with the SAME RNG state.
    Restores weights exactly (copy, not subtraction) and restores the original .grad tensors.
    Returns dict with gD, q_hat, eps, eta_hat (None if invalid), norms.
    """
    with torch.no_grad():
        if W0 is None:
            W0 = stacked_params(optimizer)
        W_norm = math.sqrt(sum(float(W.double().square().sum()) for W in W0))
        D_norm = math.sqrt(sum(float(D.double().square().sum()) for D in Ds))
        gD = float(_dot(g0, Ds))
        eps = eps_rel * W_norm / max(D_norm, 1e-30)
        saved = {p: p.grad for p in model.parameters()}
        for p in model.parameters():
            p.grad = None
        add_to_params(optimizer, Ds, eps)
    compute_grads()
    with torch.no_grad():
        g1 = stacked_grads(optimizer)
        dg = [a - b for a, b in zip(g1, g0)]
        q_hat = float(_dot(Ds, dg)) / eps
        dg_norm = math.sqrt(sum(float(x.double().square().sum()) for x in dg))
        set_params(optimizer, W0)
        for p in model.parameters():
            p.grad = saved[p]
    valid = math.isfinite(gD) and math.isfinite(q_hat) and gD > 0 and q_hat > 0
    eta_hat = gD / q_hat if valid else None
    return dict(gD=gD, q_hat=q_hat, eps=eps, eps_rel=eps_rel, eta_hat=eta_hat, valid=valid,
                W_norm=W_norm, D_norm=D_norm, dg_norm=dg_norm)


def curvature_probe_heldout(model, optimizer, Ds, compute_grads_heldout, eps_rel, W0=None):
    """
    v2 noise-debiased probe: the same quadratic step rule, but <g, D> and q are measured on a HELD-OUT minibatch B'
    (one that did not build D), so batch noise that D is aligned with does not inflate eta_hat.
      compute_grads_heldout(): recomputes all .grad on B' (same RNG state each call).
    Costs two extra forward-backward passes. Restores weights (copy) and the original .grad tensors.
    """
    with torch.no_grad():
        if W0 is None:
            W0 = stacked_params(optimizer)
        W_norm = math.sqrt(sum(float(W.double().square().sum()) for W in W0))
        D_norm = math.sqrt(sum(float(D.double().square().sum()) for D in Ds))
        eps = eps_rel * W_norm / max(D_norm, 1e-30)
        saved = {p: p.grad for p in model.parameters()}
        for p in model.parameters():
            p.grad = None
    compute_grads_heldout()
    with torch.no_grad():
        h0 = [g.clone() for g in stacked_grads(optimizer)]
        gD = float(_dot(h0, Ds))
        for p in model.parameters():
            p.grad = None
        add_to_params(optimizer, Ds, eps)
    compute_grads_heldout()
    with torch.no_grad():
        h1 = stacked_grads(optimizer)
        dg = [a - b for a, b in zip(h1, h0)]
        q_hat = float(_dot(Ds, dg)) / eps
        dg_norm = math.sqrt(sum(float(x.double().square().sum()) for x in dg))
        set_params(optimizer, W0)
        for p in model.parameters():
            p.grad = saved[p]
    valid = math.isfinite(gD) and math.isfinite(q_hat) and gD > 0 and q_hat > 0
    eta_hat = gD / q_hat if valid else None
    return dict(gD=gD, q_hat=q_hat, eps=eps, eps_rel=eps_rel, eta_hat=eta_hat, valid=valid,
                W_norm=W_norm, D_norm=D_norm, dg_norm=dg_norm, heldout=True)
