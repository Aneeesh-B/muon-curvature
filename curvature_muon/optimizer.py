"""Muon with a transactional, externally inspectable direction/probe/commit API.

Geometry follows KellerJordan/Muon (see third_party/README.md), using FP32 NS.
prepare() never advances momentum. Only step() commits momentum and weights.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import math

import torch


def rng_state():
    return (torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


@contextmanager
def replay_rng(state):
    """Replay a forward's random state without consuming the training RNG stream."""
    current = rng_state()
    try:
        torch.set_rng_state(state[0])
        if state[1]:
            torch.cuda.set_rng_state_all(state[1])
        yield
    finally:
        torch.set_rng_state(current[0])
        if current[1]:
            torch.cuda.set_rng_state_all(current[1])


@contextmanager
def displaced(params, directions, amount):
    """Exact restoration, including on exceptions (subtracting is not exact)."""
    originals = [p.detach().clone() for p in params]
    try:
        with torch.no_grad():
            for p, d in zip(params, directions):
                p.add_(d, alpha=amount)
        yield
    finally:
        with torch.no_grad():
            for p, old in zip(params, originals):
                p.copy_(old)


def dot(xs, ys):
    # FP64 scalar reductions keep cancellation out of the final sum; gradients
    # and the finite difference themselves remain in the model's precision.
    return sum((x.double() * y.double()).sum().item() for x, y in zip(xs, ys))


def orthogonalize(g, steps=5):
    if g.ndim != 2:
        raise ValueError("Muon requires 2D hidden matrices")
    x = g.float()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        a = x @ x.T
        b = -4.7750 * a + 2.0315 * (a @ a)
        x = 3.4445 * x + b @ x
    return (x.T if transposed else x).to(g.dtype)


@dataclass
class Direction:
    params: list
    gradients: list
    updates: list
    momenta: list


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.0, ns_steps=5):
        params = list(params)
        if not params or any(p.ndim != 2 for p in params):
            raise ValueError("Muon needs a nonempty collection of 2D matrices")
        if lr <= 0 or not 0 <= momentum < 1 or weight_decay < 0 or ns_steps < 1:
            raise ValueError("Invalid Muon configuration")
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps))

    @torch.no_grad()
    def prepare(self):
        ps, gs, ds, ms = [], [], [], []
        for group in self.param_groups:
            beta = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    raise RuntimeError("All Muon parameters must have gradients before prepare()")
                if not torch.isfinite(p.grad).all():
                    raise FloatingPointError("Nonfinite training gradient")
                # .get avoids even creating an empty optimizer state during probes.
                old = self.state.get(p, {}).get("momentum_buffer", torch.zeros_like(p))
                m = old.lerp(p.grad, 1 - beta)
                update = p.grad.lerp(m, beta)  # Nesterov convention of the reference
                d = orthogonalize(update, group["ns_steps"])
                d *= math.sqrt(max(1, p.shape[0] / p.shape[1]))
                ps.append(p)
                gs.append(p.grad.detach().clone())
                ds.append(d)
                ms.append(m)
        return Direction(ps, gs, ds, ms)

    @torch.no_grad()
    def step(self, direction=None):
        direction = direction if direction is not None else self.prepare()
        expected = [p for group in self.param_groups for p in group["params"]]
        if [id(p) for p in expected] != [id(p) for p in direction.params]:
            raise ValueError("Direction belongs to a different parameter collection")
        i = 0
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["momentum_buffer"] = direction.momenta[i].clone()
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(direction.updates[i], alpha=-group["lr"])
                i += 1


def probe(direction, closure, forward_rng, relative_distance=1e-3):
    """Estimate q with g(W+epsilon D) on the original batch and random state.

    closure returns a scalar loss, without backward/zero_grad. Model buffers must
    not mutate in forward (NanoGPT has no running-statistic normalization).
    Gradients in .grad and companion optimizer state are untouched.
    """
    if relative_distance <= 0:
        raise ValueError("relative_distance must be positive")
    ps, ds, gs = direction.params, direction.updates, direction.gradients
    s = dot(gs, ds)
    pn, dn = math.sqrt(dot(ps, ps)), math.sqrt(dot(ds, ds))
    result = dict(s=s, q=None, eta_raw=None, epsilon=None, gradient_difference_ratio=None)
    if not math.isfinite(dn) or dn == 0 or pn == 0:
        return dict(result, status="zero_or_invalid_norm")
    epsilon = relative_distance * pn / dn
    result["epsilon"] = epsilon
    with displaced(ps, ds, epsilon), replay_rng(forward_rng), torch.enable_grad():
        loss = closure()
        shifted = torch.autograd.grad(loss, ps)
    delta = [a - b for a, b in zip(shifted, gs)]
    q = dot(ds, delta) / epsilon
    ratio = math.sqrt(dot(delta, delta)) / max(math.sqrt(dot(gs, gs)), 1e-30)
    result.update(q=q, gradient_difference_ratio=ratio)
    if not all(math.isfinite(v) for v in (s, q, ratio)):
        status = "nonfinite"
    elif s <= 0:
        status = "non_descent"
    elif q <= 0:
        status = "nonpositive_curvature"
    elif ratio < 1e-6:
        status = "unresolved_difference"
    else:
        result["eta_raw"] = s / q
        status = "ok" if math.isfinite(result["eta_raw"]) else "nonfinite"
    return dict(result, status=status)


def select_lr(measurement, previous, minimum=1e-5, maximum=0.2, safety=1.0):
    if not 0 < minimum <= maximum or not 0 < safety <= 1:
        raise ValueError("Invalid learning-rate safeguards")
    if measurement["status"] != "ok":
        return previous, "fallback"
    raw = safety * measurement["eta_raw"]
    accepted = max(minimum, min(maximum, raw))
    return accepted, "clamped" if accepted != raw else "accepted"


def split_parameters(model):
    # named_parameters deduplicates NanoGPT's tied token embedding/output head.
    hidden, auxiliary = [], []
    for name, p in model.named_parameters():
        (hidden if name.startswith("transformer.h.") and p.ndim == 2 else auxiliary).append(p)
    assert {id(p) for p in hidden}.isdisjoint({id(p) for p in auxiliary})
    assert len(hidden) + len(auxiliary) == len(list(model.parameters()))
    return hidden, auxiliary
