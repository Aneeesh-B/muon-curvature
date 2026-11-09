"""Frozen-direction LR scans and saved-recommendation staleness diagnostics."""
import json
import math

import torch

from .optimizer import displaced, probe, replay_rng, select_lr


def append_json(path, row):
    # Strict JSON: preserve missing/nonfinite measurements as null.
    def clean(x):
        if isinstance(x, float) and not math.isfinite(x):
            return None
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, list):
            return [clean(v) for v in x]
        return x
    with open(path, "a") as stream:
        stream.write(json.dumps(clean(row), allow_nan=False) + "\n")


def scan(model, direction, batch, forward_rng, batches, config, step, anchor, saved_lr, path):
    """No optimizer steps: companion AdamW and weight decay are held fixed.

    Staleness follows a fixed-Muon trajectory, rebuilding D at each age. It asks
    whether the OLD scalar works with the CURRENT direction, not old D itself.
    """
    generator = torch.Generator().manual_seed(73000 + step)
    fresh = [batches.get("train", generator) for _ in range(config["diagnostic_batches"])]
    validation = [batches.get("val", generator) for _ in range(config["diagnostic_batches"])]
    closure = lambda: model(*batch)[1]
    measurement = probe(direction, closure, forward_rng, config["relative_distance"])
    current_lr, _ = select_lr(measurement, config["lr"], config["lr_min"], config["lr_max"], config["safety"])

    @torch.no_grad()
    def losses():
        # Replayed stochastic masks are identical for every candidate, including
        # the original probe batch. Does not consume the training RNG stream.
        with replay_rng(forward_rng):
            probe_loss = closure().item()
            fresh_loss = sum(model(*b)[1].item() for b in fresh) / len(fresh)
            val_loss = sum(model(*b)[1].item() for b in validation) / len(validation)
        return dict(probe_loss=probe_loss, fresh_train_loss=fresh_loss, val_loss=val_loss)

    base = losses()
    rates = sorted(set([0.0, saved_lr, current_lr, *config["scan_lrs"]]))
    candidates = []
    for eta in rates:
        with displaced(direction.params, direction.updates, -eta):
            values = losses()
        predicted = None
        if measurement["q"] is not None:
            predicted = -eta * measurement["s"] + 0.5 * eta**2 * measurement["q"]
        candidates.append(dict(lr=eta, predicted_delta=predicted,
                               **values, **{k+"_delta": v-base[k] for k, v in values.items()}))
    row = dict(step=step, anchor_step=anchor, age=step-anchor, saved_lr=saved_lr,
               current_lr=current_lr, measurement=measurement, base=base, candidates=candidates)
    if step == anchor:
        row["distance_sweep"] = {
            str(rho): probe(direction, closure, forward_rng, rho)
            for rho in config["probe_distances"]
        }
    append_json(path, row)
    return row
