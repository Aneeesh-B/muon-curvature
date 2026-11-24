"""Time independent finite differences against autodiff Hessian-vector products."""
import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch
from .audit import math_attention
from .budget import BudgetBatches, Trial
from .optimizer import dot, independent_probe, replay_rng, rng_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--repeats', type=int, default=12)
    args = parser.parse_args()
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = saved['training_config']
    torch.set_num_threads(config['threads'])
    batches = BudgetBatches(config['batch_size'], config['block_size'], 'cpu')
    trial = Trial(config, batches, 'fixed', 0, config['lr'])
    trial.model.load_state_dict(saved['model'])
    model = trial.model
    gen = torch.Generator().manual_seed(8400)
    training, calibration = batches.get('train', gen), batches.get('train', gen)
    model.zero_grad(set_to_none=True)
    model(*training)[1].backward()
    direction = trial.muon.prepare()
    state = rng_state()

    def finite():
        return independent_probe(direction, lambda: model(*calibration)[1], state, config['relative_distance'])

    def hvp():
        with replay_rng(state), torch.enable_grad():
            loss = model(*calibration)[1]
            gradients = torch.autograd.grad(loss, direction.params, create_graph=True)
            slope = sum((g * d).sum() for g, d in zip(gradients, direction.updates))
            product = torch.autograd.grad(slope, direction.params)
            return {'s': slope.item(), 'q': dot(direction.updates, product)}

    # Alternate order after warmup to limit drift; use the same math attention
    # backend for both because the HVP requires double-backward support.
    times = {'finite_difference': [], 'hvp': []}
    with math_attention():
        for _ in range(3):
            finite(); hvp()
        for i in range(args.repeats):
            order = [('finite_difference', finite), ('hvp', hvp)]
            if i % 2: order.reverse()
            for name, fn in order:
                start = time.perf_counter()
                result = fn()
                times[name].append(time.perf_counter() - start)
        fd, exact = finite(), hvp()
    result = dict(device='cpu', platform=platform.platform(), torch=str(torch.__version__),
                  threads=config['threads'], checkpoint=args.checkpoint, checkpoint_selection=saved['selection'],
                  parameters=sum(p.numel() for p in model.parameters()), precision='float32',
                  batch_size=config['batch_size'], block_size=config['block_size'], attention='math',
                  warmups=3, repeats=args.repeats, seconds=times,
                  median_seconds={k: statistics.median(v) for k,v in times.items()},
                  finite_difference=fd, hvp=exact,
                  relative_q_error=abs(fd['q']-exact['q'])/max(abs(exact['q']),1e-30))
    Path(args.out).write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
