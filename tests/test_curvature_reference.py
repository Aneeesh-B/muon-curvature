import contextlib
import io
import unittest

import torch

from curvature_muon.audit import exact_curvature
from curvature_muon.optimizer import Direction, dot, probe, rng_state, select_lr, split_parameters
from third_party.nanogpt.model import GPT, GPTConfig


class CurvatureReferenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(41)

    def test_coupled_quadratic_includes_cross_matrix_hessian_terms(self):
        ps = [torch.nn.Parameter(torch.randn(2, 2, dtype=torch.float64)) for _ in range(2)]
        a = torch.randn(8, 8, dtype=torch.float64)
        h = a.T @ a + torch.eye(8, dtype=torch.float64)

        def closure():
            x = torch.cat([p.flatten() for p in ps])
            return .5*x @ h @ x

        closure().backward()
        directions = [p.grad.clone()/10 for p in ps]
        direction = Direction(ps, [p.grad.clone() for p in ps], directions, [])
        vector = torch.cat([d.flatten() for d in directions])
        exact = (vector @ h @ vector).item()
        measured = probe(direction, closure, rng_state(), relative_distance=1e-5)
        self.assertEqual(measured["status"], "ok")
        self.assertAlmostEqual(measured["q"]/exact, 1.0, places=9)

    def test_nanogpt_finite_difference_converges_to_independent_hvp(self):
        with contextlib.redirect_stdout(io.StringIO()):
            model = GPT(GPTConfig(n_layer=1, n_head=2, n_embd=16, block_size=8,
                                  vocab_size=11, dropout=0.0, bias=False))
        x, y = torch.randint(11, (2, 8)), torch.randint(11, (2, 8))
        ps, _ = split_parameters(model)
        loss = model(x, y)[1]
        grads = torch.autograd.grad(loss, ps)
        norm = dot(grads, grads)**.5
        directions = [g.detach()/norm for g in grads]
        reference = exact_curvature(model, (x, y), directions)
        self.assertGreater(abs(reference["q"]), 1e-8)
        small = reference["finite_difference_sweep"]["1e-06"]["q"]
        self.assertLess(abs(small-reference["q"])/abs(reference["q"]), 1e-3)

    def test_correct_local_quadratic_rate_does_not_guarantee_actual_descent(self):
        # Locally gentle curvature, with a quartic term that dominates a larger
        # move. Even our maximum-rate cap alone cannot ensure loss decreases.
        p = torch.nn.Parameter(torch.ones(1, 1, dtype=torch.float64))
        closure = lambda: ((p-2).square()+1000*(p-1).pow(4)).sum()
        closure().backward()
        direction = Direction([p], [p.grad.clone()], [p.grad.clone()], [])
        measured = probe(direction, closure, rng_state(), 1e-4)
        self.assertEqual(measured["status"], "ok")
        self.assertAlmostEqual(measured["q"], 8.0, places=3)
        self.assertAlmostEqual(measured["eta_raw"], .5, places=3)
        rate, decision = select_lr(measured, .02)
        self.assertEqual((rate, decision), (.2, "clamped"))
        original = closure().item()
        with torch.no_grad():
            old = p.clone()
            p.add_(direction.updates[0], alpha=-rate)
            large_loss = closure().item()
            p.copy_(old)
            p.add_(direction.updates[0], alpha=-.02)
            small_loss = closure().item()
        self.assertGreater(large_loss, original)
        self.assertLess(small_loss, original)


if __name__ == "__main__":
    unittest.main()
