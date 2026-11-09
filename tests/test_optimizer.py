import copy
import unittest

import torch

from curvature_muon.optimizer import (Direction, Muon, dot, probe, replay_rng,
                                      rng_state, select_lr, split_parameters)
from third_party.nanogpt.model import GPT, GPTConfig


class CurvatureTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)

    def test_quadratic_curvature_and_minimizer(self):
        p = torch.nn.Parameter(torch.tensor([[1., 2.], [-1., 0.5]], dtype=torch.float64))
        h = torch.tensor([[2., 4.], [6., 8.]], dtype=torch.float64)
        closure = lambda: (0.5*h*p.square()).sum()
        closure().backward()
        d = p.grad.clone() / 3
        direction = Direction([p], [p.grad.clone()], [d], [])
        before, grad = p.clone(), p.grad.clone()
        m = probe(direction, closure, rng_state(), 1e-4)
        exact = dot([d], [h*d])
        self.assertEqual(m["status"], "ok")
        self.assertAlmostEqual(m["q"], exact, places=8)
        self.assertAlmostEqual(m["eta_raw"], dot([p.grad], [d])/exact, places=10)
        self.assertTrue(torch.equal(p, before))
        self.assertTrue(torch.equal(p.grad, grad))

    def test_negative_curvature_and_non_descent_fallback(self):
        p = torch.nn.Parameter(torch.ones(2, 2))
        closure = lambda: -p.square().sum()
        closure().backward()
        direction = Direction([p], [p.grad.clone()], [p.grad.clone()], [])
        m = probe(direction, closure, rng_state())
        self.assertEqual(m["status"], "nonpositive_curvature")
        self.assertEqual(select_lr(m, .02), (.02, "fallback"))
        closure = lambda: p.square().sum()
        p.grad = None
        closure().backward()
        direction = Direction([p], [p.grad.clone()], [-p.grad.clone()], [])
        self.assertEqual(probe(direction, closure, rng_state())["status"], "non_descent")

    def test_bounds_and_safety_are_explicit(self):
        self.assertEqual(select_lr(dict(status="ok", eta_raw=10), .02), (.2, "clamped"))
        self.assertEqual(select_lr(dict(status="ok", eta_raw=.01), .02, safety=.5), (.005, "accepted"))

    def test_prepare_does_not_advance_state(self):
        p = torch.nn.Parameter(torch.randn(4, 3))
        opt = Muon([p])
        p.square().sum().backward()
        a, b = opt.prepare(), opt.prepare()
        self.assertEqual(len(opt.state), 0)
        self.assertTrue(torch.equal(a.updates[0], b.updates[0]))
        expected = p.detach() - .02*a.updates[0]
        opt.step(a)
        self.assertTrue(torch.allclose(p, expected, atol=1e-7, rtol=1e-6))
        self.assertTrue(torch.allclose(opt.state[p]["momentum_buffer"], .05*p.grad))
        old = opt.state[p]["momentum_buffer"].clone()
        opt.prepare()
        self.assertTrue(torch.equal(old, opt.state[p]["momentum_buffer"]))

    def test_exception_restores_weights_gradients_and_rng(self):
        p = torch.nn.Parameter(torch.randn(3, 3))
        p.square().sum().backward()
        opt = Muon([p])
        direction = opt.prepare()
        old, grad, state = p.clone(), p.grad.clone(), rng_state()

        def fail():
            torch.rand(100)
            raise RuntimeError("intentional")

        with self.assertRaisesRegex(RuntimeError, "intentional"):
            probe(direction, fail, state)
        self.assertTrue(torch.equal(p, old))
        self.assertTrue(torch.equal(p.grad, grad))
        self.assertTrue(torch.equal(torch.get_rng_state(), state[0]))
        self.assertEqual(len(opt.state), 0)

    def test_dropout_replay_and_tied_parameter_ownership(self):
        model = GPT(GPTConfig(n_layer=1, n_head=2, n_embd=16, block_size=8,
                              vocab_size=11, dropout=.25, bias=False))
        hidden, auxiliary = split_parameters(model)
        self.assertEqual(len(hidden), 4)
        self.assertIn(id(model.lm_head.weight), [id(p) for p in auxiliary])
        self.assertNotIn(id(model.transformer.wpe.weight), [id(p) for p in hidden])
        x, y = torch.randint(11, (2, 8)), torch.randint(11, (2, 8))
        closure = lambda: model(x, y)[1]
        state = rng_state()
        loss = closure()
        loss.backward()
        grads = [p.grad.clone() for p in model.parameters()]
        after = torch.get_rng_state()
        with replay_rng(state):
            self.assertTrue(torch.equal(closure(), loss))
        opt = Muon(hidden)
        before = copy.deepcopy(model.state_dict())
        measurement = probe(opt.prepare(), closure, state)
        self.assertTrue(measurement["q"] is not None)
        for name, p in model.state_dict().items():
            self.assertTrue(torch.equal(p, before[name]))
        for p, g in zip(model.parameters(), grads):
            self.assertTrue(torch.equal(p.grad, g))
        self.assertTrue(torch.equal(torch.get_rng_state(), after))

    def test_optimizer_roundtrip(self):
        p = torch.nn.Parameter(torch.randn(3, 4))
        opt = Muon([p], weight_decay=.1)
        p.square().sum().backward()
        opt.step()
        q = torch.nn.Parameter(p.detach().clone())
        restored = Muon([q])
        restored.load_state_dict(copy.deepcopy(opt.state_dict()))
        p.grad, q.grad = None, None
        p.square().sum().backward()
        q.square().sum().backward()
        opt.step()
        restored.step()
        self.assertTrue(torch.equal(p, q))

    def test_zero_direction_and_nonfinite_training_gradients(self):
        p = torch.nn.Parameter(torch.ones(2, 2))
        direction = Direction([p], [torch.ones_like(p)], [torch.zeros_like(p)], [])
        self.assertEqual(probe(direction, lambda: p.sum(), rng_state())["status"], "zero_or_invalid_norm")
        p.grad = torch.full_like(p, float("nan"))
        with self.assertRaises(FloatingPointError):
            Muon([p]).prepare()


if __name__ == "__main__":
    unittest.main()
