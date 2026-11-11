import unittest
import torch
from curvature_muon.optimizer import Direction, independent_probe, backtrack, rng_state


class GuardedTests(unittest.TestCase):
    def test_independent_probe_uses_calibration_gradient(self):
        p = torch.nn.Parameter(torch.tensor([[2.0]]))
        d = Direction([p], [torch.tensor([[99.0]])], [torch.ones_like(p)], [])
        result = independent_probe(d, lambda: (p ** 2).sum() / 2, rng_state(), 0.01)
        self.assertAlmostEqual(result['s'], 2.0, places=5)
        self.assertAlmostEqual(result['q'], 1.0, places=4)
        self.assertAlmostEqual(p.item(), 2.0)

    def test_backtracking_restores_weights_and_reduces_step(self):
        p = torch.nn.Parameter(torch.tensor([[1.0]]))
        d = Direction([p], [], [torch.ones_like(p)], [])
        rate, result = backtrack(d, lambda: (p ** 2).sum(), rng_state(), 4.0)
        self.assertEqual(rate, 2.0)
        self.assertEqual(result['trials'], 2)
        self.assertEqual(p.item(), 1.0)

    def test_backtracking_rejects_wrong_direction(self):
        p = torch.nn.Parameter(torch.tensor([[1.0]]))
        d = Direction([p], [], [-torch.ones_like(p)], [])
        rate, result = backtrack(d, lambda: (p ** 2).sum(), rng_state(), 1.0)
        self.assertEqual(rate, 0.0)
        self.assertEqual(result['status'], 'rejected')
        self.assertEqual(p.item(), 1.0)
