import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

import torch

from curvature_muon.budget import Trial
from curvature_muon.data import DATA, ROOT
from curvature_muon.optimizer import AdaGO, orthogonalize
from curvature_muon.production import (ProductionBatches, charged, costs_for, mechanism_check, run_search,
                                       run_trajectory, step_cost)
from tests.test_budget import assert_equal

PLAN = dict(adamw_update=124375007232, newton_schulz=100705239040, muon_update=225080246272,
            probe_pass=124255731712, tuning_validation=562278678528)


class AdaGOTests(unittest.TestCase):
    def test_one_matrix_update_matches_hand_computation(self):
        torch.manual_seed(0)
        w0 = torch.randn(3, 5)
        p = torch.nn.Parameter(w0.clone())
        opt = AdaGO([p], lr=0.3, eps=1e-3, momentum=0.9, gamma=1.0, v0=1e-6)
        g1, g2 = torch.randn(3, 5)*0.1, torch.randn(3, 5)*4.0  # one under and one over gamma
        p.grad = g1.clone()
        a1 = opt.step()[0]
        m1 = 0.1*g1
        v1 = math.sqrt(1e-12 + g1.norm().item()**2)
        self.assertAlmostEqual(a1, max(1e-3, 0.3*g1.norm().item()/v1), places=6)
        expected = w0 - a1*orthogonalize(m1)
        self.assertTrue(torch.allclose(p.detach(), expected, atol=1e-6))
        p.grad = g2.clone()
        a2 = opt.step()[0]
        v2 = math.sqrt(v1**2 + 1.0)  # clipped to gamma^2
        self.assertAlmostEqual(a2, max(1e-3, 0.3*1.0/v2), places=6)
        m2 = 0.9*m1 + 0.1*g2
        self.assertTrue(torch.allclose(opt.state[p]["momentum_buffer"], m2, atol=1e-7))
        self.assertTrue(torch.allclose(p.detach(), expected - a2*orthogonalize(m2), atol=1e-5))

    def test_epsilon_floor(self):
        p = torch.nn.Parameter(torch.ones(2, 2))
        opt = AdaGO([p], lr=1e-9, eps=0.05)
        p.grad = torch.ones(2, 2)
        self.assertEqual(opt.step()[0], 0.05)


@unittest.skipUnless((DATA / "meta.json").exists(), "Prepare Tiny Shakespeare first")
class ProductionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.config = json.loads((ROOT / "configs/tiny.json").read_text())
        self.config.update(n_layer=1, n_embd=16, n_head=2, block_size=8, batch_size=2, dropout=0.0,
                           probe_start=2, probe_every=3, threads=2)
        self.batches = ProductionBatches(2, 8, "cpu")
        self.costs = costs_for(self.config, self.batches)

    def test_costs_match_the_plan_table(self):
        config = json.loads((ROOT / "configs/wide.json").read_text())
        costs = costs_for(config, ProductionBatches(16, 128, "cpu"))
        for key, value in PLAN.items():
            self.assertEqual(getattr(costs, key), value, key)
        caps = json.loads((ROOT / "configs/production.json").read_text())["caps"]
        self.assertEqual([caps[k] for k in ("low", "middle", "high")],
                         [n*costs.muon_update for n in (500, 1000, 2000)])
        self.assertEqual(costs.windows["tuning"], 217)

    def test_predicted_and_charged_work_agree(self):
        for mode, passes in (("same_batch", 1), ("independent", 2), ("guarded", 2), ("adago", 0), ("adamw", 0)):
            lr = (0.1, 5e-4) if mode == "adago" else (0.001 if mode == "adamw" else .02)
            trial = Trial(self.config, self.batches, mode, 5, lr)
            total = 0
            for _ in range(6):
                predicted = step_cost(trial, self.costs)
                due = trial.probe_due()
                record = trial.train_step()
                charge = charged(trial, record, self.costs)
                self.assertLessEqual(charge, predicted)
                if mode != "guarded":
                    self.assertEqual(charge, predicted)
                self.assertEqual(record.get("probe_passes", 0), passes if due else 0)
                total += charge
            self.assertEqual(trial.stats["probes"], 2 if passes else 0)

    def test_independent_arm_keeps_ordinary_stream_and_falls_back_to_base(self):
        fixed = Trial(self.config, self.batches, "fixed", 5, .02)
        independent = Trial(self.config, self.batches, "independent", 5, .02)
        same = Trial(self.config, self.batches, "same_batch", 5, .02)
        for _ in range(6):
            fixed.train_step(), same.train_step()
            record = independent.train_step()
            if record.get("decision") == "fallback":
                self.assertEqual(record["lr"], .02)
        assert_equal(self, fixed.generator.get_state(), independent.generator.get_state())
        assert_equal(self, fixed.generator.get_state(), same.generator.get_state())
        assert_equal(self, fixed.calibration_generator.get_state(), same.calibration_generator.get_state())
        self.assertFalse(torch.equal(fixed.calibration_generator.get_state(),
                                     independent.calibration_generator.get_state()))

    def test_mechanism_check_leaves_the_trajectory_unchanged(self):
        mechanism = dict(anchors=[2, 4], fresh_batches=2, grid=[0.0, 0.003, 0.02, 0.2])
        plain = Trial(self.config, self.batches, "fixed", 9, .02)
        observed = Trial(self.config, self.batches, "fixed", 9, .02)
        with tempfile.TemporaryDirectory() as tmp:
            for _ in range(6):
                if observed.step in mechanism["anchors"]:
                    row = mechanism_check(observed, self.batches, self.config, mechanism, 9, Path(tmp))
                    self.assertEqual(set(row["proposals"]), {"same_batch", "independent"})
                    self.assertEqual(len(row["grid"]), 4)
                    self.assertEqual(row["grid"][0]["delta"], 0.0)
                plain.train_step(), observed.train_step()
        a, b = plain.state_dict(), observed.state_dict()
        for key in ("model", "muon", "adam", "rng", "train_generator", "calibration_generator", "step"):
            assert_equal(self, a[key], b[key])

    def test_trajectory_saves_latest_update_at_or_below_each_cap(self):
        u = self.costs.muon_update
        caps = [("low", 5*u), ("high", 9*u + 3*self.costs.probe_pass)]
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            result = run_trajectory("same_batch_muon", self.config, self.batches, self.costs, caps, 3,
                                    Path(tmp) / "arm", "cpu", isolation_step=3)
            low, high = result["points"]
            # Probes at updates 2, 5 and 8 cost one hidden-matrix pass each.
            self.assertEqual(low["selected_trajectory_steps"], 4)
            self.assertEqual(low["flops"], 4*u + self.costs.probe_pass)
            self.assertEqual(high["selected_trajectory_steps"], 9)
            self.assertEqual(high["flops"], 9*u + 3*self.costs.probe_pass)
            self.assertTrue(low["eligible"] and high["eligible"])
            self.assertEqual(result["isolation"]["step"], 3)
            reference = Trial(self.config, self.batches, "same_batch", 3, .02)
            for _ in range(4):
                reference.train_step()
            saved = torch.load(Path(tmp) / "arm/weights_low.pt", weights_only=False)
            assert_equal(self, saved["model"], {k: v.cpu() for k, v in reference.model.state_dict().items()})
            # A finished campaign is skipped on rerun.
            self.assertIsNone(run_trajectory("same_batch_muon", self.config, self.batches, self.costs, caps, 3,
                                             Path(tmp) / "arm", "cpu"))

    def test_search_charges_slots_and_continues_the_winner(self):
        cap = 40*self.costs.muon_update + 25*self.costs.tuning_validation
        grid = json.loads((ROOT / "configs/production.json").read_text())["search_grids"]
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            for name in ("muon_search", "adamw_search", "adago_search"):
                result = run_search(name, self.config, self.batches, self.costs, grid[name], 8, 3, "x", cap,
                                    Path(tmp) / name, "cpu")
                self.assertEqual(len(result["candidates"]), 5)
                for candidate in result["candidates"]:
                    self.assertLessEqual(candidate["flops"], result["slot_flops"])
                    self.assertGreater(candidate["steps"], 0)
                point = result["points"][0]
                self.assertLessEqual(point["flops"], cap)
                self.assertGreater(point["selected_trajectory_steps"], result["continued_from_step"])
                self.assertEqual(point["training_characters"],
                                 sum(t["steps"] for t in result["trials"])*2*8)

    def test_test_split_is_sealed_until_unsealed(self):
        with self.assertRaises(RuntimeError):
            next(self.batches.evaluation_batches("test"))
        self.batches.unseal_test()
        self.assertEqual(sum(y.numel() for _, y in self.batches.evaluation_batches("test")),
                         self.batches.test_windows*8)


if __name__ == "__main__":
    unittest.main()
