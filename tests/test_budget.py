import contextlib
import copy
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

import torch

from curvature_muon.budget import (BudgetBatches, Trial, candidate_rates,
                                    evaluate, run_campaign, write_report)
from curvature_muon.data import DATA, ROOT
from curvature_muon.optimizer import rng_state
from curvature_muon.train import run


def assert_equal(test, a, b):
    if isinstance(a, torch.Tensor):
        test.assertTrue(torch.equal(a, b))
    elif isinstance(a, dict):
        test.assertEqual(a.keys(), b.keys())
        for key in a:
            assert_equal(test, a[key], b[key])
    elif isinstance(a, (tuple, list)):
        test.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            assert_equal(test, x, y)
    else:
        test.assertEqual(a, b)


@unittest.skipUnless((DATA / "meta.json").exists(), "Prepare Tiny Shakespeare first")
class BudgetTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.config = json.loads((ROOT / "configs/tiny.json").read_text())
        self.config.update(n_layer=1, n_embd=16, n_head=2, block_size=8, batch_size=2,
                           dropout=.2, steps=3, probe_start=0, probe_every=2,
                           threads=2, eval_batches=1, eval_every=3)
        self.experiment = json.loads((ROOT / "configs/budget.json").read_text())
        self.batches = BudgetBatches(2, 8, "cpu")

    def test_resumption_preserves_dropout_momentum_and_batch_stream(self):
        with contextlib.redirect_stdout(io.StringIO()):
            uninterrupted = Trial(self.config, self.batches, "periodic", 13, .02)
            interrupted = Trial(self.config, self.batches, "periodic", 13, .02)
            other = Trial(self.config, self.batches, "fixed", 92, .005)
            for _ in range(4):
                uninterrupted.train_step()
            interrupted.train_step()
            state = copy.deepcopy(interrupted.state_dict())
            for _ in range(2):
                other.train_step()
            resumed = Trial(self.config, self.batches, "periodic", 13, .02)
            resumed.load_state_dict(state)
            for _ in range(3):
                resumed.train_step()
            a, b = uninterrupted.state_dict(), resumed.state_dict()
            for key in ("model", "muon", "adam", "rng", "train_generator", "step"):
                assert_equal(self, a[key], b[key])
            self.assertEqual(a["stats"]["probes"], 2)
            self.assertEqual(b["stats"]["probes"], 2)

    def test_guarded_resume_preserves_calibration_stream(self):
        with contextlib.redirect_stdout(io.StringIO()):
            original = Trial(self.config, self.batches, "guarded", 13, .02)
            for _ in range(3):
                original.train_step()
            saved = copy.deepcopy(original.state_dict())
            original.train_step()
            resumed = Trial(self.config, self.batches, "guarded", 13, .02)
            resumed.load_state_dict(saved)
            resumed.train_step()
            a, b = original.state_dict(), resumed.state_dict()
            for key in ("model", "muon", "adam", "rng", "train_generator", "calibration_generator", "step"):
                assert_equal(self, a[key], b[key])

    def test_matches_existing_adaptive_trainer(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            run(self.config, Path(tmp) / "original", mode="periodic", seed=7)
            original = torch.load(Path(tmp) / "original/final.pt", weights_only=False)
            session = Trial(self.config, self.batches, "periodic", 7, .02)
            for _ in range(self.config["steps"]):
                session.train_step()
            current = session.state_dict()
            for key in ("model", "muon", "adam", "rng", "train_generator"):
                assert_equal(self, original[key], current[key])

    def test_validation_splits_and_weighted_accuracy(self):
        batches = BudgetBatches(2, 8, "cpu")
        self.assertEqual(set(batches.arrays), {"train", "tuning", "reporting"})
        midpoint = batches.split_metadata["split_at"]
        self.assertEqual(batches.split_metadata["tuning_range"][1],
                         batches.split_metadata["reporting_range"][0])
        self.assertEqual(len(batches.arrays["tuning"]), midpoint)
        # The final partial batch must contribute with its token weight.
        batches.arrays["reporting"] = torch.arange(41) % 3

        class Uniform(torch.nn.Module):
            def forward(self, x, y):
                logits = torch.zeros((*x.shape, 3))
                return logits, torch.nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten())

        model = Uniform()
        state = rng_state()
        measured = evaluate(model, batches, "reporting")
        self.assertTrue(model.training)
        self.assertEqual(measured["tokens"], 40)
        self.assertAlmostEqual(measured["loss"], math.log(3), places=6)
        self.assertEqual(measured["accuracy"], (torch.arange(1, 41) % 3 == 0).sum().item()/40)
        assert_equal(self, state, rng_state())

    def test_search_rates_include_default_and_are_reproducible(self):
        rates = candidate_rates(self.experiment, "muon_search", self.config)
        self.assertEqual(len(rates), 3)
        self.assertEqual(rates[0], .02)
        self.assertTrue(all(.001 <= value <= .1 for value in rates))
        self.assertEqual(rates, candidate_rates(self.experiment, "muon_search", self.config))
        self.assertEqual(candidate_rates(self.experiment, "adaptive_muon", self.config), [.02])

    def test_short_search_has_one_shared_cap_and_continues_winner(self):
        config = dict(self.config, dropout=0.0)
        experiment = dict(self.experiment, seconds_per_method=2.5, final_reserve_seconds=.3,
                          validation_every_seconds=.2)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            result = run_campaign(config, experiment, "muon_search", Path(tmp) / "muon_search")
            self.assertTrue(result["budget_met"], result["elapsed_seconds"])
            self.assertGreater(result["elapsed_seconds"], 1.5)
            self.assertEqual(len(result["trials"]), 3)
            winner = result["trials"][result["continued_trial_id"]]
            self.assertGreater(winner["steps"], result["continued_from_step"])
            self.assertEqual(result["reporting"]["tokens"],
                             ((len(self.batches.arrays["reporting"])-1)//8)*8)
            self.assertEqual(result["completed_steps"], sum(t["steps"] for t in result["trials"]))
            self.assertTrue((Path(tmp) / "muon_search/selected.pt").exists())
            self.assertEqual(len(write_report(tmp)), 1)
            with self.assertRaises(FileExistsError):
                run_campaign(config, experiment, "muon_search", Path(tmp) / "muon_search")


if __name__ == "__main__":
    unittest.main()
