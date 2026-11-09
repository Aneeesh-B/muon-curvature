import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

from curvature_muon.data import ROOT, DATA
from curvature_muon.train import run


@unittest.skipUnless((DATA / "meta.json").exists(), "Run python -m curvature_muon.data for integration test")
class TrainingTests(unittest.TestCase):
    def test_diagnostics_leave_complete_training_trajectory_unchanged(self):
        config = json.loads((ROOT / "configs/smoke.json").read_text())
        config.update(n_layer=1, n_head=2, n_embd=16, batch_size=2, block_size=8,
                      steps=3, dropout=.2, eval_every=3, eval_batches=1,
                      diagnostic_steps=[0], stale_ages=[0, 1], diagnostic_batches=1,
                      scan_lrs=[.01, .02], probe_distances=[.001], threads=2)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            plain, diag = Path(tmp) / "plain", Path(tmp) / "diag"
            run(config, plain, mode="fixed", seed=23)
            run(config, diag, mode="fixed", seed=23, diagnostics=True)
            # Local checkpoints produced by this test; weights_only=False loads
            # RNG/config metadata as well as tensors, never untrusted files.
            a = torch.load(plain / "final.pt", weights_only=False)
            b = torch.load(diag / "final.pt", weights_only=False)

            def equal(x, y):
                if isinstance(x, torch.Tensor):
                    self.assertTrue(torch.equal(x, y))
                elif isinstance(x, dict):
                    self.assertEqual(x.keys(), y.keys())
                    for key in x:
                        equal(x[key], y[key])
                elif isinstance(x, (list, tuple)):
                    self.assertEqual(len(x), len(y))
                    for u, v in zip(x, y):
                        equal(u, v)
                else:
                    self.assertEqual(x, y)

            for field in ("model", "muon", "adam", "rng", "train_generator"):
                equal(a[field], b[field])
            rows = [json.loads(line) for line in (diag / "scans.jsonl").read_text().splitlines()]
            self.assertEqual([row["age"] for row in rows], [0, 1])


if __name__ == "__main__":
    unittest.main()
