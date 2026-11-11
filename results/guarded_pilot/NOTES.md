# Independent-batch guarded Muon: exploratory pilot

The guarded variant constructs the direction from the usual training batch, computes both finite-difference gradients on a separately sampled training batch, and checks matrix-only loss on a third batch. It halves the proposed rate up to six times until that check does not increase loss; otherwise it sets the matrix rate to zero until the next probe. Invalid curvature falls back to the configured reference rate before the check. The companion AdamW update is not included in the check. The selected rate is held for 50 updates, so the check does not guarantee future descent. Probe work is charged to the campaign.

The single-seed, 60-second CPU pilot gave reporting losses 1.5733 for guarded Muon, 1.6044 for fixed Muon, and 2.1539 for the original adaptive method. This is exploratory: the method was chosen after earlier findings, and small unit tests ran concurrently during part of the original adaptive campaign. All three runs finished inside their caps. Guarded Muon made 23 probes: 20 valid curvature estimates, 3 non-descent cases, 22 accepted safeguards, and 1 rejection.

Separate sampling can produce overlapping examples; it does not create a disjoint dataset split. A single check batch is noisy. Next steps include multiple seeds, independent confirmation and ablations separating the effects of calibration and backtracking.

Reproduce with a new output directory: `../.venv/bin/python -m curvature_muon.budget --config configs/budget.json --out runs/guarded_pilot --seconds 60 --methods guarded_muon adaptive_muon default_muon`.
