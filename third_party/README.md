# Upstream sources

- `nanogpt/model.py` and `nanogpt/LICENSE` are unmodified from
  https://github.com/karpathy/nanoGPT/tree/3adf61e154c3fe3fca428ad6bc3818b27a3b8291 .
  Model file SHA-256 is recorded in each run manifest. This is actual NanoGPT,
  with a separate small training harness rather than a replacement GPT model.
- The Muon geometry is based on `SingleDeviceMuon` and `muon_update` at
  https://github.com/KellerJordan/Muon/blob/f98f1cacc0263b04290753e32be8d498c1efc806/muon.py .
  Its MIT license is retained in `Muon-LICENSE`. We implement the same momentum
  convention, five quintic Newton-Schulz iterations and aspect-ratio scaling
  `sqrt(max(1, rows/cols))`. Our NS arithmetic is **FP32**, unlike that source's
  BF16 arithmetic. We do not claim bitwise equivalence to upstream Muon or the
  RMS-matched layer-scaling variant in the scalable-Muon paper. All pilot Muon
  baselines and adaptive variants use exactly the same geometry here.
- Tiny Shakespeare is fetched from the char-rnn repository by
  `curvature_muon/data.py`; URL and exact content hash are saved in
  `data/shakespeare_char/meta.json`. Text is ignored by Git. Contiguous
  90/5/5 train/validation/test splits; test is prepared but never loaded by the
  experiment runner. Character IDs are sorted by Unicode code point.

NanoGPT is deprecated upstream in favor of nanochat, but is deliberately pinned
here for the small, easily audited single-device pilot requested by the user.
