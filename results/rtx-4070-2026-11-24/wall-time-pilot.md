# RTX 4070 wall-time pilot

One seed on the 9.72M NanoGPT model, with a 180-second wall-time cap per arm. Reporting loss is in nats per character. The guarded arm includes backtracking.

| Method | Reporting loss | Time (s) | Updates across trials | Cap met |
|---|---:|---:|---:|---|
| Searched AdamW | 1.4668 | 147.9 | 8,305 | Yes |
| Same-batch adaptive Muon | 2.2741 | 137.7 | 3,900 | Yes |
| Fixed Muon | 1.5276 | 138.0 | 3,608 | Yes |
| Guarded independent-batch Muon | 1.4948 | 137.6 | 3,529 | Yes |
| Searched Muon | 1.4771 | 148.5 | 3,867 | Yes |
