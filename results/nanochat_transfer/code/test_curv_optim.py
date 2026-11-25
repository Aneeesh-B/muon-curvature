"""Equivalence of the split Muon step (curv_optim) with nanochat's fused Muon step."""
import torch
from nanochat.optim import muon_step_fused
from nanochat.curv_optim import muon_direction, muon_apply

def run(shape, K=3, steps=3):
    torch.manual_seed(0)
    P1 = torch.randn(K, *shape, device="cuda") * 0.02
    P2 = P1.clone()
    mb1 = torch.zeros(K, *shape, device="cuda"); mb2 = mb1.clone()
    ss = (K, shape[0], 1) if shape[0] >= shape[1] else (K, 1, shape[1])
    sb1 = torch.zeros(ss, device="cuda"); sb2 = sb1.clone()
    red = -1 if shape[0] >= shape[1] else -2
    mom = torch.tensor(0.9); lr = torch.tensor(0.01 * max(1, shape[0] / shape[1]) ** 0.5); wd = torch.tensor(0.5); b2 = torch.tensor(0.9)
    for _ in range(steps):
        G = torch.randn(K, *shape, device="cuda")
        muon_step_fused(G.clone(), P1, mb1, sb1, mom, lr, wd, b2, 5, red)
        g = muon_direction(G.clone(), mb2, sb2, mom, b2, 5, red)
        muon_apply(P2, g, lr, wd)
    err = ((P1 - P2).norm() / P1.norm()).item()
    merr = ((mb1 - mb2).norm() / mb1.norm()).item()
    return err, merr

if __name__ == "__main__":
    for shape in [(384, 384), (1536, 384), (384, 1536), (3, 32)]:
        err, merr = run(shape)
        print(shape, f"param rel err {err:.2e}  momentum rel err {merr:.2e}")
        assert err < 1e-5 and merr < 1e-6, "split Muon differs from fused Muon"
    print("OK")
