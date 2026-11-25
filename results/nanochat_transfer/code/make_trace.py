"""Geometric mean of the per-step Muon LR of several runs -> JSON trace for --muon-lr-trace."""
import json, math, sys
out, ins = sys.argv[1], sys.argv[2:]
tr = [[s["muon_lr"] for s in json.load(open(f))["steps"]] for f in ins]
g = [math.exp(sum(math.log(t[i]) for t in tr) / len(tr)) for i in range(len(tr[0]))]
json.dump(g, open(out, "w")); print(out, len(g), g[0], max(g), g[-1])
