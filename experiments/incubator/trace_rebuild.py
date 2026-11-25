"""Rebuild each class's fading trace of use from an organism's own life log, and (APPLY=1)
switch the sitting cap with the fold on in her saved body (plan §11.12; trioron/pcll/sitting.py).

    VEE_HOME=outputs/incubator/copies/<copy> python3 experiments/incubator/trace_rebuild.py            # report only
    VEE_HOME=outputs/incubator/copies/<copy> APPLY=1 python3 experiments/incubator/trace_rebuild.py    # write body.pt

Every moment in HOME/embryo_log.pt and HOME/fetal_log.pt is placed in her classes AS THEY
STAND NOW (an approximation of her history); per sitting, trace = trace · FADE + members.
APPLY on a living organism changes the rules she lives by: only on Rocky's yes (rules 25, 27).
"""
from __future__ import annotations
import os, sys
import torch
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(__file__))
import vee_body as vb
from trioron.pcll.sitting import FADE, TRACE_FLOOR, SittingCap

body = vb.Body.load()
days = [(X, f) for f in ("embryo_log.pt", "fetal_log.pt") if os.path.exists(f"{vb.HOME}/{f}")
        for X in torch.load(f"{vb.HOME}/{f}", weights_only=False)["X"]]
print(f"{vb.HOME}: {len(days)} days in her logs")
for k, l in body.organs().items():
    m = l.mixed
    n, tr = len(m.classes), torch.zeros(len(m.classes))
    for X, f in days:
        if k == "kin" and f.startswith("embryo"):
            continue                                        # her sense of change was not born yet
        cls = body.read(X, k)[0]
        for i in range(0, len(X), vb.WINDOW):
            c = cls[i:i + vb.WINDOW]
            tr = tr * FADE + torch.bincount(c[c >= 0], minlength=n).float()
    print(f"  {k}: classes {n}; never claimed a moment she lived {int((tr == 0).sum())}; due for folding (trace < {TRACE_FLOOR}) {int((tr < TRACE_FLOOR).sum())}")
    if os.environ.get("APPLY") == "1":
        m.sitting = SittingCap(fold=True)
        m.sitting.trace = {c.name: float(t) for c, t in zip(m.classes, tr)}
if os.environ.get("APPLY") == "1":
    print("written:", body.save())
