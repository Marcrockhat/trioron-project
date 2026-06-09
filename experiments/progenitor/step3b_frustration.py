"""Step 3b — the spawning signal: LOCAL frustration, per output cell.

Rocky's correction (s025): there is no hand-wired "dog-dedicated dendrite". The
structure must spawn a dendrite *autonomously* where the data demands it. If a
dendrite never appears, the bug is that our code failed to reveal the spawning
signal — not that the wiring is wrong.

The earlier trial scored phenotypes by GLOBAL cross-entropy. That is exactly the
orphaned-global-frustration problem (design §1, §4.2): a single scalar washes out
dog (1 of 6 classes), so training aims any added capacity wherever it helps the
sum, never specifically at the frustrated class. The spawn signal has to be
**local** — measured per output cell.

This probe trains the bare linear baseline (perception → 6 outputs) and measures,
per output cell, a local frustration signal:

    residual_ce[c] — mean cross-entropy on samples whose TRUE class is c
                     (how badly cell c is still failing at convergence)
    stagnation[c]  — residual_ce[c] at the END vs the MIDPOINT of training
                     (frustration = high residual that WON'T go down: gradient
                      pressure with no comprehension — design §4.2)

A cell is a **spawn candidate** when its frustration stands out from the field
(relative rule: > mean + k·std — the cell-first recruitment rule, memory
``cell_first_cl5_relative_rule``), not by an absolute threshold. If dog crosses
and the others don't, the signal localizes the demand — and a spawn driven by
*this* signal would attach a daughter to dog on its own. If dog does NOT stand
out, the signal logic is what needs fixing.

This is measurement only — no spawning, no phenotype choice yet.

Run: python3 -m experiments.progenitor.step3b_frustration
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from trioron.core import Envelope, construct
from trioron.core.epigenome import PERCEPTION, OUTPUT, set_gene
from trioron.phenotype import default_dispatch_table

from .data import make_data, bayes_per_class

SPAWN_K = 1.0   # relative recruitment: frustration > mean + K·std → spawn candidate


def build_linear_baseline(n_in: int = 1, n_out: int = 6):
    """perception(1) → 6 linear outputs (the step-1 floor)."""
    def base(sub):
        a = sub.arena
        perc = a.alloc(n_in); outs = a.alloc(n_out)
        for p in perc.tolist():
            a.epigenome[p] = set_gene(int(a.epigenome[p].item()), PERCEPTION)
            a.rank[p] = 0
        for o in outs.tolist():
            a.epigenome[o] = set_gene(int(a.epigenome[o].item()), OUTPUT)
            a.rank[o] = 1
        a.refresh_all_phenotypes()
        a.add_edges(perc.repeat(n_out), outs.repeat_interleave(n_in))
        base.outs = outs.tolist()
    sub = construct(base=base, envelope=Envelope(),
                    dispatch_table=default_dispatch_table(), capacity=32)
    sub.outs = base.outs
    return sub


def per_class_ce(sub, d) -> list[float]:
    """Mean cross-entropy on samples of each true class (the per-output residual)."""
    with torch.no_grad():
        logits = sub(d.x)
        ce_all = F.cross_entropy(logits, d.y, reduction="none")
    return [float(ce_all[d.y == c].mean()) for c in range(len(d.names))]


def main() -> None:
    train_d = make_data(seed=0, n_per_class=256)
    test_d = make_data(seed=1, n_per_class=512)
    names = test_d.names

    sub = build_linear_baseline()
    torch.manual_seed(0); sub.prepare_training()
    opt = torch.optim.Adam(sub.trainable_tensors(), lr=0.05)

    steps = 400
    mid_residual = None
    for t in range(steps):
        opt.zero_grad()
        F.cross_entropy(sub(train_d.x), train_d.y).backward()
        opt.step()
        if t == steps // 2 - 1:
            mid_residual = per_class_ce(sub, test_d)   # frustration midpoint

    end_residual = per_class_ce(sub, test_d)
    bpc = bayes_per_class(test_d)
    with torch.no_grad():
        pred = sub(test_d.x).argmax(1)
    recall = [float((pred[test_d.y == c] == c).float().mean()) for c in range(len(names))]

    # Frustration = high residual that won't drop (end ≈ mid → stagnant).
    stagnation = [mid_residual[c] - end_residual[c] for c in range(len(names))]  # >0 = still improving
    frustration = end_residual                                                   # the residual itself is the signal

    fr = torch.tensor(frustration)
    thresh = float(fr.mean() + SPAWN_K * fr.std())

    print("Step 3b — local frustration (the spawning signal), per output cell\n")
    print(f"  {'class':>8}  {'recall':>6}  {'bayes':>6}  {'residual_ce':>11}  "
          f"{'Δmid→end':>8}  spawn?")
    for c, name in enumerate(names):
        candidate = "← SPAWN" if frustration[c] > thresh else ""
        stag = "stagnant" if abs(stagnation[c]) < 0.01 else f"{stagnation[c]:+.3f}"
        print(f"  {name:>8}  {recall[c]:>6.3f}  {bpc[c]:>6.3f}  {end_residual[c]:>11.3f}  "
              f"{stag:>8}  {candidate}")
    print(f"\n  relative spawn threshold = mean + {SPAWN_K:g}·std "
          f"= {fr.mean():.3f} + {SPAWN_K:g}·{fr.std():.3f} = {thresh:.3f}")
    cands = [names[c] for c in range(len(names)) if frustration[c] > thresh]
    print(f"  spawn candidate(s): {cands or 'NONE — signal did not localize; logic needs fixing'}")
    if cands == ["dog"]:
        print("  → dog alone crosses: the data localizes the demand. A spawn driven by\n"
              "    this signal would attach a daughter to dog autonomously.")


if __name__ == "__main__":
    main()
