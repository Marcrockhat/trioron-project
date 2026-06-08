"""Smoke: does the restored epigenetic lock λ (Fisher-EWC) reduce forgetting on the
v2 substrate? Two synthetic supervised tasks A then B on a mirror substrate; measure
task-A accuracy AFTER learning B, with vs without the EWC penalty.

Win = λ-EWC retains task A better than plain SGD (which forgets it).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.world.mirror_cells import build_mirror, _solo, INPUT_DIM
from experiments.world.tile_world import N_ACTION
from trioron.learning.epigenetic_lock import (
    accumulate_fisher, refresh_lambda, anchor, ewc_penalty, fisher_loss,
)


def _p(*a):
    print(*a, flush=True)


def make_task(seed, n=512):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, INPUT_DIM - 6, generator=g)          # raw percept dim (74)
    W = torch.randn(INPUT_DIM - 6, N_ACTION, generator=g)
    Y = (X @ W).argmax(dim=1)                               # fixed target labels
    return X, Y


@torch.no_grad()
def acc(sub, X, Y):
    return (sub(_solo(X)).argmax(dim=1) == Y).float().mean().item()


def train_task(sub, X, Y, *, epochs, lr, ewc_strength=0.0, batch=64):
    opt = torch.optim.Adam(sub.trainable_tensors(), lr=lr)
    a = sub.arena
    ce = torch.nn.functional.cross_entropy
    n = X.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = ce(sub(_solo(X[idx])), Y[idx])
            if ewc_strength > 0:
                loss = loss + ewc_strength * ewc_penalty(a)
            loss.backward()
            sub.zero_dormant_grads()
            opt.step()


def fisher_pass(sub, X, Y, batches=15, batch=64):
    """Estimate Fisher at the converged plateau (the canonical EWC protocol), roll up
    to node_lambda, then snapshot the anchor. Uses the MODEL-DISTRIBUTION target
    (fisher_loss: y ~ softmax) — the true-label Fisher vanishes at convergence and pins
    λ at the floor (Y is unused, kept for signature symmetry)."""
    a = sub.arena
    g = torch.Generator().manual_seed(7)
    for _ in range(batches):
        idx = torch.randint(0, X.shape[0], (batch,), generator=g)
        a.bias.grad = None; a.edge_weight.grad = None
        fisher_loss(sub(_solo(X[idx]))).backward()          # model-dist Fisher target
        accumulate_fisher(a)
    refresh_lambda(a)                                       # λ_i = row-sum Fisher (+floor)
    anchor(a)


def run(ewc_strength):
    sub = build_mirror(0, n_mirror=8)
    XA, YA = make_task(1)
    XB, YB = make_task(2)
    train_task(sub, XA, YA, epochs=40, lr=3e-3)
    a_after_A = acc(sub, XA, YA)
    fisher_pass(sub, XA, YA)                                # Fisher → λ → anchor at plateau
    lam = sub.arena.node_lambda
    train_task(sub, XB, YB, epochs=40, lr=3e-3, ewc_strength=ewc_strength)
    return a_after_A, acc(sub, XA, YA), acc(sub, XB, YB), float(lam.max())


def main():
    _p("=== λ-EWC forgetting smoke (v2 substrate; canonical per-node row-sum λ) ===\n")
    base_A, plain_A, plain_B, _ = run(ewc_strength=0.0)
    _p(f"  task A acc after learning A   : {base_A:.3f}")
    _p(f"  PLAIN (no λ) — A after B      : {plain_A:.3f}   (B: {plain_B:.3f})")
    for s in (1.0, 10.0, 100.0):
        _, ewc_A, ewc_B, lmax = run(ewc_strength=s)
        verdict = "retains A" if (base_A - ewc_A) < (base_A - plain_A) - 0.02 else "—"
        _p(f"  λ-EWC s={s:<5g} — A after B   : {ewc_A:.3f}   (B: {ewc_B:.3f})  "
           f"[λmax={lmax:.2e}]  {verdict}")
    _p("\n  mechanism live if a strength retains A above plain (trading some B). "
       "Orthogonal-random tasks are EWC's worst case; the real test is the shared-base world.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
