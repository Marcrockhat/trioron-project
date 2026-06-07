"""Class-incremental CL on the ORIGINAL 6 species, weight-only (1 feature).

Comparison point for how accuracy changed vs the 10-species / 4-feature taxonomy bench
(cl_incremental.py, machinery final 0.847). Same validated workhorse: manifold replay on,
credit-lock + λ-EWC off.

Run: python3 -m experiments.growth_exercise.cl_weight_only
"""
from __future__ import annotations

import torch

from trioron.core import Envelope, construct
from trioron.bases import minimal
from trioron.phenotype import default_dispatch_table
from trioron.learning.manifold import ManifoldArchive

from experiments.growth_exercise.chicken_goat import make_animals, SPECIES

SPECIES6 = ["chicken", "cat", "dog", "goat", "cow", "elephant"]
STEPS, LR, N_REPLAY = 250, 0.05, 64


def seen_bayes(test, n_seen):
    mask = test.y < n_seen
    logp = torch.stack([SPECIES[n].log_prob(test.kg[mask]) for n in SPECIES6[:n_seen]], dim=1)
    return float((logp.argmax(1) == test.y[mask]).float().mean())


def eval_seen(sub, test, n_seen):
    mask = test.y < n_seen
    with torch.no_grad():
        pred = sub(test.x[mask]).argmax(1)
    return float((pred == test.y[mask]).float().mean())


def run(train, test, replay=True):
    sub = construct(base=minimal(1, 6), envelope=Envelope(),
                    dispatch_table=default_dispatch_table(), capacity=64)
    sub.prepare_training(); sub.compile()
    archive = ManifoldArchive(sub.arena, full_cov=True) if replay else None
    opt = torch.optim.Adam(sub.trainable_tensors(), lr=LR)
    curve = []
    for t in range(6):
        Xc = train.x[train.y == t]; yc = train.y[train.y == t]
        for _ in range(STEPS):
            X, y = Xc, yc
            if replay and t > 0:
                rx, ry = [], []
                for c in range(t):
                    a = archive.get(c)
                    if a is not None:
                        rx.append(a.sample_full(N_REPLAY)); ry.append(torch.full((N_REPLAY,), c))
                if rx:
                    X = torch.cat([Xc] + rx); y = torch.cat([yc] + ry)
            opt.zero_grad()
            torch.nn.functional.cross_entropy(sub(X), y).backward()
            opt.step()
        if replay:
            archive.update_class(t, Xc)
        sub.end_task()
        curve.append(eval_seen(sub, test, t + 1))
    return curve


def main():
    tr = make_animals(SPECIES6, n_per_class=128, seed=0, log=True)
    te = make_animals(SPECIES6, n_per_class=128, seed=1, log=True)
    ceil = [seen_bayes(te, t + 1) for t in range(6)]
    naive = run(tr, te, replay=False)
    mach = run(tr, te, replay=True)
    print("6 species, WEIGHT ONLY (1 feature), class-incremental\n")
    print(f"{'task':>4} {'+species':>9} {'naive':>7} {'machinery':>10} {'Bayes':>7}")
    for t, n in enumerate(SPECIES6):
        print(f"{t:>4} {n:>9} {naive[t]:>7.3f} {mach[t]:>10.3f} {ceil[t]:>7.3f}")
    print(f"\nfinal: naive {naive[-1]:.3f} | machinery {mach[-1]:.3f} | ceiling {ceil[-1]:.3f}")


if __name__ == "__main__":
    main()
