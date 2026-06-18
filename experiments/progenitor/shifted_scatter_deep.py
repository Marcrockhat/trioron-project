"""Gradient-free DEEP scattering (S1⊕S2) on SHIFTED-MNIST (s043, NEXT#1).

The centered-MNIST run (`mnist_scatter_deep.py`) saturates the Mahalanobis
back-end (0.980), so the conv's translation-equivariance is barely exercised —
raw template-matching already wins centered. This drops each 28x28 digit at a
RANDOM (dr,dc) offset on a CANVAS x CANVAS field (the `conv_depth_shifted_mnist.py`
task, gradient-free). The object MOVES, so a tabular centroid/Mahalanobis over
fixed pixel columns cannot follow it; a translation-equivariant scattering
front-end can. The s041 Adam conv-depth saw conv2 0.842 vs raw 0.493 here.

Two questions, both internally controlled:
  - does the conv-over-raw margin WIDEN now that geometry matters? (raw vs S1)
  - does the 2nd scattering layer still LIFT when the object moves? (S1 vs S1⊕S2)

Same front-end + pooling as the centered run, so the ONLY changed variable is
the data (moving digit on a bigger canvas). Every conv runs through the real
substrate `conv.forward_batch` (lineage_root weight-tie).

Run:  python3 experiments/progenitor/shifted_scatter_deep.py
Env:  CANVAS OFFSET SEEDS S1_POOL S2_POOL
"""
from __future__ import annotations

import os
import time
import torch

from trioron.core.arena import Arena
from trioron.core.envelope import Envelope
from trioron.legacy.donorkit.datasets import DatasetBundle
from experiments.progenitor.conv_proposer import tile_patches
from experiments.progenitor.taxonomy_manifold import classify_archive
from experiments.progenitor.mnist_conv_fixed import balanced, centroid_acc
from experiments.progenitor.mnist_scatter_deep import build_bank, modulus_maps, pool_to

CANVAS = int(os.environ.get("CANVAS", "36"))
OFFSET = int(os.environ.get("OFFSET", str(CANVAS - 28)))
PER_TRAIN = 700
PER_TEST = 700

# L1 / L2 Gabor banks — identical to the centered run
K1, S1 = 9, 2
H1 = (CANVAS - K1) // S1 + 1
ORIENTS1, FREQS1 = [0, 45, 90, 135], [0.16, 0.28]
K2, S2 = 5, 1
H2 = (H1 - K2) // S2 + 1
ORIENTS2, FREQS2 = [0, 45, 90, 135], [0.16, 0.28]
S1_POOL = int(os.environ.get("S1_POOL", "5"))
S2_POOL = int(os.environ.get("S2_POOL", "2"))


def shift_onto_canvas(imgs28, g):
    """[N,784] in [0,1] -> [N, CANVAS*CANVAS], each digit at a random integer
    offset in [0,OFFSET]^2 (the conv_depth_shifted_mnist placement)."""
    n = imgs28.shape[0]
    out = torch.zeros(n, CANVAS, CANVAS)
    src = imgs28.view(n, 28, 28)
    dr = torch.randint(0, OFFSET + 1, (n,), generator=g)
    dc = torch.randint(0, OFFSET + 1, (n,), generator=g)
    for i in range(n):
        out[i, dr[i]:dr[i] + 28, dc[i]:dc[i] + 28] = src[i]
    return out.view(n, CANVAS * CANVAS)


def scatter(a1, inc1, bank1, a2, inc2, bank2, X):
    """(S1, S2): first/second-order scattering, depthwise L2 on the un-pooled U1."""
    U1 = modulus_maps(a1, inc1, bank1, X)
    S1f = torch.cat([pool_to(u, H1, S1_POOL) for u in U1], 1)
    s2 = []
    for u in U1:
        U2 = modulus_maps(a2, inc2, bank2, u)
        s2.extend(pool_to(v, H2, S2_POOL) for v in U2)
    return S1f, torch.cat(s2, 1)


def acc(Dtr, Dte, ytr, yte):
    ac = centroid_acc(Dtr, ytr, Dte, yte)
    am = (float((classify_archive(Dtr.float(), ytr, Dte.float(), 10, full_cov=True) == yte).float().mean())
          if Dtr.shape[1] < PER_TRAIN else float("nan"))
    return ac, am


def main():
    t0 = time.time()
    seeds = [int(s) for s in os.environ.get("SEEDS", "0").split(",")]
    b = DatasetBundle(["mnist"])
    Xtr_all, ytr_all = b.task_view("mnist", list(range(10)), list(range(10)), split="train").all_examples()
    Xte_all, yte_all = b.task_view("mnist", list(range(10)), list(range(10)), split="test").all_examples()

    n_l1 = len(ORIENTS1) * len(FREQS1)
    n_l2 = len(ORIENTS2) * len(FREQS2)
    print(f"SHIFTED-MNIST: 28x28 digit at random offset in [0,{OFFSET}]^2 on a "
          f"{CANVAS}x{CANVAS} canvas, gradient-free DEEP scattering (S1⊕S2)")
    print(f"  L1: K{K1} s{S1} -> {H1}x{H1}, {n_l1} ch, pool {S1_POOL}  |  "
          f"L2: K{K2} s{S2} -> {H2}x{H2}, {n_l2} ch depthwise, pool {S2_POOL}")
    print(f"  {PER_TRAIN}/class balanced | n={len(seeds)} seeds | chance 0.10\n")

    # banks built once (fixed kernels reused across seeds & splits)
    p1, _ = tile_patches(CANVAS, CANVAS, K1, S1)
    a1 = Arena(Envelope(), capacity=CANVAS * CANVAS + 2 * n_l1 * len(p1) + 128)
    a1.alloc(CANVAS * CANVAS); inc1 = list(range(CANVAS * CANVAS))
    bank1 = build_bank(a1, inc1, p1, K1, ORIENTS1, FREQS1)

    p2, _ = tile_patches(H1, H1, K2, S2)
    a2 = Arena(Envelope(), capacity=H1 * H1 + 2 * n_l2 * len(p2) + 128)
    a2.alloc(H1 * H1); inc2 = list(range(H1 * H1))
    bank2 = build_bank(a2, inc2, p2, K2, ORIENTS2, FREQS2)

    arms = ["raw pixels", "S1 only", "S2 only", "S1⊕S2 (DEPTH)"]
    rows = {k: [] for k in arms}
    for sd in seeds:
        g = torch.Generator().manual_seed(sd)
        Xtr0, ytr = balanced(Xtr_all, ytr_all, PER_TRAIN, sd)
        Xte0, yte = balanced(Xte_all, yte_all, PER_TEST, sd + 1000)
        Xtr = shift_onto_canvas(Xtr0, g)
        Xte = shift_onto_canvas(Xte0, g)
        S1tr, S2tr = scatter(a1, inc1, bank1, a2, inc2, bank2, Xtr)
        S1te, S2te = scatter(a1, inc1, bank1, a2, inc2, bank2, Xte)
        rows["raw pixels"].append(acc(Xtr, Xte, ytr, yte))
        rows["S1 only"].append(acc(S1tr, S1te, ytr, yte))
        rows["S2 only"].append(acc(S2tr, S2te, ytr, yte))
        rows["S1⊕S2 (DEPTH)"].append(acc(torch.cat([S1tr, S2tr], 1),
                                          torch.cat([S1te, S2te], 1), ytr, yte))

    print(f"  {'front-end':<28} {'centroid':>16} {'Mahalanobis':>16}")
    for k in arms:
        cen = torch.tensor([r[0] for r in rows[k]])
        mah = torch.tensor([r[1] for r in rows[k]])
        cs = f"{cen.mean():.3f}" + (f" ±{cen.std(0):.3f}" if len(seeds) > 1 else "")
        ms = ("(dim>n)" if torch.isnan(mah).any() else
              f"{mah.mean():.3f}" + (f" ±{mah.std(0):.3f}" if len(seeds) > 1 else ""))
        print(f"  {k:<28} {cs:>16} {ms:>16}")
    if len(seeds) > 1:
        d = torch.tensor([rows["S1⊕S2 (DEPTH)"][i][0] - rows["S1 only"][i][0]
                          for i in range(len(seeds))])
        print(f"\n  DEPTH lift (centroid) S1⊕S2 - S1 = {d.mean():+.3f} ±{d.std(0):.3f} "
              f"(per-seed: {', '.join(f'{x:+.3f}' for x in d)})")

    print(f"\n[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
