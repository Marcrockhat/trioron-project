"""Gradient-free DEEP scattering (conv->pool->conv) on MNIST (s043, NEXT#1a).

Builds on the s042 win (`mnist_conv_fixed.py`): a substrate-wired FIXED-Gabor
convolution beats raw pixels (centroid 0.907 / Mahalanobis 0.980 vs raw 0.815).
That win is SINGLE LAYER. This adds a second fixed-Gabor convolution on top of
the first layer's oriented-energy maps -- the s041 conv->pool->conv form, but
gradient-free (no Adam, fixed kernels). Structurally this IS the Mallat
scattering transform: the descriptor is S1 (+) S2, where

  U1 = |x * psi_l1|                      first-order envelope (modulus map)
  S1 = pool(U1)                          first-order coefficients (the s042 win)
  U2 = |U1 * psi_l2|                     second-order envelope
  S2 = pool(U2)                          second-order coefficients (NEW)

L2 is DEPTHWISE: each L1 channel's envelope U1 is independently re-convolved by
a fresh Gabor bank (this is the scattering transform; cross-channel fixed-Gabor
mixing is unprincipled). L2 reads the UN-pooled U1 (the propagated signal);
S1 is the pooled U1 -- exactly scattering's split. Every conv runs through the
real substrate `conv.forward_batch` (weight-tie by lineage_root + tap).

The falsifiable question: does the 2nd scattering layer LIFT the single-layer
0.907/0.980, or is it dead weight on centered MNIST?

Run:  python3 experiments/progenitor/mnist_scatter_deep.py
"""
from __future__ import annotations

import math
import os
import time
import torch
import torch.nn.functional as F

from trioron.core.arena import Arena
from trioron.core.envelope import Envelope
from trioron.phenotype import conv
from trioron.legacy.donorkit.datasets import DatasetBundle
from experiments.progenitor.conv_proposer import tile_patches
from experiments.progenitor.taxonomy_manifold import classify_archive
from experiments.progenitor.mnist_conv_fixed import (
    balanced, spawn_fixed_cohort, conv_map, centroid_acc,
)
import numpy as np


def gabor(k, theta_deg, freq, phase):
    """k x k Gabor, zero-DC, unit-norm. Row-major flatten matches tile_patches
    tap order (r+i)*W + (c+j). (K-parameterized version of the s042 helper.)"""
    t = math.radians(theta_deg)
    c = np.arange(k) - (k - 1) / 2
    xx, yy = np.meshgrid(c, c)
    xr = xx * math.cos(t) + yy * math.sin(t)
    yr = -xx * math.sin(t) + yy * math.cos(t)
    sigma = k / 4.0
    g = np.exp(-(xr ** 2 + yr ** 2) / (2 * sigma ** 2)) * np.cos(2 * math.pi * freq * xr + phase)
    g = g - g.mean()
    g = g / (np.linalg.norm(g) + 1e-9)
    return torch.tensor(g.reshape(-1), dtype=torch.float32)

GRID = 28
PER_TRAIN = 700
PER_TEST = 700

# L1 Gabor bank (same as the s042 win)
K1, S1 = 9, 2
H1 = (GRID - K1) // S1 + 1                # 10
ORIENTS1, FREQS1 = [0, 45, 90, 135], [0.16, 0.28]
S1_POOL = 5                               # S1 descriptor pool side

# L2 Gabor bank (depthwise, over each U1 channel)
K2, S2 = 5, 1
H2 = (H1 - K2) // S2 + 1                  # 6
ORIENTS2, FREQS2 = [0, 45, 90, 135], [0.16, 0.28]
S2_POOL = 2                              # S2 descriptor pool side (keep dim < PER_TRAIN)


def pool_to(maps, side_in, out):
    m = maps.view(-1, 1, side_in, side_in)
    return F.adaptive_avg_pool2d(m, out).reshape(len(maps), -1)


def build_bank(arena, in_cells, patches, k, orients, freqs):
    """Spawn cos/sin cohort pairs for a fixed Gabor bank; reuse across splits."""
    bank = []
    for th in orients:
        for fr in freqs:
            bc = spawn_fixed_cohort(arena, in_cells, patches, gabor(k, th, fr, 0.0))
            bs = spawn_fixed_cohort(arena, in_cells, patches, gabor(k, th, fr, -math.pi / 2))
            bank.append((bc, bs))
    return bank


def modulus_maps(arena, in_cells, bank, X):
    """[N, n_in] -> list of [N, H*H] oriented-energy (modulus) maps, one per
    (orient, freq) pair."""
    maps = []
    for bc, bs in bank:
        mc = conv_map(arena, in_cells, bc, X)
        ms = conv_map(arena, in_cells, bs, X)
        maps.append((mc ** 2 + ms ** 2).sqrt())          # MODULUS = oriented energy
    return maps


def scatter(a1, inc1, bank1, a2, inc2, bank2, X):
    """Return (S1, S2): first- and second-order scattering descriptors."""
    U1 = modulus_maps(a1, inc1, bank1, X)                 # n_l1 x [N, H1*H1]
    S1 = torch.cat([pool_to(u, H1, S1_POOL) for u in U1], 1)

    s2_chunks = []
    for u in U1:                                          # depthwise: per L1 channel
        U2 = modulus_maps(a2, inc2, bank2, u)             # n_l2 x [N, H2*H2]
        s2_chunks.extend(pool_to(v, H2, S2_POOL) for v in U2)
    S2 = torch.cat(s2_chunks, 1)
    return S1, S2


def acc(name, Dtr, Dte, ytr, yte):
    """centroid + (if dim<n) Mahalanobis, as floats."""
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
    print(f"MNIST {GRID}x{GRID}, gradient-free DEEP scattering (S1 (+) S2)")
    print(f"  L1: K{K1} s{S1} -> {H1}x{H1}, {n_l1} ch, pool {S1_POOL}  |  "
          f"L2: K{K2} s{S2} -> {H2}x{H2}, {n_l2} ch depthwise, pool {S2_POOL}")
    print(f"  {2 * PER_TRAIN}.. balanced/seed | n={len(seeds)} seeds | chance 0.10\n")

    # build banks once (kernels are fixed; reused across seeds & splits)
    p1, _ = tile_patches(GRID, GRID, K1, S1)
    a1 = Arena(Envelope(), capacity=GRID * GRID + 2 * n_l1 * len(p1) + 128)
    a1.alloc(GRID * GRID); inc1 = list(range(GRID * GRID))
    bank1 = build_bank(a1, inc1, p1, K1, ORIENTS1, FREQS1)

    p2, _ = tile_patches(H1, H1, K2, S2)
    a2 = Arena(Envelope(), capacity=H1 * H1 + 2 * n_l2 * len(p2) + 128)
    a2.alloc(H1 * H1); inc2 = list(range(H1 * H1))
    bank2 = build_bank(a2, inc2, p2, K2, ORIENTS2, FREQS2)

    arms = ["raw pixels", "S1 only (s042 win)", "S2 only (2nd order)", "S1 (+) S2 (DEPTH)"]
    rows = {k: [] for k in arms}                          # arm -> list of (cen, maha)
    for sd in seeds:
        Xtr, ytr = balanced(Xtr_all, ytr_all, PER_TRAIN, sd)
        Xte, yte = balanced(Xte_all, yte_all, PER_TEST, sd + 1000)
        S1tr, S2tr = scatter(a1, inc1, bank1, a2, inc2, bank2, Xtr)
        S1te, S2te = scatter(a1, inc1, bank1, a2, inc2, bank2, Xte)
        rows["raw pixels"].append(acc("", Xtr, Xte, ytr, yte))
        rows["S1 only (s042 win)"].append(acc("", S1tr, S1te, ytr, yte))
        rows["S2 only (2nd order)"].append(acc("", S2tr, S2te, ytr, yte))
        rows["S1 (+) S2 (DEPTH)"].append(acc("", torch.cat([S1tr, S2tr], 1),
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
        d = torch.tensor([rows["S1 (+) S2 (DEPTH)"][i][0] - rows["S1 only (s042 win)"][i][0]
                          for i in range(len(seeds))])
        print(f"\n  DEPTH lift (centroid) S1+S2 - S1 = {d.mean():+.3f} ±{d.std(0):.3f} "
              f"(per-seed: {', '.join(f'{x:+.3f}' for x in d)})")

    print(f"\n[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
