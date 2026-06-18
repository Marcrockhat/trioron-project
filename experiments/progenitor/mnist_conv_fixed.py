"""Substrate-wired convolution with a FIXED structured (Gabor) kernel vs raw
(s042, Rocky: "wire the receptors — single pixel receptor to the lens receptor,
else there's no convolution").

The earlier "lens" was z.mean() over a patch = the all-ones kernel = POOLING, with
no weights and no wiring. This builds the real thing: pixel receptors (784 input
cells) WIRED to conv cells through shared weighted edges (conv.forward_batch +
lineage_root tie — one kernel tiled across all positions). The kernel is a FIXED
Gabor bank (gradient-free, the phasor-optics spirit): per orientation a cos/sin
pair, combined by MODULUS sqrt(c^2+s^2) = oriented energy (the proper non-mean
nonlinearity, the 1st-order scattering coefficient), then spatially pooled.

Controls, all through the SAME wiring so only the kernel differs:
  raw pixels        784-d                       (the baseline the lens must beat)
  conv MEAN kernel  all-ones (= the old lens, pooling)  — should ~reproduce 0.76
  conv GABOR fixed  oriented-energy bank                — the real convolution

Back-ends: nearest centroid + real ManifoldArchive full-cov Mahalanobis.

Run:  python3 experiments/progenitor/mnist_conv_fixed.py
"""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F

from trioron.core.arena import Arena
from trioron.core.envelope import Envelope
from trioron.core.epigenome import CONV
from trioron.phenotype import conv
from trioron.legacy.donorkit.datasets import DatasetBundle
from experiments.progenitor.conv_proposer import tile_patches, _bucket_for
from experiments.progenitor.taxonomy_manifold import classify_archive

GRID = 28
K = 9
STRIDE = 2
HG = (GRID - K) // STRIDE + 1            # conv output grid side
PER_TRAIN = 700                          # > descriptor dim so Mahalanobis conditions
PER_TEST = 700
ORIENTS = [0, 45, 90, 135]
FREQS = [0.16, 0.28]


def balanced(X, y, per_class, seed):
    g = torch.Generator().manual_seed(seed)
    idx = [torch.nonzero(y == c, as_tuple=False).squeeze(1)[
        torch.randperm(int((y == c).sum()), generator=g)[:per_class]] for c in range(10)]
    return X[torch.cat(idx)], y[torch.cat(idx)]


def gabor(theta_deg, freq, phase):
    """K x K Gabor, zero-DC. Index [i(row), j(col)] -> flatten row-major matches
    tile_patches tap order (r+i)*W + (c+j)."""
    t = math.radians(theta_deg)
    c = np.arange(K) - (K - 1) / 2
    xx, yy = np.meshgrid(c, c)                       # xx[i,j]=col j, yy[i,j]=row i
    xr = xx * math.cos(t) + yy * math.sin(t)
    yr = -xx * math.sin(t) + yy * math.cos(t)
    sigma = K / 4.0
    g = np.exp(-(xr ** 2 + yr ** 2) / (2 * sigma ** 2)) * np.cos(2 * math.pi * freq * xr + phase)
    g = g - g.mean()
    g = g / (np.linalg.norm(g) + 1e-9)
    return torch.tensor(g.reshape(-1), dtype=torch.float32)


def spawn_fixed_cohort(arena, in_cells, patches, kernel):
    """One CONV cell per patch, all tied to cohort[0]'s lineage_root; cohort[0]'s
    kernel set to the FIXED `kernel` (len == K*K). conv.forward_batch tiles it."""
    ids = arena.alloc(len(patches)).tolist()
    for cc in ids:
        arena.epigenome[cc] = 1 << CONV
        arena.phenotype_cache[cc] = CONV
        arena.lineage_root[cc] = ids[0]
    for i, (cell, patch) in enumerate(zip(ids, patches)):
        src = torch.tensor([in_cells[p] for p in patch], dtype=torch.int32)
        dst = torch.full((len(patch),), cell, dtype=torch.int32)
        arena.add_edges(src, dst, kernel.clone() if i == 0 else None)
    return _bucket_for(arena, ids)


def conv_map(arena, in_cells, bucket, X):
    """Run a tied cohort -> [N, HG*HG] feature map."""
    A = torch.zeros(len(X), arena.capacity)
    A[:, in_cells] = X
    return conv.forward_batch(A, bucket, arena)


def pool(mapNP, out=5):
    m = mapNP.view(-1, 1, HG, HG)
    return F.adaptive_avg_pool2d(m, out).reshape(len(mapNP), -1)


def build_gabor_bank(arena, in_cells, patches):
    """Spawn the cos/sin cohort pairs ONCE; reuse for both splits."""
    bank = []
    for th in ORIENTS:
        for fr in FREQS:
            bc = spawn_fixed_cohort(arena, in_cells, patches, gabor(th, fr, 0.0))
            bs = spawn_fixed_cohort(arena, in_cells, patches, gabor(th, fr, -math.pi / 2))
            bank.append((bc, bs))
    return bank


def gabor_descriptor(arena, in_cells, bank, X):
    feats = []
    for bc, bs in bank:
        mc = conv_map(arena, in_cells, bc, X)
        ms = conv_map(arena, in_cells, bs, X)
        feats.append(pool((mc ** 2 + ms ** 2).sqrt()))    # MODULUS = oriented energy
    return torch.cat(feats, 1)


def centroid_acc(Dtr, ytr, Dte, yte):
    cents = torch.stack([Dtr[ytr == c].mean(0) for c in range(10)])
    return float((torch.cdist(Dte, cents).argmin(1) == yte).float().mean())


def report(name, Dtr, Dte, ytr, yte):
    ac = centroid_acc(Dtr, ytr, Dte, yte)
    if Dtr.shape[1] < PER_TRAIN:
        am = float((classify_archive(Dtr.float(), ytr, Dte.float(), 10, full_cov=True) == yte).float().mean())
        m = f"{am:>12.3f}"
    else:
        m = f"{'(dim>n)':>12}"
    print(f"  {name:<28} {Dtr.shape[1]:>5} {ac:>9.3f} {m}")


def main():
    b = DatasetBundle(["mnist"])
    Xtr, ytr = b.task_view("mnist", list(range(10)), list(range(10)), split="train").all_examples()
    Xte, yte = b.task_view("mnist", list(range(10)), list(range(10)), split="test").all_examples()
    Xtr, ytr = balanced(Xtr, ytr, PER_TRAIN, 0)
    Xte, yte = balanced(Xte, yte, PER_TEST, 1)
    print(f"MNIST {GRID}x{GRID}, WIRED conv (conv.forward_batch + lineage_root tie), "
          f"K={K} stride={STRIDE} -> {HG}x{HG}")
    print(f"  {len(Xtr)} train / {len(Xte)} test | {len(ORIENTS)} orient x {len(FREQS)} freq "
          f"| chance 0.10\n")
    print(f"  {'front-end':<28} {'dim':>5} {'centroid':>9} {'Mahalanobis':>12}")

    patches, _ = tile_patches(GRID, GRID, K, STRIDE)

    # raw baseline
    report("raw pixels", Xtr, Xte, ytr, yte)

    # control: all-ones (mean) kernel through the SAME wiring (= the old lens)
    a0 = Arena(Envelope(), capacity=GRID * GRID + len(patches) + 64)
    a0.alloc(GRID * GRID); inc0 = list(range(GRID * GRID))
    ones = torch.ones(K * K) / (K * K)
    bk0 = spawn_fixed_cohort(a0, inc0, patches, ones)
    report("conv MEAN kernel (old lens)", pool(conv_map(a0, inc0, bk0, Xtr)),
           pool(conv_map(a0, inc0, bk0, Xte)), ytr, yte)

    # the real thing: fixed Gabor bank, oriented-energy modulus (bank built once)
    nco = 2 * len(ORIENTS) * len(FREQS)
    a = Arena(Envelope(), capacity=GRID * GRID + nco * len(patches) + 128)
    a.alloc(GRID * GRID); inc = list(range(GRID * GRID))
    bank = build_gabor_bank(a, inc, patches)
    report("conv GABOR fixed (modulus)", gabor_descriptor(a, inc, bank, Xtr),
           gabor_descriptor(a, inc, bank, Xte), ytr, yte)


if __name__ == "__main__":
    main()
