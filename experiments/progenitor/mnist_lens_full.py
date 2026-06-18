"""MNIST lens vs raw at FULL 28x28 — no pre-downsample (s042, Rocky's catch).

mnist_lens.py avg-pooled to 14x14 BEFORE the lens, so the 2x2 lens was a second
pooling on an already-pooled image (and raw was the pooled 14x14). Confounded.
This runs the lens at native 28x28: stride-1 (27x27=729 patches) preserves
resolution; stride-2 (14x14=196 patches) halves it. Raw baseline is full 784-d.
Vectorized lens so full-res is fast.

  raw pixels        784-d
  lens 2x2 s1       1458-d  (no resolution loss)
  lens 2x2 s2       392-d
  lens 2x2 s1 sweep w<=2  2916-d (centroid only)

Back-ends: nearest centroid (clean, no conditioning limit) + full-cov Mahalanobis
where dim < n_per_class.

Run:  python3 experiments/progenitor/mnist_lens_full.py
"""
from __future__ import annotations

import math
import numpy as np
import torch

from trioron.legacy.donorkit.datasets import DatasetBundle
from experiments.progenitor.fingerprint_lens import adaptive_patches
from experiments.progenitor.taxonomy_manifold import classify_archive

N_QUANTA = 1000
PER_TRAIN = 1000      # > 784 so raw Mahalanobis is conditioned
PER_TEST = 800
GRID = 28


def balanced(X, y, per_class, seed):
    g = torch.Generator().manual_seed(seed)
    idx = [torch.nonzero(y == c, as_tuple=False).squeeze(1)[
        torch.randperm(int((y == c).sum()), generator=g)[:per_class]] for c in range(10)]
    idx = torch.cat(idx)
    return X[idx], y[idx]


def quantize_per_sample(X):
    lo = torch.minimum(X.amin(-1, keepdim=True), X.new_zeros(1))
    hi = X.amax(-1, keepdim=True)
    return torch.round(N_QUANTA * (X - lo) / (hi - lo).clamp_min(1e-9))


def lens_vec(X, patch_idx, carriers):
    """Vectorized: theta -> per-patch mean phasor over carriers -> [re,im]."""
    th = 2 * math.pi * quantize_per_sample(X) / N_QUANTA      # (N, D)
    tp = th[:, patch_idx]                                     # (N, P, k2)
    feats = []
    for w in carriers:
        z = torch.exp(1j * (w * tp)).mean(-1)                # (N, P) complex
        feats += [z.real, z.imag]
    return torch.cat(feats, dim=1)                           # (N, 2P*ncar)


def centroid_acc(Dtr, ytr, Dte, yte):
    cents = torch.stack([Dtr[ytr == c].mean(0) for c in range(10)])
    pred = torch.cdist(Dte, cents).argmin(1)
    return float((pred == yte).float().mean())


def main():
    b = DatasetBundle(["mnist"])
    Xtr, ytr = b.task_view("mnist", list(range(10)), list(range(10)), split="train").all_examples()
    Xte, yte = b.task_view("mnist", list(range(10)), list(range(10)), split="test").all_examples()
    Xtr, ytr = balanced(Xtr, ytr, PER_TRAIN, 0)
    Xte, yte = balanced(Xte, yte, PER_TEST, 1)
    print(f"MNIST {GRID}x{GRID} (no downsample): {len(Xtr)} train / {len(Xte)} test | chance 0.10\n")

    pi1 = torch.tensor(adaptive_patches((GRID, GRID), k=2, stride=1))
    pi2 = torch.tensor(adaptive_patches((GRID, GRID), k=2, stride=2))

    print(f"  {'front-end':<26} {'dim':>5} {'centroid':>9} {'Mahalanobis':>12}")

    def row(name, Dtr, Dte):
        ac = centroid_acc(Dtr, ytr, Dte, yte)
        if Dtr.shape[1] < PER_TRAIN:
            am = float((classify_archive(Dtr.float(), ytr, Dte.float(), 10, full_cov=True) == yte).float().mean())
            mstr = f"{am:>12.3f}"
        else:
            mstr = f"{'(dim>n)':>12}"
        print(f"  {name:<26} {Dtr.shape[1]:>5} {ac:>9.3f} {mstr}")

    row("raw pixels", Xtr, Xte)

    # patch-SIZE sweep — 2x2 is too small for 28x28 (stroke scale ~2-4px)
    for k, stride in [(2, 1), (3, 2), (4, 2), (4, 4), (7, 3), (7, 7)]:
        pi = torch.tensor(adaptive_patches((GRID, GRID), k=k, stride=stride))
        row(f"lens {k}x{k} s{stride} ({len(pi)}p)",
            lens_vec(Xtr, pi, (1.,)), lens_vec(Xte, pi, (1.,)))

    # multi-scale concat (crude scattering): k=2 + k=4 + k=7
    def multiscale(X):
        parts = [lens_vec(X, torch.tensor(adaptive_patches((GRID, GRID), k=k, stride=s)), (1.,))
                 for k, s in [(2, 2), (4, 4), (7, 7)]]
        return torch.cat(parts, dim=1)
    row("lens multiscale 2+4+7", multiscale(Xtr), multiscale(Xte))


if __name__ == "__main__":
    main()
