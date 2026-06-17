"""Fold the 12 feature-quanta into a small 2D image (3x4 / 4x3 / 2x6) and run
the 2D lens (Rocky's literal fold, s042).

Correcting taxonomy_fold2d, which expanded each feature into a 12-bin value axis
(12x12 image, dim blow-up). The simple fold: the 12 per-feature quanta ARE the
pixels, laid out as a 3x4 image; the 2x2 lens reads neighboring FEATURES. Tiny
descriptor (6 patches -> 12-d), so the back-end stays well-conditioned.

Locality lives on the feature axis, which is arbitrary unless reordered, so each
fold is tried raw and with a greedy |corr| reorder (correlated features placed
row-major-adjacent).

Reference: raw 0.365/0.901, per-feature phasor 0.471/0.876, Bayes 0.937.

Run:  python3 experiments/progenitor/taxonomy_fold34.py
"""
from __future__ import annotations

import math
import numpy as np
import torch

from experiments.progenitor.fingerprint_lens import adaptive_patches, lens_descriptor
from experiments.progenitor.data_hard import make_split, bayes_accuracy
from experiments.progenitor.taxonomy_manifold import classify_archive
from experiments.progenitor.taxonomy_fold2d import corr_order

N_QUANTA = 1000


def per_feature_q(X, lo=None, hi=None):
    if lo is None:
        lo, hi = X.amin(0), X.amax(0)
    span = (hi - lo).clamp_min(1e-9)
    return torch.round(N_QUANTA * (X - lo) / span).to(torch.int64), lo, hi


def lens_fold(Q, patches, order=None):
    if order is not None:
        Q = Q[:, order]
    return np.stack([lens_descriptor(Q[i], patches) for i in range(len(Q))])


def per_feature_desc(Q):
    th = 2 * math.pi * Q.float() / N_QUANTA
    return torch.cat([th.cos(), th.sin()], dim=1).numpy()


def centroid_acc(Dtr, ytr, Dte, yte, K):
    cents = np.stack([Dtr[ytr == c].mean(0) for c in range(K)])
    pred = ((Dte[:, None, :] - cents[None]) ** 2).sum(-1).argmin(1)
    return float((pred == yte).mean())


def run(name, Dtr, Dte, ytr_np, yte_np, K, ytr_t, yte_t):
    ac = centroid_acc(Dtr, ytr_np, Dte, yte_np, K)
    Tt, Te = torch.from_numpy(Dtr).float(), torch.from_numpy(Dte).float()
    am = float((classify_archive(Tt, ytr_t, Te, K, full_cov=True) == yte_t).float().mean())
    print(f"  {name:<30} {Dtr.shape[1]:>4} {ac:>9.3f} {am:>12.3f}")


def main():
    tr, te = make_split()
    K = len(te.names)
    ytr_np, yte_np = tr.y.numpy(), te.y.numpy()
    Qtr, lo, hi = per_feature_q(tr.x)
    Qte, _, _ = per_feature_q(te.x, lo, hi)
    order = corr_order(tr.x)
    print(f"hard taxonomy: K={K}, D=12, M=3 | Bayes {bayes_accuracy(te):.3f} "
          f"| raw 0.365/0.901, per-feature 0.471/0.876\n")
    print(f"  {'descriptor':<30} {'dim':>4} {'centroid':>9} {'Mahalanobis':>12}")

    run("per-feature phasor", per_feature_desc(Qtr), per_feature_desc(Qte),
        ytr_np, yte_np, K, tr.y, te.y)

    for shape in [(3, 4), (4, 3), (2, 6), (6, 2)]:
        P = adaptive_patches(shape, k=2)
        for od, olbl in [(None, ""), (order, " +reorder")]:
            run(f"fold {shape[0]}x{shape[1]} lens 2x2{olbl}",
                lens_fold(Qtr, P, od), lens_fold(Qte, P, od),
                ytr_np, yte_np, K, tr.y, te.y)


if __name__ == "__main__":
    main()
