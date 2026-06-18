"""Per-feature multi-carrier lens: focus the filter on ONE quantum, read it at
several carriers (Rocky, s042 — "2x2 filters focused on one quantum; trade
compute for adaptive capacity").

The spatial lens averaged 4 neighboring features per patch and lost per-feature
info (taxonomy_lens1d/fold34). The fix is the opposite of grouping: keep each
feature separate and give it MORE filters — read its single quantum at carriers
w=1,2,3,... -> [cos(w*theta), sin(w*theta)]. This is a per-feature Fourier-feature
expansion: dedicated capacity per feature, no blending. More carriers = more
compute and more capacity (the trade).

Sweeps carrier count; reports descriptor dim, weak back-end (nearest centroid)
and strong back-end (real ManifoldArchive full-cov Mahalanobis). Note: Mahalanobis
needs dim < n_per_class (128), so it degrades once carriers push the descriptor
wide — the trade is visible from both ends.

Reference: per-feature single carrier 0.471/0.876, raw 0.365/0.901, Bayes 0.937.

Run:  python3 experiments/progenitor/taxonomy_perfeat_sweep.py
"""
from __future__ import annotations

import math
import numpy as np
import torch

from experiments.progenitor.data_hard import make_split, bayes_accuracy
from experiments.progenitor.taxonomy_manifold import classify_archive

N_QUANTA = 1000


def per_feature_q(X, lo=None, hi=None):
    if lo is None:
        lo, hi = X.amin(0), X.amax(0)
    span = (hi - lo).clamp_min(1e-9)
    return torch.round(N_QUANTA * (X - lo) / span).to(torch.int64), lo, hi


def perfeat_sweep(Q, carriers):
    """Each feature's single quantum -> [cos(w*theta), sin(w*theta)] over carriers."""
    th = 2 * math.pi * Q.float() / N_QUANTA            # (N, F)
    feats = []
    for w in carriers:
        feats += [(w * th).cos(), (w * th).sin()]
    return torch.cat(feats, dim=1)                     # (N, F * 2 * |carriers|)


def centroid_acc(Dtr, ytr, Dte, yte, K):
    cents = np.stack([Dtr[ytr == c].mean(0) for c in range(K)])
    pred = ((Dte[:, None, :] - cents[None]) ** 2).sum(-1).argmin(1)
    return float((pred == yte).mean())


def main():
    tr, te = make_split()
    K = len(te.names)
    ytr_np, yte_np = tr.y.numpy(), te.y.numpy()
    Qtr, lo, hi = per_feature_q(tr.x)
    Qte, _, _ = per_feature_q(te.x, lo, hi)
    print(f"hard taxonomy: K={K}, D=12, M=3 | Bayes {bayes_accuracy(te):.3f} "
          f"| ref: per-feature(w=1) 0.471/0.876, raw 0.365/0.901\n")
    print(f"  {'carriers':<18} {'dim':>4} {'centroid':>9} {'Mahalanobis':>12}")

    for carriers in [[1], [1, 2], [1, 2, 3, 4], [1, 2, 3, 4, 6, 8],
                     [1, 2, 3, 4, 5, 6, 7, 8]]:
        Dtr = perfeat_sweep(Qtr, carriers)
        Dte = perfeat_sweep(Qte, carriers)
        ac = centroid_acc(Dtr.numpy(), ytr_np, Dte.numpy(), yte_np, K)
        am = float((classify_archive(Dtr, tr.y, Dte, K, full_cov=True) == te.y).float().mean())
        lbl = f"w<= {carriers[-1]} ({len(carriers)})"
        print(f"  {lbl:<18} {Dtr.shape[1]:>4} {ac:>9.3f} {am:>12.3f}")


if __name__ == "__main__":
    main()
