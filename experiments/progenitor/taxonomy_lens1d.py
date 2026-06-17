"""Apply the dimensionality-adaptive lens PROPERLY to the 1D taxonomy: 1xk
patches (Rocky: "shouldn't it be 1x2 or 1x3?"), s042.

taxonomy_receptor.py used a per-FEATURE phasor (24-d) and skipped the lens's
patch grouping. But the lens is dimensionality-adaptive (1D -> 1xk), so on the
12-feature taxonomy the faithful application is a 1x2 or 1x3 hypercube patch.
The patch-mean phasor also COMPRESSES: adjacent features fold into one complex
response (re,im), shrinking the descriptor.

Descriptors (per-feature global phase frame underneath all):
  per-feature  : cos/sin of each feature, no grouping        24-d  (baseline)
  lens 1x2 s1  : 1x2 patches, stride 1 (overlapping)         22-d
  lens 1x2 s2  : 1x2 patches, stride 2 (non-overlap)         12-d  (2x compress)
  lens 1x3 s1  : 1x3 patches, stride 1                       20-d
  lens 1x3 s3  : 1x3 patches, stride 3 (non-overlap)          8-d  (3x compress)

Back-ends: numpy nearest-centroid (weak) and the real ManifoldArchive full-cov
Mahalanobis (strong). Reference: raw 12-d gives centroid 0.365 / Maha 0.901;
per-feature phasor 0.471 / 0.876; Bayes 0.937.

Run:  python3 experiments/progenitor/taxonomy_lens1d.py
"""
from __future__ import annotations

import math
import numpy as np
import torch

from experiments.progenitor.fingerprint_lens import adaptive_patches, lens_descriptor
from experiments.progenitor.data_hard import make_split, bayes_accuracy
from experiments.progenitor.taxonomy_manifold import classify_archive

N_QUANTA = 1000


def per_feature_q(X, lo=None, hi=None):
    if lo is None:
        lo, hi = X.amin(0), X.amax(0)
    span = (hi - lo).clamp_min(1e-9)
    return torch.round(N_QUANTA * (X - lo) / span).to(torch.int64), lo, hi


def lens_desc(Q, patches):
    return np.stack([lens_descriptor(Q[i], patches) for i in range(len(Q))])


def per_feature_desc(Q):
    th = 2 * math.pi * Q.float() / N_QUANTA
    return torch.cat([th.cos(), th.sin()], dim=1).numpy()


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
    print(f"hard taxonomy: K={K}, D={tr.x.shape[1]}, M=3 | Bayes {bayes_accuracy(te):.3f} "
          f"| raw: centroid 0.365 / Maha 0.901\n")

    D = tr.x.shape[1]
    variants = [
        ("per-feature (24-d)", per_feature_desc(Qtr), per_feature_desc(Qte)),
    ]
    for k, stride, name in [(2, 1, "lens 1x2 s1"), (2, 2, "lens 1x2 s2"),
                            (3, 1, "lens 1x3 s1"), (3, 3, "lens 1x3 s3")]:
        P = adaptive_patches((D,), k=k, stride=stride)
        variants.append((f"{name} ({2*len(P)}-d, {len(P)} patches)",
                         lens_desc(Qtr, P), lens_desc(Qte, P)))

    print(f"  {'descriptor':<30} {'dim':>4} {'centroid':>9} {'Mahalanobis':>12}")
    for name, Dtr, Dte in variants:
        ac = centroid_acc(Dtr, ytr_np, Dte, yte_np, K)
        Tt, Te = torch.from_numpy(Dtr).float(), torch.from_numpy(Dte).float()
        am = float((classify_archive(Tt, tr.y, Te, K, full_cov=True) == te.y).float().mean())
        print(f"  {name:<30} {Dtr.shape[1]:>4} {ac:>9.3f} {am:>12.3f}")


if __name__ == "__main__":
    main()
