"""Fold the quanta into a 2D (feature x value) image to mimic an image, then run
the 2D lens (Rocky, s042).

The lens's 2x2 patch grouping is a spatial-locality prior; flat tabular features
have no locality, so the lens hurt (taxonomy_lens1d). The fix is not a smaller
patch but a better SPACE: spatialize each feature's VALUE. Place-code each
feature's quantum q_f as a soft bump along a value axis -> a sample becomes a
(F x B) image. The value axis now HAS locality (adjacent bins = similar values),
which is exactly what the 2D lens reads. Pipeline mirrors the digit toy:
image -> quantize -> 2x2 lens -> descriptor -> manifold.

Optionally reorder features (greedy |corr| chain) so the FEATURE axis also has
locality, giving the 2D patches structure on both axes.

Reference: raw 0.365/0.901, per-feature phasor 0.471/0.876, Bayes 0.937.

Run:  python3 experiments/progenitor/taxonomy_fold2d.py
"""
from __future__ import annotations

import math
import numpy as np
import torch

from trioron.core.receptor import quantize
from experiments.progenitor.fingerprint_lens import adaptive_patches, lens_descriptor
from experiments.progenitor.data_hard import make_split, bayes_accuracy
from experiments.progenitor.taxonomy_manifold import classify_archive

N_QUANTA = 1000


def per_feature_norm(X, lo=None, hi=None):
    if lo is None:
        lo, hi = X.amin(0), X.amax(0)
    span = (hi - lo).clamp_min(1e-9)
    return ((X - lo) / span).clamp(0, 1), lo, hi


def fold_image(Xn, B, sigma=1.0, order=None):
    """Each feature f -> a soft bump at column (value) bin; stack F rows -> (N,F,B)."""
    if order is not None:
        Xn = Xn[:, order]
    N, F = Xn.shape
    bins = torch.arange(B).float()
    centre = Xn.unsqueeze(-1) * (B - 1)                 # (N, F, 1)
    img = torch.exp(-0.5 * ((bins[None, None, :] - centre) / sigma) ** 2)
    return img                                          # (N, F, B)


def lens_on_images(imgs, patches):
    desc = []
    for i in range(len(imgs)):
        q = quantize(imgs[i].reshape(-1)).to(torch.int64)
        desc.append(lens_descriptor(q, patches))
    return np.stack(desc)


def per_feature_desc(Xn):
    th = 2 * math.pi * Xn                                # value in [0,1] -> phase
    return torch.cat([th.cos(), th.sin()], dim=1).numpy()


def centroid_acc(Dtr, ytr, Dte, yte, K):
    cents = np.stack([Dtr[ytr == c].mean(0) for c in range(K)])
    pred = ((Dte[:, None, :] - cents[None]) ** 2).sum(-1).argmin(1)
    return float((pred == yte).mean())


def corr_order(X):
    """Greedy chain: start at 0, append the unused feature most |corr| with last."""
    C = np.corrcoef(X.numpy().T)
    F = C.shape[0]
    used = [0]; rem = set(range(1, F))
    while rem:
        last = used[-1]
        nxt = max(rem, key=lambda j: abs(C[last, j]))
        used.append(nxt); rem.discard(nxt)
    return used


def run(name, Dtr, Dte, ytr_np, yte_np, K, ytr_t, yte_t):
    ac = centroid_acc(Dtr, ytr_np, Dte, yte_np, K)
    Tt, Te = torch.from_numpy(Dtr).float(), torch.from_numpy(Dte).float()
    am = float((classify_archive(Tt, ytr_t, Te, K, full_cov=True) == yte_t).float().mean())
    print(f"  {name:<34} {Dtr.shape[1]:>4} {ac:>9.3f} {am:>12.3f}")


def main():
    tr, te = make_split()
    K = len(te.names)
    ytr_np, yte_np = tr.y.numpy(), te.y.numpy()
    Xtr, lo, hi = per_feature_norm(tr.x)
    Xte, _, _ = per_feature_norm(te.x, lo, hi)
    print(f"hard taxonomy: K={K}, D={tr.x.shape[1]}, M=3 | Bayes {bayes_accuracy(te):.3f} "
          f"| raw 0.365/0.901, per-feature 0.471/0.876\n")
    print(f"  {'descriptor':<34} {'dim':>4} {'centroid':>9} {'Mahalanobis':>12}")

    # baseline: per-feature phasor
    run("per-feature phasor", per_feature_desc(Xtr), per_feature_desc(Xte),
        ytr_np, yte_np, K, tr.y, te.y)

    F = tr.x.shape[1]
    order = corr_order(tr.x)
    B = 12
    for k, stride, tag in [(2, 2, "2x2 s2"), (2, 3, "2x2 s3")]:
        P = adaptive_patches((F, B), k=k, stride=stride)
        for od, olbl in [(None, ""), (order, " +reorder")]:
            itr = fold_image(Xtr, B, order=od); ite = fold_image(Xte, B, order=od)
            run(f"fold2d {F}x{B} lens {tag}{olbl}",
                lens_on_images(itr, P), lens_on_images(ite, P),
                ytr_np, yte_np, K, tr.y, te.y)


if __name__ == "__main__":
    main()
