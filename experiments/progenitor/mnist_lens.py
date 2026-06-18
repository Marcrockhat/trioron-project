"""MNIST positive control: does the 2D lens WIN where locality is real? (s042)

The taxonomy arc showed the lens is a spatial-locality prior that hurts on
non-local tabular features. The clean confirmation from the other side: run the
SAME pipeline (per-sample quantize -> 2x2 lens -> manifold) on real images, where
adjacent pixels ARE correlated. Prediction: the lens beats raw pixels, especially
for the weak (nearest-centroid) back-end.

Downsampled to 14x14 (2x2 avg-pool) so the descriptors stay conditioned for the
full-cov Mahalanobis (n_per_class > dim).

  raw            : 14x14 pixels                          196-d
  lens 2x2 s2    : 2x2 patches stride 2 -> 7x7 = 49      98-d
  lens 2x2 s1    : 2x2 patches stride 1 -> 13x13 = 169   338-d (centroid only)
  lens 2x2 s2 sweep w<=4                                 392-d (centroid only)

Back-ends: nearest centroid (weak) + real ManifoldArchive full-cov Mahalanobis.

Run:  python3 experiments/progenitor/mnist_lens.py
"""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F

from trioron.core.receptor import quantize
from trioron.legacy.donorkit.datasets import DatasetBundle
from experiments.progenitor.fingerprint_lens import adaptive_patches
from experiments.progenitor.spectral_lens import spectral_descriptor
from experiments.progenitor.taxonomy_manifold import classify_archive

N_QUANTA = 1000
PER_CLASS = 300
GRID = 14


def balanced(X, y, per_class, seed):
    g = torch.Generator().manual_seed(seed)
    idx = []
    for c in range(10):
        ci = torch.nonzero(y == c, as_tuple=False).squeeze(1)
        idx.append(ci[torch.randperm(len(ci), generator=g)[:per_class]])
    idx = torch.cat(idx)
    return X[idx], y[idx]


def downsample(X):
    img = X.reshape(-1, 1, 28, 28)
    return F.avg_pool2d(img, 2).reshape(-1, GRID * GRID)   # 14x14


def lens_descs(X, patches, carriers=(1.0,)):
    out = []
    for i in range(len(X)):
        q = quantize(X[i]).to(torch.int64)
        out.append(spectral_descriptor(q, patches, carriers))
    return np.stack(out)


def centroid_acc(Dtr, ytr, Dte, yte):
    cents = np.stack([Dtr[ytr == c].mean(0) for c in range(10)])
    pred = ((Dte[:, None, :] - cents[None]) ** 2).sum(-1).argmin(1)
    return float((pred == yte).mean())


def main():
    b = DatasetBundle(["mnist"])
    Xtr_f, ytr = b.task_view("mnist", list(range(10)), list(range(10)), split="train").all_examples()
    Xte_f, yte = b.task_view("mnist", list(range(10)), list(range(10)), split="test").all_examples()
    Xtr_f, ytr = balanced(Xtr_f, ytr, PER_CLASS, 0)
    Xte_f, yte = balanced(Xte_f, yte, PER_CLASS, 1)
    Xtr, Xte = downsample(Xtr_f), downsample(Xte_f)
    ytr_np, yte_np = ytr.numpy(), yte.numpy()
    print(f"MNIST {GRID}x{GRID}: {len(Xtr)} train / {len(Xte)} test, "
          f"{PER_CLASS}/class | chance 0.10\n")
    print(f"  {'front-end':<26} {'dim':>4} {'centroid':>9} {'Mahalanobis':>12}")

    P2 = adaptive_patches((GRID, GRID), k=2, stride=2)
    P1 = adaptive_patches((GRID, GRID), k=2, stride=1)

    def row(name, Dtr, Dte, do_maha=True):
        ac = centroid_acc(Dtr, ytr_np, Dte, yte_np)
        if do_maha:
            Tt, Te = torch.from_numpy(Dtr).float(), torch.from_numpy(Dte).float()
            am = float((classify_archive(Tt, ytr, Te, 10, full_cov=True) == yte).float().mean())
            print(f"  {name:<26} {Dtr.shape[1]:>4} {ac:>9.3f} {am:>12.3f}")
        else:
            print(f"  {name:<26} {Dtr.shape[1]:>4} {ac:>9.3f} {'(skip)':>12}")

    row("raw pixels", Xtr.numpy(), Xte.numpy())
    row("lens 2x2 s2", lens_descs(Xtr, P2), lens_descs(Xte, P2))
    row("lens 2x2 s1", lens_descs(Xtr, P1), lens_descs(Xte, P1), do_maha=False)
    row("lens 2x2 s2 sweep w<=4", lens_descs(Xtr, P2, (1., 2., 3., 4.)),
        lens_descs(Xte, P2, (1., 2., 3., 4.)), do_maha=False)


if __name__ == "__main__":
    main()
