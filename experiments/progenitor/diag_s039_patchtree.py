"""s039 COMBINED patch-tree representation — both halves of Rocky's design
tested end-to-end for usefulness.

Half 1 (inversion fix): leaf feature = |center-surround| = |LoG(image)|.
The LoG kernel is zero-DC, so LoG(1-x) = -LoG(x) and |LoG| is BIT-EXACT
under background swap -> polarity/inversion invariant by construction.

Half 2 (patches): neighbour-join a tree over the leaf dimensions (data
co-activation, co-quantum agreement conditioned on activity, validated in
diag_s039_njtree: tree-nearest leaf = 1.58 px, Spearman +0.39). Each
internal CLADE of bounded size = a discovered patch; its feature = mean of
its descendant leaves (avg-pool over a tree-defined region, no grid).

A/B in the KNN harness (the organism's matched-filter readout = phasor),
NORMAL vs INVERTED test, 30-way, chance 0.033:

  raw-leaves        raw intensity quanta          (the broken baseline)
  raw + patch       + tree-aggregated patches     (patches on raw)
  cs-leaves         |LoG| quanta                  (inversion fix alone)
  cs + patch        |LoG| + tree patches          (FULL proposal)

Decisive reads: does cs survive inversion (predicted yes), and do tree
patches LIFT accuracy over leaves-only?

Run:  python3 -m experiments.progenitor.diag_s039_patchtree
Env:  S039_PER_CLASS (default 300), S039_BANDS (default 10),
      S039_PATCH_MIN/MAX (clade leaf-count window, default 3/64).
"""
from __future__ import annotations

import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from trioron.core.receptor import quantize, N_QUANTA
from trioron.legacy.donorkit.datasets import (DatasetBundle,
                                              chained_15_specs,
                                              build_task_views)

DATA_ROOT = os.environ.get(
    "AXIS6_DATA_ROOT",
    os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), "outputs", "data"))
PER_CLASS = int(os.environ.get("S039_PER_CLASS", "300"))
BANDS = int(os.environ.get("S039_BANDS", "10"))
PATCH_MIN = int(os.environ.get("S039_PATCH_MIN", "3"))
PATCH_MAX = int(os.environ.get("S039_PATCH_MAX", "64"))
ACTIVE_FRAC = 0.02
N_CLASSES = 30


def balanced_slice(views, per_class, seed):
    X = torch.cat([v.images for v in views])
    y = torch.cat([v.labels_global for v in views])
    g = torch.Generator().manual_seed(seed)
    idx = [(y == k).nonzero().squeeze(1)[
        torch.randperm(int((y == k).sum()), generator=g)][:per_class]
        for k in range(N_CLASSES)]
    idx = torch.cat(idx)
    return X[idx], y[idx]


def _log_kernel(ksize=7, sig_c=1.0, sig_s=2.0):
    half = ksize // 2
    yy, xx = torch.meshgrid(torch.arange(-half, half + 1).float(),
                            torch.arange(-half, half + 1).float(),
                            indexing="ij")
    log = (torch.exp(-(xx ** 2 + yy ** 2) / (2 * sig_c ** 2)) / sig_c ** 2
           - torch.exp(-(xx ** 2 + yy ** 2) / (2 * sig_s ** 2)) / sig_s ** 2)
    log = log - log.mean()                               # zero DC
    return (log / log.abs().sum().clamp_min(1e-6)).view(1, 1, ksize, ksize)


_LOG = _log_kernel()


def center_surround(x: torch.Tensor) -> torch.Tensor:
    """|LoG(image)| at full 28x28 res -> [N, 784]. Polarity-invariant."""
    img = x.view(-1, 1, 28, 28)
    p = _LOG.shape[-1] // 2
    img = F.pad(img, (p, p, p, p), mode="reflect")       # invariance at border
    f = F.conv2d(img, _LOG).abs()
    return f.flatten(1)


def coquantum_distance(F_feat: torch.Tensor):
    """active-conditioned co-quantum disagreement on a feature matrix."""
    q = quantize(F_feat)
    band = torch.clamp((q * BANDS // (N_QUANTA + 1)).long(), 0, BANDS - 1)
    active = ((band > 0).float().mean(0) >= ACTIVE_FRAC).nonzero().squeeze(1)
    b = band[:, active]
    N = b.shape[0]
    agree = torch.zeros(len(active), len(active))
    agree0 = None
    for k in range(BANDS):
        O = (b == k).float()
        co = O.t() @ O
        agree += co
        if k == 0:
            agree0 = co
    denom = (1.0 - agree0 / N).clamp_min(1e-9)
    D = ((1.0 - agree / N) / denom).numpy()
    D[(agree0 / N >= 1 - 1e-6).numpy()] = 1.0
    np.fill_diagonal(D, 0.0)
    return D, active.numpy()


def neighbour_join_clades(D, leaves):
    """NJ that tracks each internal node's leaf CLADE (the merged subtree).
    Returns list of clades (numpy arrays of original leaf ids)."""
    nodes = list(leaves)
    clades = {n: [n] for n in nodes}
    out = []
    Dm = D.astype(float).copy()
    while len(nodes) > 2:
        m = len(nodes)
        s = Dm.sum(1)
        Q = (m - 2) * Dm - s[:, None] - s[None, :]
        np.fill_diagonal(Q, np.inf)
        i, j = np.unravel_index(np.argmin(Q), Q.shape)
        merged = clades[nodes[i]] + clades[nodes[j]]
        out.append(np.array(merged))
        du = 0.5 * (Dm[i] + Dm[j] - Dm[i, j])
        keep = [k for k in range(m) if k != i and k != j]
        Dnew = np.empty((len(keep) + 1, len(keep) + 1))
        Dnew[:-1, :-1] = Dm[np.ix_(keep, keep)]
        Dnew[-1, :-1] = Dnew[:-1, -1] = du[keep]
        Dnew[-1, -1] = 0.0
        Dm = Dnew
        new = max(clades) + 1
        clades[new] = merged
        nodes = [nodes[k] for k in keep] + [new]
    return out


def knn_phasor(Ftr, ytr, Fte, yte, k=1):
    """cosine k-NN majority on PHASOR features (the organism readout)."""
    qtr, qte = quantize(Ftr), quantize(Fte)
    Ztr = torch.exp(1j * 2 * math.pi * qtr / N_QUANTA)
    Zte = torch.exp(1j * 2 * math.pi * qte / N_QUANTA)
    Ztr = Ztr.masked_fill((qtr == 0) | (qtr == N_QUANTA), 0)   # mask reference
    Zte = Zte.masked_fill((qte == 0) | (qte == N_QUANTA), 0)
    ntr = Ztr.abs().pow(2).sum(-1).sqrt().clamp_min(1e-9)
    nte = Zte.abs().pow(2).sum(-1).sqrt().clamp_min(1e-9)
    correct = 0
    for i in range(0, len(Zte), 1000):
        sim = (Zte[i:i + 1000] @ Ztr.conj().t()).real
        sim = sim / nte[i:i + 1000].unsqueeze(1) / ntr.unsqueeze(0)
        pred = torch.mode(ytr[sim.topk(k, 1).indices], 1).values
        correct += int((pred == yte[i:i + 1000]).sum())
    return correct / len(Zte)


def patch_features(leaf_feat, active, clades):
    """[N, A_full] leaf features -> [N, P] clade-mean patch features."""
    cols = [torch.from_numpy(c).long() for c in clades]
    return torch.stack([leaf_feat[:, c].mean(1) for c in cols], dim=1)


def main() -> None:
    bundle = DatasetBundle(["mnist", "fashion_mnist", "emnist_letters"],
                           root=DATA_ROOT, n_holdout_per_dataset=0)
    specs = chained_15_specs()
    Xtr, ytr = balanced_slice(build_task_views(bundle, specs, split="train"),
                              PER_CLASS, seed=0)
    Xte, yte = balanced_slice(build_task_views(bundle, specs, split="test"),
                              PER_CLASS, seed=1)
    Xte_inv = 1.0 - Xte
    t0 = time.time()

    feats = {}                                           # name -> leaf matrix
    feats["raw"] = (Xtr, Xte, Xte_inv)
    feats["cs"] = (center_surround(Xtr), center_surround(Xte),
                   center_surround(Xte_inv))

    print(f"s039 patch-tree — 30-way, train {tuple(Xtr.shape)} "
          f"test {tuple(Xte.shape)}  chance={1 / N_CLASSES:.3f}  "
          f"patch window [{PATCH_MIN},{PATCH_MAX}]\n")
    print(f"{'rep':>14}{'dims':>7}{'normal':>9}{'inverted':>10}")
    print("-" * 40)

    for base in ("raw", "cs"):
        Ftr, Fte, Fte_i = feats[base]
        # tree over THIS feature's dims
        D, active = coquantum_distance(Ftr)
        clades = [c for c in neighbour_join_clades(D, list(active))
                  if PATCH_MIN <= len(c) <= PATCH_MAX]
        # leaves-only
        a = knn_phasor(Ftr, ytr, Fte, yte)
        ai = knn_phasor(Ftr, ytr, Fte_i, yte)
        print(f"{base + '-leaves':>14}{Ftr.shape[1]:>7}{a:>9.3f}{ai:>10.3f}")
        # leaves + patches
        Ptr = patch_features(Ftr, active, clades)
        Pte = patch_features(Fte, active, clades)
        Pte_i = patch_features(Fte_i, active, clades)
        Gtr = torch.cat([Ftr, Ptr], 1)
        Gte = torch.cat([Fte, Pte], 1)
        Gte_i = torch.cat([Fte_i, Pte_i], 1)
        a = knn_phasor(Gtr, ytr, Gte, yte)
        ai = knn_phasor(Gtr, ytr, Gte_i, yte)
        print(f"{base + '+patch':>14}{Gtr.shape[1]:>7}{a:>9.3f}{ai:>10.3f}"
              f"   ({len(clades)} patches)")
        # control: random groupings of the SAME sizes (tree topology removed)
        g = torch.Generator().manual_seed(0)
        act_t = torch.from_numpy(active)
        perm = act_t[torch.randperm(len(active), generator=g)]
        rand, off = [], 0
        for c in clades:
            rand.append(perm[off:off + len(c)].numpy())
            off = (off + len(c)) % len(active)
        Rtr = torch.cat([Ftr, patch_features(Ftr, active, rand)], 1)
        Rte = torch.cat([Fte, patch_features(Fte, active, rand)], 1)
        a = knn_phasor(Rtr, ytr, Rte, yte)
        print(f"{base + '+randpatch':>14}{Rtr.shape[1]:>7}{a:>9.3f}"
              f"{'':>10}   (control: shuffled groups)")

    print(f"\nt={time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
