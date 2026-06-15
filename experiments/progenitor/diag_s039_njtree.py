"""s039 NEIGHBOUR-JOINING dimension tree — does the input topology EMERGE
from the data alone? (Rocky's anti-patch design.)

The keystone receptor problem: post-receptor the 784 pixels are an
UNORDERED bag — position survives only as an index, never as a PATCH, and
the substrate has no idea pixel 5 and pixel 6 are spatial neighbours
(substrate_no_spatial_memory). Hand-coding the grid (the s039 band-offset
probe) was FALSIFIED. Rocky's fix: DISCOVER the topology — build a tree
over the DIMENSIONS by neighbour-joining (phylogenetics), so internal
nodes become patches and the whole thing is modality-agnostic (1D audio,
2D image, 3D volume — NJ needs only a distance matrix, never a grid).

This is the DECISIVE validity test, BEFORE any substrate change: if a tree
built purely from data co-activation reconstructs the 2D image lattice it
was never told about, the idea is sound.

Distance = CO-QUANTUM AGREEMENT (Rocky's choice): bin each pixel's quanta
into S039_BANDS bands; d(i,j) = fraction of samples where dims i,j land in
DIFFERENT bands. Adjacent pixels co-vary (strokes span them) -> agree ->
small distance. Dead (constant-band) pixels are dropped first, like the
genesis starve step.

Validation (raw pixels, known grid coords row=f//28 col=f%28):
  * Spearman(tree patristic distance, 2D grid Euclidean distance) over
    sampled leaf pairs -- high positive = topology recovered.
  * mean grid-distance of each leaf's TREE-nearest leaf -- ~1 if the tree
    puts grid-neighbours adjacent (random baseline ~10).
  * a SHUFFLED-grid control -> correlation collapses to ~0 (the recovered
    signal is real, not an artefact of the metric).

Run:  python3 -m experiments.progenitor.diag_s039_njtree
Env:  S039_PER_CLASS (default 200), S039_BANDS (default 10),
      S039_ACTIVE_FRAC (drop pixels active in < this frac, default 0.02).
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch
from scipy.stats import spearmanr

from trioron.core.receptor import quantize, N_QUANTA
from trioron.legacy.donorkit.datasets import (DatasetBundle,
                                              chained_15_specs,
                                              build_task_views)

DATA_ROOT = os.environ.get(
    "AXIS6_DATA_ROOT",
    os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), "outputs", "data"))
PER_CLASS = int(os.environ.get("S039_PER_CLASS", "200"))
BANDS = int(os.environ.get("S039_BANDS", "10"))
ACTIVE_FRAC = float(os.environ.get("S039_ACTIVE_FRAC", "0.02"))
METRIC = os.environ.get("S039_METRIC", "active")          # active | raw
N_CLASSES = 30


def balanced_slice(views, per_class, seed):
    X = torch.cat([v.images for v in views])
    y = torch.cat([v.labels_global for v in views])
    g = torch.Generator().manual_seed(seed)
    idx = []
    for k in range(N_CLASSES):
        ck = (y == k).nonzero().squeeze(1)
        idx.append(ck[torch.randperm(len(ck), generator=g)][:per_class])
    return X[torch.cat(idx)]


def coquantum_distance(X: torch.Tensor):
    """[N, F] images -> (D [A,A] co-quantum-disagreement, active dim ids).
    band b in [0, BANDS); d(i,j) = P(band_i != band_j) over samples."""
    q = quantize(X)                                      # [N, F] 0..N_QUANTA
    band = torch.clamp((q * BANDS // (N_QUANTA + 1)).long(), 0, BANDS - 1)
    # active = not stuck in a single band (the starve step's spirit)
    nonzero_frac = (band > 0).float().mean(0)
    active = (nonzero_frac >= ACTIVE_FRAC).nonzero().squeeze(1)
    b = band[:, active]                                  # [N, A]
    N = b.shape[0]
    agree = torch.zeros(len(active), len(active))
    agree0 = None
    for k in range(BANDS):
        O = (b == k).float()                            # [N, A]
        co = O.t() @ O                                  # co-occurrence in band k
        agree += co
        if k == 0:
            agree0 = co                                 # both-background count
    disagree = 1.0 - agree / N                          # P(bands differ), all samples
    if METRIC == "active":
        # condition on "at least one active": both-bg samples always agree,
        # so all disagreement already lives in the active subset; just
        # renormalise by the active-sample fraction. Always-bg pairs -> 1.
        denom = (1.0 - agree0 / N).clamp_min(1e-9)
        D = (disagree / denom).numpy()
        both_bg = (agree0 / N >= 1.0 - 1e-6).numpy()
        D[both_bg] = 1.0
    else:
        D = disagree.numpy()
    np.fill_diagonal(D, 0.0)
    return D, active.numpy()


def neighbour_join(D: np.ndarray, leaves):
    """Classic NJ (Saitou & Nei). Returns adjacency dict node ->
    list[(neighbour, branch_len)] over leaf ids + internal ids."""
    nodes = list(leaves)
    nxt = max(leaves) + 1
    adj = {n: [] for n in nodes}
    Dm = D.astype(float).copy()
    while len(nodes) > 2:
        m = len(nodes)
        s = Dm.sum(1)
        Q = (m - 2) * Dm - s[:, None] - s[None, :]
        np.fill_diagonal(Q, np.inf)
        i, j = np.unravel_index(np.argmin(Q), Q.shape)
        li = 0.5 * Dm[i, j] + (s[i] - s[j]) / (2 * (m - 2))
        lj = Dm[i, j] - li
        u = nxt
        nxt += 1
        adj[u] = []
        for child, length in ((nodes[i], li), (nodes[j], lj)):
            length = max(length, 0.0)
            adj[u].append((child, length))
            adj[child].append((u, length))
        du = 0.5 * (Dm[i] + Dm[j] - Dm[i, j])
        keep = [k for k in range(m) if k != i and k != j]
        Dnew = np.empty((len(keep) + 1, len(keep) + 1))
        Dnew[:-1, :-1] = Dm[np.ix_(keep, keep)]
        Dnew[-1, :-1] = Dnew[:-1, -1] = du[keep]
        Dnew[-1, -1] = 0.0
        Dm = Dnew
        nodes = [nodes[k] for k in keep] + [u]
    a, b = nodes
    adj[a].append((b, Dm[0, 1]))
    adj[b].append((a, Dm[0, 1]))
    return adj


def tree_dists_from(src, adj):
    """Patristic distance from one source node to all nodes (tree BFS)."""
    dist = {src: 0.0}
    stack = [src]
    while stack:
        u = stack.pop()
        for v, w in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + w
                stack.append(v)
    return dist


def main() -> None:
    bundle = DatasetBundle(["mnist", "fashion_mnist", "emnist_letters"],
                           root=DATA_ROOT, n_holdout_per_dataset=0)
    specs = chained_15_specs()
    X = balanced_slice(build_task_views(bundle, specs, split="train"),
                       PER_CLASS, seed=0)
    print(f"s039 NJ dimension-tree — raw pixels, {tuple(X.shape)}, "
          f"bands={BANDS}\n")

    t0 = time.time()
    D, active = coquantum_distance(X)
    print(f"active dims {len(active)}/{X.shape[1]} "
          f"(dropped {X.shape[1] - len(active)} dead)  "
          f"D in [{D.min():.3f},{D.max():.3f}]  t={time.time() - t0:.1f}s")

    adj = neighbour_join(D, list(active))
    print(f"NJ tree built ({len(adj)} nodes)  t={time.time() - t0:.1f}s")

    # grid coords of each active leaf
    gr = np.array([[f // 28, f % 28] for f in active], dtype=float)

    # sample sources, collect (tree_dist, grid_dist) leaf-leaf pairs
    rng = np.random.default_rng(0)
    srcs = rng.choice(len(active), size=min(120, len(active)), replace=False)
    leaf_set = set(int(a) for a in active)
    pos = {int(a): k for k, a in enumerate(active)}
    td, gd = [], []
    nn_grid = []
    for si in srcs:
        src = int(active[si])
        dist = tree_dists_from(src, adj)
        best, best_leaf = np.inf, -1
        for leaf in leaf_set:
            if leaf == src:
                continue
            tdv = dist[leaf]
            td.append(tdv)
            gd.append(float(np.hypot(*(gr[pos[leaf]] - gr[si]))))
            if tdv < best:
                best, best_leaf = tdv, leaf
        nn_grid.append(float(np.hypot(*(gr[pos[best_leaf]] - gr[si]))))

    td, gd = np.array(td), np.array(gd)
    rho = spearmanr(td, gd).statistic
    # shuffled-grid control
    perm = rng.permutation(len(active))
    grs = gr[perm]
    gd_sh = []
    off = 0
    for si in srcs:
        for leaf in leaf_set:
            if leaf == int(active[si]):
                continue
            gd_sh.append(float(np.hypot(*(grs[pos[leaf]] - grs[si]))))
    rho_sh = spearmanr(td, np.array(gd_sh)).statistic

    print()
    print(f"Spearman(tree dist, grid dist)        : {rho:+.3f}   "
          f"<- topology recovered if strongly +")
    print(f"  shuffled-grid control               : {rho_sh:+.3f}   "
          f"<- ~0 confirms the signal is real")
    print(f"mean grid-dist of TREE-nearest leaf   : {np.mean(nn_grid):.2f}  "
          f"px  (1.0 = grid-adjacent; random ~{np.mean(gd):.1f})")


if __name__ == "__main__":
    main()
