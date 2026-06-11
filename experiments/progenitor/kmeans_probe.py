"""Does k-means recover the 32 classes, or the 96 modes? (s028 unsupervised probe)

The capacity-hard taxonomy is 32 classes × 3 modes = 96 Gaussian blobs, with 5 disruptor
classes (sigma=0.5) spread wide. If an UNSUPERVISED growth trigger clustered the inputs
and grew per-cluster, the question is whether clusters align with the 32 true classes.
This sweeps k, reporting cluster↔class agreement (ARI/NMI) and at k=32 the per-cluster
purity and how many true classes each cluster mixes.

Run: python3 -m experiments.progenitor.kmeans_probe
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score, silhouette_score,
)

from .data_hard import make_split


def main() -> None:
    tr, te = make_split()
    X = tr.x.numpy()
    y = tr.y.numpy()
    K_true = len(tr.names)
    n_modes = tr.spec.means.shape[1]
    n_blobs = K_true * n_modes
    n_dis = int((tr.spec.std > tr.spec.std.min()).sum())
    print(f"data: {X.shape[0]} samples, {X.shape[1]}-d, {K_true} classes, "
          f"{n_modes} modes/class → {n_blobs} true blobs, {n_dis} disruptor classes\n")

    print(f"  {'k':>4}  {'inertia':>9}  {'silhou':>7}  {'ARI':>6}  {'NMI':>6}   "
          f"(ARI/NMI vs the 32 TRUE classes)")
    for k in [16, 24, 32, 48, 64, 96, 128]:
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
        lab = km.labels_
        sil = silhouette_score(X, lab) if k < X.shape[0] else float("nan")
        ari = adjusted_rand_score(y, lab)
        nmi = normalized_mutual_info_score(y, lab)
        print(f"  {k:>4}  {km.inertia_:>9.1f}  {sil:>7.3f}  {ari:>6.3f}  {nmi:>6.3f}")

    # detailed look at k = 32 (the "one cluster per class" hope)
    print(f"\n=== k=32 detail (does each cluster = one class?) ===")
    km = KMeans(n_clusters=K_true, n_init=10, random_state=0).fit(X)
    lab = km.labels_
    purities, n_classes_per_cluster = [], []
    for cl in range(K_true):
        members = y[lab == cl]
        if len(members) == 0:
            continue
        cnt = Counter(members.tolist())
        top = cnt.most_common(1)[0][1]
        purities.append(top / len(members))
        n_classes_per_cluster.append(len(cnt))
    print(f"  mean cluster purity (dominant class fraction): {np.mean(purities):.3f}")
    print(f"  mean #true-classes mixed into one cluster:     {np.mean(n_classes_per_cluster):.2f}")
    # how many of the 32 true classes are the dominant ('captured') class of some cluster?
    dom = {Counter(y[lab == cl].tolist()).most_common(1)[0][0]
           for cl in range(K_true) if (lab == cl).any()}
    print(f"  distinct true classes captured as a cluster's majority: {len(dom)}/{K_true}")

    # per-class: how many clusters does ONE class get split across? (multimodal → >1)
    splits = [len(set(lab[y == c].tolist())) for c in range(K_true)]
    print(f"  mean #clusters a single class is split across:  {np.mean(splits):.2f} "
          f"(min {min(splits)}, max {max(splits)})")


if __name__ == "__main__":
    main()
