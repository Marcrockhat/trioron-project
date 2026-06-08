"""L1 — discover trioron's own taxonomic tree.

Step 3. Agglomerative clustering on the 16 fine-class L0 centroids
using cosine distance. The output is the taxonomy *the trioron itself*
imposes on these classes given its current senses, not the human
biology-derived Living/Non-living split. Subsequent levels of the
curriculum will be built FROM this discovered tree.

No training. Pure post-hoc analysis on the L1 donor we already trained.
"""
from __future__ import annotations
import argparse
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, to_tree
from scipy.spatial.distance import squareform

from trioron.senses import apply_sense, Standardizer
from experiments.cifar.datasets import load_cifar100, DEFAULT_DATA_ROOT
from experiments.cifar.bench_taxonomy_l1 import (
    LIVING_NAMES, NON_LIVING_NAMES, _resolve_names_to_ids, _binary_subset,
)
from experiments.cifar.probe_taxonomy_l1_delta import (
    _load_donor, _l0_activations,
)


def _build_per_class_mu(
    donor_path: str, data_root: str,
) -> tuple:
    net, payload = _load_donor(donor_path)
    sense_name = payload["sense"]
    std = Standardizer.from_dict(payload["standardizer"])

    name_to_id = _resolve_names_to_ids(data_root)
    living_ids = [name_to_id[n] for n in LIVING_NAMES]
    nonliving_ids = [name_to_id[n] for n in NON_LIVING_NAMES]
    train_imgs, train_labs = load_cifar100(data_root, train=True)
    Xtr_raw, _, ytr_fine = _binary_subset(
        train_imgs, train_labs, living_ids, nonliving_ids,
    )
    Xtr_sensed = std.transform(apply_sense(sense_name, Xtr_raw)).contiguous()
    h0 = _l0_activations(net, Xtr_sensed)

    ordered_names = LIVING_NAMES + NON_LIVING_NAMES
    ordered_ids = living_ids + nonliving_ids
    mus = []
    for cid in ordered_ids:
        mus.append(h0[ytr_fine == cid].mean(dim=0))
    mu_matrix = torch.stack(mus, dim=0)        # (16, 128)
    return mu_matrix, ordered_names, payload, sense_name


def _cosine_distance(M: torch.Tensor) -> np.ndarray:
    norm = M / (M.norm(dim=1, keepdim=True) + 1e-9)
    cos = (norm @ norm.t()).numpy()
    np.fill_diagonal(cos, 1.0)
    dist = 1.0 - cos
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0.0, 2.0)
    return dist


def _print_clusters(labels: np.ndarray, names: List[str], k: int) -> None:
    print(f"\n[L1-tree] === k={k} clusters ===")
    by_cluster: Dict[int, List[str]] = {}
    for label, name in zip(labels, names):
        by_cluster.setdefault(int(label), []).append(name)
    # Order clusters by size so the most populous one prints first.
    for cid in sorted(by_cluster, key=lambda c: -len(by_cluster[c])):
        members = by_cluster[cid]
        # Note any "mixed-kingdom" clusters where Living + Non-living
        # appear together — these are *trioron's* equivalences across
        # our human kingdom split.
        n_living = sum(1 for m in members if m in LIVING_NAMES)
        n_nonliving = sum(1 for m in members if m in NON_LIVING_NAMES)
        tag = ""
        if n_living and n_nonliving:
            tag = (f"  [MIXED: {n_living} Living + "
                   f"{n_nonliving} Non-living]")
        elif n_living:
            tag = "  [pure Living]"
        else:
            tag = "  [pure Non-living]"
        print(f"  c{cid}: {members}{tag}")


def _newick(node, names, dist_threshold=None) -> str:
    """Recursively walk scipy ClusterNode tree → Newick string."""
    if node.is_leaf():
        return names[node.id]
    left = _newick(node.left, names, dist_threshold)
    right = _newick(node.right, names, dist_threshold)
    return f"({left},{right}):{node.dist:.3f}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--donor-path",
        default="outputs/cifar_taxonomy/donor_l1_living_vs_nonliving.pt",
    )
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--method", default="average",
                        choices=["average", "complete", "single", "ward"])
    parser.add_argument("--cuts", type=int, nargs="+",
                        default=[2, 3, 4, 6, 8])
    args = parser.parse_args(argv)

    mu, names, payload, sense_name = _build_per_class_mu(
        args.donor_path, args.data_root,
    )
    print(f"[L1-tree] donor: {args.donor_path}")
    print(f"[L1-tree]   sense={sense_name}  arch={payload['n_nodes_per_layer']}")
    print(f"[L1-tree]   {len(names)} fine-class centroids in "
          f"{mu.shape[1]}-d L0 space")
    print(f"[L1-tree] clustering: agglomerative, "
          f"linkage={args.method}, metric=cosine")

    dist = _cosine_distance(mu)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method=args.method)

    # Cut at each requested depth.
    for k in args.cuts:
        labels = fcluster(Z, t=k, criterion="maxclust")
        _print_clusters(labels, names, k)

    # Print the full dendrogram order (a 1-d ordering induced by the
    # tree — neighbors in this list are nearest-cousins in the tree).
    print(f"\n[L1-tree] === dendrogram leaf order ===")
    tree, _ = to_tree(Z, rd=True)
    order = []

    def _walk(node):
        if node.is_leaf():
            order.append(int(node.id))
        else:
            _walk(node.left)
            _walk(node.right)

    _walk(tree)
    print("  " + " → ".join(names[i] for i in order))

    # Newick tree (compact, copy-pastable into any phylogeny viewer).
    print(f"\n[L1-tree] === Newick tree (cosine distance edge weights) ===")
    print("  " + _newick(tree, names) + ";")

    print(f"\n[L1-tree] interpretation:")
    print(f"  Read top-down: the deepest split is the trioron's coarsest")
    print(f"  perceptual division of these 16 classes. MIXED clusters at any")
    print(f"  cut depth are the spots where the human Living/Non-living")
    print(f"  taxonomy disagrees with the trioron's natural ontology.")
    print(f"  These pairs are exactly where the doc's δ-replay margin loss")
    print(f"  has most leverage — it can FORCE the boundary the senses don't")
    print(f"  carve naturally.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
