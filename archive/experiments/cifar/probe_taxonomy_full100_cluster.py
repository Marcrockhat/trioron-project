"""Discover trioron's natural taxonomy on the full 100-class CIFAR-100.

L0 is a frozen random projection (determined by l0_seed only), so we
don't need to train any donor on the full set first — we hijack the
L0 layer of an existing l0_seed=42 donor and forward all 50K full-
class training images through it.

Steps:
  1. Apply sense_classical to full CIFAR-100 train (50K × 33-d).
  2. Fit a Standardizer on this full distribution (different scale
     than the 16-class subset's standardizer).
  3. Forward through donor's L0 → 50K × 128-d.
  4. Compute per-fine-class μ (100 × 128-d).
  5. Agglomerative clustering on cosine distance, show k cuts.
  6. Show whether trioron's natural clusters align with CIFAR's stock
     20 superclasses.
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

from trioron.network import TrioronNetwork
from trioron.senses import apply_sense, Standardizer
from experiments.cifar.datasets import DEFAULT_DATA_ROOT, load_cifar100
from experiments.cifar.hierarchical_train import FINE_TO_COARSE


def _load_l0(donor_path: str) -> tuple:
    """Load just the L0 layer from a donor."""
    payload = torch.load(donor_path, map_location="cpu", weights_only=False)
    n_nodes = list(payload["n_nodes_per_layer"])
    layer_specs = []
    prev = int(payload["input_dim"])
    for i, n in enumerate(n_nodes):
        act = "linear" if i == len(n_nodes) - 1 else "relu"
        layer_specs.append((prev, int(n), act))
        prev = int(n)
    net = TrioronNetwork(layer_specs)
    net.load_state_dict(payload["state_dict"])
    net.eval()
    return net, payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--donor-path",
        default="outputs/cifar_taxonomy/donor_l1_perceptual_4way_seed42.pt",
        help="Any l0_seed=42 donor — only its L0 weights are used.",
    )
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cuts", type=int, nargs="+",
                        default=[4, 8, 10, 16, 20])
    parser.add_argument("--method", default="average",
                        choices=["average", "complete", "single", "ward"])
    parser.add_argument("--show-cluster", type=int, default=10,
                        help="Print full member lists for the given k cut.")
    args = parser.parse_args(argv)

    net, payload = _load_l0(args.donor_path)
    sense_name = payload["sense"]
    print(f"[full100] using L0 from {args.donor_path}")
    print(f"  arch={payload['n_nodes_per_layer']}  l0_seed={payload['l0_seed']}")

    # Resolve fine class names.
    from torchvision.datasets import CIFAR100
    ds = CIFAR100(root=args.data_root, train=True, download=False)
    fine_names = list(ds.classes)
    print(f"[full100] {len(fine_names)} fine classes")

    # Apply sense + fit standardizer fresh on full data.
    train_imgs, train_labs = load_cifar100(args.data_root, train=True)
    print(f"[full100] applying sense to {train_imgs.shape[0]} train images...")
    Xtr_sensed = apply_sense(sense_name, train_imgs)
    std = Standardizer.fit(Xtr_sensed)
    Xtr_sensed = std.transform(Xtr_sensed).contiguous()

    # Forward through L0 (frozen random projection).
    print(f"[full100] forwarding through L0...")
    h0 = torch.empty(Xtr_sensed.shape[0],
                     payload["n_nodes_per_layer"][0],
                     dtype=torch.float32)
    batch = 1024
    with torch.no_grad():
        for i in range(0, Xtr_sensed.shape[0], batch):
            h0[i:i + batch] = net.layers[0](Xtr_sensed[i:i + batch])
    print(f"  L0 activations: {tuple(h0.shape)}  "
          f"mean={h0.mean().item():+.4f}  std={h0.std().item():.4f}")

    # Per-fine-class μ.
    K = len(fine_names)
    L0_dim = h0.shape[1]
    mu = torch.zeros(K, L0_dim)
    for c in range(K):
        m = (train_labs == c)
        if m.sum() == 0:
            continue
        mu[c] = h0[m].mean(dim=0)

    # Save standardizer + centroids for reuse in the actual pipeline.
    std_path = "outputs/cifar_taxonomy/standardizer_full100_classical.pt"
    os.makedirs(os.path.dirname(std_path), exist_ok=True)
    torch.save({
        "sense": sense_name,
        "standardizer": std.to_dict(),
        "fine_class_names": fine_names,
        "centroids": mu,
    }, std_path)
    print(f"[full100] saved {std_path}")

    # Cosine distance matrix.
    norm = mu / (mu.norm(dim=1, keepdim=True) + 1e-9)
    cos = (norm @ norm.t()).numpy()
    np.fill_diagonal(cos, 1.0)
    dist = 1.0 - cos
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0.0, 2.0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method=args.method)

    # Cluster sizes at various cuts.
    print(f"\n[full100] === cluster sizes at various k ===")
    for k in args.cuts:
        labels = fcluster(Z, t=k, criterion="maxclust")
        sizes = sorted([(int(c), int((labels == c).sum())) for c in set(labels)],
                       key=lambda x: -x[1])
        size_dist = [s for _, s in sizes]
        print(f"  k={k:>3d}: clusters of sizes {size_dist[:10]}{'...' if len(size_dist) > 10 else ''}")

    # Save a cluster assignment file at every requested k, so the
    # pipeline can pick up any cut without re-running discovery.
    for k in args.cuts:
        labels = fcluster(Z, t=k, criterion="maxclust")
        labels0 = (labels - 1).tolist()
        clusters: Dict[int, List[str]] = {}
        for cid, name in zip(labels0, fine_names):
            clusters.setdefault(int(cid), []).append(name)
        ordered = [clusters[cid] for cid in sorted(clusters)]
        out = f"outputs/cifar_taxonomy/cluster_assignment_full100_k{k}.pt"
        torch.save({
            "k": k,
            "clusters": ordered,
            "fine_to_cluster": labels0,
            "method": args.method,
            "metric": "cosine",
        }, out)
        print(f"[full100] saved {out} (k={k}, {len(ordered)} clusters)")

    # Show full member list for chosen k.
    k = args.show_cluster
    labels = fcluster(Z, t=k, criterion="maxclust")
    print(f"\n[full100] === k={k} clusters (members) ===")
    by_cluster: Dict[int, List[str]] = {}
    for label, name in zip(labels, fine_names):
        by_cluster.setdefault(int(label), []).append(name)
    # Order by size, descending.
    for cid in sorted(by_cluster, key=lambda c: -len(by_cluster[c])):
        members = by_cluster[cid]
        print(f"  c{cid:>2d} ({len(members):>2d} members): {members}")

    # CIFAR-100 stock superclass mapping for comparison.
    print(f"\n[full100] === alignment with CIFAR-100 stock 20 superclasses ===")
    coarse_names = [
        "aquatic_mammals", "fish", "flowers", "food_containers",
        "fruit_and_veg", "household_electrical", "household_furniture",
        "insects", "large_carnivores", "large_man_made_outdoor",
        "large_natural_outdoor", "large_omnivores_herbivores",
        "medium_mammals", "non_insect_invertebrates", "people",
        "reptiles", "small_mammals", "trees", "vehicles_1", "vehicles_2",
    ]
    # For each fine class, get the trioron-discovered cluster id.
    fine_to_trioron_cluster = {i: int(labels[i]) for i in range(K)}
    fine_to_cifar_super = {i: FINE_TO_COARSE[i] for i in range(K)}
    # Adjusted Rand Index.
    from collections import Counter
    n11 = n10 = n01 = n00 = 0
    for i in range(K):
        for j in range(i + 1, K):
            tri_same = (fine_to_trioron_cluster[i] == fine_to_trioron_cluster[j])
            cifar_same = (fine_to_cifar_super[i] == fine_to_cifar_super[j])
            if tri_same and cifar_same:
                n11 += 1
            elif tri_same and not cifar_same:
                n10 += 1
            elif not tri_same and cifar_same:
                n01 += 1
            else:
                n00 += 1
    n_total = K * (K - 1) // 2
    rand_idx = (n11 + n00) / n_total if n_total else 0.0
    print(f"  (k={k}) Rand index between trioron clusters and CIFAR superclasses:  {rand_idx:.4f}")
    print(f"    pair-agreement: same-in-both={n11}  diff-in-both={n00}")
    print(f"    disagreement:   tri-same/CIFAR-diff={n10}  tri-diff/CIFAR-same={n01}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
