"""L1 macroclassifier for full 100-class CIFAR-100.

Trains a single trioron donor on the full CIFAR-100 train set with
fine-class labels remapped to k macroclusters (from the trioron-
discovered cluster assignment file). All 50K train images, k-way head.

Args:
  --cluster-file: outputs/cifar_taxonomy/cluster_assignment_full100_k{k}.pt
  --seed: trioron + numpy seed (42, 43, 44 for n=3)
  --out-path: where to save the L1 macroclassifier donor
"""
from __future__ import annotations
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from trioron.api import TaskData, TrioronConfig, build_donor
from trioron.network import TrioronNetwork
from trioron.senses import apply_sense, Standardizer
from experiments.cifar.datasets import DEFAULT_DATA_ROOT, load_cifar100


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster-file",
                        default="outputs/cifar_taxonomy/cluster_assignment_full100_k20.pt")
    parser.add_argument("--sense", default="classical")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--cap-bytes", type=int, default=200_000)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-path", default=None)
    args = parser.parse_args(argv)

    # Load cluster assignment.
    ca = torch.load(args.cluster_file, map_location="cpu", weights_only=False)
    K = ca["k"]
    fine_to_cluster = ca["fine_to_cluster"]   # 100-long list of cluster ids
    print(f"[full100-l1] cluster file: {args.cluster_file}")
    print(f"[full100-l1]   k={K}  cluster sizes: "
          f"{sorted([len(c) for c in ca['clusters']], reverse=True)}")

    if args.out_path is None:
        args.out_path = f"outputs/cifar_taxonomy/donor_full100_l1_k{K}_seed{args.seed}.pt"

    # Load full CIFAR-100.
    train_imgs, train_labs = load_cifar100(args.data_root, train=True)
    test_imgs, test_labs = load_cifar100(args.data_root, train=False)
    print(f"[full100-l1] train: {train_imgs.shape[0]}  test: {test_imgs.shape[0]}")

    # Map fine → cluster.
    cluster_t = torch.tensor(fine_to_cluster, dtype=torch.long)
    ytr_macro = cluster_t[train_labs.long()]
    yte_macro = cluster_t[test_labs.long()]

    # Apply sense + standardize on full training set.
    t0 = time.time()
    Xtr_sensed = apply_sense(args.sense, train_imgs)
    Xte_sensed = apply_sense(args.sense, test_imgs)
    std = Standardizer.fit(Xtr_sensed)
    Xtr_sensed = std.transform(Xtr_sensed).contiguous()
    Xte_sensed = std.transform(Xte_sensed).contiguous()
    print(f"[full100-l1] sense+standardize ({time.time()-t0:.1f}s)")

    tasks = [TaskData(
        name=f"full100_L1_k{K}",
        X_train=Xtr_sensed,
        y_train=ytr_macro,
        X_test=Xte_sensed,
        y_test=yte_macro,
        classes=list(range(K)),
    )]

    out_path = os.path.abspath(args.out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cfg = TrioronConfig(cap_bytes=args.cap_bytes)
    t0 = time.time()
    build_donor(
        label=f"cifar100_full100_L1_k{K}",
        tasks=tasks,
        seed=args.seed,
        epochs_per_task=args.epochs,
        config=cfg,
        out_path=out_path,
    )
    print(f"[full100-l1] build_donor done ({time.time()-t0:.1f}s)")

    payload = torch.load(out_path, map_location="cpu", weights_only=False)
    payload["sense"] = args.sense
    payload["standardizer"] = std.to_dict()
    payload["taxonomy_level"] = 1
    payload["taxonomy_k"] = K
    payload["fine_to_cluster"] = fine_to_cluster
    payload["clusters"] = ca["clusters"]
    torch.save(payload, out_path)

    # Eval.
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
    with torch.no_grad():
        pred = net(Xte_sensed).argmax(dim=1)
    acc = (pred == yte_macro).float().mean().item()
    largest = max(
        (yte_macro == c).float().mean().item() for c in range(K)
    )
    print(f"\n[full100-l1] === results ===")
    print(f"  L1 macro acc:                 {acc:.4f}")
    print(f"  always-predict-largest:       {largest:.4f}")
    print(f"  margin over largest baseline: {acc - largest:+.4f}")
    print(f"  uniform chance (1/{K}):         {1/K:.4f}")
    print(f"\n[full100-l1] [SAVE] {out_path}  "
          f"({os.path.getsize(out_path)/1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
