"""L2 — within-cluster expert trioron, generic cluster.

Trains a fine-class trioron on a single L1 macrocluster's fines.
Wraps the central-object-specific bench by accepting any cluster from
PERCEPTUAL_GROUPS:

  --cluster compact-object     → {chair, bottle, cup}                  (3-way)
  --cluster central-object     → 9 fines                               (9-way)
  --cluster horizontal-landscape → {dolphin, mountain}                 (2-way)
  --cluster vertical-landscape → {oak_tree, castle}                    (2-way)

Used to build the panel of L2 experts the dual-trioron 16-way eval
routes through.
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
from experiments.cifar.bench_taxonomy_l1 import _resolve_names_to_ids
from experiments.cifar.bench_taxonomy_l1_perceptual import PERCEPTUAL_GROUPS


def _build_subset(data_root: str, names):
    name_to_id = _resolve_names_to_ids(data_root)
    fine_ids = [name_to_id[n] for n in names]
    id_to_local = {fid: i for i, fid in enumerate(fine_ids)}

    train_imgs, train_labs = load_cifar100(data_root, train=True)
    test_imgs, test_labs = load_cifar100(data_root, train=False)

    def _filter(images, labels):
        keep = torch.zeros(labels.shape[0], dtype=torch.bool)
        for c in fine_ids:
            keep |= labels == c
        X = images[keep]
        y_fine = labels[keep]
        y_local = torch.tensor(
            [id_to_local[int(c)] for c in y_fine.tolist()],
            dtype=torch.long,
        )
        return X, y_local, y_fine

    Xtr, ytr_local, ytr_fine = _filter(train_imgs, train_labs)
    Xte, yte_local, yte_fine = _filter(test_imgs, test_labs)
    return Xtr, ytr_local, ytr_fine, Xte, yte_local, yte_fine, fine_ids


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster", required=True,
                        choices=[v["label"] for v in PERCEPTUAL_GROUPS.values()])
    parser.add_argument("--sense", default="classical")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--cap-bytes", type=int, default=64_000)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-path", default=None)
    args = parser.parse_args(argv)

    # Pick the cluster's fine names.
    cluster_names = None
    for v in PERCEPTUAL_GROUPS.values():
        if v["label"] == args.cluster:
            cluster_names = v["names"]
            break
    assert cluster_names is not None
    K = len(cluster_names)

    if args.out_path is None:
        cluster_slug = args.cluster.replace("-", "_")
        args.out_path = (
            f"outputs/cifar_taxonomy/donor_l2_{cluster_slug}_{K}way"
            f"_seed{args.seed}.pt"
        )

    Xtr, ytr_local, ytr_fine, Xte, yte_local, yte_fine, fine_ids = _build_subset(
        args.data_root, cluster_names,
    )
    print(f"[L2-{args.cluster}] sense={args.sense}  seed={args.seed}  "
          f"epochs={args.epochs}  cap_bytes={args.cap_bytes}")
    print(f"[L2-{args.cluster}] {K} fine classes: {cluster_names}")
    print(f"[L2-{args.cluster}] train: {Xtr.shape[0]} imgs  "
          f"test: {Xte.shape[0]} imgs  chance: 1/{K}={1/K:.4f}")

    Xtr_s = apply_sense(args.sense, Xtr)
    Xte_s = apply_sense(args.sense, Xte)
    std = Standardizer.fit(Xtr_s)
    Xtr_s = std.transform(Xtr_s).contiguous()
    Xte_s = std.transform(Xte_s).contiguous()

    tasks = [TaskData(
        name=f"L2_{args.cluster}_{K}way",
        X_train=Xtr_s, y_train=ytr_local,
        X_test=Xte_s, y_test=yte_local,
        classes=list(range(K)),
    )]

    out_path = os.path.abspath(args.out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cfg = TrioronConfig(cap_bytes=args.cap_bytes)
    t0 = time.time()
    build_donor(
        label=f"cifar100_taxonomy_l2_{args.cluster}",
        tasks=tasks,
        seed=args.seed,
        epochs_per_task=args.epochs,
        config=cfg,
        out_path=out_path,
    )
    print(f"[L2-{args.cluster}] build_donor done ({time.time()-t0:.1f}s)")

    # Bake metadata.
    payload = torch.load(out_path, map_location="cpu", weights_only=False)
    payload["sense"] = args.sense
    payload["standardizer"] = std.to_dict()
    payload["taxonomy_level"] = 2
    payload["taxonomy_level_name"] = f"L2_{args.cluster}_{K}way"
    payload["fine_class_names"] = list(cluster_names)
    payload["fine_class_ids"] = list(fine_ids)
    payload["cluster"] = args.cluster
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
        pred = net(Xte_s).argmax(dim=1)
    acc = (pred == yte_local).float().mean().item()
    print(f"\n[L2-{args.cluster}] === results ===")
    print(f"  acc:                {acc:.4f}")
    print(f"  chance:             {1/K:.4f}")
    print(f"  margin over chance: {acc - 1/K:+.4f}")
    print(f"  per-class:")
    for i, n in enumerate(cluster_names):
        n_i = int((yte_local == i).sum().item())
        if n_i == 0:
            continue
        acc_i = (pred[yte_local == i] == i).float().mean().item()
        print(f"    {n:<14s}  n={n_i}  acc={acc_i:.4f}")
    print(f"\n[L2-{args.cluster}] [SAVE] {out_path}  "
          f"({os.path.getsize(out_path)/1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
