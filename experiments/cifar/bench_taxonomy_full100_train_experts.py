"""Train all L2 experts for the full-100 pipeline.

Iterates over the cluster assignment file and trains one L2 expert per
multi-class cluster. Singleton clusters (size=1) are skipped — at
hierarchical inference time, an L1 macro-prediction of a singleton's
cluster directly maps to that singleton's fine class.
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


def _train_one(cluster_id, cluster_names, args, name_to_id,
                train_imgs, train_labs, test_imgs, test_labs):
    K = len(cluster_names)
    if K < 2:
        print(f"[full100-l2] cluster c{cluster_id} has {K} members "
              f"({cluster_names}) — skipping (singleton)")
        return None

    fine_ids = [name_to_id[n] for n in cluster_names]
    id_to_local = {fid: i for i, fid in enumerate(fine_ids)}

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
        return X, y_local

    Xtr, ytr_local = _filter(train_imgs, train_labs)
    Xte, yte_local = _filter(test_imgs, test_labs)
    Xtr_s = apply_sense(args.sense, Xtr)
    Xte_s = apply_sense(args.sense, Xte)
    std = Standardizer.fit(Xtr_s)
    Xtr_s = std.transform(Xtr_s).contiguous()
    Xte_s = std.transform(Xte_s).contiguous()

    tasks = [TaskData(
        name=f"full100_L2_c{cluster_id}_{K}way",
        X_train=Xtr_s, y_train=ytr_local,
        X_test=Xte_s, y_test=yte_local,
        classes=list(range(K)),
    )]

    out_path = (f"outputs/cifar_taxonomy/donor_full100_l2_c{cluster_id:02d}_"
                f"{K}way_seed{args.seed}.pt")
    cfg = TrioronConfig(cap_bytes=args.cap_bytes)
    t0 = time.time()
    build_donor(
        label=f"cifar100_full100_L2_c{cluster_id}",
        tasks=tasks, seed=args.seed,
        epochs_per_task=args.epochs,
        config=cfg, out_path=out_path,
    )
    payload = torch.load(out_path, map_location="cpu", weights_only=False)
    payload["sense"] = args.sense
    payload["standardizer"] = std.to_dict()
    payload["taxonomy_level"] = 2
    payload["fine_class_names"] = list(cluster_names)
    payload["fine_class_ids"] = list(fine_ids)
    payload["cluster"] = f"c{cluster_id:02d}"
    payload["cluster_id"] = cluster_id
    torch.save(payload, out_path)

    net_layers = []
    prev = int(payload["input_dim"])
    for i, n in enumerate(payload["n_nodes_per_layer"]):
        act = "linear" if i == len(payload["n_nodes_per_layer"]) - 1 else "relu"
        net_layers.append((prev, int(n), act))
        prev = int(n)
    net = TrioronNetwork(net_layers)
    net.load_state_dict(payload["state_dict"])
    net.eval()
    with torch.no_grad():
        pred = net(Xte_s).argmax(dim=1)
    acc = (pred == yte_local).float().mean().item()
    print(f"[full100-l2] c{cluster_id:02d} ({K}-way, members={cluster_names[:3]}"
          f"{'…' if K > 3 else ''}): acc={acc:.4f}  "
          f"chance={1/K:.4f}  ({time.time()-t0:.1f}s)")
    return out_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster-file",
                        default="outputs/cifar_taxonomy/cluster_assignment_full100_k20.pt")
    parser.add_argument("--sense", default="classical")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--cap-bytes", type=int, default=64_000)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    args = parser.parse_args(argv)

    ca = torch.load(args.cluster_file, map_location="cpu", weights_only=False)
    print(f"[full100-l2] cluster file: {args.cluster_file}  k={ca['k']}")

    name_to_id = _resolve_names_to_ids(args.data_root)
    train_imgs, train_labs = load_cifar100(args.data_root, train=True)
    test_imgs, test_labs = load_cifar100(args.data_root, train=False)

    saved = []
    skipped = []
    for cid, names in enumerate(ca["clusters"]):
        path = _train_one(cid, names, args, name_to_id,
                          train_imgs, train_labs, test_imgs, test_labs)
        if path is None:
            skipped.append((cid, names))
        else:
            saved.append(path)
    print(f"\n[full100-l2] trained {len(saved)} experts; "
          f"skipped {len(skipped)} singletons "
          f"({[(c, n[0]) for c, n in skipped]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
