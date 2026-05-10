"""L2 — within central-object refinement, classical senses.

The central-object macrocluster (from L1's discovered taxonomy) holds
9 visually-similar fine classes that the trioron groups together at
the coarsest level. The contrastive-method hypothesis says: this is
where the trioron's senses don't naturally separate, so this is where
δ-replay margin loss should have most leverage.

This bench trains a 9-class single-task donor on those fines alone,
no L1 prior, no contrastive yet. The accuracy answers:

  * How well can trioron discriminate within-cluster with classical
    senses alone? (chance = 1/9 = 0.111)
  * Which fine pairs within central-object are most confusable?
    (per-class accuracy + confusion matrix → identifies the pairs
    that contrastive method should target.)
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from typing import List

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from trioron.api import TaskData, TrioronConfig, build_donor
from trioron.network import TrioronNetwork
from trioron.senses import apply_sense, Standardizer
from experiments.cifar.datasets import load_cifar100, DEFAULT_DATA_ROOT
from experiments.cifar.bench_taxonomy_l1 import _resolve_names_to_ids


CENTRAL_OBJECT_NAMES = [
    "wolf", "man", "clock",
    "motorcycle", "pickup_truck",
    "rose", "spider", "butterfly", "mushroom",
]


def _build_subset(data_root: str, names: List[str]) -> tuple:
    name_to_id = _resolve_names_to_ids(data_root)
    fine_ids = [name_to_id[n] for n in names]
    id_to_local = {fid: i for i, fid in enumerate(fine_ids)}

    train_imgs, train_labs = load_cifar100(data_root, train=True)
    test_imgs, test_labs = load_cifar100(data_root, train=False)

    def _filter(images: torch.Tensor, labels: torch.Tensor):
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
    parser.add_argument("--sense", default="classical")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--cap-bytes", type=int, default=64_000)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--out-path",
        default="outputs/cifar_taxonomy/donor_l2_central_object_9way.pt",
    )
    args = parser.parse_args(argv)

    Xtr, ytr_local, ytr_fine, Xte, yte_local, yte_fine, fine_ids = (
        _build_subset(args.data_root, CENTRAL_OBJECT_NAMES)
    )
    print(f"[L2-central] sense={args.sense}  epochs={args.epochs}  "
          f"cap_bytes={args.cap_bytes}")
    print(f"[L2-central] train: {Xtr.shape[0]} imgs across "
          f"{len(CENTRAL_OBJECT_NAMES)} fine classes "
          f"({CENTRAL_OBJECT_NAMES})")
    print(f"[L2-central] test:  {Xte.shape[0]} imgs")
    print(f"[L2-central] chance (uniform 9-way): {1/9:.4f}")

    t0 = time.time()
    Xtr_sensed = apply_sense(args.sense, Xtr)
    Xte_sensed = apply_sense(args.sense, Xte)
    std = Standardizer.fit(Xtr_sensed)
    Xtr_sensed = std.transform(Xtr_sensed).contiguous()
    Xte_sensed = std.transform(Xte_sensed).contiguous()
    print(f"[L2-central] sense+standardize ({time.time()-t0:.1f}s): "
          f"{tuple(Xtr_sensed.shape)} train")

    tasks = [TaskData(
        name="L2_central_object_9way",
        X_train=Xtr_sensed,
        y_train=ytr_local,
        X_test=Xte_sensed,
        y_test=yte_local,
        classes=list(range(len(CENTRAL_OBJECT_NAMES))),
    )]

    out_path = os.path.abspath(args.out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cfg = TrioronConfig(cap_bytes=args.cap_bytes)
    t0 = time.time()
    build_donor(
        label="cifar100_taxonomy_l2_central_object",
        tasks=tasks,
        seed=args.seed,
        epochs_per_task=args.epochs,
        config=cfg,
        out_path=out_path,
    )
    print(f"[L2-central] build_donor done ({time.time()-t0:.1f}s)")

    payload = torch.load(out_path, map_location="cpu", weights_only=False)
    payload["sense"] = args.sense
    payload["standardizer"] = std.to_dict()
    payload["taxonomy_level"] = 2
    payload["taxonomy_level_name"] = "L2_central_object_9way"
    payload["fine_class_names"] = list(CENTRAL_OBJECT_NAMES)
    payload["fine_class_ids"] = list(fine_ids)
    torch.save(payload, out_path)

    # Per-class confusion + most-confused pair identification.
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
        logits = net(Xte_sensed)
    pred = logits.argmax(dim=1)
    acc = (pred == yte_local).float().mean().item()

    K = len(CENTRAL_OBJECT_NAMES)
    cm = torch.zeros(K, K, dtype=torch.long)
    for t, p in zip(yte_local.tolist(), pred.tolist()):
        cm[int(t), int(p)] += 1

    print(f"\n[L2-central] === results ===")
    print(f"  overall acc:        {acc:.4f}")
    print(f"  chance:             {1/K:.4f}")
    print(f"  margin over chance: {acc - 1/K:+.4f}")
    print(f"\n  per-class accuracy:")
    for i, n in enumerate(CENTRAL_OBJECT_NAMES):
        n_i = int((yte_local == i).sum().item())
        acc_i = cm[i, i].item() / max(n_i, 1)
        print(f"    {n:<14s}  n={n_i}  acc={acc_i:.4f}")

    # Most-confused pairs.
    pair_conf = []
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            c = cm[i, j].item()
            if c > 0:
                pair_conf.append(
                    (c, CENTRAL_OBJECT_NAMES[i], CENTRAL_OBJECT_NAMES[j])
                )
    pair_conf.sort(reverse=True)
    print(f"\n  top-10 most-confused (true → predicted):")
    for c, ti, pj in pair_conf[:10]:
        print(f"    {ti:<14s} → {pj:<14s}   {c:>3d} confusions")

    print(f"\n[L2-central] [SAVE] {out_path}  "
          f"({os.path.getsize(out_path)/1024:.1f} KB)  "
          f"input_dim={payload['input_dim']}  "
          f"head_size={payload['n_nodes_per_layer'][-1]}  "
          f"arch={payload['n_nodes_per_layer']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
