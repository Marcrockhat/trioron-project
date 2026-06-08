"""L2 — within-central-object refinement via api.extend (developmental
growth of the trioron substrate).

This is the version Rocky asked for: not a fresh L2 donor on the 9
central-object fines (that's `bench_taxonomy_l2_central_object.py` —
the from-scratch baseline), but the L1 perceptual donor *grown* to
encode the within-central-object distinctions on top of its existing
4-way macroclass structure.

Design:

  * base_tasks: the imbalanced L1 perceptual 4-way task on the 16-
    class subset (all 16 fines mapped to 4 macro labels [0..3]).
    The L1 donor was trained on exactly this curriculum.

  * new_tasks: the 9-way central-object refinement. New class labels
    [4..12] (disjoint from L1's [0..3]) — wolf=4, man=5, clock=6,
    motorcycle=7, pickup_truck=8, rose=9, spider=10, butterfly=11,
    mushroom=12. 9 fines, 4500 train + 900 test images.

  * api.extend handles: ship-wake-extend (boundary consolidation
    dream over base_tasks → archive-lock), permanent int8 quant on
    archived rows, growth-budget lift, then training on new_tasks.

After extension, the trioron has a 13-column head: 4 macro logits +
9 fine logits. Task-aware eval restricts to the relevant column
subset for each level.
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from trioron.api import TaskData, TrioronConfig, extend
from trioron.network import TrioronNetwork
from trioron.senses import apply_sense, Standardizer
from experiments.cifar.datasets import load_cifar100, DEFAULT_DATA_ROOT
from experiments.cifar.bench_taxonomy_l1 import _resolve_names_to_ids
from experiments.cifar.bench_taxonomy_l1_perceptual import (
    PERCEPTUAL_GROUPS, _build_subset as _build_l1_subset,
)
from experiments.cifar.bench_taxonomy_l2_central_object import (
    CENTRAL_OBJECT_NAMES,
)


def _build_l1_tasks(data_root: str, std: Standardizer, sense_name: str):
    Xtr, ytr_perc, _, Xte, yte_perc, _, _, _ = _build_l1_subset(data_root)
    Xtr_sensed = std.transform(apply_sense(sense_name, Xtr)).contiguous()
    Xte_sensed = std.transform(apply_sense(sense_name, Xte)).contiguous()
    return [TaskData(
        name="L1_perceptual_4way",
        X_train=Xtr_sensed, y_train=ytr_perc,
        X_test=Xte_sensed,  y_test=yte_perc,
        classes=list(PERCEPTUAL_GROUPS),
    )]


def _build_l2_tasks(data_root: str, std: Standardizer, sense_name: str,
                    base_offset: int):
    """L2 9-way central-object refinement. Labels offset by `base_offset`
    (= 4 = number of L1 classes) so they're disjoint from L1's [0..3]."""
    name_to_id = _resolve_names_to_ids(data_root)
    fine_ids = [name_to_id[n] for n in CENTRAL_OBJECT_NAMES]
    id_to_extlabel = {fid: i + base_offset
                      for i, fid in enumerate(fine_ids)}

    train_imgs, train_labs = load_cifar100(data_root, train=True)
    test_imgs, test_labs = load_cifar100(data_root, train=False)

    def _filter(images, labels):
        keep = torch.zeros(labels.shape[0], dtype=torch.bool)
        for c in fine_ids:
            keep |= labels == c
        X = images[keep]
        y = torch.tensor(
            [id_to_extlabel[int(c)] for c in labels[keep].tolist()],
            dtype=torch.long,
        )
        return X, y

    Xtr, ytr = _filter(train_imgs, train_labs)
    Xte, yte = _filter(test_imgs, test_labs)
    Xtr_sensed = std.transform(apply_sense(sense_name, Xtr)).contiguous()
    Xte_sensed = std.transform(apply_sense(sense_name, Xte)).contiguous()
    return [TaskData(
        name="L2_central_object_refine",
        X_train=Xtr_sensed, y_train=ytr,
        X_test=Xte_sensed,  y_test=yte,
        classes=list(range(base_offset, base_offset + len(fine_ids))),
    )], CENTRAL_OBJECT_NAMES


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--l1-donor-path",
        default="outputs/cifar_taxonomy/donor_l1_perceptual_4way.pt",
        help="Imbalanced 4-way L1 donor (16-class subset) — substrate "
             "for the expansion. Balanced L1 donor only saw 8 fines and "
             "would force re-training during extend.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--extension-cap-bytes", type=int, default=128_000,
                        help="Lift cap from L1's 64K to 128K for L2 growth.")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--out-path",
        default="outputs/cifar_taxonomy/donor_l2_expanded_from_l1.pt",
    )
    args = parser.parse_args(argv)

    payload = torch.load(args.l1_donor_path, map_location="cpu",
                         weights_only=False)
    sense_name = payload["sense"]
    std = Standardizer.from_dict(payload["standardizer"])
    base_classes = payload["classes_covered"]
    print(f"[L2-expand] L1 donor: {args.l1_donor_path}")
    print(f"[L2-expand]   sense={sense_name}  arch={payload['n_nodes_per_layer']}")
    print(f"[L2-expand]   L1 classes (base): {base_classes}")

    base_tasks = _build_l1_tasks(args.data_root, std, sense_name)
    new_tasks, fine_names = _build_l2_tasks(
        args.data_root, std, sense_name,
        base_offset=len(base_classes),
    )
    print(f"[L2-expand] base_tasks: 1 task with classes "
          f"{base_tasks[0].classes}, train {base_tasks[0].X_train.shape[0]}")
    print(f"[L2-expand] new_tasks:  1 task with classes "
          f"{new_tasks[0].classes}, train {new_tasks[0].X_train.shape[0]}")
    print(f"[L2-expand]   L2 fine names: {fine_names}")

    out_path = os.path.abspath(args.out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    t0 = time.time()
    extend(
        donor_path=args.l1_donor_path,
        base_tasks=base_tasks,
        new_tasks=new_tasks,
        out_path=out_path,
        extension_cap_bytes=args.extension_cap_bytes,
        epochs_per_task=args.epochs,
        permanent_int8=True,
    )
    print(f"[L2-expand] api.extend done ({time.time()-t0:.1f}s)")

    # Manual eval — restrict softmax to L1 columns vs L2 columns.
    payload = torch.load(out_path, map_location="cpu", weights_only=False)
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

    # L1 eval — task-aware on the 4 macro columns.
    Xte_l1, yte_l1 = base_tasks[0].X_test, base_tasks[0].y_test
    with torch.no_grad():
        logits_l1 = net(Xte_l1)[:, base_classes]
    pred_l1_local = logits_l1.argmax(dim=1)
    pred_l1 = torch.tensor([base_classes[i] for i in pred_l1_local.tolist()],
                           dtype=yte_l1.dtype)
    acc_l1 = (pred_l1 == yte_l1).float().mean().item()

    # L2 eval — task-aware on the 9 fine columns.
    l2_classes = new_tasks[0].classes
    Xte_l2, yte_l2 = new_tasks[0].X_test, new_tasks[0].y_test
    with torch.no_grad():
        logits_l2 = net(Xte_l2)[:, l2_classes]
    pred_l2_local = logits_l2.argmax(dim=1)
    pred_l2 = torch.tensor([l2_classes[i] for i in pred_l2_local.tolist()],
                           dtype=yte_l2.dtype)
    acc_l2 = (pred_l2 == yte_l2).float().mean().item()

    # Per-class L2 accuracy.
    K2 = len(l2_classes)
    cm_l2 = torch.zeros(K2, K2, dtype=torch.long)
    yte_l2_local = (yte_l2 - len(base_classes))
    pred_l2_localv = (pred_l2 - len(base_classes))
    for t, p in zip(yte_l2_local.tolist(), pred_l2_localv.tolist()):
        cm_l2[int(t), int(p)] += 1

    print(f"\n[L2-expand] === results ===")
    print(f"  L1 task-aware (4-way macro):  {acc_l1:.4f}  "
          f"(prior n=1 imbalanced donor: 0.7831; chance 0.25)")
    print(f"  L2 task-aware (9-way fine):   {acc_l2:.4f}  "
          f"(fresh L2 baseline n=1: 0.4389; chance 0.111)")
    print(f"  total head columns: {payload['n_nodes_per_layer'][-1]}  "
          f"arch={payload['n_nodes_per_layer']}")
    print(f"\n  L2 per-class accuracy:")
    for i, n in enumerate(fine_names):
        n_i = int((yte_l2_local == i).sum().item())
        acc_i = cm_l2[i, i].item() / max(n_i, 1)
        print(f"    {n:<14s}  n={n_i}  acc={acc_i:.4f}")

    # Most-confused L2 pairs.
    pair_conf = []
    for i in range(K2):
        for j in range(K2):
            if i == j:
                continue
            c = cm_l2[i, j].item()
            if c > 0:
                pair_conf.append((c, fine_names[i], fine_names[j]))
    pair_conf.sort(reverse=True)
    print(f"\n  L2 top-10 confusions (true → predicted):")
    for c, ti, pj in pair_conf[:10]:
        print(f"    {ti:<14s} → {pj:<14s}   {c:>3d}")

    print(f"\n[L2-expand] [SAVE] {out_path}  "
          f"({os.path.getsize(out_path)/1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
