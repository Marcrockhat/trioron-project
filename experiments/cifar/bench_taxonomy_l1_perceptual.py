"""L1 — train donor on trioron's OWN 4-way perceptual taxonomy.

The data-driven version of L1. Replaces the human "Living vs Non-living"
binary (which the cluster probe showed is invisible to classical senses)
with the four perceptual macroclasses agglomerative clustering on L0
centroids actually discovered:

  0 — compact-object         {chair, bottle, cup}                  (3)
  1 — central-object         {wolf, man, clock, motorcycle,        (9)
                              pickup_truck, rose, spider,
                              butterfly, mushroom}
  2 — horizontal-landscape   {dolphin, mountain}                   (2)
  3 — vertical-landscape     {oak_tree, castle}                    (2)

Same 16 fine classes, same classical senses, same trioron config —
only the training labels differ. If aligning the curriculum with the
trioron's native ontology helps, we expect a much wider margin over
the always-predict-largest baseline (9/16=0.5625) than the 2-way
binary's margin over chance (0.50 → 0.7462).
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

from trioron.api import TaskData, TrioronConfig, build_donor
from trioron.senses import apply_sense, Standardizer
from experiments.cifar.datasets import load_cifar100, DEFAULT_DATA_ROOT
from experiments.cifar.bench_taxonomy_l1 import _resolve_names_to_ids


# Trioron's discovered 4-way taxonomy (from probe_taxonomy_l1_cluster
# at k=4, linkage=average, metric=cosine on the L1 donor's L0 space).
PERCEPTUAL_GROUPS: Dict[int, Dict[str, object]] = {
    0: {"label": "compact-object",
        "names": ["chair", "bottle", "cup"]},
    1: {"label": "central-object",
        "names": ["wolf", "man", "clock", "motorcycle", "pickup_truck",
                  "rose", "spider", "butterfly", "mushroom"]},
    2: {"label": "horizontal-landscape",
        "names": ["dolphin", "mountain"]},
    3: {"label": "vertical-landscape",
        "names": ["oak_tree", "castle"]},
}


def _build_subset(
    data_root: str,
) -> tuple:
    """Return (Xtr, ytr_perc, ytr_fine, Xte, yte_perc, yte_fine,
                fine_to_perc, name_to_id)."""
    name_to_id = _resolve_names_to_ids(data_root)
    fine_to_perc: Dict[int, int] = {}
    for perc_id, info in PERCEPTUAL_GROUPS.items():
        for n in info["names"]:
            fine_to_perc[name_to_id[n]] = perc_id
    keep_ids = sorted(fine_to_perc)

    train_imgs, train_labs = load_cifar100(data_root, train=True)
    test_imgs, test_labs = load_cifar100(data_root, train=False)

    def _filter(images: torch.Tensor, labels: torch.Tensor):
        keep = torch.zeros(labels.shape[0], dtype=torch.bool)
        for c in keep_ids:
            keep |= labels == c
        X = images[keep]
        y_fine = labels[keep]
        y_perc = torch.tensor(
            [fine_to_perc[int(c)] for c in y_fine.tolist()],
            dtype=torch.long,
        )
        return X, y_perc, y_fine

    Xtr, ytr_perc, ytr_fine = _filter(train_imgs, train_labs)
    Xte, yte_perc, yte_fine = _filter(test_imgs, test_labs)
    return Xtr, ytr_perc, ytr_fine, Xte, yte_perc, yte_fine, fine_to_perc, name_to_id


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sense", default="classical")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--cap-bytes", type=int, default=64_000)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--out-path",
        default="outputs/cifar_taxonomy/donor_l1_perceptual_4way.pt",
    )
    args = parser.parse_args(argv)

    Xtr, ytr_perc, ytr_fine, Xte, yte_perc, yte_fine, fine_to_perc, name_to_id = (
        _build_subset(args.data_root)
    )
    print(f"[L1-4way] sense={args.sense}  epochs={args.epochs}  "
          f"cap_bytes={args.cap_bytes}")
    print(f"[L1-4way] train: {Xtr.shape[0]} imgs "
          f"({len(fine_to_perc)} fine classes)")
    for pid, info in PERCEPTUAL_GROUPS.items():
        n_tr = int((ytr_perc == pid).sum().item())
        n_te = int((yte_perc == pid).sum().item())
        print(f"  perc={pid} ({info['label']:<22s})  "
              f"fines={info['names']}  "
              f"train={n_tr}  test={n_te}")

    # Sense + standardize.
    t0 = time.time()
    Xtr_sensed = apply_sense(args.sense, Xtr)
    Xte_sensed = apply_sense(args.sense, Xte)
    std = Standardizer.fit(Xtr_sensed)
    Xtr_sensed = std.transform(Xtr_sensed).contiguous()
    Xte_sensed = std.transform(Xte_sensed).contiguous()
    print(f"[L1-4way] sense+standardize ({time.time()-t0:.1f}s): "
          f"{tuple(Xtr_sensed.shape)} train")

    tasks = [TaskData(
        name="L1_perceptual_4way",
        X_train=Xtr_sensed,
        y_train=ytr_perc,
        X_test=Xte_sensed,
        y_test=yte_perc,
        classes=list(PERCEPTUAL_GROUPS),
    )]

    out_path = os.path.abspath(args.out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cfg = TrioronConfig(cap_bytes=args.cap_bytes)
    t0 = time.time()
    build_donor(
        label="cifar100_taxonomy_l1_4way",
        tasks=tasks,
        seed=args.seed,
        epochs_per_task=args.epochs,
        config=cfg,
        out_path=out_path,
    )
    print(f"[L1-4way] build_donor done ({time.time()-t0:.1f}s)")

    # Bake metadata.
    payload = torch.load(out_path, map_location="cpu", weights_only=False)
    payload["sense"] = args.sense
    payload["standardizer"] = std.to_dict()
    payload["taxonomy_level"] = 1
    payload["taxonomy_level_name"] = "L1_perceptual_4way"
    payload["perceptual_groups"] = {
        str(k): {"label": v["label"], "names": v["names"]}
        for k, v in PERCEPTUAL_GROUPS.items()
    }
    payload["fine_to_perc"] = {str(k): v for k, v in fine_to_perc.items()}
    torch.save(payload, out_path)
    size_kb = os.path.getsize(out_path) / 1024.0

    # Per-class confusion on test set.
    from trioron.network import TrioronNetwork
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
    acc_overall = (pred == yte_perc).float().mean().item()
    largest_baseline = max(
        (yte_perc == pid).float().mean().item()
        for pid in PERCEPTUAL_GROUPS
    )
    print(f"\n[L1-4way] === results ===")
    print(f"  overall accuracy:                 {acc_overall:.4f}")
    print(f"  always-predict-largest baseline:  {largest_baseline:.4f}")
    print(f"  margin over largest-baseline:     "
          f"{acc_overall - largest_baseline:+.4f}")
    print(f"  comparison to L1 binary (0.7462 with chance 0.50): "
          f"binary margin was +0.2462")
    print(f"  per-class accuracy:")
    for pid, info in PERCEPTUAL_GROUPS.items():
        mask = yte_perc == pid
        if mask.sum() == 0:
            continue
        acc_c = (pred[mask] == pid).float().mean().item()
        print(f"    perc={pid} ({info['label']:<22s})  n={int(mask.sum())}  "
              f"acc={acc_c:.4f}")

    print(f"\n[L1-4way] [SAVE] {out_path}  ({size_kb:.1f} KB)  "
          f"input_dim={payload['input_dim']}  "
          f"head_size={payload['n_nodes_per_layer'][-1]}  "
          f"arch={payload['n_nodes_per_layer']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
