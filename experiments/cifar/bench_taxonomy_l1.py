"""L1 — Kingdom split: Living vs Non-living.

The first level of a Linnaean-style taxonomic curriculum. Classical
senses only (no MobileNet / cortex), to keep the pipeline pure-trioron.

Step 1 of the contrastive-method scaffold:
  * No contrastive replay yet (vanilla trioron training).
  * No multi-level growth yet (single task, single phase).
  * Goal: pipeline validation. Confirm that classical-sense input through
    a 2-class trioron head can separate Living from Non-living above
    chance, before adding the contrastive machinery.

Class list (16 fine classes, 8 per side, balanced across kingdoms within
each side so the contrast isn't accidentally just "wolves vs cars"):

  Living  (label 0): wolf, dolphin, butterfly, spider, man,
                     oak_tree, rose, mushroom
  Non-liv (label 1): motorcycle, pickup_truck, chair, bottle,
                     castle, clock, mountain, cup
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
from trioron.senses import apply_sense, sense_dim, Standardizer
from experiments.cifar.datasets import load_cifar100, DEFAULT_DATA_ROOT


# Class names → CIFAR-100 fine indices via torchvision metadata; names
# kept as the source of truth so the script reads at the taxonomic
# level, not at the integer-id level.
LIVING_NAMES = [
    "wolf", "dolphin", "butterfly", "spider",
    "man", "oak_tree", "rose", "mushroom",
]
NON_LIVING_NAMES = [
    "motorcycle", "pickup_truck", "chair", "bottle",
    "castle", "clock", "mountain", "cup",
]


def _resolve_names_to_ids(root: str) -> dict:
    """Get CIFAR-100 {fine_class_name: int_id} from the torchvision dataset."""
    from torchvision.datasets import CIFAR100
    ds = CIFAR100(root=root, train=True, download=True)
    return {name: i for i, name in enumerate(ds.classes)}


def _binary_subset(
    images: torch.Tensor,
    labels: torch.Tensor,
    living_ids: List[int],
    nonliving_ids: List[int],
) -> tuple:
    """Filter to (living ∪ non-living) and remap fine labels to {0, 1}."""
    living_set = set(int(c) for c in living_ids)
    nonliving_set = set(int(c) for c in nonliving_ids)
    keep = torch.zeros(labels.shape[0], dtype=torch.bool)
    for c in living_set | nonliving_set:
        keep |= labels == int(c)
    X = images[keep]
    y_fine = labels[keep]
    y_bin = torch.where(
        torch.tensor([int(c) in living_set for c in y_fine.tolist()]),
        torch.zeros_like(y_fine),
        torch.ones_like(y_fine),
    )
    return X, y_bin, y_fine


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sense", default="classical",
                        help="Default = classical (33-d closed-form senses). "
                             "Override with cortex / sensorium / etc. for ablation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--cap-bytes", type=int, default=64_000)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--out-path",
        default="outputs/cifar_taxonomy/donor_l1_living_vs_nonliving.pt",
    )
    args = parser.parse_args(argv)

    name_to_id = _resolve_names_to_ids(args.data_root)
    living_ids = [name_to_id[n] for n in LIVING_NAMES]
    nonliving_ids = [name_to_id[n] for n in NON_LIVING_NAMES]
    print(f"[L1] living classes:     {LIVING_NAMES}")
    print(f"[L1]   ids:              {living_ids}")
    print(f"[L1] non-living classes: {NON_LIVING_NAMES}")
    print(f"[L1]   ids:              {nonliving_ids}")
    print(f"[L1] sense={args.sense}  input_dim={sense_dim(args.sense)}  "
          f"epochs={args.epochs}  cap_bytes={args.cap_bytes}")

    # Load and subset.
    train_imgs, train_labs = load_cifar100(args.data_root, train=True)
    test_imgs, test_labs = load_cifar100(args.data_root, train=False)
    Xtr_raw, ytr_bin, ytr_fine = _binary_subset(
        train_imgs, train_labs, living_ids, nonliving_ids,
    )
    Xte_raw, yte_bin, yte_fine = _binary_subset(
        test_imgs, test_labs, living_ids, nonliving_ids,
    )
    n_living_tr = int((ytr_bin == 0).sum().item())
    n_nonliving_tr = int((ytr_bin == 1).sum().item())
    print(f"[L1] train: {Xtr_raw.shape[0]} imgs "
          f"({n_living_tr} Living, {n_nonliving_tr} Non-living)")
    print(f"[L1] test:  {Xte_raw.shape[0]} imgs "
          f"({int((yte_bin == 0).sum())} Living, "
          f"{int((yte_bin == 1).sum())} Non-living)")

    # Apply sense.
    t0 = time.time()
    Xtr_sensed = apply_sense(args.sense, Xtr_raw)
    Xte_sensed = apply_sense(args.sense, Xte_raw)
    std = Standardizer.fit(Xtr_sensed)
    Xtr_sensed = std.transform(Xtr_sensed).contiguous()
    Xte_sensed = std.transform(Xte_sensed).contiguous()
    print(f"[L1] sense+standardize: {Xtr_sensed.shape} train, "
          f"{Xte_sensed.shape} test  ({time.time()-t0:.1f}s)")

    # Single-task binary head: classes [0, 1].
    tasks = [TaskData(
        name="L1_living_vs_nonliving",
        X_train=Xtr_sensed,
        y_train=ytr_bin,
        X_test=Xte_sensed,
        y_test=yte_bin,
        classes=[0, 1],
    )]

    out_path = os.path.abspath(args.out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cfg = TrioronConfig(cap_bytes=args.cap_bytes)
    t0 = time.time()
    build_donor(
        label="cifar100_taxonomy_l1",
        tasks=tasks,
        seed=args.seed,
        epochs_per_task=args.epochs,
        config=cfg,
        out_path=out_path,
    )
    print(f"[L1] build_donor done ({time.time()-t0:.1f}s)")

    # Bake metadata for downstream level builders.
    payload = torch.load(out_path, map_location="cpu", weights_only=False)
    payload["sense"] = args.sense
    payload["standardizer"] = std.to_dict()
    payload["taxonomy_level"] = 1
    payload["taxonomy_level_name"] = "L1_living_vs_nonliving"
    payload["binary_label_map"] = {
        "0_living": LIVING_NAMES,
        "1_nonliving": NON_LIVING_NAMES,
    }
    torch.save(payload, out_path)

    size_kb = os.path.getsize(out_path) / 1024.0
    print(f"[L1] [SAVE] {out_path}  ({size_kb:.1f} KB)  "
          f"input_dim={payload['input_dim']}  "
          f"head_size={payload['n_nodes_per_layer'][-1]}  "
          f"arch={payload['n_nodes_per_layer']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
