"""Train one CIFAR-100 primitive donor for a single sense.

Each donor is trained on the readings of ONE sense over a slice of
CIFAR-100, with the standardizer that brought the sense to z-scored
form saved alongside the network so inference applies the same
transform. Donors are independent (each has its own L0 random
projection); they're fused at conduct time by `SensoryConductor`.
"""
from __future__ import annotations
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from trioron.api import build_donor, TrioronConfig, AdvancedConfig
from trioron.senses import SENSES, sense_dim
from experiments.cifar.datasets import (
    build_sense_tasks, SLICES, DEFAULT_DATA_ROOT,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sense", required=True, choices=sorted(SENSES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument(
        "--slice", choices=sorted(SLICES), default="first",
        help="Which class slice to train on: 'first' = 25 classes / 5 tasks; "
             "'full' = 100 classes / 20 tasks.",
    )
    parser.add_argument(
        "--cap-bytes", type=int, default=None,
        help="Trainable parameter byte cap. Default scales with slice: "
             "64K for 'first' (16K params), 200K for 'full' (50K params).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Default scales with slice: outputs/cifar_donors for 'first', "
             "outputs/cifar_donors_full for 'full'.",
    )
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    args = parser.parse_args(argv)

    class_groups = SLICES[args.slice]
    if args.cap_bytes is None:
        args.cap_bytes = 64_000 if args.slice == "first" else 200_000
    if args.out_dir is None:
        sub = "cifar_donors" if args.slice == "first" else "cifar_donors_full"
        args.out_dir = os.path.join(os.path.dirname(DEFAULT_DATA_ROOT), sub)

    sd = sense_dim(args.sense)
    print(f"[train_donor_cifar] sense={args.sense}  seed={args.seed}  "
          f"epochs={args.epochs}  input_dim={sd}  cap_bytes={args.cap_bytes}  "
          f"slice={args.slice}  n_tasks={len(class_groups)}  "
          f"n_classes={sum(len(g) for g in class_groups)}")
    t0 = time.time()
    tasks, std = build_sense_tasks(
        args.sense, class_groups, root=args.data_root,
    )
    print(f"[train_donor_cifar] curriculum: {len(tasks)} tasks, "
          f"classes={sorted({c for t in tasks for c in t.classes})}  "
          f"({time.time()-t0:.1f}s data prep)")

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"sense_donor_{args.sense}.pt")

    cfg = TrioronConfig(cap_bytes=args.cap_bytes)
    t0 = time.time()
    build_donor(
        label=f"cifar100_{args.sense}",
        tasks=tasks,
        seed=args.seed,
        epochs_per_task=args.epochs,
        config=cfg,
        out_path=out_path,
    )
    print(f"[train_donor_cifar] build_donor done ({time.time()-t0:.1f}s)")

    # Bake the sense name + standardizer into the donor checkpoint so
    # inference can reconstruct the exact (sense, standardizer, net)
    # triple without out-of-band metadata.
    payload = torch.load(out_path, map_location="cpu", weights_only=False)
    payload["sense"] = args.sense
    payload["standardizer"] = std.to_dict()
    torch.save(payload, out_path)

    size_kb = os.path.getsize(out_path) / 1024.0
    print(f"[SAVE] {out_path}  ({size_kb:.1f} KB)  "
          f"input_dim={payload['input_dim']}  "
          f"head_size={payload['n_nodes_per_layer'][-1]}  "
          f"arch={payload['n_nodes_per_layer']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
