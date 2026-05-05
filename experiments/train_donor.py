"""Train one donor (skill pack) on a configurable chained-15 sub-block.

Generalization of `experiments/poc_dual_organism.py` — same arm and
training protocol, but parameterized so we can produce arbitrary skill
packs (digits, fashion, emnist letters K..Z, etc.) at the same shared L0
seed for absorption testing. Saves a self-contained donor checkpoint at
`outputs/poc_donor_<label>.pt`.

Run examples:
  # MNIST 0..9 (chained_15_specs[0:5], global classes 0..9)
  python3 -m experiments.train_donor --label digits

  # Fashion 0..9 (chained_15_specs[5:10], global classes 10..19)
  python3 -m experiments.train_donor --label fashion

  # EMNIST letters A..J (chained_15_specs[10:15], global classes 20..29)
  python3 -m experiments.train_donor --label emnist
"""
from __future__ import annotations
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from experiments import bench_chained_15task as bench
from experiments.datasets import (
    DatasetBundle, build_task_views, chained_15_specs,
    chained_extension_specs,
)

# Manifold replay must be ON so the donor accumulates its (μ,σ) archive.
bench.MANIFOLD_REPLAY_ENABLED = True
bench.HIPPOCAMPAL_ENABLED = False
bench.HIPPOCAMPAL_SYNTHETIC = False
bench.REHEARSAL_ENABLED = False
bench.LWF_ENABLED = False
bench.BRAINSTEM_ENABLED = False
bench.ENGRAM_ENABLED = False
bench.DIFFERENTIAL_ENABLED = False


def _emnist_kt_specs():
    # EMNIST letters K..T — local 10..19, global 30..39 (5 binary tasks).
    return chained_extension_specs(
        n_tasks=5, start_class_offset=30, start_local_class=10,
    )


def _emnist_uz_specs():
    # EMNIST letters U..Z (partial) — local 20..25, global 40..45 (3 binary tasks).
    # Only 6 letters left after K..T; gives a 6-class third EMNIST donor.
    return chained_extension_specs(
        n_tasks=3, start_class_offset=40, start_local_class=20,
    )


SPLIT_BLOCKS = {
    # label : (specs_fn, dataset_name for DatasetBundle)
    "digits":    (lambda: chained_15_specs()[0:5],   "mnist"),
    "fashion":   (lambda: chained_15_specs()[5:10],  "fashion_mnist"),
    "emnist":    (lambda: chained_15_specs()[10:15], "emnist_letters"),
    "emnist_kt": (_emnist_kt_specs,                  "emnist_letters"),
    "emnist_uz": (_emnist_uz_specs,                  "emnist_letters"),
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label", required=True, choices=sorted(SPLIT_BLOCKS),
        help="Donor split identifier (one of: digits, fashion, emnist).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Shared L0 seed (must match across all donors in a population).",
    )
    parser.add_argument(
        "--epochs", type=int, default=bench.N_EPOCHS_PER_TASK,
        help=f"Epochs per task (default {bench.N_EPOCHS_PER_TASK}).",
    )
    parser.add_argument("--data-root", default=bench.DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--out-dir", default=None,
        help="Directory to save the donor checkpoint (default <repo>/outputs).",
    )
    args = parser.parse_args(argv)

    specs_fn, ds_name = SPLIT_BLOCKS[args.label]
    specs_subset = specs_fn()
    classes_covered = sorted({c for s in specs_subset for c in s.global_classes})

    print(f"train_donor: label={args.label}  seed={args.seed}  "
          f"arm=grown_uncapped_dream  epochs/task={args.epochs}")
    print(f"  tasks   = {[s.name for s in specs_subset]}")
    print(f"  classes = {classes_covered}  dataset={ds_name}")

    bundle = DatasetBundle(
        [ds_name], root=args.data_root, n_holdout_per_dataset=0,
    )
    train_views = build_task_views(bundle, specs_subset, split="train")
    eval_views = build_task_views(bundle, specs_subset, split="test")
    task_class_lists = [s.global_classes for s in specs_subset]

    r = bench.run_arm(
        "grown_uncapped_dream",
        seed=args.seed,
        n_epochs_per_task=args.epochs,
        train_views=train_views,
        eval_views=eval_views,
        task_class_lists=task_class_lists,
        infancy_view=None,
        n_passes=1,
        return_state=True,
    )
    net = r["net"]
    mb = r["manifold"]
    if mb is None:
        raise RuntimeError(
            "Manifold buffer is None — MANIFOLD_REPLAY_ENABLED must be True."
        )

    arc = mb.stored_classes()
    if arc != classes_covered:
        raise AssertionError(
            f"archive class mismatch: {arc} vs expected {classes_covered}"
        )

    print()
    print(f"[SCORE] {args.label}  task-aware {r['final_accuracy_aware']:.4f}  "
          f"full {r['final_accuracy']:.4f}  domain {r['final_accuracy_domain']:.4f}")

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs",
    )
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "label": args.label,
        "arm": "grown_uncapped_dream",
        "l0_seed": args.seed,
        "input_dim": net.layers[0].fan_in,
        "l0_width": net.layers[0].n_nodes,
        "n_layers": len(net.layers),
        "head_size": net.layers[-1].n_nodes,
        "n_nodes_per_layer": list(net.n_nodes_per_layer()),
        "classes_covered": classes_covered,
        "task_specs": [
            {"name": s.name, "dataset_name": s.dataset_name,
             "local_classes": list(s.local_classes),
             "global_classes": list(s.global_classes)}
            for s in specs_subset
        ],
        "state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
        "manifold_stats": {
            int(c): (mu.detach().cpu(), sg.detach().cpu())
            for c, (mu, sg) in mb._stats.items()
        },
        "final_accuracy_aware": r["final_accuracy_aware"],
        "final_accuracy": r["final_accuracy"],
        "final_accuracy_domain": r["final_accuracy_domain"],
    }
    path = os.path.join(out_dir, f"poc_donor_{args.label}.pt")
    torch.save(payload, path)
    size_kb = os.path.getsize(path) / 1024.0
    print(f"[SAVE] {path}  ({size_kb:.1f} KB)  head={payload['head_size']}  "
          f"arch={payload['n_nodes_per_layer']}  "
          f"archive_classes={sorted(payload['manifold_stats'].keys())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
