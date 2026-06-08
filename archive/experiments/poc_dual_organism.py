"""Dual-organism PoC for the absorption / stem-cell mechanism.

Trains two donors with a SHARED L0 seed on disjoint chained-15 task
subsets:

  donor_digits   : chained_15_specs()[0:5]   (MNIST,   global classes 0..9)
  donor_fashion  : chained_15_specs()[5:10]  (Fashion, global classes 10..19)

Each donor is saved as a self-contained skill pack — state_dict +
ManifoldBuffer (μ,σ) per class + class layout + L0 seed — so a recipient
can later transplant the donor into a fresh L1 branch slot without
re-training. This script only PRODUCES donors; the multi-branch container
and absorption protocol are deferred to a later PR.

Sanity assertions:
  - L0 W and b are byte-identical across donors (shared seed + frozen L0).
  - Each donor's manifold archive contains EXACTLY its own training
    classes (no leakage).

Run:
  python3 -m experiments.poc_dual_organism \\
      > outputs/poc_dual_organism_seed42.log 2>&1
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
)

# Manifold replay must be ON so each donor accumulates its (μ,σ) archive.
# Other rehearsal mechanisms off — keeps the donor's saved state pure
# (manifold-only is the trioron-native pseudo-rehearsal and the only
# storage that's actually portable across organisms).
bench.MANIFOLD_REPLAY_ENABLED = True
bench.HIPPOCAMPAL_ENABLED = False
bench.HIPPOCAMPAL_SYNTHETIC = False
bench.REHEARSAL_ENABLED = False
bench.LWF_ENABLED = False
bench.BRAINSTEM_ENABLED = False
bench.ENGRAM_ENABLED = False
bench.DIFFERENTIAL_ENABLED = False


def train_donor(label, specs_subset, *, seed, bundle, n_epochs):
    train_views = build_task_views(bundle, specs_subset, split="train")
    eval_views = build_task_views(bundle, specs_subset, split="test")
    task_class_lists = [s.global_classes for s in specs_subset]
    print()
    print("#" * 78)
    print(f"#   DONOR: {label}  (seed={seed}, n_tasks={len(specs_subset)})")
    print("#" * 78)
    return bench.run_arm(
        "grown_uncapped_dream",
        seed=seed,
        n_epochs_per_task=n_epochs,
        train_views=train_views,
        eval_views=eval_views,
        task_class_lists=task_class_lists,
        infancy_view=None,
        n_passes=1,
        return_state=True,
    )


def save_donor(out_dir, label, r, specs_subset, classes_covered, l0_seed):
    net = r["net"]
    mb = r["manifold"]
    if mb is None:
        raise RuntimeError(
            f"donor_{label}: manifold buffer is None — "
            "MANIFOLD_REPLAY_ENABLED must be True for this PoC."
        )
    manifold_stats = {
        int(c): (mu.detach().cpu(), sg.detach().cpu())
        for c, (mu, sg) in mb._stats.items()
    }
    payload = {
        "label": label,
        "arm": "grown_uncapped_dream",
        "l0_seed": l0_seed,
        "input_dim": net.layers[0].fan_in,
        "l0_width": net.layers[0].n_nodes,
        "n_layers": len(net.layers),
        "head_size": net.layers[-1].n_nodes,
        "n_nodes_per_layer": list(net.n_nodes_per_layer()),
        "classes_covered": list(classes_covered),
        "task_specs": [
            {
                "name": s.name,
                "dataset_name": s.dataset_name,
                "local_classes": list(s.local_classes),
                "global_classes": list(s.global_classes),
            }
            for s in specs_subset
        ],
        "state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
        "manifold_stats": manifold_stats,
        "final_accuracy_aware": r["final_accuracy_aware"],
        "final_accuracy": r["final_accuracy"],
        "final_accuracy_domain": r["final_accuracy_domain"],
    }
    path = os.path.join(out_dir, f"poc_donor_{label}.pt")
    torch.save(payload, path)
    size_kb = os.path.getsize(path) / 1024.0
    print(f"[SAVE] {path}  ({size_kb:.1f} KB)  "
          f"head={payload['head_size']}  "
          f"arch={payload['n_nodes_per_layer']}  "
          f"archive_classes={sorted(manifold_stats.keys())}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Shared L0 seed for both donors (default 42).",
    )
    parser.add_argument(
        "--epochs", type=int, default=bench.N_EPOCHS_PER_TASK,
        help=f"Epochs per task (default {bench.N_EPOCHS_PER_TASK}).",
    )
    parser.add_argument("--data-root", default=bench.DEFAULT_DATA_ROOT)
    args = parser.parse_args(argv)

    bundle = DatasetBundle(
        ["mnist", "fashion_mnist", "emnist_letters"],
        root=args.data_root,
        n_holdout_per_dataset=bench.N_INFANCY_PER_DATASET,
    )
    all_specs = chained_15_specs()
    digits_specs = all_specs[0:5]    # MNIST   (global 0..9)
    fashion_specs = all_specs[5:10]  # Fashion (global 10..19)

    print("PoC dual-organism training")
    print(f"  shared L0 seed     = {args.seed}")
    print(f"  epochs/task        = {args.epochs}")
    print(f"  arm                = grown_uncapped_dream")
    print(f"  donor_digits       = {[s.name for s in digits_specs]}")
    print(f"  donor_fashion      = {[s.name for s in fashion_specs]}")
    print(f"  manifold_replay    = {bench.MANIFOLD_REPLAY_ENABLED}")

    r_d = train_donor(
        "digits", digits_specs,
        seed=args.seed, bundle=bundle, n_epochs=args.epochs,
    )
    r_f = train_donor(
        "fashion", fashion_specs,
        seed=args.seed, bundle=bundle, n_epochs=args.epochs,
    )

    net_d, net_f = r_d["net"], r_f["net"]
    mb_d, mb_f = r_d["manifold"], r_f["manifold"]

    # Sanity 1: shared L0 → byte-identical L0 W,b across donors.
    l0_d_W = net_d.layers[0].W.detach().cpu()
    l0_f_W = net_f.layers[0].W.detach().cpu()
    l0_d_b = net_d.layers[0].b.detach().cpu()
    l0_f_b = net_f.layers[0].b.detach().cpu()
    print()
    print("=" * 78)
    print("SANITY")
    print("=" * 78)
    if not torch.equal(l0_d_W, l0_f_W):
        raise AssertionError("L0 W mismatch — shared-seed sanity FAILED")
    if not torch.equal(l0_d_b, l0_f_b):
        raise AssertionError("L0 b mismatch — shared-seed sanity FAILED")
    print(f"[OK] L0 W byte-identical across donors  shape={tuple(l0_d_W.shape)}")
    print(f"[OK] L0 b byte-identical across donors  shape={tuple(l0_d_b.shape)}")

    # Sanity 2: each donor's archive covers exactly its training classes.
    digits_classes = sorted({c for s in digits_specs for c in s.global_classes})
    fashion_classes = sorted({c for s in fashion_specs for c in s.global_classes})
    arc_d = mb_d.stored_classes() if mb_d is not None else []
    arc_f = mb_f.stored_classes() if mb_f is not None else []
    print(f"[?]  donor_digits  archive = {arc_d}  (expected {digits_classes})")
    print(f"[?]  donor_fashion archive = {arc_f}  (expected {fashion_classes})")
    if arc_d != digits_classes:
        raise AssertionError("donor_digits archive class list mismatch")
    if arc_f != fashion_classes:
        raise AssertionError("donor_fashion archive class list mismatch")
    print("[OK] archive class lists match training-class subsets")

    # Headline accuracy.
    print()
    print("=" * 78)
    print("SCORES (final-row, last pass)")
    print("=" * 78)
    print(f"donor_digits   task-aware {r_d['final_accuracy_aware']:.4f}  "
          f"full {r_d['final_accuracy']:.4f}  "
          f"domain {r_d['final_accuracy_domain']:.4f}")
    print(f"donor_fashion  task-aware {r_f['final_accuracy_aware']:.4f}  "
          f"full {r_f['final_accuracy']:.4f}  "
          f"domain {r_f['final_accuracy_domain']:.4f}")

    # Persist.
    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs",
    )
    os.makedirs(out_dir, exist_ok=True)
    print()
    print("=" * 78)
    print("PERSIST")
    print("=" * 78)
    save_donor(out_dir, "digits", r_d, digits_specs,
               digits_classes, l0_seed=args.seed)
    save_donor(out_dir, "fashion", r_f, fashion_specs,
               fashion_classes, l0_seed=args.seed)
    print("\nPoC complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
