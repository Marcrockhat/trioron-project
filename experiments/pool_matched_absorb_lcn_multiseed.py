"""Multi-seed driver for pool_matched_absorb_lcn_test.

Single-seed (seed=42) produced A-task full-softmax lift +0.193, sitting
0.007 below the v2 paper's claimed "+0.20" floor. This driver runs the
sentinel across n seeds (default 5) so the headline can be reported as
mean ± σ rather than a single-seed number.

Each run uses the same seed for donors A and B (paper §3.10 shared-L0
invariant). The shared seed varies across runs.
"""
from __future__ import annotations

import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from experiments.pool_matched_absorb_lcn_test import (  # noqa: E402
    train_donor_lcn,
    measure_locality,
)
from experiments.axis6_credit_chained15 import (  # noqa: E402
    extend_l1_lcn_mask,
)
from experiments.datasets import (  # noqa: E402
    DatasetBundle, build_task_views, chained_15_specs, DEFAULT_DATA_ROOT,
)
from experiments.pool_matched_absorption import (  # noqa: E402
    pool_matched_absorb, merge_manifold,
)
from experiments.pool_matched_absorb_test import (  # noqa: E402
    settle_head_via_manifold, eval_on_tasks,
)


SEEDS = [42, 43, 44, 1, 7, 11, 13, 17, 19, 23, 29, 31]


def _locality_mean(L1) -> float:
    if not hasattr(L1, "W_lcn_mask"):
        return float("nan")
    n = L1.n_nodes
    mask = L1.W_lcn_mask[:n]
    W_abs = L1.W[:n].detach().abs()
    in_window = (mask > 0.1).float()
    total_mass = W_abs.sum(dim=1)
    in_window_mass = (W_abs * in_window).sum(dim=1)
    return (in_window_mass / total_mass.clamp(min=1e-8)).mean().item()


def run_seed(seed: int, train_views, eval_views, task_class_lists) -> dict:
    donor_a, _credit_a, manifold_a, trained_a = train_donor_lcn(
        seed=seed, task_indices=[0, 1],
        train_views=train_views, eval_views=eval_views,
        task_class_lists=task_class_lists, n_epochs=4, label=f"A_seed{seed}",
    )
    donor_b, _credit_b, manifold_b, trained_b = train_donor_lcn(
        seed=seed, task_indices=[2, 3],
        train_views=train_views, eval_views=eval_views,
        task_class_lists=task_class_lists, n_epochs=4, label=f"B_seed{seed}",
    )

    a_self_ta, a_self_full = eval_on_tasks(
        donor_a, eval_views, task_class_lists, [0, 1], label_prefix="    "
    )
    b_self_ta, b_self_full = eval_on_tasks(
        donor_b, eval_views, task_class_lists, [2, 3], label_prefix="    "
    )

    pool_matched_absorb(
        donor_a, donor_b, grid_size=4, pool_capacity=None,
        snap_to_pool_centroid=False,
        donor_trained_classes=trained_b,
        recipient_trained_classes=trained_a,
    )
    extend_l1_lcn_mask(donor_a)
    locality = _locality_mean(donor_a.layers[1])

    merge_manifold(manifold_a, manifold_b)
    combined_seen = sorted(set(trained_a) | set(trained_b))
    settle_head_via_manifold(
        donor_a, manifold_a, combined_seen,
        n_steps=200, batch_size=64, lr=0.01, noise_scale=1.0,
    )

    ab_a_ta, ab_a_full = eval_on_tasks(
        donor_a, eval_views, task_class_lists, [0, 1], label_prefix="    "
    )
    ab_b_ta, ab_b_full = eval_on_tasks(
        donor_a, eval_views, task_class_lists, [2, 3], label_prefix="    "
    )

    return {
        "seed": seed,
        "a_self_ta": a_self_ta, "a_self_full": a_self_full,
        "b_self_ta": b_self_ta, "b_self_full": b_self_full,
        "ab_a_ta": ab_a_ta, "ab_a_full": ab_a_full,
        "ab_b_ta": ab_b_ta, "ab_b_full": ab_b_full,
        "delta_ta_a": ab_a_ta - a_self_ta,
        "delta_ta_b": ab_b_ta - b_self_ta,
        "delta_full_a": ab_a_full - a_self_full,
        "delta_full_b": ab_b_full - b_self_full,
        "locality": locality,
    }


def _stat(label: str, values: list) -> str:
    m = statistics.mean(values)
    s = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"  {label:>34}  mean={m:+.4f}  σ={s:.4f}  n={len(values)}"


def main(seeds=None) -> int:
    seeds = seeds or SEEDS
    print("=" * 78)
    print(f"Multi-seed absorption sentinel (n={len(seeds)} seeds: {seeds})")
    print("=" * 78)

    bundle = DatasetBundle(
        ["mnist", "fashion_mnist", "emnist_letters"],
        root=DEFAULT_DATA_ROOT, n_holdout_per_dataset=0,
    )
    specs = chained_15_specs()
    train_views = build_task_views(bundle, specs, split="train")
    eval_views = build_task_views(bundle, specs, split="test")
    task_class_lists = [s.global_classes for s in specs]

    rows = []
    t_total = time.time()
    for seed in seeds:
        print(f"\n--- seed={seed} ---")
        t0 = time.time()
        row = run_seed(seed, train_views, eval_views, task_class_lists)
        rows.append(row)
        print(f"  seed={seed} in {time.time()-t0:.0f}s  "
              f"Δta_a={row['delta_ta_a']:+.3f}  Δta_b={row['delta_ta_b']:+.3f}  "
              f"Δfull_a={row['delta_full_a']:+.3f}  Δfull_b={row['delta_full_b']:+.3f}  "
              f"loc={row['locality']:.3f}")

    print(f"\nTotal: {time.time()-t_total:.0f}s")
    print("\n" + "=" * 78)
    print(f"AGGREGATE (n={len(rows)})")
    print("=" * 78)
    print(_stat("task-aware Δ on A-tasks",
               [r["delta_ta_a"] for r in rows]))
    print(_stat("task-aware Δ on B-tasks",
               [r["delta_ta_b"] for r in rows]))
    print(_stat("full-softmax lift on A-tasks",
               [r["delta_full_a"] for r in rows]))
    print(_stat("full-softmax lift on B-tasks",
               [r["delta_full_b"] for r in rows]))
    print(_stat("post-absorb locality",
               [r["locality"] for r in rows]))

    print("\nv2 paper §4.2 claims vs measured:")
    delta_a = [r["delta_ta_a"] for r in rows]
    delta_b = [r["delta_ta_b"] for r in rows]
    full_a = [r["delta_full_a"] for r in rows]
    full_b = [r["delta_full_b"] for r in rows]
    loc = [r["locality"] for r in rows]
    print(f"  task-aware preserved within Δ ≤ 0.026  "
          f"→ worst Δ across both arms = "
          f"{min(min(delta_a), min(delta_b)):+.4f}")
    print(f"  full-softmax lift +0.20 to +0.31      "
          f"→ A mean {statistics.mean(full_a):+.4f}  "
          f"B mean {statistics.mean(full_b):+.4f}")
    print(f"  post-absorb locality ≥ 0.78           "
          f"→ mean {statistics.mean(loc):.4f}  "
          f"min {min(loc):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
