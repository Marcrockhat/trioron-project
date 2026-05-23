"""Head-to-head: single-attempt head-settle vs N-attempt retry-and-pick.

Same scaffolding as the multiseed sentinel, but for each seed we run
TWO settle paths from the same post-absorb checkpoint:

  1. baseline — one settle pass under torch.manual_seed(seed)
  2. retry   — n_attempts independent settle passes; the trial with the
               highest held-out synthetic-sample accuracy wins

The retry compute is opt-in absorption-time work; both paths share the
same absorbed substrate, so any lift is attributable to head-settle
variance reduction, not to additional learning.

Default config: n_attempts=10, n=12 seeds (matches the original
multiseed). Override via env: ABSORB_RETRY_N (attempts), ABSORB_SEEDS
(comma-separated seed list).
"""
from __future__ import annotations

import os
import statistics
import sys
import time

import torch

from experiments.datasets import (
    DatasetBundle,
    DEFAULT_DATA_ROOT,
    chained_15_specs,
    build_task_views,
)
from experiments.pool_matched_absorb_lcn_multiseed import SEEDS, _locality_mean
from experiments.pool_matched_absorb_lcn_test import (
    train_donor_lcn,
    extend_l1_lcn_mask,
)
from experiments.pool_matched_absorb_test import eval_on_tasks
from experiments.pool_matched_absorption import (
    pool_matched_absorb,
    merge_manifold,
)
from trioron.api import settle_head_via_manifold, settle_head_with_retry


SETTLE_KWARGS = dict(n_steps=200, batch_size=64, lr=0.01, noise_scale=1.0)


def run_seed(
    seed: int,
    n_attempts: int,
    train_views,
    eval_views,
    task_class_lists,
) -> dict:
    donor_a, _, manifold_a, trained_a = train_donor_lcn(
        seed=seed,
        task_indices=[0, 1],
        train_views=train_views,
        eval_views=eval_views,
        task_class_lists=task_class_lists,
        n_epochs=4,
        label=f"A_seed{seed}",
    )
    donor_b, _, manifold_b, trained_b = train_donor_lcn(
        seed=seed,
        task_indices=[2, 3],
        train_views=train_views,
        eval_views=eval_views,
        task_class_lists=task_class_lists,
        n_epochs=4,
        label=f"B_seed{seed}",
    )

    a_self_ta, a_self_full = eval_on_tasks(
        donor_a, eval_views, task_class_lists, [0, 1], label_prefix="    ",
    )
    b_self_ta, b_self_full = eval_on_tasks(
        donor_b, eval_views, task_class_lists, [2, 3], label_prefix="    ",
    )

    pool_matched_absorb(
        donor_a,
        donor_b,
        grid_size=4,
        pool_capacity=None,
        snap_to_pool_centroid=False,
        donor_trained_classes=trained_b,
        recipient_trained_classes=trained_a,
    )
    extend_l1_lcn_mask(donor_a)
    locality = _locality_mean(donor_a.layers[1])
    merge_manifold(manifold_a, manifold_b)
    combined_seen = sorted(set(trained_a) | set(trained_b))

    # Snapshot the post-absorb head; both paths start here.
    head = donor_a.layers[2]
    W_snap = head.W.detach().clone()
    b_snap = head.b.detach().clone()

    # --- Path A: baseline single settle ---
    torch.manual_seed(seed)
    settle_head_via_manifold(donor_a, manifold_a, combined_seen, **SETTLE_KWARGS)
    base_a_ta, base_a_full = eval_on_tasks(
        donor_a, eval_views, task_class_lists, [0, 1], label_prefix="    ",
    )
    base_b_ta, base_b_full = eval_on_tasks(
        donor_a, eval_views, task_class_lists, [2, 3], label_prefix="    ",
    )

    # --- Path B: retry N attempts ---
    with torch.no_grad():
        head.W.data.copy_(W_snap)
        head.b.data.copy_(b_snap)
    best_trial, best_score = settle_head_with_retry(
        donor_a,
        manifold_a,
        combined_seen,
        n_attempts=n_attempts,
        score_samples=256,
        seed_offset=seed * 1000,
        settle_kwargs=SETTLE_KWARGS,
    )
    retry_a_ta, retry_a_full = eval_on_tasks(
        donor_a, eval_views, task_class_lists, [0, 1], label_prefix="    ",
    )
    retry_b_ta, retry_b_full = eval_on_tasks(
        donor_a, eval_views, task_class_lists, [2, 3], label_prefix="    ",
    )

    return {
        "seed": seed,
        "locality": locality,
        "a_self_ta": a_self_ta,
        "a_self_full": a_self_full,
        "b_self_ta": b_self_ta,
        "b_self_full": b_self_full,
        "base_a_ta": base_a_ta,
        "base_a_full": base_a_full,
        "base_b_ta": base_b_ta,
        "base_b_full": base_b_full,
        "retry_a_ta": retry_a_ta,
        "retry_a_full": retry_a_full,
        "retry_b_ta": retry_b_ta,
        "retry_b_full": retry_b_full,
        "best_trial": best_trial,
        "best_score": best_score,
    }


def _stat(label: str, values: list) -> str:
    m = statistics.mean(values)
    s = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"  {label:>36}  mean={m:+.4f}  σ={s:.4f}  n={len(values)}"


def main(seeds=None, n_attempts=None) -> int:
    seeds = seeds or _parse_seeds() or SEEDS
    n_attempts = n_attempts or int(os.environ.get("ABSORB_RETRY_N", "10"))

    print("=" * 78)
    print(
        f"Retry head-to-head (n_attempts={n_attempts}, n_seeds={len(seeds)}: {seeds})"
    )
    print("=" * 78)

    bundle = DatasetBundle(
        ["mnist", "fashion_mnist", "emnist_letters"],
        root=DEFAULT_DATA_ROOT,
        n_holdout_per_dataset=0,
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
        row = run_seed(seed, n_attempts, train_views, eval_views, task_class_lists)
        rows.append(row)
        base_a = row["base_a_full"]
        base_b = row["base_b_full"]
        ret_a = row["retry_a_full"]
        ret_b = row["retry_b_full"]
        print(
            f"  seed={seed} in {time.time()-t0:.0f}s  "
            f"base_full=({base_a:.3f},{base_b:.3f})  "
            f"retry_full=({ret_a:.3f},{ret_b:.3f})  "
            f"Δ_a={ret_a-base_a:+.3f}  Δ_b={ret_b-base_b:+.3f}  "
            f"best_trial={row['best_trial']}  best_score={row['best_score']:.3f}"
        )

    print(f"\nTotal: {time.time()-t_total:.0f}s")
    print("\n" + "=" * 78)
    print(f"AGGREGATE (n={len(rows)})  baseline vs retry-{n_attempts}")
    print("=" * 78)

    base_a_full = [r["base_a_full"] for r in rows]
    base_b_full = [r["base_b_full"] for r in rows]
    retry_a_full = [r["retry_a_full"] for r in rows]
    retry_b_full = [r["retry_b_full"] for r in rows]
    delta_a = [r["retry_a_full"] - r["base_a_full"] for r in rows]
    delta_b = [r["retry_b_full"] - r["base_b_full"] for r in rows]

    print(_stat("baseline A-tasks full", base_a_full))
    print(_stat("retry    A-tasks full", retry_a_full))
    print(_stat("Δ_full A (retry − baseline)", delta_a))
    print()
    print(_stat("baseline B-tasks full", base_b_full))
    print(_stat("retry    B-tasks full", retry_b_full))
    print(_stat("Δ_full B (retry − baseline)", delta_b))
    print()
    base_ta_a = [r["base_a_ta"] for r in rows]
    retry_ta_a = [r["retry_a_ta"] for r in rows]
    base_ta_b = [r["base_b_ta"] for r in rows]
    retry_ta_b = [r["retry_b_ta"] for r in rows]
    print(_stat("baseline A task-aware", base_ta_a))
    print(_stat("retry    A task-aware", retry_ta_a))
    print(_stat("baseline B task-aware", base_ta_b))
    print(_stat("retry    B task-aware", retry_ta_b))

    n_wins_a = sum(1 for d in delta_a if d > 0)
    n_wins_b = sum(1 for d in delta_b if d > 0)
    print(
        f"\nretry beat baseline on A-tasks in {n_wins_a}/{len(rows)} seeds; "
        f"on B-tasks in {n_wins_b}/{len(rows)} seeds."
    )
    return 0


def _parse_seeds():
    raw = os.environ.get("ABSORB_SEEDS")
    if not raw:
        return None
    return [int(s) for s in raw.split(",") if s.strip()]


if __name__ == "__main__":
    sys.exit(main())
