"""Head-to-head: exact-position absorption vs intra-pool position-jitter.

For each seed, train two donors and run TWO absorption paths from the
same trained donors:

  1. baseline — exact donor positions (current behaviour)
  2. jitter   — Gaussian σ added to each absorbed cell's (x, y)

Both paths use the same post-absorb manifold merge + head-settle.
Any lift is attributable to locus-collision relief, not to head
dynamics. Donors are trained once per seed; the absorption + head-
settle steps are forked.

Default config: jitter σ = 0.05 (20 % of the pool extent at
grid_size=4), n=12 seeds (same as the multiseed sentinel). Override
via env: ABSORB_JITTER (float σ), ABSORB_SEEDS (csv list).
"""
from __future__ import annotations

import copy
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
from trioron.api import settle_head_via_manifold


SETTLE_KWARGS = dict(n_steps=200, batch_size=64, lr=0.01, noise_scale=1.0)


def _absorb_and_settle(
    donor_a,
    donor_b,
    manifold_a,
    manifold_b_clone,
    trained_a,
    trained_b,
    position_jitter: float,
    settle_seed: int,
):
    """Absorb donor_b into donor_a (in place) and run a settle.

    Returns the locality value for the post-absorb L1.
    """
    pool_matched_absorb(
        donor_a,
        donor_b,
        grid_size=4,
        pool_capacity=None,
        snap_to_pool_centroid=False,
        donor_trained_classes=trained_b,
        recipient_trained_classes=trained_a,
        position_jitter=position_jitter,
    )
    extend_l1_lcn_mask(donor_a)
    locality = _locality_mean(donor_a.layers[1])
    merge_manifold(manifold_a, manifold_b_clone)
    combined_seen = sorted(set(trained_a) | set(trained_b))
    torch.manual_seed(settle_seed)
    settle_head_via_manifold(donor_a, manifold_a, combined_seen, **SETTLE_KWARGS)
    return locality


def run_seed(
    seed: int,
    jitter_sigma: float,
    train_views,
    eval_views,
    task_class_lists,
) -> dict:
    # Train donors once.
    donor_a_seed, _, manifold_a_seed, trained_a = train_donor_lcn(
        seed=seed,
        task_indices=[0, 1],
        train_views=train_views,
        eval_views=eval_views,
        task_class_lists=task_class_lists,
        n_epochs=4,
        label=f"A_seed{seed}",
    )
    donor_b_seed, _, manifold_b_seed, trained_b = train_donor_lcn(
        seed=seed,
        task_indices=[2, 3],
        train_views=train_views,
        eval_views=eval_views,
        task_class_lists=task_class_lists,
        n_epochs=4,
        label=f"B_seed{seed}",
    )

    # --- Path A: baseline (exact positions) ---
    donor_a = copy.deepcopy(donor_a_seed)
    donor_b = copy.deepcopy(donor_b_seed)
    manifold_a = copy.deepcopy(manifold_a_seed)
    manifold_b = copy.deepcopy(manifold_b_seed)
    loc_base = _absorb_and_settle(
        donor_a, donor_b, manifold_a, manifold_b,
        trained_a, trained_b,
        position_jitter=0.0, settle_seed=seed,
    )
    base_a_ta, base_a_full = eval_on_tasks(
        donor_a, eval_views, task_class_lists, [0, 1], label_prefix="    ",
    )
    base_b_ta, base_b_full = eval_on_tasks(
        donor_a, eval_views, task_class_lists, [2, 3], label_prefix="    ",
    )

    # --- Path B: jitter ---
    donor_a = copy.deepcopy(donor_a_seed)
    donor_b = copy.deepcopy(donor_b_seed)
    manifold_a = copy.deepcopy(manifold_a_seed)
    manifold_b = copy.deepcopy(manifold_b_seed)
    # Use a deterministic generator-state for the jitter so the jitter
    # path is reproducible per seed. Seeding here advances the global
    # RNG that absorb_cell reads inside the torch.no_grad() block.
    torch.manual_seed(seed * 7919 + 11)
    loc_jit = _absorb_and_settle(
        donor_a, donor_b, manifold_a, manifold_b,
        trained_a, trained_b,
        position_jitter=jitter_sigma, settle_seed=seed,
    )
    jit_a_ta, jit_a_full = eval_on_tasks(
        donor_a, eval_views, task_class_lists, [0, 1], label_prefix="    ",
    )
    jit_b_ta, jit_b_full = eval_on_tasks(
        donor_a, eval_views, task_class_lists, [2, 3], label_prefix="    ",
    )

    return {
        "seed": seed,
        "loc_base": loc_base,
        "loc_jit": loc_jit,
        "base_a_ta": base_a_ta,
        "base_a_full": base_a_full,
        "base_b_ta": base_b_ta,
        "base_b_full": base_b_full,
        "jit_a_ta": jit_a_ta,
        "jit_a_full": jit_a_full,
        "jit_b_ta": jit_b_ta,
        "jit_b_full": jit_b_full,
    }


def _stat(label: str, values: list) -> str:
    m = statistics.mean(values)
    s = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"  {label:>36}  mean={m:+.4f}  σ={s:.4f}  n={len(values)}"


def main(seeds=None, jitter_sigma=None) -> int:
    seeds = seeds or _parse_seeds() or SEEDS
    if jitter_sigma is None:
        jitter_sigma = float(os.environ.get("ABSORB_JITTER", "0.05"))

    print("=" * 78)
    print(
        f"Jitter head-to-head (σ={jitter_sigma}, n_seeds={len(seeds)}: {seeds})"
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
        row = run_seed(seed, jitter_sigma, train_views, eval_views, task_class_lists)
        rows.append(row)
        ba = row["base_a_full"]; bb = row["base_b_full"]
        ja = row["jit_a_full"]; jb = row["jit_b_full"]
        print(
            f"  seed={seed} in {time.time()-t0:.0f}s  "
            f"base_full=({ba:.3f},{bb:.3f})  "
            f"jit_full=({ja:.3f},{jb:.3f})  "
            f"Δ_a={ja-ba:+.3f}  Δ_b={jb-bb:+.3f}"
        )

    print(f"\nTotal: {time.time()-t_total:.0f}s")
    print("\n" + "=" * 78)
    print(f"AGGREGATE (n={len(rows)})  baseline vs jitter σ={jitter_sigma}")
    print("=" * 78)
    base_a = [r["base_a_full"] for r in rows]
    base_b = [r["base_b_full"] for r in rows]
    jit_a = [r["jit_a_full"] for r in rows]
    jit_b = [r["jit_b_full"] for r in rows]
    delta_a = [r["jit_a_full"] - r["base_a_full"] for r in rows]
    delta_b = [r["jit_b_full"] - r["base_b_full"] for r in rows]
    print(_stat("baseline A-tasks full", base_a))
    print(_stat("jitter   A-tasks full", jit_a))
    print(_stat("Δ_full A (jitter − baseline)", delta_a))
    print()
    print(_stat("baseline B-tasks full", base_b))
    print(_stat("jitter   B-tasks full", jit_b))
    print(_stat("Δ_full B (jitter − baseline)", delta_b))
    print()
    print(_stat("baseline A task-aware", [r["base_a_ta"] for r in rows]))
    print(_stat("jitter   A task-aware", [r["jit_a_ta"] for r in rows]))
    print(_stat("baseline B task-aware", [r["base_b_ta"] for r in rows]))
    print(_stat("jitter   B task-aware", [r["jit_b_ta"] for r in rows]))
    print()
    print(_stat("baseline locality", [r["loc_base"] for r in rows]))
    print(_stat("jitter   locality", [r["loc_jit"] for r in rows]))

    n_wins_a = sum(1 for d in delta_a if d > 0)
    n_wins_b = sum(1 for d in delta_b if d > 0)
    print(
        f"\njitter beat baseline on A-tasks in {n_wins_a}/{len(rows)} seeds; "
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
