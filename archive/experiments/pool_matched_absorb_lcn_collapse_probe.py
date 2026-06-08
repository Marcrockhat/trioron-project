"""Probe the seed=7 absorption collapse.

n=5 sentinel showed seed=7 produces Δfull_b = −0.157 (loss, not lift) on
donor B's tasks. Seed=44 gives a clean +0.309 lift on the same arm.
Both other seeds (42, 43, 1) sit between.

This script instruments the seed=7 vs seed=44 paths with three checkpoints:
  A. donor B solo eval (baseline)
  B. post-absorb / pre-head-settle eval  (does pool_matched_absorb alone
     already break B's task-2 mass?)
  C. post-head-settle eval (does the manifold-replay calibration pass
     overwrite useful head columns?)

Also dumps per-class head-weight magnitude before/after head-settle so we
can see WHICH classes' head columns got reshuffled.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from experiments.pool_matched_absorb_lcn_test import (  # noqa: E402
    train_donor_lcn,
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


def head_col_norms(net) -> torch.Tensor:
    """L2 norm of each class column in the head weight (post-L1 → class)."""
    head = net.layers[-1]
    return head.W.detach().norm(dim=1)  # head.W shape: (n_classes, fan_in_into_head)


def probe_seed(seed: int, train_views, eval_views, task_class_lists) -> None:
    print(f"\n{'=' * 78}")
    print(f"PROBE seed={seed}")
    print(f"{'=' * 78}")

    donor_a, _, manifold_a, trained_a = train_donor_lcn(
        seed=seed, task_indices=[0, 1],
        train_views=train_views, eval_views=eval_views,
        task_class_lists=task_class_lists, n_epochs=4, label=f"A_seed{seed}",
    )
    donor_b, _, manifold_b, trained_b = train_donor_lcn(
        seed=seed, task_indices=[2, 3],
        train_views=train_views, eval_views=eval_views,
        task_class_lists=task_class_lists, n_epochs=4, label=f"B_seed{seed}",
    )

    print(f"\n[A] donor B solo on B-tasks:")
    b_solo_ta, b_solo_full = eval_on_tasks(
        donor_b, eval_views, task_class_lists, [2, 3], label_prefix="    "
    )

    head_b_solo = head_col_norms(donor_b)
    print(f"  donor B head-col norms (classes 0..7): "
          + " ".join(f"{c}:{head_b_solo[c]:.2f}" for c in range(8)))

    # Absorb
    pool_matched_absorb(
        donor_a, donor_b, grid_size=4, pool_capacity=None,
        snap_to_pool_centroid=False, donor_trained_classes=trained_b,
    )
    extend_l1_lcn_mask(donor_a)
    merge_manifold(manifold_a, manifold_b)
    combined_seen = sorted(set(trained_a) | set(trained_b))

    head_pre_settle = head_col_norms(donor_a)
    print(f"\n  donor A head-col norms POST-absorb PRE-settle: "
          + " ".join(f"{c}:{head_pre_settle[c]:.2f}" for c in range(8)))

    print(f"\n[B] POST-absorb PRE-head-settle on B-tasks:")
    pre_settle_ta, pre_settle_full = eval_on_tasks(
        donor_a, eval_views, task_class_lists, [2, 3], label_prefix="    "
    )

    # Head settle
    settle_head_via_manifold(
        donor_a, manifold_a, combined_seen,
        n_steps=200, batch_size=64, lr=0.01, noise_scale=1.0,
    )
    head_post_settle = head_col_norms(donor_a)
    print(f"\n  donor A head-col norms POST-settle: "
          + " ".join(f"{c}:{head_post_settle[c]:.2f}" for c in range(8)))

    delta = head_post_settle - head_pre_settle
    print(f"  Δ head-col norm settle - pre   : "
          + " ".join(f"{c}:{delta[c]:+.2f}" for c in range(8)))

    print(f"\n[C] POST-head-settle on B-tasks:")
    post_settle_ta, post_settle_full = eval_on_tasks(
        donor_a, eval_views, task_class_lists, [2, 3], label_prefix="    "
    )

    print(f"\nVERDICT seed={seed} on B-tasks:")
    print(f"  B solo                          ta={b_solo_ta:.3f}  full={b_solo_full:.3f}")
    print(f"  POST-absorb PRE-settle          ta={pre_settle_ta:.3f}  full={pre_settle_full:.3f}  "
          f"Δ_full={pre_settle_full - b_solo_full:+.3f}")
    print(f"  POST-head-settle                ta={post_settle_ta:.3f}  full={post_settle_full:.3f}  "
          f"Δ_full={post_settle_full - b_solo_full:+.3f}")
    print(f"  Δ from settle alone             Δ_full_from_settle="
          f"{post_settle_full - pre_settle_full:+.3f}")

    # Per-task in the post-settle output (we want to know if the loss is
    # concentrated on task 2 or task 3)
    print(f"\n  Per-task breakdown POST-settle (already printed above as [C])")


def main() -> int:
    bundle = DatasetBundle(
        ["mnist", "fashion_mnist", "emnist_letters"],
        root=DEFAULT_DATA_ROOT, n_holdout_per_dataset=0,
    )
    specs = chained_15_specs()
    train_views = build_task_views(bundle, specs, split="train")
    eval_views = build_task_views(bundle, specs, split="test")
    task_class_lists = [s.global_classes for s in specs]

    # Run seeds spanning the n=5 spectrum
    for seed in [44, 42, 7]:
        probe_seed(seed, train_views, eval_views, task_class_lists)
    return 0


if __name__ == "__main__":
    sys.exit(main())
