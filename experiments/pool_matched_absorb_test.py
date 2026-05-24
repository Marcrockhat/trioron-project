"""End-to-end test for pool-matched L1 absorption.

Setup:
  - Donor A: trained on chained-15 tasks {0, 1}                (mnist 0..3)
  - Donor B: trained on chained-15 tasks {2, 3}                (mnist 4..7)
  - Both use seed=42 for L0 (R·S handshake → same L0 factor)
  - Both built via axis6_credit_chained15.build_net

Procedure:
  1. Train both donors independently.
  2. Eval each donor on its own tasks (BEFORE absorption): baseline.
  3. Eval donor A on donor B's tasks: should be near chance.
  4. Eval donor B on donor A's tasks: should be near chance.
  5. Absorb donor B's L1 cells into donor A via pool_matched_absorb.
  6. Eval combined (A+B) on ALL four task class lists.
  7. Headline: did A's classes survive? Did B's classes transfer losslessly?

Comparison: also train a CONTROL recipient on all four tasks {0,1,2,3}
sequentially with regular chained-15 mechanisms, for reference.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from experiments.axis6_credit_chained15 import (
    build_net, train_one_task, evaluate_task_aware, evaluate_full,
    ManifoldStore, cosine_logits,
)
from experiments.datasets import (
    DatasetBundle, build_task_views, chained_15_specs, DEFAULT_DATA_ROOT,
)
from experiments.pool_matched_absorption import (
    pool_matched_absorb, pool_id, merge_manifold,
)
from trioron import masked_cross_entropy
from trioron.manifold import settle_head_via_manifold  # noqa: F401


def train_donor(
    seed: int,
    task_indices: list,
    train_views, eval_views, task_class_lists,
    n_epochs: int = 4,
    label: str = "donor",
):
    """Run chained-15 train_one_task loop on a subset of tasks. Returns
    (net, cell_credit, manifold, trained_classes_list)."""
    net = build_net(seed=seed, lcn_enabled=True, l1_lcn_mode="off")
    L1 = net.layers[1]
    cell_credit = torch.zeros(L1.n_nodes, dtype=torch.bool)
    manifold = ManifoldStore(n_l0=net.layers[0].n_nodes)
    seen_classes = []
    for task_idx in task_indices:
        for c in task_class_lists[task_idx]:
            if c not in seen_classes:
                seen_classes.append(c)
        r = train_one_task(
            net=net, view=train_views[task_idx],
            task_global_classes=task_class_lists[task_idx],
            n_epochs=n_epochs, batch_size=256, lr=0.01,
            axis6=True, diff_floor=0.0005, diff_b_thr=1.0,
            field_sigma=0.15, field_dt=0.1, stress_tolerance=0.0,
            spawn_cap=5, cooldown_steps=20,
            l1_credit_mask=cell_credit,
            track_engagement=True,
            manifold=manifold,
            lambda_manifold=1.0, manifold_total_batch=64,
            manifold_noise_scale=1.0,
            seen_classes_global=list(seen_classes),
            lambda_diff=0.0, W_snapshot=None,
            verbose=False,
        )
        if r.engagement_frac is not None:
            ef = r.engagement_frac
            if cell_credit.numel() < L1.n_nodes:
                cell_credit = torch.cat([
                    cell_credit,
                    torch.zeros(L1.n_nodes - cell_credit.numel(), dtype=torch.bool),
                ])
            if ef.numel() < L1.n_nodes:
                ef = torch.cat([ef, torch.zeros(L1.n_nodes - ef.numel())])
            cell_credit = cell_credit | (ef > 0.3)
        manifold.store_task(net, train_views[task_idx], task_class_lists[task_idx])
    print(f"  [{label}] trained on tasks {task_indices}  L1.n_nodes={L1.n_nodes}  "
          f"frozen={int(cell_credit[:L1.n_nodes].sum())}")
    return net, cell_credit, manifold, list(seen_classes)


def eval_on_tasks(net, eval_views, task_class_lists, task_indices, label_prefix=""):
    """Eval task_aware on each requested task index. Returns mean."""
    rows = []
    for k in task_indices:
        ta = evaluate_task_aware(net, eval_views[k], task_class_lists[k])
        full = evaluate_full(net, eval_views[k])
        rows.append((k, ta, full))
    mean_ta = sum(r[1] for r in rows) / len(rows)
    mean_full = sum(r[2] for r in rows) / len(rows)
    for k, ta, full in rows:
        print(f"  {label_prefix}task {k}: task_aware={ta:.3f}  full={full:.3f}")
    print(f"  {label_prefix}MEAN  : task_aware={mean_ta:.3f}  full={mean_full:.3f}")
    return mean_ta, mean_full


def main() -> int:
    print("=" * 78)
    print("Pool-matched L1 absorption: end-to-end test")
    print("=" * 78)

    bundle = DatasetBundle(
        ["mnist", "fashion_mnist", "emnist_letters"],
        root=DEFAULT_DATA_ROOT, n_holdout_per_dataset=0,
    )
    specs = chained_15_specs()
    train_views = build_task_views(bundle, specs, split="train")
    eval_views = build_task_views(bundle, specs, split="test")
    task_class_lists = [s.global_classes for s in specs]
    print(f"  tasks 0..3 cover classes: " + " | ".join(
        f"{i}:{task_class_lists[i]}" for i in range(4)
    ))

    # Same seed for L0 (R·S handshake). Different L1 seeds give different
    # L1 init / training trajectories — donor A and donor B have distinct
    # cells. For now we use the same seed for both — the spawn jitter +
    # different tasks still produces distinct L1 populations.
    print("\n--- Phase 1: train donor A (tasks 0, 1) ---")
    t0 = time.time()
    donor_a, credit_a, manifold_a, trained_a = train_donor(
        seed=42, task_indices=[0, 1],
        train_views=train_views, eval_views=eval_views,
        task_class_lists=task_class_lists, n_epochs=4, label="donor A",
    )
    print(f"  donor A trained in {time.time()-t0:.0f}s  trained_classes={trained_a}")

    print("\n--- Phase 2: train donor B (tasks 2, 3) ---")
    t0 = time.time()
    donor_b, credit_b, manifold_b, trained_b = train_donor(
        seed=42, task_indices=[2, 3],
        train_views=train_views, eval_views=eval_views,
        task_class_lists=task_class_lists, n_epochs=4, label="donor B",
    )
    print(f"  donor B trained in {time.time()-t0:.0f}s  trained_classes={trained_b}")

    print("\n--- Phase 3: PRE-absorption baselines ---")
    print("  donor A on its own tasks (0, 1):")
    a_self_ta, a_self_full = eval_on_tasks(
        donor_a, eval_views, task_class_lists, [0, 1], label_prefix="    "
    )
    print("  donor A on donor B's tasks (2, 3) — should be near chance:")
    a_cross_ta, a_cross_full = eval_on_tasks(
        donor_a, eval_views, task_class_lists, [2, 3], label_prefix="    "
    )
    print("  donor B on its own tasks (2, 3):")
    b_self_ta, b_self_full = eval_on_tasks(
        donor_b, eval_views, task_class_lists, [2, 3], label_prefix="    "
    )
    print("  donor B on donor A's tasks (0, 1) — should be near chance:")
    b_cross_ta, b_cross_full = eval_on_tasks(
        donor_b, eval_views, task_class_lists, [0, 1], label_prefix="    "
    )

    print("\n--- Phase 4: pool-matched absorption (B → A) ---")
    print(f"  donor A pre-absorb L1.n_nodes = {donor_a.layers[1].n_nodes}")
    print(f"  donor B           L1.n_nodes = {donor_b.layers[1].n_nodes}")
    absorbed = pool_matched_absorb(
        donor_a, donor_b, grid_size=4, pool_capacity=None,
        snap_to_pool_centroid=False,
        donor_trained_classes=trained_b,
    )
    print(f"  absorbed {len(absorbed)}/{donor_b.layers[1].n_nodes} donor B cells")
    print(f"  donor A post-absorb L1.n_nodes = {donor_a.layers[1].n_nodes}")

    print("\n  --- Manifold merge (donor B's class signatures → A's manifold) ---")
    n_added = merge_manifold(manifold_a, manifold_b)
    combined_seen = sorted(set(trained_a) | set(trained_b))
    print(f"  added {n_added} new class signatures; combined manifold has "
          f"{len(manifold_a.mu_per_class)} classes (seen union = {combined_seen})")

    print("\n  --- Head-settle pass (manifold replay, head-only) ---")
    t0 = time.time()
    final_loss = settle_head_via_manifold(
        donor_a, manifold_a, combined_seen,
        n_steps=200, batch_size=64, lr=0.01, noise_scale=1.0,
    )
    print(f"  head-settle 200 steps in {time.time()-t0:.1f}s  final_loss={final_loss:.4f}")

    # Show pool distribution after absorption.
    pool_dist = {}
    for i in range(donor_a.layers[1].n_nodes):
        pid = pool_id(donor_a.layers[1].cell_position[i, :2].tolist(), 4)
        pool_dist[pid] = pool_dist.get(pid, 0) + 1
    print("  post-absorb pool distribution:")
    for pid in sorted(pool_dist.keys()):
        print(f"    pool {pid:>2}: {pool_dist[pid]:>3} cells")

    print("\n--- Phase 5: POST-absorption eval (combined A+B) ---")
    print("  Combined on A's tasks (0, 1):")
    ab_a_ta, ab_a_full = eval_on_tasks(
        donor_a, eval_views, task_class_lists, [0, 1], label_prefix="    "
    )
    print("  Combined on B's tasks (2, 3):")
    ab_b_ta, ab_b_full = eval_on_tasks(
        donor_a, eval_views, task_class_lists, [2, 3], label_prefix="    "
    )

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(
        f"{'arm':>28}  {'task_aware':>12}  {'full':>8}  {'Δ vs solo':>10}"
    )
    print("-" * 70)
    print(f"  {'donor A solo on A-tasks':>28}  {a_self_ta:>12.3f}  {a_self_full:>8.3f}")
    print(f"  {'A+B (absorbed) on A-tasks':>28}  {ab_a_ta:>12.3f}  {ab_a_full:>8.3f}  "
          f"{ab_a_ta - a_self_ta:>+9.3f}")
    print(f"  {'donor B solo on B-tasks':>28}  {b_self_ta:>12.3f}  {b_self_full:>8.3f}")
    print(f"  {'A+B (absorbed) on B-tasks':>28}  {ab_b_ta:>12.3f}  {ab_b_full:>8.3f}  "
          f"{ab_b_ta - b_self_ta:>+9.3f}")
    print()
    print(
        f"  A-classes survival : {ab_a_ta:.3f} (was {a_self_ta:.3f}) "
        f"Δ={ab_a_ta - a_self_ta:+.3f}"
    )
    print(
        f"  B-classes transfer : {ab_b_ta:.3f} (donor B solo {b_self_ta:.3f}) "
        f"Δ={ab_b_ta - b_self_ta:+.3f}"
    )
    print()
    if ab_a_ta >= a_self_ta - 0.05 and ab_b_ta >= b_self_ta - 0.05:
        print("  → PASS: A survives AND B transfers within 0.05 of solo baselines")
    elif ab_b_ta >= b_self_ta - 0.05:
        print("  → PARTIAL: B transferred losslessly, but A degraded")
    elif ab_a_ta >= a_self_ta - 0.05:
        print("  → PARTIAL: A survived, but B didn't transfer losslessly")
    else:
        print("  → FAIL: both A and B degraded post-absorption")
    return 0


if __name__ == "__main__":
    sys.exit(main())
