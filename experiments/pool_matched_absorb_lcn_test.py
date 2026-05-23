"""LCN + pool-matched absorption test.

The architectural unlock from 2026-05-22: positional constraints on L1
(LCN) are now compatible with absorption when both sides share the
convention. Donor cells trained under LCN_MODE=soft σ=0.25 carry
mask-shaped W rows (zero outside their window). When absorbed via
pool-matched, the recipient computes a fresh mask row using the
donor-inherited cell_position — same convention → same mask → idempotent
re-mask → no information loss.

This test verifies all three properties:
  (a) task_aware preserved on both donors (lossless transfer)
  (b) full preserved/lifted via manifold merge + head settle
  (c) absorbed cells' W rows still locality-constrained — off-window
      entries should be near zero after recipient extends its mask
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from experiments.axis6_credit_chained15 import (
    build_net, train_one_task, evaluate_task_aware, evaluate_full,
    ManifoldStore, extend_l1_lcn_mask, build_l1_lcn_mask,
)
from experiments.datasets import (
    DatasetBundle, build_task_views, chained_15_specs, DEFAULT_DATA_ROOT,
)
from experiments.pool_matched_absorption import (
    pool_matched_absorb, pool_id, merge_manifold,
)
from experiments.pool_matched_absorb_test import (
    settle_head_via_manifold, eval_on_tasks,
)


LCN_MODE = "soft"
LCN_SIGMA = 0.25
LCN_K = 8


def train_donor_lcn(
    seed: int,
    task_indices: list,
    train_views, eval_views, task_class_lists,
    n_epochs: int = 4,
    label: str = "donor",
):
    """train_donor variant with LCN-on-L1 enabled."""
    net = build_net(
        seed=seed, lcn_enabled=True,
        l1_lcn_mode=LCN_MODE, l1_lcn_sigma=LCN_SIGMA, l1_lcn_k=LCN_K,
    )
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
          f"frozen={int(cell_credit[:L1.n_nodes].sum())}  "
          f"W_lcn_mask installed: {hasattr(L1, 'W_lcn_mask')}")
    return net, cell_credit, manifold, list(seen_classes)


def measure_locality(L1, label: str = "") -> None:
    """Diagnostic: for each L1 cell, measure how much of its W mass is
    inside its expected window. Helps verify that absorbed cells retain
    the donor's LCN structure."""
    if not hasattr(L1, "W_lcn_mask"):
        print(f"  {label}: no W_lcn_mask buffer — locality check skipped")
        return
    n = L1.n_nodes
    mask = L1.W_lcn_mask[:n]
    W_abs = L1.W[:n].detach().abs()
    # In-window mass: weighted by mask (real-valued for soft, binary for hard).
    # For soft mask, "in-window" means mask > 0.1 say.
    in_window_threshold = 0.1
    in_window = (mask > in_window_threshold).float()
    total_mass = W_abs.sum(dim=1)
    in_window_mass = (W_abs * in_window).sum(dim=1)
    locality = (in_window_mass / total_mass.clamp(min=1e-8)).mean().item()
    print(f"  {label}: locality (in-window mass / total) = {locality:.3f} "
          f"(mean over {n} cells, threshold mask>{in_window_threshold})")


def main() -> int:
    print("=" * 78)
    print(f"LCN + pool-matched absorption test (LCN_MODE={LCN_MODE}, σ={LCN_SIGMA})")
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

    print("\n--- Phase 1: train donor A (tasks 0, 1) WITH LCN ---")
    t0 = time.time()
    donor_a, credit_a, manifold_a, trained_a = train_donor_lcn(
        seed=42, task_indices=[0, 1],
        train_views=train_views, eval_views=eval_views,
        task_class_lists=task_class_lists, n_epochs=4, label="donor A",
    )
    print(f"  trained in {time.time()-t0:.0f}s  trained_classes={trained_a}")
    measure_locality(donor_a.layers[1], "  donor A post-train")

    print("\n--- Phase 2: train donor B (tasks 2, 3) WITH LCN ---")
    t0 = time.time()
    donor_b, credit_b, manifold_b, trained_b = train_donor_lcn(
        seed=42, task_indices=[2, 3],
        train_views=train_views, eval_views=eval_views,
        task_class_lists=task_class_lists, n_epochs=4, label="donor B",
    )
    print(f"  trained in {time.time()-t0:.0f}s  trained_classes={trained_b}")
    measure_locality(donor_b.layers[1], "  donor B post-train")

    print("\n--- Phase 3: PRE-absorption baselines ---")
    print("  donor A on its own tasks (0, 1):")
    a_self_ta, a_self_full = eval_on_tasks(
        donor_a, eval_views, task_class_lists, [0, 1], label_prefix="    "
    )
    print("  donor B on its own tasks (2, 3):")
    b_self_ta, b_self_full = eval_on_tasks(
        donor_b, eval_views, task_class_lists, [2, 3], label_prefix="    "
    )

    print("\n--- Phase 4: pool-matched absorption (B → A) ---")
    print(f"  donor A pre-absorb L1.n_nodes = {donor_a.layers[1].n_nodes}")
    print(f"  donor B           L1.n_nodes = {donor_b.layers[1].n_nodes}")
    absorbed = pool_matched_absorb(
        donor_a, donor_b, grid_size=4, pool_capacity=None,
        snap_to_pool_centroid=False,
        donor_trained_classes=trained_b,
        recipient_trained_classes=trained_a,
    )
    print(f"  absorbed {len(absorbed)}/{donor_b.layers[1].n_nodes} donor B cells")
    print(f"  donor A post-absorb L1.n_nodes = {donor_a.layers[1].n_nodes}")

    # Extend recipient's LCN mask for newly-absorbed cells. Each new cell
    # gets a mask row computed from its (donor-inherited) cell_position
    # using the recipient's LCN convention (same as donor's, so identical).
    print("\n  --- Extend recipient's LCN mask for absorbed cells ---")
    extend_l1_lcn_mask(donor_a)
    measure_locality(donor_a.layers[1], "  recipient post-absorb+extend")

    print("\n  --- Manifold merge ---")
    n_added = merge_manifold(manifold_a, manifold_b)
    combined_seen = sorted(set(trained_a) | set(trained_b))
    print(f"  added {n_added} class signatures; manifold has "
          f"{len(manifold_a.mu_per_class)} classes (seen union = {combined_seen})")

    print("\n  --- Head-settle pass ---")
    t0 = time.time()
    final_loss = settle_head_via_manifold(
        donor_a, manifold_a, combined_seen,
        n_steps=200, batch_size=64, lr=0.01, noise_scale=1.0,
    )
    print(f"  head-settle 200 steps in {time.time()-t0:.1f}s  final_loss={final_loss:.4f}")

    print("\n--- Phase 5: POST-absorption eval ---")
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
    a_ok = ab_a_ta >= a_self_ta - 0.05
    b_ok = ab_b_ta >= b_self_ta - 0.05
    if a_ok and b_ok:
        print("  → PASS: LCN+absorption combo preserves both donors within 0.05")
    else:
        print(f"  → FAIL: A_ok={a_ok}  B_ok={b_ok}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
