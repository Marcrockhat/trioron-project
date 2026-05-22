"""Multi-task position-evolution probe.

Runs the chained-15 curriculum task-by-task, dumping L1 cell-position
stats (xy spread, per-cell trajectory, collapse indicator) at the end
of EVERY task. Compares ALIGN on/off at the same low weight so we can
see whether:

  (a) per-task collapse + between-task redistribution (positions move
      a lot but stay spread across the substrate over the curriculum)
  (b) progressive accumulation (positions migrate steadily toward a
      stable topographic layout)
  (c) collapse without recovery (cells converge to a single point)
  (d) no effect (positions just track baseline noise)

Reuses train_one_task directly so the per-task end gives us a clean
inspection point. Runs a configurable subset of tasks (default 6 so
we cross the dataset switch at task 5 → fashion_mnist, which often
shakes up cell engagement patterns).

Env knobs (with defaults tuned for visible-but-not-catastrophic):
  PROBE_ALIGN_WEIGHT=0.05
  PROBE_N_TASKS=6
  PROBE_EPOCHS=4
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def _summarize_positions(L1) -> dict:
    """Return spread/cluster diagnostics for L1.cell_position[:n_nodes]."""
    n = L1.n_nodes
    pos = L1.cell_position[:n, :2].detach()
    return {
        "n": n,
        "x_min": float(pos[:, 0].min()),
        "x_max": float(pos[:, 0].max()),
        "y_min": float(pos[:, 1].min()),
        "y_max": float(pos[:, 1].max()),
        "x_std": float(pos[:, 0].std()) if n > 1 else 0.0,
        "y_std": float(pos[:, 1].std()) if n > 1 else 0.0,
        "centroid_x": float(pos[:, 0].mean()),
        "centroid_y": float(pos[:, 1].mean()),
    }


def _mean_pairwise_distance(L1) -> float:
    n = L1.n_nodes
    pos = L1.cell_position[:n, :2].detach()
    if n < 2:
        return 0.0
    diff = pos.unsqueeze(1) - pos.unsqueeze(0)
    dist = diff.norm(dim=-1)
    dist.fill_diagonal_(0.0)
    return float(dist.sum() / (n * (n - 1)))


def run_curriculum(align: bool, label: str) -> None:
    print("\n" + "=" * 78)
    print(f"PROBE: {label}")
    print("=" * 78)

    weight = os.environ.get("PROBE_ALIGN_WEIGHT", "0.05")
    n_tasks = int(os.environ.get("PROBE_N_TASKS", "6"))
    n_epochs = int(os.environ.get("PROBE_EPOCHS", "4"))

    os.environ["AXIS6_ALIGN"] = "1" if align else "0"
    os.environ["AXIS6_ALIGN_WEIGHT"] = weight
    os.environ["AXIS6_FINITE_SPACE"] = "1"
    os.environ["AXIS6_REPULSE_MIN_SEP"] = "0.15"
    os.environ["AXIS6_REPULSE_STEPS"] = "16"
    os.environ["AXIS6_REPULSE_STRENGTH"] = "0.5"
    os.environ["AXIS6_BOUND_REGION"] = "unit_box"

    import importlib
    import experiments.axis6_credit_chained15 as mod
    importlib.reload(mod)
    from experiments.datasets import (
        DatasetBundle, build_task_views, chained_15_specs, DEFAULT_DATA_ROOT,
    )

    bundle = DatasetBundle(
        ["mnist", "fashion_mnist", "emnist_letters"],
        root=DEFAULT_DATA_ROOT, n_holdout_per_dataset=0,
    )
    specs = chained_15_specs()[:n_tasks]
    train_views = build_task_views(bundle, specs, split="train")
    eval_views = build_task_views(bundle, specs, split="test")
    task_class_lists = [s.global_classes for s in specs]

    net = mod.build_net(seed=0, lcn_enabled=True)
    L1 = net.layers[1]
    init_positions = L1.cell_position[: L1.n_nodes, :2].clone()

    initial_stats = _summarize_positions(L1)
    initial_pairwise = _mean_pairwise_distance(L1)
    print(f"\n[task 0 — INITIAL]  n_cells={initial_stats['n']}")
    print(
        f"  x=[{initial_stats['x_min']:.3f}, {initial_stats['x_max']:.3f}]  "
        f"(width={initial_stats['x_max']-initial_stats['x_min']:.3f}, "
        f"std={initial_stats['x_std']:.3f})"
    )
    print(
        f"  y=[{initial_stats['y_min']:.3f}, {initial_stats['y_max']:.3f}]  "
        f"(width={initial_stats['y_max']-initial_stats['y_min']:.3f}, "
        f"std={initial_stats['y_std']:.3f})"
    )
    print(f"  centroid=({initial_stats['centroid_x']:.3f}, "
          f"{initial_stats['centroid_y']:.3f})")
    print(f"  mean pairwise distance = {initial_pairwise:.4f}")

    cell_credit = torch.zeros(L1.n_nodes, dtype=torch.bool)
    manifold = mod.ManifoldStore(n_l0=net.layers[0].n_nodes)
    seen_classes = []

    print(
        f"\n{'task':>5}  {'n':>4}  {'spawns':>6}  {'frozen':>6}  "
        f"{'x_width':>8}  {'y_width':>8}  {'x_std':>7}  {'y_std':>7}  "
        f"{'pair_dist':>9}  {'init_drift':>10}  {'task_aware':>10}  {'full':>6}"
    )
    print("-" * 120)

    for task_idx in range(n_tasks):
        for c in task_class_lists[task_idx]:
            if c not in seen_classes:
                seen_classes.append(c)
        res = mod.train_one_task(
            net=net, view=train_views[task_idx],
            task_global_classes=task_class_lists[task_idx],
            n_epochs=n_epochs, batch_size=256, lr=0.01,
            axis6=True, diff_floor=0.0005, diff_b_thr=1.0,
            field_sigma=0.15, field_dt=0.1, stress_tolerance=0.0,
            spawn_cap=5, cooldown_steps=20,
            l1_credit_mask=cell_credit,
            track_engagement=True,
            manifold=manifold,
            lambda_manifold=1.0,
            manifold_total_batch=64,
            manifold_noise_scale=1.0,
            seen_classes_global=list(seen_classes),
            lambda_diff=0.0,
            W_snapshot=None,
            verbose=False,
        )
        spawn_events = sum(
            1 for _, msg in res.events if msg.startswith("AXIS6 spawn:")
        )
        # Update credit.
        if res.engagement_frac is not None:
            ef = res.engagement_frac
            if cell_credit.numel() < L1.n_nodes:
                pad = torch.zeros(L1.n_nodes - cell_credit.numel(), dtype=torch.bool)
                cell_credit = torch.cat([cell_credit, pad])
            if ef.numel() < L1.n_nodes:
                ef = torch.cat([ef, torch.zeros(L1.n_nodes - ef.numel())])
            cell_credit = cell_credit | (ef > 0.3)
        # Snapshot manifold for this task's classes.
        manifold.store_task(net, train_views[task_idx], task_class_lists[task_idx])

        # Position stats.
        stats = _summarize_positions(L1)
        pairwise = _mean_pairwise_distance(L1)
        # Drift from initial — only for cells that existed at init.
        n_init = init_positions.shape[0]
        drift = (L1.cell_position[:n_init, :2].detach() - init_positions).abs()
        init_drift = float(drift.mean())

        # Eval this task.
        task_aware = mod.evaluate_task_aware(
            net, eval_views[task_idx], task_class_lists[task_idx],
            lambda_diff=0.0, credit_mask=cell_credit, W_snapshot=None,
        )
        full = mod.evaluate_full(
            net, eval_views[task_idx],
            lambda_diff=0.0, credit_mask=cell_credit, W_snapshot=None,
        )

        n_frozen = int(cell_credit[:L1.n_nodes].sum().item())
        print(
            f"{task_idx+1:>5}  {stats['n']:>4}  {spawn_events:>6}  "
            f"{n_frozen:>6}  "
            f"{stats['x_max']-stats['x_min']:>8.3f}  "
            f"{stats['y_max']-stats['y_min']:>8.3f}  "
            f"{stats['x_std']:>7.3f}  {stats['y_std']:>7.3f}  "
            f"{pairwise:>9.4f}  {init_drift:>10.4f}  "
            f"{task_aware:>10.3f}  {full:>6.3f}"
        )


def main() -> int:
    run_curriculum(align=False, label="ALIGN=off (control, finite_space=on)")
    run_curriculum(align=True, label="ALIGN=on (finite_space=on)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
