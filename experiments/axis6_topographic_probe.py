"""Topographic emergence probe.

For each L1 cell after training, compute:

  preferred_centroid_i = sum_{image, pixel} pixel_pos[p] * intensity[p] * activation[i]
                        / sum_{image, pixel}        intensity[p] * activation[i]

This is "where on the 28x28 image does cell i fire hardest" — the
activation-weighted center of mass over the input pixel grid.

Then compare against the cell's cell_position. If alignment loss has
produced a topographic map, cells whose POSITION is at (x,y) should
prefer image content at (x,y) — i.e., correlation between cell_position
and preferred_centroid.

Compares ALIGN=on vs ALIGN=off. Both also have finite_space=on.

Reports:
  - Per-cell drift from grid init
  - Pearson correlation between (cell_position_x, preferred_x) and same for y
  - Mean distance |cell_position - preferred_centroid|
  - Visualization (text grid) of where cells "live" in image space
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def _pixel_positions(img_hw: int = 28) -> torch.Tensor:
    """(img_hw*img_hw, 2) tensor of pixel positions in [0,1]^2.

    Order matches row-major flatten (i = row*W + col): pixel i lives at
    (col=i%W+0.5)/W, (row=i//W+0.5)/H — same convention as the LCN code.
    """
    px = (torch.arange(img_hw).float() + 0.5) / img_hw
    py = (torch.arange(img_hw).float() + 0.5) / img_hw
    rows, cols = torch.meshgrid(py, px, indexing="ij")
    return torch.stack([cols, rows], dim=-1).reshape(-1, 2)


def compute_w_weighted_l0_centroids(net) -> torch.Tensor:
    """For each L1 cell i, return the centroid of L0 cell positions
    weighted by |W[i, j]|. This is "where in L0-space does cell i read
    from on average." Since L0 uses an LCN grid in image space, the L0
    cell positions ARE image-space positions; this centroid IS the cell's
    image-space preferred region under its current weights.

    Unlike activation-weighted centroids over input pixels (which collapse
    to image-center on centered MNIST), this directly measures the
    architecture's read pattern and varies meaningfully per cell.
    """
    L0 = net.layers[0]
    L1 = net.layers[1]
    n_l1 = L1.n_nodes
    l0_pos = L0.cell_position[: L0.n_nodes, :2].detach()    # (n_l0, 2)
    W = L1.W[:n_l1].detach().abs()                            # (n_l1, n_l0)
    den = W.sum(dim=1, keepdim=True).clamp(min=1e-8)         # (n_l1, 1)
    centroid = (W @ l0_pos) / den                             # (n_l1, 2)
    return centroid


def _pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x - x.mean()
    y = y - y.mean()
    denom = (x.norm() * y.norm()).clamp(min=1e-8)
    return float((x * y).sum() / denom)


def _ascii_grid(positions: torch.Tensor, grid_size: int = 12) -> str:
    """Render (n, 2) positions in [0,1]² as an ASCII heatmap."""
    counts = torch.zeros(grid_size, grid_size)
    for i in range(positions.shape[0]):
        x, y = positions[i].tolist()
        cx = min(grid_size - 1, max(0, int(x * grid_size)))
        cy = min(grid_size - 1, max(0, int(y * grid_size)))
        counts[cy, cx] += 1
    lines = []
    chars = " .:-=+*#%@"
    cmax = counts.max().item() or 1.0
    for row in counts.flip(0):  # flip so y=0 is bottom
        line = ""
        for v in row.tolist():
            idx = min(len(chars) - 1, int(v / cmax * (len(chars) - 1)))
            line += chars[idx]
        lines.append(line)
    return "\n".join(lines)


def run_probe(align: bool, label: str) -> dict:
    print("\n" + "=" * 78)
    print(f"PROBE: {label}")
    print("=" * 78)

    n_tasks = int(os.environ.get("PROBE_N_TASKS", "15"))
    n_epochs = int(os.environ.get("PROBE_EPOCHS", "4"))
    weight = os.environ.get("PROBE_ALIGN_WEIGHT", "0.01")

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

    # Run training manually (run_arm doesn't return net; we need it here).
    net = mod.build_net(seed=0, lcn_enabled=True, l1_lcn_mode="off")
    L1 = net.layers[1]
    init_positions = L1.cell_position[: L1.n_nodes, :2].clone()
    cell_credit = torch.zeros(L1.n_nodes, dtype=torch.bool)
    manifold = mod.ManifoldStore(n_l0=net.layers[0].n_nodes)
    seen_classes = []
    for task_idx in range(n_tasks):
        for c in task_class_lists[task_idx]:
            if c not in seen_classes:
                seen_classes.append(c)
        r = mod.train_one_task(
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

    # Now probe.
    print(f"\n[trained]  L1.n_nodes={L1.n_nodes}  "
          f"frozen={int(cell_credit[:L1.n_nodes].sum())}")

    preferred = compute_w_weighted_l0_centroids(net)
    cell_pos = L1.cell_position[: L1.n_nodes, :2].clone()

    # Correlation: cell_position_x vs preferred_x, same for y.
    rho_x = _pearson(cell_pos[:, 0], preferred[:, 0])
    rho_y = _pearson(cell_pos[:, 1], preferred[:, 1])
    # Per-cell distance between assigned position and preferred centroid.
    diff = (cell_pos - preferred).norm(dim=1)
    # Drift for the initial cells (the ones that existed at init).
    n_init = init_positions.shape[0]
    if cell_pos.shape[0] >= n_init:
        drift_from_init = (cell_pos[:n_init] - init_positions).norm(dim=1)
    else:
        drift_from_init = torch.tensor([0.0])

    print(f"\n  cell_position vs preferred_centroid:")
    print(f"    Pearson(x, preferred_x)  = {rho_x:+.3f}")
    print(f"    Pearson(y, preferred_y)  = {rho_y:+.3f}")
    print(f"    mean |pos - preferred|   = {float(diff.mean()):.3f}")
    print(f"    max  |pos - preferred|   = {float(diff.max()):.3f}")
    print(f"    mean |pos - init_pos|    = {float(drift_from_init.mean()):.3f}  "
          f"(initial cells only)")

    print(f"\n  cell_position layout (12x12 grid, top=y=1, bottom=y=0):")
    print(_ascii_grid(cell_pos))
    print(f"\n  preferred_centroid layout (same axes):")
    print(_ascii_grid(preferred))
    return {
        "rho_x": rho_x,
        "rho_y": rho_y,
        "mean_diff": float(diff.mean()),
        "mean_drift": float(drift_from_init.mean()),
    }


def main() -> int:
    r_off = run_probe(align=False, label="ALIGN=off  (control)")
    r_on = run_probe(align=True, label="ALIGN=on   (Boquila aux loss)")
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(
        f"  {'config':>20}  {'Pearson_x':>10}  {'Pearson_y':>10}  "
        f"{'mean|diff|':>10}  {'init_drift':>10}"
    )
    print(
        f"  {'ALIGN=off':>20}  {r_off['rho_x']:>10.3f}  {r_off['rho_y']:>10.3f}  "
        f"{r_off['mean_diff']:>10.3f}  {r_off['mean_drift']:>10.3f}"
    )
    print(
        f"  {'ALIGN=on':>20}  {r_on['rho_x']:>10.3f}  {r_on['rho_y']:>10.3f}  "
        f"{r_on['mean_diff']:>10.3f}  {r_on['mean_drift']:>10.3f}"
    )
    print()
    if abs(r_on['rho_x']) - abs(r_off['rho_x']) > 0.15:
        print("  → ALIGN produces stronger position-vs-preferred correlation on X")
    if abs(r_on['rho_y']) - abs(r_off['rho_y']) > 0.15:
        print("  → ALIGN produces stronger position-vs-preferred correlation on Y")
    return 0


if __name__ == "__main__":
    sys.exit(main())
