"""Diagnostic: verify Boquila alignment actually migrates positions.

Runs a single seed, single task, dumps L1 cell positions BEFORE and
AFTER training, plus position-drift stats. Compares ALIGN on vs off so
the drift attributable to alignment is visible.

If alignment is doing anything, ALIGN=on should produce MORE drift than
ALIGN=off (where positions only change via finite-space repulsion, if
that's also on).
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def run_probe(align: bool, finite_space: bool) -> None:
    # Force-set env BEFORE importing the bench module so module-level
    # constants pick up the right values. Need to reload if testing
    # multiple configs in the same process — easier to use subprocess.
    label = (
        f"ALIGN={'on' if align else 'off'}  "
        f"FINITE_SPACE={'on' if finite_space else 'off'}"
    )
    print("\n" + "=" * 72)
    print(f"PROBE: {label}")
    print("=" * 72)

    os.environ["AXIS6_ALIGN"] = "1" if align else "0"
    os.environ["AXIS6_ALIGN_WEIGHT"] = os.environ.get("PROBE_ALIGN_WEIGHT", "0.05")
    os.environ["AXIS6_FINITE_SPACE"] = "1" if finite_space else "0"
    os.environ["AXIS6_REPULSE_MIN_SEP"] = "0.15"
    os.environ["AXIS6_REPULSE_STEPS"] = "16"
    os.environ["AXIS6_REPULSE_STRENGTH"] = "0.5"
    os.environ["AXIS6_BOUND_REGION"] = "unit_box"

    # Re-import to pick up env (importlib.reload is fragile; subprocess
    # is cleaner but for a quick check we just import here once).
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
    specs = chained_15_specs()
    train_views = build_task_views(bundle, specs, split="train")
    eval_views = build_task_views(bundle, specs, split="test")
    task_class_lists = [s.global_classes for s in specs]

    hp = dict(
        diff_floor=0.0005, diff_b_thr=1.0,
        field_sigma=0.15, field_dt=0.1,
        stress_tolerance=0.0,
        spawn_cap=5, cooldown_steps=20,
        credit_thr=0.3,
        lambda_manifold=1.0,
        manifold_total_batch=64,
        manifold_noise_scale=1.0,
        lcn_enabled=True,
        lambda_diff=0.0,
        l1_lcn_mode="off",
        l1_lcn_sigma=0.25,
        l1_lcn_k=8,
    )
    net = mod.build_net(seed=0, lcn_enabled=True)
    L1 = net.layers[1]

    init_positions = L1.cell_position[: L1.n_nodes].clone()
    print(f"\n[init] L1.n_nodes={L1.n_nodes}")
    print(
        f"  position xy range: x=[{init_positions[:, 0].min().item():.3f}, "
        f"{init_positions[:, 0].max().item():.3f}]  "
        f"y=[{init_positions[:, 1].min().item():.3f}, "
        f"{init_positions[:, 1].max().item():.3f}]"
    )
    align_param = getattr(L1, "cell_position_learnable", None)
    print(f"  cell_position_learnable installed: {align_param is not None}")

    # Run just task 0 (mnist 0/1) for 4 epochs to get strong alignment signal.
    res = mod.run_arm(
        seed=0,
        arm_label=label,
        axis6=True, credit_enabled=True, manifold_enabled=True,
        train_views=train_views[:1], eval_views=eval_views[:1],
        task_class_lists=task_class_lists[:1],
        n_epochs=4, batch_size=256, lr=0.01,
        hp=hp,
        starting_net=net,
        verbose=False,
    )

    final_n = L1.n_nodes
    final_positions = L1.cell_position[:final_n].clone()
    # Compare positions of the cells that existed at init.
    init_n = init_positions.shape[0]
    drift = (final_positions[:init_n] - init_positions).abs()
    print(f"\n[after 1 task, 4 ep]  L1.n_nodes={final_n}  spawns={res['n_spawns']}")
    print(
        f"  init-cell drift (xy):  mean={drift[:, :2].mean().item():.4f}  "
        f"max={drift[:, :2].max().item():.4f}"
    )
    print(
        f"  final xy range: x=[{final_positions[:, 0].min().item():.3f}, "
        f"{final_positions[:, 0].max().item():.3f}]  "
        f"y=[{final_positions[:, 1].min().item():.3f}, "
        f"{final_positions[:, 1].max().item():.3f}]"
    )
    print(
        f"  task_aware={res['mean_final_task_aware']:.3f}  "
        f"full={res['mean_final_full']:.3f}"
    )


def main() -> int:
    # Order matters — env is set per-call and module reloaded.
    run_probe(align=False, finite_space=False)
    run_probe(align=False, finite_space=True)
    run_probe(align=True, finite_space=False)
    run_probe(align=True, finite_space=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
