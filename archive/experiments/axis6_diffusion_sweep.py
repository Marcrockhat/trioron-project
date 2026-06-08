"""Sweep λ_diff for the diffusion-of-W mechanism on L1.

Reuses axis6_credit_chained15.run_arm, varies AXIS6_LAMBDA_DIFF.
AXIS6_credit_manifold arm only — the diffusion mechanism is meant
to be additive on top of the credit + manifold + cosine stack.
"""
from __future__ import annotations

import os
import sys
import time
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.axis6_credit_chained15 import (
    DatasetBundle, build_task_views, chained_15_specs, run_arm, mean_std,
)
from experiments.datasets import DEFAULT_DATA_ROOT


def main() -> int:
    lambdas_env = os.environ.get("AXIS6_LAMBDAS", "0.0,0.05,0.1,0.2,0.3")
    lambdas = [float(x) for x in lambdas_env.split(",")]
    n_seeds = int(os.environ.get("AXIS6_N_SEEDS", "1"))
    n_epochs = int(os.environ.get("AXIS6_EPOCHS", "4"))
    batch_size = int(os.environ.get("AXIS6_BATCH", "256"))
    lr = float(os.environ.get("AXIS6_LR", "0.01"))
    credit_thr = float(os.environ.get("AXIS6_CREDIT_THR", "0.3"))
    diff_floor = float(os.environ.get("AXIS6_FLOOR", "0.0005"))
    field_sigma = float(os.environ.get("AXIS6_SIGMA", "0.15"))
    spawn_cap = int(os.environ.get("AXIS6_SPAWN_CAP", "5"))
    cooldown_steps = int(os.environ.get("AXIS6_COOLDOWN", "20"))
    lcn_enabled = os.environ.get("AXIS6_LCN", "1") == "1"
    lambda_manifold = float(os.environ.get("AXIS6_LAMBDA_MANIFOLD", "1.0"))
    manifold_total_batch = int(os.environ.get("AXIS6_MANIFOLD_BATCH", "64"))
    manifold_noise_scale = float(os.environ.get("AXIS6_MANIFOLD_NOISE", "1.0"))

    print("=" * 88)
    print(f"L1 diffusion-of-W sweep — λ_diff ∈ {lambdas}, n={n_seeds} seeds, "
          f"{n_epochs} epochs, LCN={'on' if lcn_enabled else 'off'}")
    print("=" * 88)

    bundle = DatasetBundle(
        ["mnist", "fashion_mnist", "emnist_letters"],
        root=DEFAULT_DATA_ROOT, n_holdout_per_dataset=0,
    )
    specs = chained_15_specs()
    train_views = build_task_views(bundle, specs, split="train")
    eval_views = build_task_views(bundle, specs, split="test")
    task_class_lists = [s.global_classes for s in specs]

    results: dict = {}
    for lam in lambdas:
        per_seed = []
        for seed in range(n_seeds):
            hp = dict(
                diff_floor=diff_floor, diff_b_thr=1.0,
                field_sigma=field_sigma, field_dt=0.1,
                stress_tolerance=0.0,
                spawn_cap=spawn_cap, cooldown_steps=cooldown_steps,
                credit_thr=credit_thr,
                lambda_manifold=lambda_manifold,
                manifold_total_batch=manifold_total_batch,
                manifold_noise_scale=manifold_noise_scale,
                lcn_enabled=lcn_enabled,
                lambda_diff=lam,
            )
            t0 = time.time()
            r = run_arm(
                seed=seed,
                arm_label="AXIS6_credit_manifold",
                axis6=True, credit_enabled=True, manifold_enabled=True,
                train_views=train_views, eval_views=eval_views,
                task_class_lists=task_class_lists,
                n_epochs=n_epochs, batch_size=batch_size, lr=lr,
                hp=hp, verbose=False,
            )
            elapsed = time.time() - t0
            per_seed.append(r)
            print(
                f"[λ={lam:>4.2f} seed={seed}]  "
                f"task_aware={r['mean_final_task_aware']:.3f}  "
                f"full={r['mean_final_full']:.3f}  "
                f"max|drift|={r['max_drift_task_aware']:.3f}  "
                f"H={r['n_hidden_L1']}  spawns={r['n_spawns']}  "
                f"frozen={r['n_frozen']}  ({elapsed:.0f}s)"
            )
        results[lam] = per_seed

    print("\n" + "=" * 88)
    print(f"AGGREGATE — λ_diff sweep (n={n_seeds} seeds)")
    print("=" * 88)
    print(
        f"  {'λ_diff':>6}  {'task_aware':>14}  {'full':>14}  "
        f"{'max|drift|':>14}  {'H_L1':>10}  {'frozen':>10}"
    )
    for lam in lambdas:
        rs = results[lam]
        ta_m, ta_s = mean_std([r["mean_final_task_aware"] for r in rs])
        f_m, f_s = mean_std([r["mean_final_full"] for r in rs])
        md_m, md_s = mean_std([r["max_drift_task_aware"] for r in rs])
        h_m, h_s = mean_std([float(r["n_hidden_L1"]) for r in rs])
        fr_m, fr_s = mean_std([float(r["n_frozen"]) for r in rs])
        print(
            f"  {lam:>6.2f}  {ta_m:>6.3f} ± {ta_s:>5.3f}  "
            f"{f_m:>6.3f} ± {f_s:>5.3f}  "
            f"{md_m:>6.3f} ± {md_s:>5.3f}  "
            f"{h_m:>5.1f} ± {h_s:>3.1f}  "
            f"{fr_m:>5.1f} ± {fr_s:>3.1f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
