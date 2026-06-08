"""Trioron 2.0 Axis 6 + credit — spawn_cap Pareto sweep on 10-task curriculum.

Probes the plasticity ↔ substrate-size trade-off. AXIS6_credit only (the
other two arms are insensitive to spawn_cap at the level that matters).
Hypothesis: at spawn_cap=3, complex late-curriculum tasks (checker,
annulus) under-fit because they get only ~3 freshly-spawnable plastic
cells. Raising spawn_cap should lift those ceilings at the cost of H_final.
At what spawn_cap do checker / annulus saturate?
"""
from __future__ import annotations

import os
import sys
from typing import List

from experiments.axis6_credit_10task import (
    make_curriculum_10,
    run_one_arm,
    mean_std,
)


def main() -> int:
    diff_mode = os.environ.get("AXIS6_MODE", "absolute")
    diff_k = float(os.environ.get("AXIS6_K", "1.0"))
    diff_floor = float(os.environ.get("AXIS6_FLOOR", "0.005"))
    diff_b_thr = float(os.environ.get("AXIS6_B_THR", "1.0"))
    field_sigma = float(os.environ.get("AXIS6_SIGMA", "1.5"))
    field_dt = float(os.environ.get("AXIS6_DT", "0.1"))
    stress_tolerance = float(os.environ.get("AXIS6_TOL", "0.0"))
    cooldown_steps = int(os.environ.get("AXIS6_COOLDOWN", "50"))
    H_init = int(os.environ.get("AXIS6_H_INIT", "4"))
    credit_thr = float(os.environ.get("AXIS6_CREDIT_THR", "0.1"))
    n_seeds = int(os.environ.get("AXIS6_N_SEEDS", "3"))

    spawn_caps_env = os.environ.get("AXIS6_SPAWN_CAPS", "2,3,5,8,12")
    spawn_caps = [int(x) for x in spawn_caps_env.split(",")]

    epochs_per_task = [200, 400, 400, 400, 400, 400, 400, 400, 400, 400]
    lr = 0.05

    X, tasks = make_curriculum_10(n=300, seed=0)
    task_names = [t[0] for t in tasks]
    print("=" * 100)
    print(f"AXIS6_credit spawn_cap sweep — caps={spawn_caps}, n={n_seeds} seeds, 10 tasks")
    print(f"H_init={H_init}  credit_thr={credit_thr}  cooldown={cooldown_steps}  "
          f"field_sigma={field_sigma}  floor={diff_floor}")
    print(f"tasks: {task_names}")
    print("=" * 100)

    # Per-spawn_cap aggregated results, plus per-task ceilings.
    all_results: dict = {}
    for cap in spawn_caps:
        hp = dict(
            diff_mode=diff_mode, diff_k=diff_k, diff_floor=diff_floor,
            diff_b_thr=diff_b_thr, field_sigma=field_sigma, field_dt=field_dt,
            stress_tolerance=stress_tolerance, spawn_cap=cap,
            cooldown_steps=cooldown_steps, credit_thr=credit_thr,
        )
        per_seed = []
        for seed in range(n_seeds):
            r = run_one_arm(
                seed=seed, arm_label="AXIS6_credit",
                axis6=True, credit_enabled=True,
                X=X, tasks=tasks, epochs_per_task=epochs_per_task,
                lr=lr, H_init=H_init, hp=hp, verbose=False,
            )
            per_seed.append(r)
            print(
                f"  [spawn_cap={cap:>2}  seed={seed}]  "
                f"max|drift|={r['max_drift']:.3f}  "
                f"sum_final={r['sum_final_acc']:.3f}  "
                f"mean_final={r['mean_final_acc']:.3f}  "
                f"H={r['n_hidden']}  spawns={r['n_spawns']}  "
                f"frozen={r['n_frozen']}"
            )
        all_results[cap] = per_seed

    print("\n" + "=" * 100)
    print(f"AGGREGATE — spawn_cap sweep (mean ± std across {n_seeds} seeds)")
    print("=" * 100)
    print(
        f"  {'spawn_cap':>10}  {'max|drift|':>14}  {'sum_final':>14}  "
        f"{'mean_final':>14}  {'H_final':>10}  {'frozen':>10}"
    )
    for cap in spawn_caps:
        rs = all_results[cap]
        md_m, md_s = mean_std([r["max_drift"] for r in rs])
        sf_m, sf_s = mean_std([r["sum_final_acc"] for r in rs])
        mf_m, mf_s = mean_std([r["mean_final_acc"] for r in rs])
        h_m, h_s = mean_std([float(r["n_hidden"]) for r in rs])
        fr_m, fr_s = mean_std([float(r["n_frozen"]) for r in rs])
        print(
            f"  {cap:>10}  {md_m:>6.3f} ± {md_s:>5.3f}  "
            f"{sf_m:>6.3f} ± {sf_s:>5.3f}  "
            f"{mf_m:>6.3f} ± {mf_s:>5.3f}  "
            f"{h_m:>5.1f} ± {h_s:>3.1f}  "
            f"{fr_m:>5.1f} ± {fr_s:>3.1f}"
        )

    print(f"\n  Per-task final acc (mean across {n_seeds} seeds) — columns = spawn_cap:")
    header = f"  {'task':<14}  " + "  ".join(f"cap{cap:>2}" for cap in spawn_caps)
    print(header)
    for k, name in enumerate(task_names):
        cells = []
        for cap in spawn_caps:
            vals = [r["final_acc"][k] for r in all_results[cap]]
            m, _ = mean_std(vals)
            cells.append(f"{m:>5.3f}")
        print(f"  {name:<14}  " + "  ".join(cells))

    csv_path = os.environ.get(
        "AXIS6_CSV", "outputs/axis6_credit_spawn_sweep_n3.csv"
    )
    try:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w") as f:
            f.write("spawn_cap,seed,task,first_acc,final_acc,drift,max_drift,sum_final,n_hidden,n_spawns,n_frozen\n")
            for cap in spawn_caps:
                for r in all_results[cap]:
                    for k, name in enumerate(task_names):
                        f.write(
                            f"{cap},{r['seed']},{name},"
                            f"{r['first_acc'][k]:.4f},{r['final_acc'][k]:.4f},"
                            f"{r['drift'][k]:.4f},{r['max_drift']:.4f},"
                            f"{r['sum_final_acc']:.4f},{r['n_hidden']},"
                            f"{r['n_spawns']},{r['n_frozen']}\n"
                        )
        print(f"\n  CSV → {csv_path}")
    except Exception as e:
        print(f"  CSV write failed: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
