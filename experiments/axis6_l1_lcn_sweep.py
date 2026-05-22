"""Co-design sweep: LCN-on-L1 × diffusion-of-W on the chained-15 bench.

The hypothesis under test is that L1's diffusion-of-W mechanism only
emerges into convolution-like organization when L1 cells also READ from
positionally-local L0 cells. The diffusion-of-W null result (commit
1768e61) diagnosed input-side locality as the missing half; this
sweep settles the co-design question:

    arms (3 mask modes) × (variable λ_diff)
      mask ∈ {off, soft, hard}        (AXIS6_L1_LCN_MODE)
      λ_diff ∈ {0.0, 0.1, 0.2}        (AXIS6_LAMBDAS)

  - mask=off, λ=0      : pre-existing baseline
  - mask=off, λ>0      : prior null (diffusion alone)
  - mask∈{soft,hard}, λ=0   : locality alone
  - mask∈{soft,hard}, λ>0   : co-design (emergent conv)

A constructive interaction (co-design > both isolated mechanisms) is
the gate for declaring conv-by-self-organization shipping. Falsified
otherwise; record as "diffusion+LCN don't co-design on chained-15".

Reuses axis6_credit_chained15.run_arm so all bench machinery (credit
freeze, manifold replay, cosine head, Axis 6 spawn, W_snapshot) is
shared with the baseline.
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


# Sentinel hp values used for "ignored" axis-of-the-other-mode in the cell list.
L1_SIGMA_NA = 0.0
L1_K_NA = 0


def main() -> int:
    modes_env = os.environ.get("AXIS6_L1_LCN_MODES", "off,soft,hard")
    modes = [m.strip().lower() for m in modes_env.split(",") if m.strip()]
    for m in modes:
        if m not in {"off", "soft", "hard"}:
            raise ValueError(f"AXIS6_L1_LCN_MODES contains invalid {m!r}")
    lambdas_env = os.environ.get("AXIS6_LAMBDAS", "0.0,0.1,0.2")
    lambdas = [float(x) for x in lambdas_env.split(",")]
    # Per-mode hyperparameter sweeps:
    #   soft → vary σ (Gaussian falloff width in image space)
    #   hard → vary K (top-k window size, in # of L0 cells)
    #   off  → single cell (sigma/K ignored)
    sigmas_env = os.environ.get("AXIS6_L1_LCN_SIGMAS", "0.25")
    sigmas = [float(x) for x in sigmas_env.split(",")]
    ks_env = os.environ.get("AXIS6_L1_LCN_KS", "8")
    ks = [int(x) for x in ks_env.split(",")]
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

    # Materialize the cell list: (mode, lambda, sigma, k, label).
    cells: List[tuple] = []
    for mode in modes:
        if mode == "off":
            per_mode_hps = [(L1_SIGMA_NA, L1_K_NA, "—")]
        elif mode == "soft":
            per_mode_hps = [(s, L1_K_NA, f"σ={s}") for s in sigmas]
        elif mode == "hard":
            per_mode_hps = [(L1_SIGMA_NA, k, f"K={k}") for k in ks]
        for lam in lambdas:
            for sigma, k, hp_label in per_mode_hps:
                cells.append((mode, lam, sigma, k, hp_label))

    print("=" * 96)
    print(
        f"L1 co-design sweep — modes ∈ {modes}, λ_diff ∈ {lambdas}, "
        f"σ ∈ {sigmas}, K ∈ {ks}, n={n_seeds} seeds, {n_epochs} ep, "
        f"L0_LCN={'on' if lcn_enabled else 'off'}, cells={len(cells)}"
    )
    print("=" * 96)

    bundle = DatasetBundle(
        ["mnist", "fashion_mnist", "emnist_letters"],
        root=DEFAULT_DATA_ROOT, n_holdout_per_dataset=0,
    )
    specs = chained_15_specs()
    train_views = build_task_views(bundle, specs, split="train")
    eval_views = build_task_views(bundle, specs, split="test")
    task_class_lists = [s.global_classes for s in specs]

    results: dict = {}
    for mode, lam, sigma, k, hp_label in cells:
        cell_key = (mode, lam, sigma, k)
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
                l1_lcn_mode=mode,
                l1_lcn_sigma=sigma,
                l1_lcn_k=k,
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
                f"[mask={mode:>4}  λ={lam:>4.2f}  {hp_label:>7}  seed={seed}]  "
                f"task_aware={r['mean_final_task_aware']:.3f}  "
                f"full={r['mean_final_full']:.3f}  "
                f"max|drift|={r['max_drift_task_aware']:.3f}  "
                f"H={r['n_hidden_L1']}  spawns={r['n_spawns']}  "
                f"frozen={r['n_frozen']}  ({elapsed:.0f}s)"
            )
        results[cell_key] = per_seed

    print("\n" + "=" * 96)
    print(f"AGGREGATE — mask × λ_diff × hp (n={n_seeds} seeds)")
    print("=" * 96)
    print(
        f"  {'mask':>4}  {'λ_diff':>6}  {'hp':>8}  {'task_aware':>14}  "
        f"{'full':>14}  {'max|drift|':>14}  {'H_L1':>10}  {'frozen':>10}"
    )
    for mode, lam, sigma, k, hp_label in cells:
        rs = results[(mode, lam, sigma, k)]
        ta_m, ta_s = mean_std([r["mean_final_task_aware"] for r in rs])
        f_m, f_s = mean_std([r["mean_final_full"] for r in rs])
        md_m, md_s = mean_std([r["max_drift_task_aware"] for r in rs])
        h_m, h_s = mean_std([float(r["n_hidden_L1"]) for r in rs])
        fr_m, fr_s = mean_std([float(r["n_frozen"]) for r in rs])
        print(
            f"  {mode:>4}  {lam:>6.2f}  {hp_label:>8}  "
            f"{ta_m:>6.3f} ± {ta_s:>5.3f}  "
            f"{f_m:>6.3f} ± {f_s:>5.3f}  "
            f"{md_m:>6.3f} ± {md_s:>5.3f}  "
            f"{h_m:>5.1f} ± {h_s:>3.1f}  "
            f"{fr_m:>5.1f} ± {fr_s:>3.1f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
