"""HAT (Serra et al. 2018) baseline — multi-seed, 50-task curriculum.

Per next_session_plan.md: HAT is the closest published task-aware
continual-learning baseline to the §11 risk register's "basically
DEN with extra steps". The literature consensus (Parisi 2019, van de
Ven 2019, Mai 2022) places HAT ≈ DEN on standard benchmarks. If grown
beats HAT robustly we plausibly close the DEN line by transitivity;
if grown only ties HAT, the DEN comparison stays genuinely open.

Setup:
  - HAT × {H=8, H=12, H=16} × 3 seeds = 9 runs.
  - Same 50-task / 12-dim curriculum as bench_50task_seeds.
  - Hyperparameters from the original paper:
      s_max = 400, s_min = 1/400, sparsity_coef = 0.75, emb_clip = 6.0.
  - Compare against grown's already-collected seed distribution from
    bench_50task_seeds_run1 (mean 0.053 ± 0.021, params 247-264).

Notes:
  - HAT, like PackNet, uses task ID at inference (per-task embedding).
    Same literature-standard asymmetry that PackNet enjoys; kept in.
  - Tracker exposes cumulative_mask_density per masked layer at the end
    of training; logged for diagnostics. If density saturates (≈1.0)
    long before all 50 tasks complete, that's a capacity-exhaustion
    signal worth reporting in the result memory.
"""
from __future__ import annotations
import csv
import os
import statistics
import sys
import time
from typing import Dict, List, Tuple

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trioron.curriculum import ParameterizedContrastiveCurriculum

from experiments.bench_50task import (
    STATE_DIM,
    N_SINGLE,
    N_COMPOUND,
    build_50task_pairs,
    make_fixed_eval_batches,
    make_network,
    run_hat_curriculum,
)


SEEDS = [0, 1, 2]
HAT_HIDDEN_SIZES = [8, 12, 16]

HAT_S_MIN = 1.0 / 400.0
HAT_S_MAX = 400.0
HAT_SPARSITY_COEF = 0.75
HAT_EMB_CLIP = 6.0

# Grown's already-collected distribution from bench_50task_seeds_run1.log
# (commit e5915db). Used for cross-comparison without re-running grown.
GROWN_AVG_FINALS = [0.0297, 0.0689, 0.0603]
GROWN_FINAL_LATENTS = [4, 3, 3]
GROWN_FINAL_PARAMS = [247, 264, 252]
TARGET_LATENT = int(round(sum(GROWN_FINAL_LATENTS) / len(GROWN_FINAL_LATENTS)))


def _agg(xs: List[float]) -> Tuple[float, float]:
    if len(xs) < 2:
        return (xs[0] if xs else float("nan"), 0.0)
    return statistics.mean(xs), statistics.stdev(xs)


def main() -> int:
    pair_specs = build_50task_pairs(
        state_dim=STATE_DIM, n_single=N_SINGLE, n_compound=N_COMPOUND, seed=0,
    )
    pair_names = [p.name for p in pair_specs]

    def cur_factory(seed):
        return ParameterizedContrastiveCurriculum(
            state_dim=STATE_DIM, pair_specs=pair_specs, seed=seed)

    eval_batches = make_fixed_eval_batches(pair_names, cur_factory)

    print("=" * 78)
    print("bench_50task_hat_seeds — HAT × 3 widths × 3 seeds")
    print("=" * 78)
    print(f"Seeds:  {SEEDS}")
    print(f"Widths: {HAT_HIDDEN_SIZES}")
    print(f"HAT params: s∈[{HAT_S_MIN:.4f},{HAT_S_MAX}] "
          f"sparsity_coef={HAT_SPARSITY_COEF} emb_clip={HAT_EMB_CLIP}")
    print(f"Target latent (from grown's distribution): {TARGET_LATENT}")
    print(f"50-task curriculum: 12 single + 38 compound on 12-dim state")
    print()

    rows: List[dict] = []
    results_by_H: Dict[int, List[dict]] = {H: [] for H in HAT_HIDDEN_SIZES}
    t_start = time.monotonic()

    for H in HAT_HIDDEN_SIZES:
        for s in SEEDS:
            torch.manual_seed(s + 4000 + H)
            net = make_network(STATE_DIM, H, TARGET_LATENT)
            result = run_hat_curriculum(
                net, label=f"hat_H{H}_seed{s}",
                train_cur=cur_factory(seed=s + 17 * H),
                eval_batches=eval_batches, pair_names=pair_names,
                s_min=HAT_S_MIN, s_max=HAT_S_MAX,
                sparsity_coef=HAT_SPARSITY_COEF, emb_clip=HAT_EMB_CLIP,
            )
            rows.append({
                "arch": "hat",
                "H": H,
                "seed": s,
                "n_params_final": result["final_n_params"],
                "final_latent": result["final_latent"],
                "avg_final_loss": result["avg_final_loss"],
                "avg_forgetting": result["avg_forgetting"],
                "wall_clock_seconds": result["wall_clock_seconds"],
                "hat_cum_density_layer0": result["hat_cumulative_density"][0],
                "hat_cum_density_layer1": result["hat_cumulative_density"][1],
            })
            results_by_H[H].append(result)

    elapsed_total = time.monotonic() - t_start

    # ----- Report -----
    print()
    print("=" * 78)
    print(f"bench_50task_hat_seeds — Final Report  "
          f"(total wall-clock {elapsed_total/60:.1f} min)")
    print("=" * 78)
    print()
    print(f"HAT ({len(SEEDS)} seeds × {len(HAT_HIDDEN_SIZES)} widths):")
    print(f"  {'H':>4} {'params':>8} {'avg_final mean±std':>22} "
          f"{'cum_density(L0,L1)':>22} {'per-seed avg_final':>40}")
    for H in HAT_HIDDEN_SIZES:
        results = results_by_H[H]
        finals = [r["avg_final_loss"] for r in results]
        params = results[0]["final_n_params"]
        f_mean, f_std = _agg(finals)
        d0 = sum(r["hat_cumulative_density"][0] for r in results) / len(results)
        d1 = sum(r["hat_cumulative_density"][1] for r in results) / len(results)
        finals_str = "[" + ", ".join(f"{x:.4f}" for x in finals) + "]"
        print(f"  {H:>4d} {params:>8d}    {f_mean:.4f} ± {f_std:.4f}"
              f"     ({d0:.2f},{d1:.2f})    {finals_str}")
    print()

    g_mean, g_std = _agg(GROWN_AVG_FINALS)
    print(f"Grown distribution (from bench_50task_seeds_run1):")
    print(f"  per-seed: {GROWN_AVG_FINALS} → mean ± std: {g_mean:.4f} ± {g_std:.4f}")
    print()

    # Headline: grown vs matched HAT (closest in params)
    matched_H = min(
        HAT_HIDDEN_SIZES,
        key=lambda H: abs(results_by_H[H][0]["final_n_params"] -
                          (sum(GROWN_FINAL_PARAMS) / len(GROWN_FINAL_PARAMS))),
    )
    matched_finals = [r["avg_final_loss"] for r in results_by_H[matched_H]]
    m_mean, m_std = _agg(matched_finals)
    print(f"Headline: grown vs matched HAT (H={matched_H}, "
          f"~{results_by_H[matched_H][0]['final_n_params']} params):")
    print(f"  grown:   {g_mean:.4f} ± {g_std:.4f}")
    print(f"  HAT:     {m_mean:.4f} ± {m_std:.4f}")
    rel = (m_mean - g_mean) / g_mean if g_mean > 0 else 0
    multiplier = m_mean / g_mean if g_mean > 0 else 0
    print(f"  HAT's mean is {multiplier:.2f}× grown's mean "
          f"(grown {rel*100:+.0f}% lower)")
    if g_std + m_std > 0:
        gap_in_std = (m_mean - g_mean) / max(g_std + m_std, 1e-9)
        print(f"  gap is {gap_in_std:+.1f}× combined std — "
              f"{'robust' if gap_in_std > 2 else 'borderline' if gap_in_std > 1 else 'within noise'}")
    print()

    # Best-of HAT comparison
    best_H = min(
        HAT_HIDDEN_SIZES,
        key=lambda H: statistics.mean([r["avg_final_loss"] for r in results_by_H[H]]),
    )
    best_finals = [r["avg_final_loss"] for r in results_by_H[best_H]]
    bp_mean, bp_std = _agg(best_finals)
    bp_params = results_by_H[best_H][0]["final_n_params"]
    print(f"Best HAT width: H={best_H} ({bp_params} params, "
          f"{bp_params / (sum(GROWN_FINAL_PARAMS) / len(GROWN_FINAL_PARAMS)):.2f}× "
          f"grown's avg params):")
    print(f"  grown:   {g_mean:.4f} ± {g_std:.4f}")
    print(f"  HAT:     {bp_mean:.4f} ± {bp_std:.4f}")
    if g_mean > 0:
        print(f"  HAT's mean is {bp_mean/g_mean:.2f}× grown's mean")

    # CSV
    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs"
    )
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "bench_50task_hat_seeds_log.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arch", "H", "seed", "n_params_final", "final_latent",
                    "avg_final_loss", "avg_forgetting",
                    "wall_clock_seconds",
                    "hat_cum_density_layer0", "hat_cum_density_layer1"])
        for r in rows:
            w.writerow([r["arch"], r["H"], r["seed"], r["n_params_final"],
                        r["final_latent"], f"{r['avg_final_loss']:.6f}",
                        f"{r['avg_forgetting']:.6f}",
                        f"{r['wall_clock_seconds']:.2f}",
                        f"{r['hat_cum_density_layer0']:.4f}",
                        f"{r['hat_cum_density_layer1']:.4f}"])
    print(f"\n  log: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
