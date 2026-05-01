"""HAT 6-seed extension — seeds [3,4,5] across 3 widths, post-hoc combined
with the cached 3-seed run from bench_50task_hat_seeds (commit 52cb67d).

Purpose: the original 3-seed bench landed grown 2.36× ahead on means at
matched H=8 but only +0.8σ — central-tendency win, not σ-robust under
the multi-seed standard the PackNet line cleared (2.8σ). HAT's per-seed
variance is enormous (range 4× at H=8), so √2-tightening σ by doubling
seeds is the cheapest path to converting "central-tendency win" into
"σ-robust win".

Setup:
  - HAT × {H=8, H=12, H=16} × seeds {3, 4, 5} = 9 new runs.
  - Hyperparameters identical to bench_50task_hat_seeds (paper defaults:
    s_max=400, s_min=1/400, sparsity_coef=0.75, emb_clip=6.0).
  - Same seeding scheme: torch.manual_seed(s + 4000 + H), curriculum
    seed s + 17*H. Just larger s values.
  - After running, loads outputs/bench_50task_hat_seeds_log.csv and
    produces the combined 6-seed report (original 3-seed CSV is left
    untouched; new CSV at bench_50task_hat_extra_seeds_log.csv).

Frustration disabled (per next_session_plan.md). Grown is the cached
3-seed distribution from bench_50task_seeds_run1 (the multi-seed
extension applies to HAT only — grown's 0.053 ± 0.021 is already the
multi-seed reference number).
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


EXTRA_SEEDS = [3, 4, 5]
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

CACHED_CSV = "bench_50task_hat_seeds_log.csv"


def _agg(xs: List[float]) -> Tuple[float, float]:
    if len(xs) < 2:
        return (xs[0] if xs else float("nan"), 0.0)
    return statistics.mean(xs), statistics.stdev(xs)


def _load_cached_rows(out_dir: str) -> List[dict]:
    """Pull the original 3-seed runs (seeds 0,1,2) from the cached CSV."""
    path = os.path.join(out_dir, CACHED_CSV)
    rows: List[dict] = []
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append({
                "arch": r["arch"],
                "H": int(r["H"]),
                "seed": int(r["seed"]),
                "n_params_final": int(r["n_params_final"]),
                "final_latent": int(r["final_latent"]),
                "avg_final_loss": float(r["avg_final_loss"]),
                "avg_forgetting": float(r["avg_forgetting"]),
                "wall_clock_seconds": float(r["wall_clock_seconds"]),
                "hat_cum_density_layer0": float(r["hat_cum_density_layer0"]),
                "hat_cum_density_layer1": float(r["hat_cum_density_layer1"]),
            })
    return rows


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
    print("bench_50task_hat_extra_seeds — HAT × 3 widths × 3 EXTRA seeds")
    print("=" * 78)
    print(f"Extra seeds: {EXTRA_SEEDS}  (cached: 0, 1, 2 → combined 6-seed report)")
    print(f"Widths:      {HAT_HIDDEN_SIZES}")
    print(f"HAT params:  s∈[{HAT_S_MIN:.4f},{HAT_S_MAX}] "
          f"sparsity_coef={HAT_SPARSITY_COEF} emb_clip={HAT_EMB_CLIP}")
    print(f"Target latent (from grown's distribution): {TARGET_LATENT}")
    print(f"50-task curriculum: 12 single + 38 compound on 12-dim state")
    print()

    new_rows: List[dict] = []
    new_results_by_H: Dict[int, List[dict]] = {H: [] for H in HAT_HIDDEN_SIZES}
    t_start = time.monotonic()

    for H in HAT_HIDDEN_SIZES:
        for s in EXTRA_SEEDS:
            torch.manual_seed(s + 4000 + H)
            net = make_network(STATE_DIM, H, TARGET_LATENT)
            result = run_hat_curriculum(
                net, label=f"hat_H{H}_seed{s}",
                train_cur=cur_factory(seed=s + 17 * H),
                eval_batches=eval_batches, pair_names=pair_names,
                s_min=HAT_S_MIN, s_max=HAT_S_MAX,
                sparsity_coef=HAT_SPARSITY_COEF, emb_clip=HAT_EMB_CLIP,
            )
            new_rows.append({
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
            new_results_by_H[H].append(result)

    elapsed_total = time.monotonic() - t_start

    # ----- Write CSV (new seeds only — keep cached CSV untouched) -----
    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs"
    )
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "bench_50task_hat_extra_seeds_log.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arch", "H", "seed", "n_params_final", "final_latent",
                    "avg_final_loss", "avg_forgetting",
                    "wall_clock_seconds",
                    "hat_cum_density_layer0", "hat_cum_density_layer1"])
        for r in new_rows:
            w.writerow([r["arch"], r["H"], r["seed"], r["n_params_final"],
                        r["final_latent"], f"{r['avg_final_loss']:.6f}",
                        f"{r['avg_forgetting']:.6f}",
                        f"{r['wall_clock_seconds']:.2f}",
                        f"{r['hat_cum_density_layer0']:.4f}",
                        f"{r['hat_cum_density_layer1']:.4f}"])

    # ----- Combined report (cached + new) -----
    cached_rows = _load_cached_rows(out_dir)
    combined_by_H: Dict[int, List[dict]] = {H: [] for H in HAT_HIDDEN_SIZES}
    for row in cached_rows:
        if row["H"] in combined_by_H:
            combined_by_H[row["H"]].append(row)
    for row in new_rows:
        combined_by_H[row["H"]].append(row)

    print()
    print("=" * 78)
    print(f"bench_50task_hat_extra_seeds — Final Report  "
          f"(extra-run wall-clock {elapsed_total/60:.1f} min)")
    print("=" * 78)
    print()

    # Section 1: extra seeds in isolation (sanity check vs cached)
    print(f"NEW seeds {EXTRA_SEEDS} only (sanity vs cached 3-seed):")
    print(f"  {'H':>4} {'params':>8} {'avg_final mean±std':>22} "
          f"{'per-seed avg_final':>40}")
    for H in HAT_HIDDEN_SIZES:
        finals = [r["avg_final_loss"] for r in new_rows if r["H"] == H]
        params = next(r["n_params_final"] for r in new_rows if r["H"] == H)
        f_mean, f_std = _agg(finals)
        finals_str = "[" + ", ".join(f"{x:.4f}" for x in finals) + "]"
        print(f"  {H:>4d} {params:>8d}    {f_mean:.4f} ± {f_std:.4f}     {finals_str}")
    print()

    # Section 2: combined 6-seed picture
    print(f"COMBINED 6-seed picture (seeds 0..5):")
    print(f"  {'H':>4} {'params':>8} {'avg_final mean±std':>22} "
          f"{'cum_density(L0,L1)':>22}")
    print(f"  {'':>4} {'':>8} {'':>22} {'':>22}    per-seed avg_final")
    for H in HAT_HIDDEN_SIZES:
        results = combined_by_H[H]
        finals = sorted([(r["seed"], r["avg_final_loss"]) for r in results])
        finals_vals = [v for _, v in finals]
        params = results[0]["n_params_final"]
        f_mean, f_std = _agg(finals_vals)
        d0 = sum(r["hat_cum_density_layer0"] for r in results) / len(results)
        d1 = sum(r["hat_cum_density_layer1"] for r in results) / len(results)
        finals_str = "[" + ", ".join(f"s{s}={v:.4f}" for s, v in finals) + "]"
        print(f"  {H:>4d} {params:>8d}    {f_mean:.4f} ± {f_std:.4f}     "
              f"({d0:.2f},{d1:.2f})")
        print(f"  {'':>4} {'':>8}    {finals_str}")
    print()

    g_mean, g_std = _agg(GROWN_AVG_FINALS)
    print(f"Grown distribution (cached, 3 seeds — bench_50task_seeds_run1):")
    print(f"  per-seed: {GROWN_AVG_FINALS} → mean ± std: {g_mean:.4f} ± {g_std:.4f}")
    print()

    # Headline: matched HAT
    matched_H = min(
        HAT_HIDDEN_SIZES,
        key=lambda H: abs(combined_by_H[H][0]["n_params_final"] -
                          (sum(GROWN_FINAL_PARAMS) / len(GROWN_FINAL_PARAMS))),
    )
    matched_finals = [r["avg_final_loss"] for r in combined_by_H[matched_H]]
    m_mean, m_std = _agg(matched_finals)
    matched_params = combined_by_H[matched_H][0]["n_params_final"]
    print(f"Headline (6-seed): grown vs matched HAT (H={matched_H}, "
          f"~{matched_params} params):")
    print(f"  grown:   {g_mean:.4f} ± {g_std:.4f}  (3 seeds)")
    print(f"  HAT:     {m_mean:.4f} ± {m_std:.4f}  (6 seeds)")
    if g_mean > 0:
        rel = (m_mean - g_mean) / g_mean
        multiplier = m_mean / g_mean
        print(f"  HAT's mean is {multiplier:.2f}× grown's mean "
              f"(grown {rel*100:+.0f}% lower)")
    if g_std + m_std > 0:
        gap_in_std = (m_mean - g_mean) / max(g_std + m_std, 1e-9)
        verdict = ("ROBUST" if gap_in_std > 2 else
                   "borderline" if gap_in_std > 1 else "within noise")
        print(f"  gap is {gap_in_std:+.1f}× combined std — {verdict}")
    print()

    # Best-of HAT comparison
    best_H = min(
        HAT_HIDDEN_SIZES,
        key=lambda H: statistics.mean(
            [r["avg_final_loss"] for r in combined_by_H[H]]),
    )
    best_finals = [r["avg_final_loss"] for r in combined_by_H[best_H]]
    bp_mean, bp_std = _agg(best_finals)
    bp_params = combined_by_H[best_H][0]["n_params_final"]
    avg_grown_params = sum(GROWN_FINAL_PARAMS) / len(GROWN_FINAL_PARAMS)
    print(f"Best HAT width (6-seed): H={best_H} ({bp_params} params, "
          f"{bp_params / avg_grown_params:.2f}× grown's avg params):")
    print(f"  grown:   {g_mean:.4f} ± {g_std:.4f}")
    print(f"  HAT:     {bp_mean:.4f} ± {bp_std:.4f}")
    if g_mean > 0:
        print(f"  HAT's mean is {bp_mean/g_mean:.2f}× grown's mean")
    if g_std + bp_std > 0:
        gap_in_std = (bp_mean - g_mean) / max(g_std + bp_std, 1e-9)
        verdict = ("ROBUST" if gap_in_std > 2 else
                   "borderline" if gap_in_std > 1 else "within noise")
        print(f"  gap is {gap_in_std:+.1f}× combined std — {verdict}")

    print(f"\n  log: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
