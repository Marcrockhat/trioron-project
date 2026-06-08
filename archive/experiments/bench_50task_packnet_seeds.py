"""Multi-seed PackNet sweep on the bench_50task curriculum.

Companion to bench_50task_seeds.py. The bench_50task_seeds run revised
grown vs fixed-EWC numbers downward (47% → 44% at matched, 27% vs
best, with substantial variance bands). The PackNet line of bench_50task
showed grown beating PackNet ~10×, but that was single-seed too — and
while the per-task fragmentation argument is structural (mechanism, not
noise), the precise multiplier should be confirmed before quoting.

Scope:
  - PackNet × {H=8, H=12, H=16} × 3 seeds = 9 runs.
  - Compare against bench_50task_seeds' already-collected grown
    distribution (mean 0.053 ± 0.021).

Reuses helpers from bench_50task. Same curriculum, same hyperparameters.
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
    run_packnet_curriculum,
)


SEEDS = [0, 1, 2]
PACKNET_HIDDEN_SIZES = [8, 12, 16]

# Grown's already-collected distribution from bench_50task_seeds_run1.log
# (commit e5915db). Used for cross-comparison without re-running grown.
GROWN_AVG_FINALS = [0.0297, 0.0689, 0.0603]
GROWN_FINAL_LATENTS = [4, 3, 3]
GROWN_FINAL_PARAMS = [247, 264, 252]  # approx, from log
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
    print("bench_50task_packnet_seeds — PackNet × 3 widths × 3 seeds")
    print("=" * 78)
    print(f"Seeds:  {SEEDS}")
    print(f"Widths: {PACKNET_HIDDEN_SIZES}")
    print(f"Target latent (from grown's distribution): {TARGET_LATENT}")
    print(f"50-task curriculum: 12 single + 38 compound on 12-dim state")
    print()

    rows: List[dict] = []
    results_by_H: Dict[int, List[dict]] = {H: [] for H in PACKNET_HIDDEN_SIZES}
    t_start = time.monotonic()

    for H in PACKNET_HIDDEN_SIZES:
        for s in SEEDS:
            torch.manual_seed(s + 2000 + H)
            net = make_network(STATE_DIM, H, TARGET_LATENT)
            result = run_packnet_curriculum(
                net, label=f"packnet_H{H}_seed{s}",
                train_cur=cur_factory(seed=s + 13 * H),
                eval_batches=eval_batches, pair_names=pair_names,
            )
            rows.append({
                "arch": "packnet",
                "H": H,
                "seed": s,
                "n_params_final": result["final_n_params"],
                "final_latent": result["final_latent"],
                "avg_final_loss": result["avg_final_loss"],
                "avg_forgetting": result["avg_forgetting"],
                "wall_clock_seconds": result["wall_clock_seconds"],
                "packnet_min_capacity": min(result["packnet_capacity"]),
                "packnet_max_capacity": max(result["packnet_capacity"]),
            })
            results_by_H[H].append(result)

    elapsed_total = time.monotonic() - t_start

    # ----- Report -----
    print()
    print("=" * 78)
    print(f"bench_50task_packnet_seeds — Final Report  "
          f"(total wall-clock {elapsed_total/60:.1f} min)")
    print("=" * 78)
    print()
    print(f"PackNet ({len(SEEDS)} seeds × {len(PACKNET_HIDDEN_SIZES)} widths):")
    print(f"  {'H':>4} {'params':>8} {'avg_final mean±std':>22} "
          f"{'per-task cap':>16} {'per-seed avg_final':>40}")
    for H in PACKNET_HIDDEN_SIZES:
        results = results_by_H[H]
        finals = [r["avg_final_loss"] for r in results]
        params = results[0]["final_n_params"]
        f_mean, f_std = _agg(finals)
        # per-task capacity ranges across seeds
        all_min = min(min(r["packnet_capacity"]) for r in results)
        all_max = max(max(r["packnet_capacity"]) for r in results)
        finals_str = "[" + ", ".join(f"{x:.4f}" for x in finals) + "]"
        print(f"  {H:>4d} {params:>8d}    {f_mean:.4f} ± {f_std:.4f}"
              f"   {all_min:>3d}-{all_max:<3d}    {finals_str}")
    print()

    # Headline: grown vs PackNet at matched params (approx 247-265, closest is H=8 with 203)
    g_mean, g_std = _agg(GROWN_AVG_FINALS)
    print(f"Grown distribution (from bench_50task_seeds_run1):")
    print(f"  per-seed: {GROWN_AVG_FINALS} → mean ± std: {g_mean:.4f} ± {g_std:.4f}")
    print(f"  final latents: {GROWN_FINAL_LATENTS}, final params: {GROWN_FINAL_PARAMS}")
    print()

    matched_H = min(
        PACKNET_HIDDEN_SIZES,
        key=lambda H: abs(results_by_H[H][0]["final_n_params"] -
                          (sum(GROWN_FINAL_PARAMS) / len(GROWN_FINAL_PARAMS))),
    )
    matched_finals = [r["avg_final_loss"] for r in results_by_H[matched_H]]
    m_mean, m_std = _agg(matched_finals)
    print(f"Headline: grown vs matched PackNet (H={matched_H}, "
          f"~{results_by_H[matched_H][0]['final_n_params']} params):")
    print(f"  grown:   {g_mean:.4f} ± {g_std:.4f}")
    print(f"  PackNet: {m_mean:.4f} ± {m_std:.4f}")
    rel = (m_mean - g_mean) / g_mean if g_mean > 0 else 0
    multiplier = m_mean / g_mean if g_mean > 0 else 0
    print(f"  PackNet's mean is {multiplier:.1f}× grown's mean "
          f"(grown {rel*100:.0f}% lower)")
    if g_std + m_std > 0:
        gap_in_std = (m_mean - g_mean) / max(g_std + m_std, 1e-9)
        print(f"  gap is {gap_in_std:.1f}× the combined std — "
              f"{'robust' if gap_in_std > 2 else 'borderline' if gap_in_std > 1 else 'within noise'}")
    print()

    best_pn_H = min(
        PACKNET_HIDDEN_SIZES,
        key=lambda H: statistics.mean([r["avg_final_loss"] for r in results_by_H[H]]),
    )
    best_finals = [r["avg_final_loss"] for r in results_by_H[best_pn_H]]
    bp_mean, bp_std = _agg(best_finals)
    bp_params = results_by_H[best_pn_H][0]["final_n_params"]
    print(f"Best PackNet width: H={best_pn_H} ({bp_params} params, "
          f"{bp_params / (sum(GROWN_FINAL_PARAMS) / len(GROWN_FINAL_PARAMS)):.1f}× "
          f"grown's avg params):")
    print(f"  grown:   {g_mean:.4f} ± {g_std:.4f}")
    print(f"  PackNet: {bp_mean:.4f} ± {bp_std:.4f}")
    if g_mean > 0:
        print(f"  PackNet's mean is {bp_mean/g_mean:.1f}× grown's mean")

    # CSV
    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs"
    )
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "bench_50task_packnet_seeds_log.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arch", "H", "seed", "n_params_final", "final_latent",
                    "avg_final_loss", "avg_forgetting",
                    "wall_clock_seconds",
                    "packnet_min_capacity", "packnet_max_capacity"])
        for r in rows:
            w.writerow([r["arch"], r["H"], r["seed"], r["n_params_final"],
                        r["final_latent"], f"{r['avg_final_loss']:.6f}",
                        f"{r['avg_forgetting']:.6f}",
                        f"{r['wall_clock_seconds']:.2f}",
                        r["packnet_min_capacity"],
                        r["packnet_max_capacity"]])
    print(f"\n  log: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
