"""Multi-seed sweep over the bench_50task fixed-EWC sweep.

bench_50task showed grown 47% better than the matched fixed-EWC at
50 tasks, but the fixed-EWC sweep was non-monotonic in H:

  H=8  (212 params): 0.044  ← matched, best
  H=12 (364 params): 0.082  ← worst
  H=16 (548 params): 0.062  ← mid

A bigger network sometimes performing WORSE under EWC suggests an
EWC-strength × width interaction. This wrapper runs each condition
with 3 different seeds and reports mean ± std so we can tell whether
the non-monotonicity is real or seed noise.

Scope:
  - grown × 3 seeds
  - fixed-EWC × {H=8, H=12, H=16} × 3 seeds  (9 runs)
  - PackNet skipped — its 50-task failure mode (per-task allocation
    too sparse) is mechanism, not noise. Re-running adds time without
    new evidence.

Reuses the same hyperparameters and helpers from bench_50task.py.
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
    N_PAIRS,
    HIDDEN,
    LATENT_INIT_GROWN,
    build_50task_pairs,
    make_fixed_eval_batches,
    make_network,
    run_ewc_curriculum,
)


SEEDS = [0, 1, 2]
FIXED_HIDDEN_SIZES = [8, 12, 16]


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
    print("bench_50task_seeds — multi-seed sweep over the bench_50task conditions")
    print("=" * 78)
    print(f"Seeds:  {SEEDS}")
    print(f"Fixed-EWC widths: {FIXED_HIDDEN_SIZES}")
    print(f"50-task curriculum: 12 single + 38 compound on 12-dim state")
    print()

    # All runs collected as flat list with metadata
    rows: List[dict] = []
    t_start = time.monotonic()

    # --- Grown × seeds ---
    grown_avg_finals: List[float] = []
    grown_forgettings: List[float] = []
    grown_n_params: List[int] = []
    grown_n_fires: List[int] = []
    grown_seconds: List[float] = []
    grown_final_latents: List[int] = []
    for s in SEEDS:
        torch.manual_seed(s)
        net = make_network(STATE_DIM, HIDDEN, LATENT_INIT_GROWN)
        result = run_ewc_curriculum(
            net, label=f"grown_seed{s}",
            do_growth=True, do_pruning=True,
            train_cur=cur_factory(seed=s),
            eval_batches=eval_batches, pair_names=pair_names,
        )
        rows.append({
            "arch": "grown",
            "H": HIDDEN,
            "seed": s,
            "n_params_final": result["final_n_params"],
            "final_latent": result["final_latent"],
            "avg_final_loss": result["avg_final_loss"],
            "avg_forgetting": result["avg_forgetting"],
            "n_divisions_allowed": len([f for f in result["fires"] if f.get("allowed")]),
            "n_prunes": len(result["prunes"]),
            "wall_clock_seconds": result["wall_clock_seconds"],
        })
        grown_avg_finals.append(result["avg_final_loss"])
        grown_forgettings.append(result["avg_forgetting"])
        grown_n_params.append(result["final_n_params"])
        grown_n_fires.append(rows[-1]["n_divisions_allowed"])
        grown_seconds.append(result["wall_clock_seconds"])
        grown_final_latents.append(result["final_latent"])

    # --- Fixed-EWC × widths × seeds ---
    # Pick a target_latent for fixed nets — average of grown's final latents.
    # Round to int. Most likely all 3 grown runs hit the same final latent.
    target_latent = int(round(sum(grown_final_latents) / len(grown_final_latents)))
    print(f"\n[seed-sweep] grown final latents across seeds: {grown_final_latents}")
    print(f"[seed-sweep] using target_latent={target_latent} for fixed-EWC sweep")

    fixed_results_by_H: Dict[int, List[dict]] = {}
    for H in FIXED_HIDDEN_SIZES:
        fixed_results_by_H[H] = []
        for s in SEEDS:
            torch.manual_seed(s + 1000)
            net = make_network(STATE_DIM, H, target_latent)
            result = run_ewc_curriculum(
                net, label=f"fixed_ewc_H{H}_seed{s}",
                do_growth=False, do_pruning=False,
                train_cur=cur_factory(seed=s + 7 * H),
                eval_batches=eval_batches, pair_names=pair_names,
            )
            row = {
                "arch": "fixed_ewc",
                "H": H,
                "seed": s,
                "n_params_final": result["final_n_params"],
                "final_latent": result["final_latent"],
                "avg_final_loss": result["avg_final_loss"],
                "avg_forgetting": result["avg_forgetting"],
                "n_divisions_allowed": 0,
                "n_prunes": 0,
                "wall_clock_seconds": result["wall_clock_seconds"],
            }
            rows.append(row)
            fixed_results_by_H[H].append(result)

    # --- Report ---
    elapsed_total = time.monotonic() - t_start
    print()
    print("=" * 78)
    print(f"bench_50task_seeds — Final Report  (total wall-clock {elapsed_total/60:.1f} min)")
    print("=" * 78)
    print()

    g_mean, g_std = _agg(grown_avg_finals)
    g_fmean, g_fstd = _agg(grown_forgettings)
    g_pmean, g_pstd = _agg([float(p) for p in grown_n_params])
    g_smean, g_sstd = _agg(grown_seconds)
    print(f"Grown ({len(SEEDS)} seeds):")
    print(f"  per-seed avg_final:  {[f'{x:.4f}' for x in grown_avg_finals]}")
    print(f"  mean ± std:          {g_mean:.4f} ± {g_std:.4f}")
    print(f"  forgetting:          {g_fmean:.4f} ± {g_fstd:.4f}")
    print(f"  final params:        {g_pmean:.0f} ± {g_pstd:.0f}")
    print(f"  divisions per seed:  {grown_n_fires}")
    print(f"  final latent per seed: {grown_final_latents}")
    print(f"  wall-clock per seed: {[f'{x:.0f}s' for x in grown_seconds]}")
    print()

    print(f"Fixed-EWC ({len(SEEDS)} seeds × {len(FIXED_HIDDEN_SIZES)} widths):")
    print(f"  {'H':>4} {'params':>8} {'avg_final mean±std':>22} "
          f"{'forget mean±std':>20} {'per-seed avg_final':>40}")
    for H in FIXED_HIDDEN_SIZES:
        results = fixed_results_by_H[H]
        finals = [r["avg_final_loss"] for r in results]
        forgets = [r["avg_forgetting"] for r in results]
        params = results[0]["final_n_params"]
        f_mean, f_std = _agg(finals)
        fg_mean, fg_std = _agg(forgets)
        finals_str = "[" + ", ".join(f"{x:.4f}" for x in finals) + "]"
        print(f"  {H:>4d} {params:>8d}    {f_mean:.4f} ± {f_std:.4f}"
              f"    {fg_mean:+.4f} ± {fg_std:.4f}    {finals_str}")
    print()

    # Decision: is the non-monotonicity from bench_50task real?
    h12 = [r["avg_final_loss"] for r in fixed_results_by_H[12]]
    h8 = [r["avg_final_loss"] for r in fixed_results_by_H[8]]
    h16 = [r["avg_final_loss"] for r in fixed_results_by_H[16]]
    h12_mean, h12_std = _agg(h12)
    h8_mean, _ = _agg(h8)
    h16_mean, _ = _agg(h16)
    print("Non-monotonicity check (was H=12 worse than both H=8 and H=16 in single-seed?):")
    print(f"  H=8  mean: {h8_mean:.4f}")
    print(f"  H=12 mean: {h12_mean:.4f} ± {h12_std:.4f}")
    print(f"  H=16 mean: {h16_mean:.4f}")
    if h12_mean > h8_mean and h12_mean > h16_mean:
        delta_low = h12_mean - h8_mean
        delta_high = h12_mean - h16_mean
        print(f"  H=12 IS still worst by {delta_low:.4f} vs H=8, "
              f"{delta_high:.4f} vs H=16")
        if h12_std < min(delta_low, delta_high) / 2:
            print(f"  Effect is larger than the std band — likely real, "
                  f"not seed noise.")
        else:
            print(f"  Effect is within ~2× the std — could still be noise.")
    else:
        print(f"  Order is now monotonic (or H=12 is no longer worst).")
        print(f"  The single-seed bench_50task non-monotonicity was probably "
              f"seed noise.")

    # Headline: grown vs matched fixed-EWC across seeds
    print()
    print("Headline (grown vs matched fixed-EWC, across seeds):")
    matched_H = min(
        FIXED_HIDDEN_SIZES,
        key=lambda H: abs(fixed_results_by_H[H][0]["final_n_params"] - g_pmean),
    )
    matched_finals = [r["avg_final_loss"] for r in fixed_results_by_H[matched_H]]
    m_mean, m_std = _agg(matched_finals)
    print(f"  matched H={matched_H} (~{fixed_results_by_H[matched_H][0]['final_n_params']} params)")
    print(f"  grown:        {g_mean:.4f} ± {g_std:.4f}")
    print(f"  fixed-EWC:    {m_mean:.4f} ± {m_std:.4f}")
    rel = (m_mean - g_mean) / g_mean if g_mean > 0 else 0
    print(f"  grown is {rel*100:.1f}% lower than matched fixed-EWC mean")
    if g_std + m_std > 0:
        gap_in_std = (m_mean - g_mean) / max(g_std + m_std, 1e-9)
        print(f"  gap is {gap_in_std:.1f}× the combined std — "
              f"{'robust' if gap_in_std > 2 else 'borderline'}")

    # CSV log
    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs"
    )
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "bench_50task_seeds_log.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arch", "H", "seed", "n_params_final", "final_latent",
                    "avg_final_loss", "avg_forgetting",
                    "n_divisions_allowed", "n_prunes", "wall_clock_seconds"])
        for r in rows:
            w.writerow([r["arch"], r["H"], r["seed"], r["n_params_final"],
                        r["final_latent"], f"{r['avg_final_loss']:.6f}",
                        f"{r['avg_forgetting']:.6f}",
                        r["n_divisions_allowed"], r["n_prunes"],
                        f"{r['wall_clock_seconds']:.2f}"])
    print(f"\n  log: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
