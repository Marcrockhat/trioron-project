"""Phase 4.5 dreaming-phase bench — grown WITH dreaming × 3 seeds.

Per next_session_plan.md and phase_4_5_dreaming_phase.md, this is the
architectural answer to the HAT 6-seed result (commit 1cf1b3e): grown
beats HAT 2.20× on means at matched H=8 but only 1.0σ. Six seeds did
not push past 2σ; HAT's per-seed variance is structurally large.

The bet for dreaming: topological compression should tighten grown's
effective parameter count, widening the matched-params margin by
making grown's "matched" line meet HAT at a smaller param budget than
the no-dreaming baseline. The σ-claim should follow.

Setup:
  - Grown WITH dreaming × 3 seeds. dreaming_config = harness defaults
    (replay_fraction=0.25, replay_steps_per_pair=200, cos_threshold=0.95,
    u_threshold=1e-3). EWC strength inside replay = 1000 = EWC_INTERTASK.
    Frustration off (per spec).
  - 50-task / 12-dim curriculum identical to bench_50task_seeds and
    bench_50task_hat_extra_seeds — direct comparability.
  - Compared post-hoc against cached:
      * grown 3-seed without dreaming (bench_50task_seeds_run1.log,
        commit e5915db): mean 0.0530 ± 0.0206, params ~252.
      * HAT 6-seed at H=8 (bench_50task_hat_extra_seeds + cached
        bench_50task_hat_seeds): mean 0.1166 ± 0.0429, 203 params.

Hypothesis to test (a-priori interpretation rule):
  - PASS: dreaming widens grown vs HAT (matched) gap from 1.0σ to ≥ 2σ
    AND grown's n_params trajectory shows compression-driven drops
    (visible per-task in the CSV / log).
  - PARTIAL: σ-gap improves (e.g. 1.0σ → 1.5σ) and per-task n_params
    drops are visible — mechanism works but more seeds / threshold
    tuning needed.
  - FAIL: σ-gap unchanged or worse, or n_params trajectory shows no
    compression — mechanism does not deliver under defaults.
  - WORST: grown WITH dreaming performs worse than grown WITHOUT —
    the mechanism is anti-architectural like frustration was.

The n_params trajectory is the smoking-gun signal that compression is
*doing* something, distinct from any final-loss change.
"""
from __future__ import annotations
import csv
import os
import random
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
    HIDDEN,
    LATENT_INIT_GROWN,
    DREAM_REPLAY_FRACTION,
    DREAM_REPLAY_STEPS,
    DREAM_COS_THRESHOLD,
    DREAM_U_THRESHOLD,
    DREAM_EWC_STRENGTH,
    DREAM_BATCH,
    build_50task_pairs,
    make_fixed_eval_batches,
    make_network,
    run_ewc_curriculum,
)


SEEDS = [0, 1, 2]

# Cached comparison numbers (do NOT re-run grown-no-dreaming or HAT here;
# we compare post-hoc to make the bench cheap and exactly comparable).

# Grown without dreaming (3 seeds) — bench_50task_seeds_run1.log (e5915db).
GROWN_BASELINE_AVG_FINALS = [0.0297, 0.0689, 0.0603]
GROWN_BASELINE_FINAL_LATENTS = [4, 3, 3]
GROWN_BASELINE_FINAL_PARAMS = [247, 264, 252]

# HAT 6-seed combined (from bench_50task_hat_seeds + extra) at H=8.
HAT_H8_AVG_FINALS = [
    0.0525, 0.1468, 0.1751,  # cached seeds 0,1,2
    0.1202, 0.0896, 0.1156,  # extra seeds 3,4,5
]
HAT_H8_PARAMS = 203

DREAMING_CONFIG = {
    "replay_fraction": DREAM_REPLAY_FRACTION,
    "replay_steps_per_pair": DREAM_REPLAY_STEPS,
    "replay_batch": DREAM_BATCH,
    "ewc_strength": DREAM_EWC_STRENGTH,
    "cos_threshold": DREAM_COS_THRESHOLD,
    "u_threshold": DREAM_U_THRESHOLD,
    "skip_output_layer": True,
}


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
    print("bench_50task_dreaming_seeds — grown WITH dreaming × 3 seeds")
    print("=" * 78)
    print(f"Seeds: {SEEDS}")
    print(f"Dreaming config: replay={DREAMING_CONFIG['replay_fraction']:.0%} of past × "
          f"{DREAMING_CONFIG['replay_steps_per_pair']} steps; "
          f"cos_threshold={DREAMING_CONFIG['cos_threshold']}; "
          f"u_threshold={DREAMING_CONFIG['u_threshold']}; "
          f"ewc_strength={DREAMING_CONFIG['ewc_strength']}")
    print(f"50-task curriculum: 12 single + 38 compound on 12-dim state")
    print()

    rows: List[dict] = []
    results: List[dict] = []
    t_start = time.monotonic()

    for s in SEEDS:
        torch.manual_seed(s + 9000)
        net = make_network(STATE_DIM, HIDDEN, LATENT_INIT_GROWN)
        result = run_ewc_curriculum(
            net, label=f"grown_dream_seed{s}",
            do_growth=True, do_pruning=True,
            train_cur=cur_factory(seed=s + 23),
            eval_batches=eval_batches, pair_names=pair_names,
            dreaming_config=DREAMING_CONFIG,
            dreaming_rng=random.Random(s + 31),
        )
        n_merges = sum(d["n_merges"] for d in result.get("dreams", []))
        n_purges = sum(d["n_purges"] for d in result.get("dreams", []))
        rows.append({
            "arch": "grown_dream",
            "H": HIDDEN,
            "seed": s,
            "n_params_final": result["final_n_params"],
            "final_latent": result["final_latent"],
            "avg_final_loss": result["avg_final_loss"],
            "avg_forgetting": result["avg_forgetting"],
            "n_divisions_allowed": len(
                [f for f in result["fires"] if f.get("allowed")]),
            "n_prunes_in_training": len(result["prunes"]),
            "n_merges": n_merges,
            "n_purges": n_purges,
            "wall_clock_seconds": result["wall_clock_seconds"],
        })
        results.append(result)

    elapsed_total = time.monotonic() - t_start

    # ----- Report -----
    print()
    print("=" * 78)
    print(f"bench_50task_dreaming_seeds — Final Report  "
          f"(total wall-clock {elapsed_total/60:.1f} min)")
    print("=" * 78)
    print()

    finals = [r["avg_final_loss"] for r in rows]
    final_params = [r["n_params_final"] for r in rows]
    final_latents = [r["final_latent"] for r in rows]
    n_merges = [r["n_merges"] for r in rows]
    n_purges = [r["n_purges"] for r in rows]
    wall_clocks = [r["wall_clock_seconds"] for r in rows]

    f_mean, f_std = _agg(finals)
    p_mean, p_std = _agg([float(p) for p in final_params])
    print(f"Grown WITH dreaming ({len(SEEDS)} seeds):")
    print(f"  per-seed avg_final:    {[f'{x:.4f}' for x in finals]}")
    print(f"  mean ± std:            {f_mean:.4f} ± {f_std:.4f}")
    print(f"  final params:          {p_mean:.0f} ± {p_std:.0f}  per-seed: {final_params}")
    print(f"  final latent per seed: {final_latents}")
    print(f"  total merges per seed: {n_merges}")
    print(f"  total purges per seed: {n_purges}")
    print(f"  wall-clock per seed:   {[f'{x:.0f}s' for x in wall_clocks]}")
    print()

    # n_params trajectory — the compression mechanism made observable.
    print("n_params trajectory (per-task, post-dreaming) per seed:")
    for s_idx, s in enumerate(SEEDS):
        trajectory = results[s_idx].get("n_params_per_task", [])
        if not trajectory:
            continue
        # Compress to a single line — sample at every 5th task + last.
        sampled = [(t + 1, trajectory[t])
                   for t in range(0, len(trajectory), 5)]
        if sampled[-1][0] != len(trajectory):
            sampled.append((len(trajectory), trajectory[-1]))
        sampled_str = "  ".join(f"t{t}:{p}" for t, p in sampled)
        print(f"  seed{s}:  {sampled_str}")
    print()

    # ----- Comparison vs cached baselines -----

    # Grown without dreaming (cached 3-seed)
    g_base_mean, g_base_std = _agg(GROWN_BASELINE_AVG_FINALS)
    g_base_p_mean, _ = _agg([float(p) for p in GROWN_BASELINE_FINAL_PARAMS])

    # HAT H=8 6-seed combined
    hat_mean, hat_std = _agg(HAT_H8_AVG_FINALS)

    print(f"Cached baselines (post-hoc comparison):")
    print(f"  Grown (no dreaming, 3 seeds): "
          f"{g_base_mean:.4f} ± {g_base_std:.4f}  "
          f"(per-seed {GROWN_BASELINE_AVG_FINALS}, "
          f"~{g_base_p_mean:.0f} params)")
    print(f"  HAT H=8 (6 seeds, {HAT_H8_PARAMS} params): "
          f"{hat_mean:.4f} ± {hat_std:.4f}")
    print()

    # Headline 1: grown-WITH vs grown-WITHOUT dreaming
    print(f"Headline 1: dreaming vs no-dreaming (intra-grown):")
    print(f"  no-dream:  {g_base_mean:.4f} ± {g_base_std:.4f}  "
          f"(~{g_base_p_mean:.0f} params)")
    print(f"  +dream:    {f_mean:.4f} ± {f_std:.4f}  "
          f"(~{p_mean:.0f} params)")
    if g_base_mean > 0:
        delta_pct = (f_mean - g_base_mean) / g_base_mean * 100
        print(f"  Δ:         {delta_pct:+.0f}% on mean  "
              f"(negative = dreaming improved)")
    if g_base_std + f_std > 0:
        gap = (g_base_mean - f_mean) / max(g_base_std + f_std, 1e-9)
        verdict = ("ROBUST IMPROVEMENT" if gap > 2
                   else "borderline improvement" if gap > 1
                   else "within noise" if gap > -1
                   else "borderline regression" if gap > -2
                   else "ROBUST REGRESSION")
        print(f"  σ-gap:     {gap:+.1f}σ  → {verdict}")
    print()

    # Headline 2: grown+dreaming vs HAT (matched on H=8 ~203 params).
    print(f"Headline 2: grown+dreaming vs HAT H=8 (matched-ish, target gap → ≥2σ):")
    print(f"  grown+dream: {f_mean:.4f} ± {f_std:.4f}  "
          f"(~{p_mean:.0f} params)")
    print(f"  HAT H=8:     {hat_mean:.4f} ± {hat_std:.4f}  "
          f"({HAT_H8_PARAMS} params)")
    if f_mean > 0:
        mult = hat_mean / f_mean
        print(f"  HAT mean is {mult:.2f}× grown+dream's mean")
    if f_std + hat_std > 0:
        gap = (hat_mean - f_mean) / max(f_std + hat_std, 1e-9)
        verdict = ("ROBUST" if gap > 2
                   else "borderline" if gap > 1 else "within noise")
        print(f"  σ-gap:       {gap:+.1f}σ  → {verdict}")
        if gap > 2:
            print(f"  PASS: dreaming closed the σ-claim against HAT.")
        elif gap > 1:
            print(f"  PARTIAL: improved over no-dream's 1.0σ but not "
                  f"σ-robust at 2σ.")
        else:
            print(f"  FAIL or REGRESSION at 2σ.")

    # ----- CSV log -----
    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs"
    )
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "bench_50task_dreaming_seeds_log.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "arch", "H", "seed", "n_params_final", "final_latent",
            "avg_final_loss", "avg_forgetting",
            "n_divisions_allowed", "n_prunes_in_training",
            "n_merges", "n_purges", "wall_clock_seconds",
        ])
        for r in rows:
            w.writerow([
                r["arch"], r["H"], r["seed"], r["n_params_final"],
                r["final_latent"], f"{r['avg_final_loss']:.6f}",
                f"{r['avg_forgetting']:.6f}",
                r["n_divisions_allowed"], r["n_prunes_in_training"],
                r["n_merges"], r["n_purges"],
                f"{r['wall_clock_seconds']:.2f}",
            ])

    # Trajectory CSV — separate, one row per (seed, task) so per-task
    # n_params is plottable.
    traj_path = os.path.join(
        out_dir, "bench_50task_dreaming_seeds_trajectory.csv"
    )
    with open(traj_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["seed", "task_idx", "pair_name",
                    "n_params_after_task", "n_merges_at_task",
                    "n_purges_at_task"])
        for s_idx, s in enumerate(SEEDS):
            traj = results[s_idx].get("n_params_per_task", [])
            dreams_by_task = {d["task_idx"]: d for d in results[s_idx].get(
                "dreams", [])}
            for t_idx in range(len(traj)):
                d = dreams_by_task.get(t_idx, {})
                w.writerow([
                    s, t_idx, pair_names[t_idx], traj[t_idx],
                    d.get("n_merges", 0), d.get("n_purges", 0),
                ])

    print(f"\n  log:        {csv_path}")
    print(f"  trajectory: {traj_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
