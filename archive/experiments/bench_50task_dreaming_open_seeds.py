"""Phase 4.5 dreaming-phase bench (open variant) — output-layer compression
ENABLED + cosine probe instrumentation.

The first dreaming bench (commit 2ad7fa4) fired zero merges across all 3
seeds because:
  - Hidden layers (12, 12) are Kaiming-random and only shrink → near-
    orthogonal pairwise cosines, far below 0.95.
  - The output (latent) layer IS the only one that grows (grow_layer
    targets len(layers)-1) → the only place redundancy can emerge.
  - But compress(skip_output_layer=True) was the harness default, so
    compression had no path to fire. The architectural bet (growth +
    dreaming beats freeze-based CL via topological compression) was
    not exercised.

This bench fixes that:
  - skip_output_layer=False — compression considers the output layer
    where redundancy actually emerges.
  - cos_threshold lowered from 0.95 → 0.85 — function preservation
    degrades modestly (1-0.85 = 0.15 angular error vs 0.05 at 0.95)
    but replay afterwards cleans up most of it. The probe trajectory
    will tell us if 0.85 is reasonable for this curriculum.
  - Probe instrumentation: per-task per-layer max off-diagonal
    W_anchor cosine, captured by dreaming_block.pre_compress_max_cosines
    and forwarded through the harness. Reported as a per-seed
    trajectory + a CSV.

A-priori interpretation rule:
  PASS:    σ-gap vs HAT moves from 1.0σ → ≥ 2σ AND merges > 0 visible
           in the dreams log.
  PARTIAL: σ-gap improves to ~1.5σ + merges fire — mechanism works,
           threshold tuning would tighten further.
  FAIL:    σ-gap unchanged or no merges — defaults still wrong; the
           probe trajectory tells us what threshold WOULD have fired.
  WORST:   grown+open-dream worse than grown alone → enabling output-
           layer compression hurts (latent collapse breaks the
           representation rather than compressing redundancy).

The probe trajectory is the diagnostic gap the first bench couldn't
fill — even at FAIL, we'll know whether the issue is "threshold still
too tight" (max cosines hover at 0.7) vs "no redundancy emerges"
(max cosines flat near 0).

Wall-clock estimate: ~25 min for 3 seeds (similar to first dreaming
bench).
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
    DREAM_U_THRESHOLD,
    DREAM_EWC_STRENGTH,
    DREAM_BATCH,
    build_50task_pairs,
    make_fixed_eval_batches,
    make_network,
    run_ewc_curriculum,
)


SEEDS = [0, 1, 2]

# Lowered from 0.95 → 0.85: outputs of grow_layer are PCA-of-residuals
# directions (task-specific), so post-EWC W_anchors of two latent dims
# may share structure when tasks share dimensional content (compound
# tasks reusing single-task dims). 0.85 catches "mostly the same
# direction" without going so loose that we merge unrelated dims.
COS_THRESHOLD_OPEN = 0.85

# Cached comparison numbers (do NOT re-run grown-no-dreaming or HAT).
GROWN_BASELINE_AVG_FINALS = [0.0297, 0.0689, 0.0603]
GROWN_BASELINE_FINAL_PARAMS = [247, 264, 252]

# Cached first-dreaming-bench (closed-output) — for direct comparison
# of "what changes when we enable output-layer compression".
GROWN_DREAM_CLOSED_AVG_FINALS = [0.0272, 0.0248, 0.0799]
GROWN_DREAM_CLOSED_FINAL_PARAMS = [254, 323, 245]

# HAT 6-seed combined at H=8.
HAT_H8_AVG_FINALS = [
    0.0525, 0.1468, 0.1751,
    0.1202, 0.0896, 0.1156,
]
HAT_H8_PARAMS = 203

DREAMING_CONFIG_OPEN = {
    "replay_fraction": DREAM_REPLAY_FRACTION,
    "replay_steps_per_pair": DREAM_REPLAY_STEPS,
    "replay_batch": DREAM_BATCH,
    "ewc_strength": DREAM_EWC_STRENGTH,
    "cos_threshold": COS_THRESHOLD_OPEN,
    "u_threshold": DREAM_U_THRESHOLD,
    "skip_output_layer": False,
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
    print("bench_50task_dreaming_open_seeds — output-layer compression ENABLED")
    print("=" * 78)
    print(f"Seeds: {SEEDS}")
    print(f"Dreaming config: skip_output_layer=False  "
          f"cos_threshold={COS_THRESHOLD_OPEN}  "
          f"replay={DREAMING_CONFIG_OPEN['replay_fraction']:.0%}×"
          f"{DREAMING_CONFIG_OPEN['replay_steps_per_pair']}  "
          f"u_threshold={DREAMING_CONFIG_OPEN['u_threshold']}  "
          f"ewc_strength={DREAMING_CONFIG_OPEN['ewc_strength']}")
    print(f"50-task curriculum: 12 single + 38 compound on 12-dim state")
    print()

    rows: List[dict] = []
    results: List[dict] = []
    t_start = time.monotonic()

    for s in SEEDS:
        torch.manual_seed(s + 11000)
        net = make_network(STATE_DIM, HIDDEN, LATENT_INIT_GROWN)
        result = run_ewc_curriculum(
            net, label=f"grown_open_seed{s}",
            do_growth=True, do_pruning=True,
            train_cur=cur_factory(seed=s + 41),
            eval_batches=eval_batches, pair_names=pair_names,
            dreaming_config=DREAMING_CONFIG_OPEN,
            dreaming_rng=random.Random(s + 53),
        )
        n_merges = sum(d["n_merges"] for d in result.get("dreams", []))
        n_purges = sum(d["n_purges"] for d in result.get("dreams", []))
        rows.append({
            "arch": "grown_open",
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
    print(f"bench_50task_dreaming_open_seeds — Final Report  "
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
    print(f"Grown WITH open-output dreaming ({len(SEEDS)} seeds):")
    print(f"  per-seed avg_final:    {[f'{x:.4f}' for x in finals]}")
    print(f"  mean ± std:            {f_mean:.4f} ± {f_std:.4f}")
    print(f"  final params:          {p_mean:.0f} ± {p_std:.0f}  "
          f"per-seed: {final_params}")
    print(f"  final latent per seed: {final_latents}")
    print(f"  total merges per seed: {n_merges}  "
          f"(closed-bench had [0, 0, 0])")
    print(f"  total purges per seed: {n_purges}")
    print(f"  wall-clock per seed:   {[f'{x:.0f}s' for x in wall_clocks]}")
    print()

    # ----- Cosine trajectory (the key new diagnostic) -----
    print("max W_anchor cosine probe per task per layer (pre-compress):")
    for s_idx, s in enumerate(SEEDS):
        dreams = results[s_idx].get("dreams", [])
        if not dreams:
            continue
        # Aggregate by layer.
        layers_seen = sorted({L for d in dreams
                              for L, _ in d.get("pre_compress_max_cosines", [])})
        print(f"  seed{s}:")
        for L in layers_seen:
            cos_per_task = [
                next((c for ll, c in d.get("pre_compress_max_cosines", [])
                      if ll == L), float("nan"))
                for d in dreams
            ]
            valid = [c for c in cos_per_task if c == c and c != float("-inf")]
            if not valid:
                print(f"    layer {L}: no valid samples")
                continue
            sampled_idxs = [0, len(cos_per_task) // 4,
                            len(cos_per_task) // 2,
                            3 * len(cos_per_task) // 4,
                            len(cos_per_task) - 1]
            sampled_idxs = sorted(set(i for i in sampled_idxs
                                      if 0 <= i < len(cos_per_task)))
            sampled = [(i + 1, cos_per_task[i]) for i in sampled_idxs]
            samp_str = "  ".join(
                f"t{t}:{c:.2f}" if c == c and c != float("-inf")
                else f"t{t}:--"
                for t, c in sampled
            )
            print(f"    layer {L}: max_cos over curriculum  "
                  f"min={min(valid):.3f}  max={max(valid):.3f}  "
                  f"mean={sum(valid)/len(valid):.3f}")
            print(f"      sample: {samp_str}")
    print()

    # n_params trajectory.
    print("n_params trajectory (per-task, post-dreaming) per seed:")
    for s_idx, s in enumerate(SEEDS):
        trajectory = results[s_idx].get("n_params_per_task", [])
        if not trajectory:
            continue
        sampled = [(t + 1, trajectory[t])
                   for t in range(0, len(trajectory), 5)]
        if sampled[-1][0] != len(trajectory):
            sampled.append((len(trajectory), trajectory[-1]))
        sampled_str = "  ".join(f"t{t}:{p}" for t, p in sampled)
        print(f"  seed{s}:  {sampled_str}")
    print()

    # ----- Comparison vs cached baselines -----
    g_base_mean, g_base_std = _agg(GROWN_BASELINE_AVG_FINALS)
    g_base_p_mean, _ = _agg([float(p) for p in GROWN_BASELINE_FINAL_PARAMS])
    g_closed_mean, g_closed_std = _agg(GROWN_DREAM_CLOSED_AVG_FINALS)
    hat_mean, hat_std = _agg(HAT_H8_AVG_FINALS)

    print(f"Cached baselines:")
    print(f"  grown no-dream    : {g_base_mean:.4f} ± {g_base_std:.4f}  "
          f"(~{g_base_p_mean:.0f}p)")
    print(f"  grown closed-dream: {g_closed_mean:.4f} ± {g_closed_std:.4f}  "
          f"(prev bench, skip_output=True)")
    print(f"  HAT H=8 6-seed    : {hat_mean:.4f} ± {hat_std:.4f}  "
          f"({HAT_H8_PARAMS}p)")
    print()

    # Headline 1: open-dream vs no-dream.
    print(f"Headline 1: open-dream vs no-dream:")
    print(f"  no-dream     : {g_base_mean:.4f} ± {g_base_std:.4f}  "
          f"(~{g_base_p_mean:.0f}p)")
    print(f"  open-dream   : {f_mean:.4f} ± {f_std:.4f}  "
          f"(~{p_mean:.0f}p)")
    if g_base_mean > 0:
        delta_pct = (f_mean - g_base_mean) / g_base_mean * 100
        print(f"  Δ            : {delta_pct:+.0f}% on mean  "
              f"(negative = open-dream improved)")
    if g_base_std + f_std > 0:
        gap = (g_base_mean - f_mean) / max(g_base_std + f_std, 1e-9)
        verdict = ("ROBUST IMPROVEMENT" if gap > 2
                   else "borderline improvement" if gap > 1
                   else "within noise" if gap > -1
                   else "borderline regression" if gap > -2
                   else "ROBUST REGRESSION")
        print(f"  σ-gap        : {gap:+.1f}σ  → {verdict}")
    print()

    # Headline 2: open-dream vs closed-dream (intra-dreaming compare).
    print(f"Headline 2: open-dream vs closed-dream "
          f"(does enabling output-layer compression help?):")
    print(f"  closed-dream : {g_closed_mean:.4f} ± {g_closed_std:.4f}")
    print(f"  open-dream   : {f_mean:.4f} ± {f_std:.4f}")
    if g_closed_mean > 0:
        delta_pct = (f_mean - g_closed_mean) / g_closed_mean * 100
        print(f"  Δ            : {delta_pct:+.0f}% on mean")
    if g_closed_std + f_std > 0:
        gap = (g_closed_mean - f_mean) / max(g_closed_std + f_std, 1e-9)
        print(f"  σ-gap        : {gap:+.1f}σ")
    print()

    # Headline 3: open-dream vs HAT (the architectural target).
    print(f"Headline 3: open-dream vs HAT H=8 (target gap → ≥2σ):")
    print(f"  open-dream   : {f_mean:.4f} ± {f_std:.4f}  (~{p_mean:.0f}p)")
    print(f"  HAT H=8      : {hat_mean:.4f} ± {hat_std:.4f}  "
          f"({HAT_H8_PARAMS}p)")
    if f_mean > 0:
        mult = hat_mean / f_mean
        print(f"  HAT mean is {mult:.2f}× open-dream's mean")
    if f_std + hat_std > 0:
        gap = (hat_mean - f_mean) / max(f_std + hat_std, 1e-9)
        verdict = ("ROBUST" if gap > 2
                   else "borderline" if gap > 1 else "within noise")
        print(f"  σ-gap        : {gap:+.1f}σ  → {verdict}")
        if gap > 2:
            print(f"  PASS: open-dream closes the σ-claim against HAT.")
        elif gap > 1:
            print(f"  PARTIAL: improved over closed-dream's 1.0σ but not "
                  f"σ-robust at 2σ.")
        else:
            print(f"  FAIL or REGRESSION at 2σ.")
    print()

    # ----- CSVs -----
    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs"
    )
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "bench_50task_dreaming_open_seeds_log.csv")
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

    # Cosine probe trajectory CSV: one row per (seed, task, layer) so
    # the cosine landscape is plottable per-layer over the curriculum.
    probe_path = os.path.join(
        out_dir, "bench_50task_dreaming_open_probe.csv"
    )
    with open(probe_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["seed", "task_idx", "pair_name", "layer_idx",
                    "max_off_diag_cosine", "n_merges_at_task",
                    "n_purges_at_task", "n_params_after_task"])
        for s_idx, s in enumerate(SEEDS):
            traj = results[s_idx].get("n_params_per_task", [])
            for d in results[s_idx].get("dreams", []):
                t_idx = d["task_idx"]
                p_after = traj[t_idx] if t_idx < len(traj) else -1
                for L, c in d.get("pre_compress_max_cosines", []):
                    w.writerow([
                        s, t_idx, d["pair_name"], L,
                        f"{c:.6f}" if c != float("-inf") else "-inf",
                        d.get("n_merges", 0), d.get("n_purges", 0),
                        p_after,
                    ])

    print(f"  log:   {csv_path}")
    print(f"  probe: {probe_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
