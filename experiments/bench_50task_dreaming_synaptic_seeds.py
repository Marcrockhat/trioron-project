"""Phase 4.5 dreaming-phase bench (synaptic downscale + activation signal).

Fourth dreaming bench. Mirrors bench_50task_dreaming_act_seeds (same
activation-correlation detector, same probe construction, same
curriculum) but swaps the compression mechanism from MERGE to
SYNAPTIC DOWNSCALE — the substrate-preserving variant agreed mid-bench
on 2026-05-01:

  Rocky: "Brain cells such as neurons are too precious to be destroyed."

The merge math (incoming-mean + outgoing-sum + delete victim) was
doubly destructive when the redundancy signal is activation-
correlation:
  1. Averaging the incoming Ws degrades the surviving peer when
     activation-correlated nodes have DIFFERENT W (the
     orthogonal-W-correlated-act case is real, covered by
     test_dreaming.py:478).
  2. Deleting the victim destroys substrate that future tasks could
     re-recruit.

Synaptic downscale (per b4b0156):
  - Layer L+1: peer's outgoing column += victim's; victim's outgoing
    column zeroed; fisher there zeroed (so EWC doesn't pin zero —
    re-recruitment is unconstrained).
  - Layer L: victim's row (W_in / bias / anchor / lam / fisher / u)
    untouched. Architecture / param count UNCHANGED.

Two design decisions for THIS bench, distinct from the merge bench:

  ac_threshold = 0.85 (down from 0.95).
    The act-bench probe (commit 9195404) showed natural ceiling at
    0.85-0.92 mean on layer 1, with ac_threshold=0.95 the merge bench
    fired only ONE event across 3 seeds × 50 tasks — almost no test
    of the mechanism. With downscale we can be aggressive (no
    destructive cost), so push the threshold below the natural ceiling
    and let the mechanism actually fire across the curriculum.

  skip_output_layer = True (unchanged).
    Output-layer downscale is anyway a no-op (no L+1 to redirect to);
    explicitly skipping documents intent and saves the probe forward.

A-priori interpretation rule:
  PASS:    σ-gap vs HAT moves from 1.1σ → ≥ 2σ AND downscale events
           > 0 visible across seeds. The substrate-preserving
           treatment + correct diagnostic closes the σ-claim.
  PARTIAL: σ-gap improves to ~1.5-1.9σ + events fire — mechanism
           works directionally but threshold or replay budget
           tuning would tighten further.
  FAIL (mechanism-doesn't-help): σ-gap unchanged or worse vs act-merge
           bench (1.1σ) AND events fire. Downscale doesn't add value
           over merge — the redundancy detection is identifying real
           pairs but the consolidation isn't paying off
           representationally. Would suggest the activation-correlation
           pairs are actually carrying load that gets lost when
           consolidated, regardless of mechanism.
  FAIL (events-don't-fire): if even ac_threshold=0.85 produces no
           events, the post-replay activation cosines are below
           0.85 (probe will tell us). Re-tune.

Param-count comparisons CHANGE SHAPE under downscale: arch is
unchanged from no-dream (~250-300p across seeds depending on growth
+ in-training pruning), so any improvement comes from consolidation
quality alone, not from param savings.

Wall-clock estimate: ~25 min for 3 seeds (downscale itself is cheaper
than merge — no Parameter replacement — but the activation probe
forward is the dominant cost and unchanged).
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

AC_THRESHOLD = 0.85           # Below the natural ceiling — the
                              # downscale equivalent of "actually let
                              # the mechanism fire".
PROBE_BATCH_SIZE = 128

# Cached comparison numbers (do NOT re-run grown-no-dreaming or HAT).
GROWN_BASELINE_AVG_FINALS = [0.0297, 0.0689, 0.0603]
GROWN_BASELINE_FINAL_PARAMS = [247, 264, 252]

# Closed-output W-cosine dreaming bench (skip_output=True, cos=0.95):
# zero merges across all 3 seeds.
GROWN_DREAM_CLOSED_AVG_FINALS = [0.0272, 0.0248, 0.0799]
GROWN_DREAM_CLOSED_FINAL_PARAMS = [254, 323, 245]

# Open-output W-cosine dreaming bench (skip_output=False, cos=0.85):
# 1 merge total.
GROWN_DREAM_OPEN_AVG_FINALS = [0.0954, 0.0706, 0.0222]
GROWN_DREAM_OPEN_FINAL_PARAMS = [259, 296, 299]

# Activation-signal MERGE bench (skip_output=True, ac=0.95): 1 merge
# total — threshold above natural ceiling.
GROWN_DREAM_ACT_AVG_FINALS = [0.0332, 0.0692, 0.0221]
GROWN_DREAM_ACT_FINAL_PARAMS = [269, 296, 299]

# HAT 6-seed combined at H=8.
HAT_H8_AVG_FINALS = [
    0.0525, 0.1468, 0.1751,
    0.1202, 0.0896, 0.1156,
]
HAT_H8_PARAMS = 203

DREAMING_CONFIG_SYNAPTIC = {
    "replay_fraction": DREAM_REPLAY_FRACTION,
    "replay_steps_per_pair": DREAM_REPLAY_STEPS,
    "replay_batch": DREAM_BATCH,
    "ewc_strength": DREAM_EWC_STRENGTH,
    "cos_threshold": 0.95,           # ignored under "activation" signal,
                                     # kept so any inadvertent fall-back
                                     # doesn't silently change behavior.
    "u_threshold": DREAM_U_THRESHOLD,
    "skip_output_layer": True,
    "redundancy_signal": "activation",
    "ac_threshold": AC_THRESHOLD,
    "probe_batch_size": PROBE_BATCH_SIZE,
    "compression_action": "downscale",  # ← THE switch this bench tests.
}


def _agg(xs: List[float]) -> Tuple[float, float]:
    if len(xs) < 2:
        return (xs[0] if xs else float("nan"), 0.0)
    return statistics.mean(xs), statistics.stdev(xs)


def _trajectory_summary(per_task: List[float]) -> str:
    valid = [c for c in per_task if c == c and c != float("-inf")]
    if not valid:
        return "no valid samples"
    return (f"min={min(valid):.3f}  max={max(valid):.3f}  "
            f"mean={sum(valid)/len(valid):.3f}")


def _trajectory_sample(per_task: List[float], n: int = 5) -> str:
    if not per_task:
        return ""
    idxs = sorted(set([
        0,
        len(per_task) // 4,
        len(per_task) // 2,
        3 * len(per_task) // 4,
        len(per_task) - 1,
    ]))[:n]
    return "  ".join(
        f"t{i + 1}:{per_task[i]:.2f}"
        if (per_task[i] == per_task[i] and per_task[i] != float("-inf"))
        else f"t{i + 1}:--"
        for i in idxs
    )


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
    print("bench_50task_dreaming_synaptic_seeds — SYNAPTIC DOWNSCALE")
    print("                                       (substrate-preserving)")
    print("=" * 78)
    print(f"Seeds: {SEEDS}")
    print(f"Dreaming config: redundancy_signal=activation  "
          f"compression_action=downscale")
    print(f"                 ac_threshold={AC_THRESHOLD}  "
          f"probe_batch_size={PROBE_BATCH_SIZE}  "
          f"skip_output_layer=True")
    print(f"                 replay={DREAMING_CONFIG_SYNAPTIC['replay_fraction']:.0%}×"
          f"{DREAMING_CONFIG_SYNAPTIC['replay_steps_per_pair']}  "
          f"u_threshold={DREAMING_CONFIG_SYNAPTIC['u_threshold']}  "
          f"ewc_strength={DREAMING_CONFIG_SYNAPTIC['ewc_strength']}")
    print(f"50-task curriculum: 12 single + 38 compound on 12-dim state")
    print()

    rows: List[dict] = []
    results: List[dict] = []
    t_start = time.monotonic()

    for s in SEEDS:
        torch.manual_seed(s + 11000)
        net = make_network(STATE_DIM, HIDDEN, LATENT_INIT_GROWN)
        result = run_ewc_curriculum(
            net, label=f"grown_synaptic_seed{s}",
            do_growth=True, do_pruning=True,
            train_cur=cur_factory(seed=s + 41),
            eval_batches=eval_batches, pair_names=pair_names,
            dreaming_config=DREAMING_CONFIG_SYNAPTIC,
            dreaming_rng=random.Random(s + 53),
        )
        # n_merges in the dreams summary covers both merge AND downscale
        # events (same MergeEvent dataclass, distinguished by .action).
        n_events = sum(d["n_merges"] for d in result.get("dreams", []))
        n_purges = sum(d["n_purges"] for d in result.get("dreams", []))
        rows.append({
            "arch": "grown_synaptic",
            "H": HIDDEN,
            "seed": s,
            "n_params_final": result["final_n_params"],
            "final_latent": result["final_latent"],
            "avg_final_loss": result["avg_final_loss"],
            "avg_forgetting": result["avg_forgetting"],
            "n_divisions_allowed": len(
                [f for f in result["fires"] if f.get("allowed")]),
            "n_prunes_in_training": len(result["prunes"]),
            "n_downscales": n_events,
            "n_purges": n_purges,
            "wall_clock_seconds": result["wall_clock_seconds"],
        })
        results.append(result)

    elapsed_total = time.monotonic() - t_start

    # ----- Report -----
    print()
    print("=" * 78)
    print(f"bench_50task_dreaming_synaptic_seeds — Final Report  "
          f"(total wall-clock {elapsed_total/60:.1f} min)")
    print("=" * 78)
    print()

    finals = [r["avg_final_loss"] for r in rows]
    final_params = [r["n_params_final"] for r in rows]
    final_latents = [r["final_latent"] for r in rows]
    n_downscales = [r["n_downscales"] for r in rows]
    n_purges = [r["n_purges"] for r in rows]
    wall_clocks = [r["wall_clock_seconds"] for r in rows]

    f_mean, f_std = _agg(finals)
    p_mean, p_std = _agg([float(p) for p in final_params])
    print(f"Grown WITH synaptic-downscale dreaming ({len(SEEDS)} seeds):")
    print(f"  per-seed avg_final:    {[f'{x:.4f}' for x in finals]}")
    print(f"  mean ± std:            {f_mean:.4f} ± {f_std:.4f}")
    print(f"  final params:          {p_mean:.0f} ± {p_std:.0f}  "
          f"per-seed: {final_params}")
    print(f"  final latent per seed: {final_latents}")
    print(f"  total downscales/seed: {n_downscales}  "
          f"(act-merge bench: [0,1,0])")
    print(f"  total purges per seed: {n_purges}")
    print(f"  wall-clock per seed:   {[f'{x:.0f}s' for x in wall_clocks]}")
    print()

    # ----- Probe trajectory: BOTH W cosine and activation cosine -----
    print("Probe trajectory (pre-compress, per task per layer):")
    for s_idx, s in enumerate(SEEDS):
        dreams = results[s_idx].get("dreams", [])
        if not dreams:
            continue
        layers_seen = sorted({
            L for d in dreams
            for src in (d.get("pre_compress_max_cosines", []),
                        d.get("pre_compress_max_activation_cosines", []))
            for L, _ in src
        })
        print(f"  seed{s}:")
        for L in layers_seen:
            w_traj = [
                next((c for ll, c in d.get("pre_compress_max_cosines", [])
                      if ll == L), float("nan"))
                for d in dreams
            ]
            a_traj = [
                next((c for ll, c in
                      d.get("pre_compress_max_activation_cosines", [])
                      if ll == L), float("nan"))
                for d in dreams
            ]
            print(f"    layer {L}:")
            print(f"      W-cos       : {_trajectory_summary(w_traj)}")
            print(f"        sample    : {_trajectory_sample(w_traj)}")
            print(f"      act-cos     : {_trajectory_summary(a_traj)}")
            print(f"        sample    : {_trajectory_sample(a_traj)}")
    print()

    # n_params trajectory — should be flat under downscale (only growth
    # and in-training pruning move it; downscale doesn't).
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
    g_open_mean, g_open_std = _agg(GROWN_DREAM_OPEN_AVG_FINALS)
    g_act_mean, g_act_std = _agg(GROWN_DREAM_ACT_AVG_FINALS)
    hat_mean, hat_std = _agg(HAT_H8_AVG_FINALS)

    print(f"Cached baselines:")
    print(f"  grown no-dream     : {g_base_mean:.4f} ± {g_base_std:.4f}  "
          f"(~{g_base_p_mean:.0f}p)")
    print(f"  grown closed W-dream: {g_closed_mean:.4f} ± {g_closed_std:.4f}  "
          f"(zero merges)")
    print(f"  grown open   W-dream: {g_open_mean:.4f} ± {g_open_std:.4f}  "
          f"(1 merge total)")
    print(f"  grown act    -dream: {g_act_mean:.4f} ± {g_act_std:.4f}  "
          f"(MERGE math, 1 event)")
    print(f"  HAT H=8 6-seed     : {hat_mean:.4f} ± {hat_std:.4f}  "
          f"({HAT_H8_PARAMS}p)")
    print()

    # Headline 1: synaptic-dream vs no-dream.
    print(f"Headline 1: synaptic-dream vs no-dream:")
    print(f"  no-dream      : {g_base_mean:.4f} ± {g_base_std:.4f}  "
          f"(~{g_base_p_mean:.0f}p)")
    print(f"  synaptic-dream: {f_mean:.4f} ± {f_std:.4f}  (~{p_mean:.0f}p)")
    if g_base_mean > 0:
        delta_pct = (f_mean - g_base_mean) / g_base_mean * 100
        print(f"  Δ             : {delta_pct:+.0f}% on mean  "
              f"(negative = synaptic improved)")
    if g_base_std + f_std > 0:
        gap = (g_base_mean - f_mean) / max(g_base_std + f_std, 1e-9)
        verdict = ("ROBUST IMPROVEMENT" if gap > 2
                   else "borderline improvement" if gap > 1
                   else "within noise" if gap > -1
                   else "borderline regression" if gap > -2
                   else "ROBUST REGRESSION")
        print(f"  σ-gap         : {gap:+.1f}σ  → {verdict}")
    print()

    # Headline 2: synaptic-dream vs act-merge dream — same diagnostic,
    # different treatment.
    print(f"Headline 2: synaptic-dream vs act-MERGE-dream "
          f"(same activation signal, different mechanism):")
    print(f"  act-merge     : {g_act_mean:.4f} ± {g_act_std:.4f}  "
          f"(threshold 0.95)")
    print(f"  synaptic      : {f_mean:.4f} ± {f_std:.4f}  "
          f"(threshold {AC_THRESHOLD})")
    if g_act_mean > 0:
        delta_pct = (f_mean - g_act_mean) / g_act_mean * 100
        print(f"  Δ             : {delta_pct:+.0f}% on mean")
    if g_act_std + f_std > 0:
        gap = (g_act_mean - f_mean) / max(g_act_std + f_std, 1e-9)
        print(f"  σ-gap         : {gap:+.1f}σ")
    print()

    # Headline 3: synaptic-dream vs HAT (the architectural target).
    print(f"Headline 3: synaptic-dream vs HAT H=8 (target gap → ≥2σ):")
    print(f"  synaptic     : {f_mean:.4f} ± {f_std:.4f}  (~{p_mean:.0f}p)")
    print(f"  HAT H=8      : {hat_mean:.4f} ± {hat_std:.4f}  "
          f"({HAT_H8_PARAMS}p)")
    if f_mean > 0:
        mult = hat_mean / f_mean
        print(f"  HAT mean is {mult:.2f}× synaptic-dream's mean")
    if f_std + hat_std > 0:
        gap = (hat_mean - f_mean) / max(f_std + hat_std, 1e-9)
        verdict = ("ROBUST" if gap > 2
                   else "borderline" if gap > 1 else "within noise")
        print(f"  σ-gap        : {gap:+.1f}σ  → {verdict}")
        if gap > 2:
            print(f"  PASS: synaptic-dream closes the σ-claim against HAT.")
        elif gap > 1:
            print(f"  PARTIAL: improved but not σ-robust at 2σ.")
        else:
            print(f"  FAIL or REGRESSION at 2σ.")
    print()

    # ----- CSVs -----
    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs"
    )
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(
        out_dir, "bench_50task_dreaming_synaptic_seeds_log.csv"
    )
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "arch", "H", "seed", "n_params_final", "final_latent",
            "avg_final_loss", "avg_forgetting",
            "n_divisions_allowed", "n_prunes_in_training",
            "n_downscales", "n_purges", "wall_clock_seconds",
        ])
        for r in rows:
            w.writerow([
                r["arch"], r["H"], r["seed"], r["n_params_final"],
                r["final_latent"], f"{r['avg_final_loss']:.6f}",
                f"{r['avg_forgetting']:.6f}",
                r["n_divisions_allowed"], r["n_prunes_in_training"],
                r["n_downscales"], r["n_purges"],
                f"{r['wall_clock_seconds']:.2f}",
            ])

    probe_path = os.path.join(
        out_dir, "bench_50task_dreaming_synaptic_probe.csv"
    )
    with open(probe_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["seed", "task_idx", "pair_name", "layer_idx",
                    "max_w_cos", "max_act_cos",
                    "n_downscales_at_task", "n_purges_at_task",
                    "n_params_after_task"])
        for s_idx, s in enumerate(SEEDS):
            traj = results[s_idx].get("n_params_per_task", [])
            for d in results[s_idx].get("dreams", []):
                t_idx = d["task_idx"]
                p_after = traj[t_idx] if t_idx < len(traj) else -1
                w_by_layer = dict(d.get("pre_compress_max_cosines", []))
                a_by_layer = dict(
                    d.get("pre_compress_max_activation_cosines", []))
                layers = sorted(set(w_by_layer) | set(a_by_layer))
                for L in layers:
                    wc = w_by_layer.get(L, float("nan"))
                    ac = a_by_layer.get(L, float("nan"))
                    w.writerow([
                        s, t_idx, d["pair_name"], L,
                        f"{wc:.6f}" if (wc == wc and wc != float("-inf"))
                        else "nan",
                        f"{ac:.6f}" if (ac == ac and ac != float("-inf"))
                        else "nan",
                        d.get("n_merges", 0), d.get("n_purges", 0),
                        p_after,
                    ])

    print(f"  log:   {csv_path}")
    print(f"  probe: {probe_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
