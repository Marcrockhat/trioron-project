"""Phase 4.5 dreaming-phase bench (activation-correlation signal).

Third dreaming bench. Replaces the W_anchor cosine redundancy detector
with Pearson cosine of post-activation column vectors over a probe
batch (built once per dreaming block from past pair samples). The
W-cosine signal couldn't find redundancy in this architecture (probe
data from bench_50task_dreaming_open_seeds was conclusive — output-
layer cosines are negative-to-near-zero by construction; hidden-layer
cosines are partial direction-sharing, not functional redundancy).

Activation correlation asks the right question: "do these two nodes
produce the same activation pattern across the data distribution?"
Two nodes with orthogonal weights but correlated outputs across a
representative slice ARE redundant — the W signal misses this; the
activation signal catches it.

Config (per next_session_plan.md, dated 2026-05-01, agreed with Rocky
2026-05-01 mid-session):
  - redundancy_signal = "activation"
  - ac_threshold = 0.95
  - probe_batch_size = 128
  - skip_output_layer = True (default per plan; output-layer activation
    cosine is the next experiment if hidden activations don't fire)
  - All other dreaming params identical to closed/open bench: replay
    25% × 200 steps × batch=32, ewc_strength=1000, u_threshold=1e-3.

A-priori interpretation rule:
  PASS:    σ-gap vs HAT moves from 1.0σ → ≥ 2σ AND merges > 0 visible
           in the dreams log. The activation signal is the right one.
  PARTIAL: σ-gap improves to ~1.5σ + merges fire — mechanism works,
           threshold tuning would tighten further. Worth a follow-up.
  FAIL (signal-still-wrong): σ-gap unchanged AND merges = 0 + activation
           cosines flat — even the activation signal can't find
           redundancy in this architecture; latent dims are
           truly-functionally-distinct AND hidden-layer activation
           patterns are uncorrelated. The σ-rescue may need a
           different mechanism entirely (e.g., shrinkage-based
           compression — find dims that contribute little to current
           loss, merge into a peer).
  FAIL (signal-right-but-merges-hurt): merges fire but mean regresses
           — activation correlation catches redundancy that's
           load-bearing for the representation; the linear-merge math
           (incoming-mean + outgoing-sum) is too lossy when activation-
           correlated nodes have different W. Would suggest a
           learned-merge variant or replay budget increase.

The activation-cosine probe trajectory + the W-cosine probe trajectory
report side-by-side per task per layer. Even at FAIL, the dual probe
tells us WHY: high activation cos but no merges → threshold too tight;
flat activation cos → no functional redundancy AT ALL.

Wall-clock estimate: ~30 min for 3 seeds (probe forward passes add
~20% overhead vs the W-cosine bench).
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

AC_THRESHOLD = 0.95
PROBE_BATCH_SIZE = 128

# Cached comparison numbers (do NOT re-run grown-no-dreaming or HAT).
GROWN_BASELINE_AVG_FINALS = [0.0297, 0.0689, 0.0603]
GROWN_BASELINE_FINAL_PARAMS = [247, 264, 252]

# Closed-output W-cosine dreaming bench (skip_output=True, cos=0.95):
# zero merges across all 3 seeds.
GROWN_DREAM_CLOSED_AVG_FINALS = [0.0272, 0.0248, 0.0799]
GROWN_DREAM_CLOSED_FINAL_PARAMS = [254, 323, 245]

# Open-output W-cosine dreaming bench (skip_output=False, cos=0.85):
# 1 merge total (kaiming-noise coincidence on layer 1, seed 1, task 1).
GROWN_DREAM_OPEN_AVG_FINALS = [0.0954, 0.0706, 0.0222]
GROWN_DREAM_OPEN_FINAL_PARAMS = [259, 296, 299]

# HAT 6-seed combined at H=8.
HAT_H8_AVG_FINALS = [
    0.0525, 0.1468, 0.1751,
    0.1202, 0.0896, 0.1156,
]
HAT_H8_PARAMS = 203

DREAMING_CONFIG_ACT = {
    "replay_fraction": DREAM_REPLAY_FRACTION,
    "replay_steps_per_pair": DREAM_REPLAY_STEPS,
    "replay_batch": DREAM_BATCH,
    "ewc_strength": DREAM_EWC_STRENGTH,
    "cos_threshold": 0.95,           # ignored under "activation" signal,
                                     # kept so any inadvertent fall-back
                                     # doesn't silently change behavior.
    "u_threshold": DREAM_U_THRESHOLD,
    "skip_output_layer": True,       # output-layer activation probe is
                                     # the next experiment.
    "redundancy_signal": "activation",
    "ac_threshold": AC_THRESHOLD,
    "probe_batch_size": PROBE_BATCH_SIZE,
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
    print("bench_50task_dreaming_act_seeds — ACTIVATION-CORRELATION signal")
    print("=" * 78)
    print(f"Seeds: {SEEDS}")
    print(f"Dreaming config: redundancy_signal=activation  "
          f"ac_threshold={AC_THRESHOLD}  "
          f"probe_batch_size={PROBE_BATCH_SIZE}  "
          f"skip_output_layer=True")
    print(f"                replay={DREAMING_CONFIG_ACT['replay_fraction']:.0%}×"
          f"{DREAMING_CONFIG_ACT['replay_steps_per_pair']}  "
          f"u_threshold={DREAMING_CONFIG_ACT['u_threshold']}  "
          f"ewc_strength={DREAMING_CONFIG_ACT['ewc_strength']}")
    print(f"50-task curriculum: 12 single + 38 compound on 12-dim state")
    print()

    rows: List[dict] = []
    results: List[dict] = []
    t_start = time.monotonic()

    for s in SEEDS:
        torch.manual_seed(s + 11000)
        net = make_network(STATE_DIM, HIDDEN, LATENT_INIT_GROWN)
        result = run_ewc_curriculum(
            net, label=f"grown_act_seed{s}",
            do_growth=True, do_pruning=True,
            train_cur=cur_factory(seed=s + 41),
            eval_batches=eval_batches, pair_names=pair_names,
            dreaming_config=DREAMING_CONFIG_ACT,
            dreaming_rng=random.Random(s + 53),
        )
        n_merges = sum(d["n_merges"] for d in result.get("dreams", []))
        n_purges = sum(d["n_purges"] for d in result.get("dreams", []))
        rows.append({
            "arch": "grown_act",
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
    print(f"bench_50task_dreaming_act_seeds — Final Report  "
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
    print(f"Grown WITH activation-signal dreaming ({len(SEEDS)} seeds):")
    print(f"  per-seed avg_final:    {[f'{x:.4f}' for x in finals]}")
    print(f"  mean ± std:            {f_mean:.4f} ± {f_std:.4f}")
    print(f"  final params:          {p_mean:.0f} ± {p_std:.0f}  "
          f"per-seed: {final_params}")
    print(f"  final latent per seed: {final_latents}")
    print(f"  total merges per seed: {n_merges}  "
          f"(closed had [0,0,0]; open had [0,1,0])")
    print(f"  total purges per seed: {n_purges}")
    print(f"  wall-clock per seed:   {[f'{x:.0f}s' for x in wall_clocks]}")
    print()

    # ----- Probe trajectory: BOTH W cosine and activation cosine -----
    print("Probe trajectory (pre-compress, per task per layer):")
    for s_idx, s in enumerate(SEEDS):
        dreams = results[s_idx].get("dreams", [])
        if not dreams:
            continue
        # Collect all layer indices observed across either probe.
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
    g_open_mean, g_open_std = _agg(GROWN_DREAM_OPEN_AVG_FINALS)
    hat_mean, hat_std = _agg(HAT_H8_AVG_FINALS)

    print(f"Cached baselines:")
    print(f"  grown no-dream    : {g_base_mean:.4f} ± {g_base_std:.4f}  "
          f"(~{g_base_p_mean:.0f}p)")
    print(f"  grown closed-dream: {g_closed_mean:.4f} ± {g_closed_std:.4f}  "
          f"(W cos, skip_output=True)")
    print(f"  grown open-dream  : {g_open_mean:.4f} ± {g_open_std:.4f}  "
          f"(W cos, skip_output=False)")
    print(f"  HAT H=8 6-seed    : {hat_mean:.4f} ± {hat_std:.4f}  "
          f"({HAT_H8_PARAMS}p)")
    print()

    # Headline 1: activation-dream vs no-dream.
    print(f"Headline 1: activation-dream vs no-dream:")
    print(f"  no-dream     : {g_base_mean:.4f} ± {g_base_std:.4f}  "
          f"(~{g_base_p_mean:.0f}p)")
    print(f"  act-dream    : {f_mean:.4f} ± {f_std:.4f}  "
          f"(~{p_mean:.0f}p)")
    if g_base_mean > 0:
        delta_pct = (f_mean - g_base_mean) / g_base_mean * 100
        print(f"  Δ            : {delta_pct:+.0f}% on mean  "
              f"(negative = act-dream improved)")
    if g_base_std + f_std > 0:
        gap = (g_base_mean - f_mean) / max(g_base_std + f_std, 1e-9)
        verdict = ("ROBUST IMPROVEMENT" if gap > 2
                   else "borderline improvement" if gap > 1
                   else "within noise" if gap > -1
                   else "borderline regression" if gap > -2
                   else "ROBUST REGRESSION")
        print(f"  σ-gap        : {gap:+.1f}σ  → {verdict}")
    print()

    # Headline 2: act-dream vs prior W-dream variants.
    print(f"Headline 2: act-dream vs prior W-cosine dream variants:")
    print(f"  closed W-dream : {g_closed_mean:.4f} ± {g_closed_std:.4f}")
    print(f"  open   W-dream : {g_open_mean:.4f} ± {g_open_std:.4f}")
    print(f"  act-dream      : {f_mean:.4f} ± {f_std:.4f}")
    if g_closed_mean > 0:
        delta_pct = (f_mean - g_closed_mean) / g_closed_mean * 100
        print(f"  Δ vs closed    : {delta_pct:+.0f}% on mean")
    if g_open_mean > 0:
        delta_pct = (f_mean - g_open_mean) / g_open_mean * 100
        print(f"  Δ vs open      : {delta_pct:+.0f}% on mean")
    print()

    # Headline 3: act-dream vs HAT (the architectural target).
    print(f"Headline 3: act-dream vs HAT H=8 (target gap → ≥2σ):")
    print(f"  act-dream    : {f_mean:.4f} ± {f_std:.4f}  (~{p_mean:.0f}p)")
    print(f"  HAT H=8      : {hat_mean:.4f} ± {hat_std:.4f}  "
          f"({HAT_H8_PARAMS}p)")
    if f_mean > 0:
        mult = hat_mean / f_mean
        print(f"  HAT mean is {mult:.2f}× act-dream's mean")
    if f_std + hat_std > 0:
        gap = (hat_mean - f_mean) / max(f_std + hat_std, 1e-9)
        verdict = ("ROBUST" if gap > 2
                   else "borderline" if gap > 1 else "within noise")
        print(f"  σ-gap        : {gap:+.1f}σ  → {verdict}")
        if gap > 2:
            print(f"  PASS: act-dream closes the σ-claim against HAT.")
        elif gap > 1:
            print(f"  PARTIAL: improved over closed/open W-dream's 1.0σ "
                  f"but not σ-robust at 2σ.")
        else:
            print(f"  FAIL or REGRESSION at 2σ.")
    print()

    # ----- CSVs -----
    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs"
    )
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "bench_50task_dreaming_act_seeds_log.csv")
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

    # Probe trajectory CSV: one row per (seed, task, layer) carrying
    # BOTH the W-cosine and the activation-cosine probe so the
    # landscape is plottable side-by-side.
    probe_path = os.path.join(
        out_dir, "bench_50task_dreaming_act_probe.csv"
    )
    with open(probe_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["seed", "task_idx", "pair_name", "layer_idx",
                    "max_w_cos", "max_act_cos",
                    "n_merges_at_task", "n_purges_at_task",
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
