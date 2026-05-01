"""Phase 4.5 dreaming-phase SWEEP harness — synaptic downscale + activation
signal — parameterized over `ac_threshold` and `seeds`.

Same code paths as bench_50task_dreaming_synaptic_seeds.py (the original
single-config bench at ac_threshold=0.85). Difference: this script accepts
`--ac-thresholds A,B,C` and `--seeds X,Y,Z` from the CLI and iterates the
Cartesian product, so a single invocation can be either:

  - a probe sweep (single seed across multiple thresholds), or
  - a 3-seed bench at one chosen threshold,

without duplicating the ~470-line module per configuration.

Why the sweep exists (next_session_plan.md, 2026-05-01):

  Synaptic-downscale at ac_threshold=0.85 fired 94/76/147 events per seed
  → +139% regression vs no-dream (sign FLIPPED vs HAT). Activation-MERGE
  at ac_threshold=0.95 fired only 0/1/0 events → almost no test of the
  mechanism. The hypothesis is a sweet spot in {0.92, 0.95, 0.97} where
  events fire 5-15 times per 50 tasks (covered redundancy without
  overwhelming the 200-step replay budget).

A-priori reading rule:
  PASS:    σ-gap vs HAT moves from 1.1σ → ≥ 2σ AND events fire 5-15 per
           50 tasks. Sweet-spot threshold confirmed.
  PARTIAL: σ-gap improves to ~1.5-1.9σ + events in range. Treatment
           helps but threshold or replay budget would tighten further.
  FAIL:    No threshold in the sweep produces both events-in-range AND
           σ-gap improvement vs the act-merge bench's 1.1σ. Move to
           Experiment 2 (per-block downscale cap).

Usage:
  # Probe sweep — single seed across the three thresholds (~30 min).
  python3 -m experiments.bench_50task_dreaming_synaptic_sweep \\
      --ac-thresholds 0.92,0.95,0.97 --seeds 0

  # 3-seed bench at the chosen sweet-spot threshold (~25 min).
  python3 -m experiments.bench_50task_dreaming_synaptic_sweep \\
      --ac-thresholds 0.94 --seeds 0,1,2

  # Phase 4.5 Experiment 2 — per-block downscale cap. Re-runs the
  # aggressive ac=0.85 threshold but with cap=1 or cap=2 so replay
  # absorbs each consolidation before the next one drifts on top.
  python3 -m experiments.bench_50task_dreaming_synaptic_sweep \\
      --ac-thresholds 0.85 --seeds 0,1,2 --max-downscales-per-layer 1
"""
from __future__ import annotations
import argparse
import csv
import os
import random
import statistics
import sys
import time
from typing import List, Tuple

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


PROBE_BATCH_SIZE = 128

# Cached comparison numbers (unchanged from synaptic-seeds bench).
GROWN_BASELINE_AVG_FINALS = [0.0297, 0.0689, 0.0603]
GROWN_BASELINE_FINAL_PARAMS = [247, 264, 252]

GROWN_DREAM_ACT_AVG_FINALS = [0.0332, 0.0692, 0.0221]   # ac=0.95 MERGE
GROWN_DREAM_ACT_FINAL_PARAMS = [269, 296, 299]

GROWN_DREAM_SYNAPTIC_85_AVG_FINALS = [0.0701, 0.1280, 0.2050]  # ac=0.85 DOWNSCALE
GROWN_DREAM_SYNAPTIC_85_FINAL_PARAMS = [263, 289, 298]
GROWN_DREAM_SYNAPTIC_85_EVENTS = [94, 76, 147]

HAT_H8_AVG_FINALS = [
    0.0525, 0.1468, 0.1751,
    0.1202, 0.0896, 0.1156,
]
HAT_H8_PARAMS = 203


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


def _make_dreaming_config(
    ac_threshold: float,
    max_downscales_per_layer: "int | None" = None,
) -> dict:
    return {
        "replay_fraction": DREAM_REPLAY_FRACTION,
        "replay_steps_per_pair": DREAM_REPLAY_STEPS,
        "replay_batch": DREAM_BATCH,
        "ewc_strength": DREAM_EWC_STRENGTH,
        "cos_threshold": 0.95,
        "u_threshold": DREAM_U_THRESHOLD,
        "skip_output_layer": True,
        "redundancy_signal": "activation",
        "ac_threshold": ac_threshold,
        "probe_batch_size": PROBE_BATCH_SIZE,
        "compression_action": "downscale",
        "max_downscales_per_layer": max_downscales_per_layer,
    }


def _parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ac-thresholds", type=_parse_float_list, required=True,
                   help="Comma-separated list of ac_threshold values, e.g. 0.92,0.95,0.97")
    p.add_argument("--seeds", type=_parse_int_list, required=True,
                   help="Comma-separated list of seeds, e.g. 0  or  0,1,2")
    p.add_argument("--max-downscales-per-layer", type=int, default=None,
                   help="Phase 4.5 Experiment 2: cap on downscale events per "
                        "layer per dreaming block. Omit for uncapped (default).")
    p.add_argument("--tag", type=str, default="",
                   help="Optional output-file tag suffix; if empty, derived from args")
    args = p.parse_args(argv)

    thresholds = args.ac_thresholds
    seeds = args.seeds
    max_downscales_per_layer = args.max_downscales_per_layer

    # Derive a default tag if none provided.
    if args.tag:
        tag = args.tag
    else:
        th_part = (f"th{thresholds[0]:.2f}"
                   if len(thresholds) == 1
                   else f"th{thresholds[0]:.2f}-{thresholds[-1]:.2f}")
        s_part = (f"s{seeds[0]}"
                  if len(seeds) == 1
                  else f"s{seeds[0]}-{seeds[-1]}")
        cap_part = (f"_cap{max_downscales_per_layer}"
                    if max_downscales_per_layer is not None
                    else "")
        tag = f"{th_part}_{s_part}{cap_part}"

    pair_specs = build_50task_pairs(
        state_dim=STATE_DIM, n_single=N_SINGLE, n_compound=N_COMPOUND, seed=0,
    )
    pair_names = [p.name for p in pair_specs]

    def cur_factory(seed):
        return ParameterizedContrastiveCurriculum(
            state_dim=STATE_DIM, pair_specs=pair_specs, seed=seed)

    eval_batches = make_fixed_eval_batches(pair_names, cur_factory)

    print("=" * 78)
    print("bench_50task_dreaming_synaptic_sweep — SYNAPTIC DOWNSCALE")
    print("                                       (substrate-preserving)")
    print("=" * 78)
    print(f"Thresholds: {thresholds}")
    print(f"Seeds:      {seeds}")
    print(f"Cap:        max_downscales_per_layer="
          f"{max_downscales_per_layer if max_downscales_per_layer is not None else 'None (uncapped)'}")
    print(f"Tag:        {tag}")
    print(f"Replay:     {DREAM_REPLAY_FRACTION:.0%} × {DREAM_REPLAY_STEPS} "
          f"steps   ewc_strength={DREAM_EWC_STRENGTH}   "
          f"u_threshold={DREAM_U_THRESHOLD}")
    print(f"Probe:      batch={PROBE_BATCH_SIZE}   skip_output_layer=True")
    print(f"50-task curriculum: 12 single + 38 compound on 12-dim state")
    total_runs = len(thresholds) * len(seeds)
    print(f"Total runs: {total_runs} ({len(thresholds)} thresholds × "
          f"{len(seeds)} seeds)")
    print()

    rows: List[dict] = []
    results_by_key: dict = {}    # (threshold, seed) -> result
    t_start = time.monotonic()
    run_idx = 0

    for ac_threshold in thresholds:
        for s in seeds:
            run_idx += 1
            print()
            print("-" * 78)
            print(f"Run {run_idx}/{total_runs}: ac_threshold={ac_threshold}  "
                  f"seed={s}")
            print("-" * 78)
            torch.manual_seed(s + 11000)
            net = make_network(STATE_DIM, HIDDEN, LATENT_INIT_GROWN)
            dreaming_config = _make_dreaming_config(
                ac_threshold,
                max_downscales_per_layer=max_downscales_per_layer,
            )
            result = run_ewc_curriculum(
                net,
                label=f"grown_synaptic_th{ac_threshold:.2f}_seed{s}",
                do_growth=True, do_pruning=True,
                train_cur=cur_factory(seed=s + 41),
                eval_batches=eval_batches, pair_names=pair_names,
                dreaming_config=dreaming_config,
                dreaming_rng=random.Random(s + 53),
            )
            n_events = sum(d["n_merges"] for d in result.get("dreams", []))
            n_purges = sum(d["n_purges"] for d in result.get("dreams", []))
            rows.append({
                "ac_threshold": ac_threshold,
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
            results_by_key[(ac_threshold, s)] = result
            print(f"  -> seed{s} @ ac={ac_threshold:.2f}: "
                  f"avg_final={result['avg_final_loss']:.4f}  "
                  f"events={n_events}  purges={n_purges}  "
                  f"params={result['final_n_params']}")

    elapsed_total = time.monotonic() - t_start

    # ===== Per-threshold report =====
    print()
    print("=" * 78)
    print(f"bench_50task_dreaming_synaptic_sweep — Final Report  "
          f"(total wall-clock {elapsed_total/60:.1f} min)")
    print("=" * 78)
    print()

    g_base_mean, g_base_std = _agg(GROWN_BASELINE_AVG_FINALS)
    g_act_mean, g_act_std = _agg(GROWN_DREAM_ACT_AVG_FINALS)
    g_syn85_mean, g_syn85_std = _agg(GROWN_DREAM_SYNAPTIC_85_AVG_FINALS)
    hat_mean, hat_std = _agg(HAT_H8_AVG_FINALS)

    print(f"Cached baselines:")
    print(f"  no-dream            : {g_base_mean:.4f} ± {g_base_std:.4f}")
    print(f"  act-MERGE @ 0.95    : {g_act_mean:.4f} ± {g_act_std:.4f}  "
          f"(events: 0/1/0)")
    print(f"  synaptic @ 0.85     : {g_syn85_mean:.4f} ± {g_syn85_std:.4f}  "
          f"(events: {GROWN_DREAM_SYNAPTIC_85_EVENTS})")
    print(f"  HAT H=8 6-seed      : {hat_mean:.4f} ± {hat_std:.4f}  "
          f"({HAT_H8_PARAMS}p)")
    print()

    # Per-threshold aggregation across the seeds run for that threshold.
    print(f"{'ac_thr':>8}  {'seeds':>15}  {'mean ± std':>20}  "
          f"{'events (per seed)':>22}  {'purges':>14}  {'σ-vs-HAT':>10}  "
          f"{'verdict':>22}")
    print("-" * 130)
    for ac_threshold in thresholds:
        th_rows = [r for r in rows if r["ac_threshold"] == ac_threshold]
        if not th_rows:
            continue
        finals = [r["avg_final_loss"] for r in th_rows]
        events = [r["n_downscales"] for r in th_rows]
        purges = [r["n_purges"] for r in th_rows]
        seed_list = [r["seed"] for r in th_rows]
        f_mean, f_std = _agg(finals)
        gap = ((hat_mean - f_mean) / max(f_std + hat_std, 1e-9)
               if (f_std + hat_std) > 0 else float("nan"))
        # Sweet-spot heuristic on EVENTS. The sweet spot is 5-15 per 50 tasks.
        events_mean = sum(events) / len(events)
        if events_mean < 2:
            event_verdict = "too few"
        elif events_mean <= 15:
            event_verdict = "in range (5-15)"
        elif events_mean <= 40:
            event_verdict = "elevated"
        else:
            event_verdict = "too many"
        sigma_verdict = ("PASS ≥2σ" if gap > 2
                         else "PARTIAL" if gap > 1
                         else "within noise" if gap > -1
                         else "REGRESSION")
        verdict = f"{event_verdict} / {sigma_verdict}"
        print(f"{ac_threshold:>8.2f}  {str(seed_list):>15}  "
              f"{f_mean:>10.4f} ± {f_std:.4f}  {str(events):>22}  "
              f"{str(purges):>14}  {gap:>+8.1f}σ  {verdict:>22}")
    print()

    # ===== Per-run probe trajectory dump (compact) =====
    print("Probe trajectory (pre-compress, max activation cosine, per-task):")
    for ac_threshold in thresholds:
        for s in seeds:
            key = (ac_threshold, s)
            result = results_by_key.get(key)
            if result is None:
                continue
            dreams = result.get("dreams", [])
            if not dreams:
                continue
            layers_seen = sorted({
                L for d in dreams
                for L, _ in d.get("pre_compress_max_activation_cosines", [])
            })
            print(f"  ac={ac_threshold:.2f} seed={s}:")
            for L in layers_seen:
                a_traj = [
                    next((c for ll, c in
                          d.get("pre_compress_max_activation_cosines", [])
                          if ll == L), float("nan"))
                    for d in dreams
                ]
                print(f"    layer {L} act-cos: {_trajectory_summary(a_traj)}")
                print(f"      sample        : {_trajectory_sample(a_traj)}")
    print()

    # ===== CSV outputs =====
    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs"
    )
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(
        out_dir, f"bench_50task_dreaming_synaptic_sweep_{tag}_log.csv"
    )
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "ac_threshold", "seed", "n_params_final", "final_latent",
            "avg_final_loss", "avg_forgetting",
            "n_divisions_allowed", "n_prunes_in_training",
            "n_downscales", "n_purges", "wall_clock_seconds",
        ])
        for r in rows:
            w.writerow([
                f"{r['ac_threshold']:.4f}", r["seed"], r["n_params_final"],
                r["final_latent"], f"{r['avg_final_loss']:.6f}",
                f"{r['avg_forgetting']:.6f}",
                r["n_divisions_allowed"], r["n_prunes_in_training"],
                r["n_downscales"], r["n_purges"],
                f"{r['wall_clock_seconds']:.2f}",
            ])

    probe_path = os.path.join(
        out_dir, f"bench_50task_dreaming_synaptic_sweep_{tag}_probe.csv"
    )
    with open(probe_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ac_threshold", "seed", "task_idx", "pair_name",
                    "layer_idx", "max_w_cos", "max_act_cos",
                    "n_downscales_at_task", "n_purges_at_task",
                    "n_params_after_task"])
        for ac_threshold in thresholds:
            for s in seeds:
                key = (ac_threshold, s)
                result = results_by_key.get(key)
                if result is None:
                    continue
                traj = result.get("n_params_per_task", [])
                for d in result.get("dreams", []):
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
                            f"{ac_threshold:.4f}", s, t_idx, d["pair_name"], L,
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
