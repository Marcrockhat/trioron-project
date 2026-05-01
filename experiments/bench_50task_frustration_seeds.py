"""Frustration multiplier × all four conditions × 3 seeds, matched-width.

Spec from next_session_plan.md:

  Per-pair plateau counter that scales the per-pair gradient when loss
  has been stuck for N windows. Apply UNIFORMLY to grown / fixed-EWC /
  Online-EWC / PackNet so the comparison measures architecture ×
  optimizer interaction, not just one combo. If frustration helps
  everyone equally it's a free-lunch optimizer trick; if it helps
  grown more, the architectural claim strengthens.

Setup:
  - Same 50-task / 12-dim curriculum as bench_50task_seeds.
  - All four arms × 3 seeds, frustration ON throughout.
  - Width = matched (H=8 for fixed/online/packnet; matches what was
    "matched" in each prior seeded sweep). Single width to keep
    wall-clock reasonable; if frustration shows a strong effect a
    follow-up can sweep widths.

Comparison strategy:
  - Re-run grown WITH frustration so any optimizer-state interactions
    with growth/pruning are captured fresh.
  - For the three baselines (fixed-EWC, Online-EWC, PackNet) we compare
    against numbers already cached from the prior multi-seed runs (see
    GROWN_BASE / FIXED_EWC_BASE / ONLINE_EWC_BASE / PACKNET_BASE
    constants below). If you want a fully apples-to-apples re-run
    without frustration, set RERUN_BASELINES=True and add ~30 minutes.

Hyperparameters chosen so the multiplier can actually engage within a
1500-step task:
  - window=400 matches TRIGGER_W (the growth trigger's window)
  - threshold=1 → first plateau window past warmup engages the boost
  - eps_loss=0.001 matches TRIGGER_EPS_LOSS
  - gain=1.0, max_mult=4.0 (so a deep plateau caps at 4× signal)

Output: outputs/bench_50task_frustration_seeds_log.csv plus the
per-arm tracker diagnostic (boosted_pairs, total_boosted_windows) so
we can see whether the multiplier ever fired.
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
from trioron.frustration import FrustrationTracker

from experiments.bench_50task import (
    STATE_DIM,
    N_SINGLE,
    N_COMPOUND,
    HIDDEN,
    LATENT_INIT_GROWN,
    build_50task_pairs,
    make_fixed_eval_batches,
    make_network,
    run_ewc_curriculum,
    run_packnet_curriculum,
)
from experiments.bench_50task_online_ewc import run_online_ewc_curriculum


SEEDS = [0, 1, 2]
MATCHED_H = 8  # matches "matched" width in all three baseline sweeps

FRUSTRATION_WINDOW = 400
FRUSTRATION_THRESHOLD = 1
FRUSTRATION_EPS_LOSS = 0.001
FRUSTRATION_GAIN = 1.0
FRUSTRATION_MAX_MULT = 4.0


# -- Cached baselines from prior multi-seed runs (frustration OFF). -----------
# bench_50task_seeds_run1.log (commit e5915db):
GROWN_BASE = {
    "per_seed_avg_final": [0.0297, 0.0689, 0.0603],
    "final_latents":      [4, 3, 3],
    "final_params":       [247, 264, 252],
}
FIXED_EWC_BASE = {  # H=8, matched, per-seed from bench_50task_seeds_run1.log
    "per_seed_avg_final": [0.1080, 0.0791, 0.0980],  # = 0.0950 ± 0.0147
    "params":             203,
}
# bench_50task_packnet_seeds_run1.log (commit 0f105fd), H=8:
PACKNET_BASE = {
    "per_seed_avg_final": [0.1709, 0.2175, 0.1692],  # = 0.1859 ± 0.0274
    "params":             203,
}
# bench_50task_online_ewc_run1.log (commit 81dd822), H=8:
ONLINE_EWC_BASE = {
    "per_seed_avg_final": [0.0506, 0.0887, 0.0595],  # = 0.0662 ± 0.0200
    "params":             203,
}


def _agg(xs: List[float]) -> Tuple[float, float]:
    if len(xs) < 2:
        return (xs[0] if xs else float("nan"), 0.0)
    return statistics.mean(xs), statistics.stdev(xs)


def _make_tracker() -> FrustrationTracker:
    return FrustrationTracker(
        window=FRUSTRATION_WINDOW,
        threshold=FRUSTRATION_THRESHOLD,
        eps_loss=FRUSTRATION_EPS_LOSS,
        gain=FRUSTRATION_GAIN,
        max_mult=FRUSTRATION_MAX_MULT,
    )


def _diag(tracker: FrustrationTracker) -> Dict[str, object]:
    return {
        "boosted_pairs": tracker.boosted_pairs(),
        "total_boosted_windows": tracker.total_boosted_windows(),
        "n_pairs_tracked": len(tracker._stuck),
    }


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
    print("bench_50task_frustration_seeds — frustration × {grown, fixed-EWC,")
    print("                                  Online-EWC, PackNet} × 3 seeds")
    print("=" * 78)
    print(f"Seeds:        {SEEDS}")
    print(f"Matched H:    {MATCHED_H}  (single width; matches each baseline's matched arm)")
    print(f"Frustration:  W={FRUSTRATION_WINDOW} threshold={FRUSTRATION_THRESHOLD} "
          f"eps_loss={FRUSTRATION_EPS_LOSS} gain={FRUSTRATION_GAIN} "
          f"max_mult={FRUSTRATION_MAX_MULT}")
    print()

    rows: List[dict] = []
    diagnostics: Dict[str, List[dict]] = {
        "grown": [], "fixed_ewc": [], "online_ewc": [], "packnet": [],
    }

    t_start = time.monotonic()

    # --- Grown × seeds (with frustration) ---
    grown_finals: List[float] = []
    grown_params: List[int] = []
    grown_latents: List[int] = []
    grown_results: List[dict] = []
    for s in SEEDS:
        torch.manual_seed(s)
        net = make_network(STATE_DIM, HIDDEN, LATENT_INIT_GROWN)
        tracker = _make_tracker()
        result = run_ewc_curriculum(
            net, label=f"grown_frust_seed{s}",
            do_growth=True, do_pruning=True,
            train_cur=cur_factory(seed=s),
            eval_batches=eval_batches, pair_names=pair_names,
            frustration=tracker,
        )
        diag = _diag(tracker)
        diagnostics["grown"].append(diag)
        rows.append({
            "arch": "grown_frust", "H": HIDDEN, "seed": s,
            "n_params_final": result["final_n_params"],
            "final_latent": result["final_latent"],
            "avg_final_loss": result["avg_final_loss"],
            "avg_forgetting": result["avg_forgetting"],
            "n_divisions_allowed": len([f for f in result["fires"] if f.get("allowed")]),
            "n_prunes": len(result["prunes"]),
            "wall_clock_seconds": result["wall_clock_seconds"],
            "frustration_total_boosted_windows": diag["total_boosted_windows"],
            "frustration_n_boosted_pairs": len(diag["boosted_pairs"]),
        })
        grown_finals.append(result["avg_final_loss"])
        grown_params.append(result["final_n_params"])
        grown_latents.append(result["final_latent"])
        grown_results.append(result)
        print(f"\n[grown_frust seed={s}] tracker diag: "
              f"boosted_windows={diag['total_boosted_windows']}, "
              f"boosted_pairs={diag['boosted_pairs']}")

    # --- Fixed-EWC × seeds (with frustration) ---
    target_latent = int(round(sum(grown_latents) / len(grown_latents)))
    print(f"\n[frustration-bench] target_latent={target_latent} for fixed-EWC / "
          f"Online-EWC / PackNet")

    fixed_finals: List[float] = []
    for s in SEEDS:
        torch.manual_seed(s + 1000)
        net = make_network(STATE_DIM, MATCHED_H, target_latent)
        tracker = _make_tracker()
        result = run_ewc_curriculum(
            net, label=f"fixed_ewc_H{MATCHED_H}_frust_seed{s}",
            do_growth=False, do_pruning=False,
            train_cur=cur_factory(seed=s + 7 * MATCHED_H),
            eval_batches=eval_batches, pair_names=pair_names,
            frustration=tracker,
        )
        diag = _diag(tracker)
        diagnostics["fixed_ewc"].append(diag)
        rows.append({
            "arch": "fixed_ewc_frust", "H": MATCHED_H, "seed": s,
            "n_params_final": result["final_n_params"],
            "final_latent": result["final_latent"],
            "avg_final_loss": result["avg_final_loss"],
            "avg_forgetting": result["avg_forgetting"],
            "n_divisions_allowed": 0, "n_prunes": 0,
            "wall_clock_seconds": result["wall_clock_seconds"],
            "frustration_total_boosted_windows": diag["total_boosted_windows"],
            "frustration_n_boosted_pairs": len(diag["boosted_pairs"]),
        })
        fixed_finals.append(result["avg_final_loss"])
        print(f"\n[fixed_ewc seed={s}] tracker diag: "
              f"boosted_windows={diag['total_boosted_windows']}, "
              f"boosted_pairs={diag['boosted_pairs']}")

    # --- Online-EWC × seeds (with frustration) ---
    online_finals: List[float] = []
    for s in SEEDS:
        torch.manual_seed(s + 3000 + MATCHED_H)
        net = make_network(STATE_DIM, MATCHED_H, target_latent)
        tracker = _make_tracker()
        result = run_online_ewc_curriculum(
            net, label=f"online_ewc_H{MATCHED_H}_frust_seed{s}",
            train_cur=cur_factory(seed=s + 7 * MATCHED_H + 100),
            eval_batches=eval_batches, pair_names=pair_names,
            frustration=tracker,
        )
        diag = _diag(tracker)
        diagnostics["online_ewc"].append(diag)
        rows.append({
            "arch": "online_ewc_frust", "H": MATCHED_H, "seed": s,
            "n_params_final": result["final_n_params"],
            "final_latent": result["final_latent"],
            "avg_final_loss": result["avg_final_loss"],
            "avg_forgetting": result["avg_forgetting"],
            "n_divisions_allowed": 0, "n_prunes": 0,
            "wall_clock_seconds": result["wall_clock_seconds"],
            "frustration_total_boosted_windows": diag["total_boosted_windows"],
            "frustration_n_boosted_pairs": len(diag["boosted_pairs"]),
        })
        online_finals.append(result["avg_final_loss"])
        print(f"\n[online_ewc seed={s}] tracker diag: "
              f"boosted_windows={diag['total_boosted_windows']}, "
              f"boosted_pairs={diag['boosted_pairs']}")

    # --- PackNet × seeds (with frustration) ---
    packnet_finals: List[float] = []
    for s in SEEDS:
        torch.manual_seed(s + 2000 + MATCHED_H)
        net = make_network(STATE_DIM, MATCHED_H, target_latent)
        tracker = _make_tracker()
        result = run_packnet_curriculum(
            net, label=f"packnet_H{MATCHED_H}_frust_seed{s}",
            train_cur=cur_factory(seed=s + 13 * MATCHED_H),
            eval_batches=eval_batches, pair_names=pair_names,
            frustration=tracker,
        )
        diag = _diag(tracker)
        diagnostics["packnet"].append(diag)
        rows.append({
            "arch": "packnet_frust", "H": MATCHED_H, "seed": s,
            "n_params_final": result["final_n_params"],
            "final_latent": result["final_latent"],
            "avg_final_loss": result["avg_final_loss"],
            "avg_forgetting": result["avg_forgetting"],
            "n_divisions_allowed": 0, "n_prunes": 0,
            "wall_clock_seconds": result["wall_clock_seconds"],
            "frustration_total_boosted_windows": diag["total_boosted_windows"],
            "frustration_n_boosted_pairs": len(diag["boosted_pairs"]),
        })
        packnet_finals.append(result["avg_final_loss"])
        print(f"\n[packnet seed={s}] tracker diag: "
              f"boosted_windows={diag['total_boosted_windows']}, "
              f"boosted_pairs={diag['boosted_pairs']}")

    elapsed_total = time.monotonic() - t_start

    # ----- Report -----
    print()
    print("=" * 78)
    print(f"bench_50task_frustration_seeds — Final Report  "
          f"(total wall-clock {elapsed_total/60:.1f} min)")
    print("=" * 78)
    print()

    def summarize(name, on, off):
        on_mean, on_std = _agg(on)
        off_mean, off_std = _agg(off)
        delta = off_mean - on_mean
        pct = 100.0 * delta / off_mean if off_mean > 0 else 0.0
        gap = (delta / max(on_std + off_std, 1e-9)) if (on_std + off_std) > 0 else 0.0
        verdict = (
            "ROBUST help" if gap > 2 else
            "borderline help" if gap > 1 else
            "within noise" if abs(gap) <= 1 else
            "borderline HURT" if gap > -2 else "ROBUST HURT"
        )
        print(f"  {name:>18}: ON {on_mean:.4f} ± {on_std:.4f}  | "
              f"OFF (cached) {off_mean:.4f} ± {off_std:.4f}  | "
              f"Δ = {delta:+.4f} ({pct:+.1f}%)  gap = {gap:+.1f}σ  {verdict}")

    print("Per-arm: frustration ON vs cached OFF baseline")
    summarize("grown",      grown_finals,      GROWN_BASE["per_seed_avg_final"])
    summarize("fixed-EWC",  fixed_finals,      FIXED_EWC_BASE["per_seed_avg_final"])
    summarize("Online-EWC", online_finals,     ONLINE_EWC_BASE["per_seed_avg_final"])
    summarize("PackNet",    packnet_finals,    PACKNET_BASE["per_seed_avg_final"])
    print()

    # Headline: did the gap to grown change?
    g_on_mean, g_on_std = _agg(grown_finals)
    g_off_mean, g_off_std = _agg(GROWN_BASE["per_seed_avg_final"])
    print("Headline (matched H=8, frustration ON):")
    for label, finals, off in [
        ("fixed-EWC",  fixed_finals,  FIXED_EWC_BASE["per_seed_avg_final"]),
        ("Online-EWC", online_finals, ONLINE_EWC_BASE["per_seed_avg_final"]),
        ("PackNet",    packnet_finals, PACKNET_BASE["per_seed_avg_final"]),
    ]:
        on_mean, on_std = _agg(finals)
        off_mean, off_std = _agg(off)
        # Gap-to-grown ON vs OFF
        on_gap_pct = 100.0 * (on_mean - g_on_mean) / g_on_mean if g_on_mean > 0 else 0.0
        off_gap_pct = 100.0 * (off_mean - g_off_mean) / g_off_mean if g_off_mean > 0 else 0.0
        delta_pct = on_gap_pct - off_gap_pct
        on_combined = max(g_on_std + on_std, 1e-9)
        off_combined = max(g_off_std + off_std, 1e-9)
        on_sigma = (on_mean - g_on_mean) / on_combined
        off_sigma = (off_mean - g_off_mean) / off_combined
        print(f"  vs {label:>10}:  ON  grown {g_on_mean:.4f} vs {label} {on_mean:.4f}  "
              f"({on_gap_pct:+.0f}%, {on_sigma:+.1f}σ)")
        print(f"             OFF  grown {g_off_mean:.4f} vs {label} {off_mean:.4f}  "
              f"({off_gap_pct:+.0f}%, {off_sigma:+.1f}σ)")
        print(f"             Δ-gap = {delta_pct:+.0f}%-points  "
              f"(Δσ = {on_sigma - off_sigma:+.1f})")

    # Diagnostic: did frustration even fire?
    print()
    print("Frustration diagnostics (did the multiplier engage?):")
    for arm, ds in diagnostics.items():
        bw = [d["total_boosted_windows"] for d in ds]
        nbp = [len(d["boosted_pairs"]) for d in ds]
        print(f"  {arm:>12}:  boosted_windows per seed = {bw}  "
              f"boosted_pairs per seed = {nbp}")
    print()

    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs"
    )
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "bench_50task_frustration_seeds_log.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "arch", "H", "seed", "n_params_final", "final_latent",
            "avg_final_loss", "avg_forgetting",
            "n_divisions_allowed", "n_prunes", "wall_clock_seconds",
            "frustration_total_boosted_windows",
            "frustration_n_boosted_pairs",
        ])
        for r in rows:
            w.writerow([
                r["arch"], r["H"], r["seed"], r["n_params_final"],
                r["final_latent"], f"{r['avg_final_loss']:.6f}",
                f"{r['avg_forgetting']:.6f}",
                r["n_divisions_allowed"], r["n_prunes"],
                f"{r['wall_clock_seconds']:.2f}",
                r["frustration_total_boosted_windows"],
                r["frustration_n_boosted_pairs"],
            ])
    print(f"  log: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
