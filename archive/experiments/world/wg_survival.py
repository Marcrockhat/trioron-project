"""How long does the WG-consolidated organism SURVIVE in the arena?

The consolidation benches measure retention as deterministic battery policy-agreement
(no episode noise). This asks the other question Rocky wants: take the organism AFTER
the full solo→fire→consolidate→water pipeline and run it in the real TileWorld, reporting
mean survival (out of max_steps), cold-deaths, and fire-occupancy via fire_taming.evaluate.

WG = the native |w·g| soft epigenetic lock λ. FULL-LOCK (hard freeze) and PLAIN are the
reference points. NOTE: whole-episode survival swings ±15 (predator/spawn luck) — read
the means, not sub-±15 deltas (that's why the battery exists). seeds × evaluate's 40 eps.
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.world.fire_taming import evaluate
from experiments.world.mirror_cells import _solo
from experiments.world.consolidate_base import run_arm, build_battery
from experiments.world.fire_taming import fire_oracle
from experiments.world.quest import water_master


def _p(*a):
    print(*a, flush=True)


def policy_act(sub):
    @torch.no_grad()
    def act(w, p):
        return int(sub(_solo(p.unsqueeze(0)))[0].argmax())
    return act


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--solo-ep", type=int, default=120)
    ap.add_argument("--fire-ep", type=int, default=150)
    ap.add_argument("--water-ep", type=int, default=150)
    ap.add_argument("--eval-eps", type=int, default=40)
    ap.add_argument("--ewc-strength", type=float, default=1.0)
    args = ap.parse_args()

    _p("=== WG-consolidated organism: ARENA SURVIVAL (not battery agreement) ===")
    _p("    survival out of 300 steps; whole-episode metric swings ±15 — read means.\n")

    cold_bat = build_battery(fire_oracle, which="cold")
    thirst_bat = build_battery(water_master, which="thirsty")

    arms = (("lambda_wg", "LAMBDA-WG"), ("full", "FULL-LOCK"), ("none", "PLAIN"))
    results = {label: [] for _, label in arms}
    for mode, label in arms:
        _p(f"########## {label} ##########")
        for s in range(args.seeds):
            torch.manual_seed(s)                        # battery/explore determinism
            r = run_arm(mode, s, solo_ep=args.solo_ep, fire_ep=args.fire_ep,
                        water_ep=args.water_ep, cold_bat=cold_bat, warm_bat=None,
                        thirst_bat=thirst_bat, ewc_strength=args.ewc_strength,
                        keep_sub=True)
            surv, causes, fire_use = evaluate(policy_act(r["_sub"]),
                                              f"{label} s{s}", seeds=args.eval_eps)
            results[label].append(surv)

    _p(f"\n=== SURVIVAL SUMMARY (mean steps/300 over {args.seeds} seeds × "
       f"{args.eval_eps} eps) ===")
    for _, label in arms:
        xs = results[label]
        sd = statistics.pstdev(xs) if len(xs) > 1 else 0.0
        _p(f"  {label:>12s}: {statistics.mean(xs):5.1f} +/- {sd:4.1f}  "
           f"(seeds: {', '.join(f'{x:.0f}' for x in xs)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
