"""Confirmation (session 015): is the imitation/survival ceiling the TEACHER'S stochasticity?

imitation_ceiling.py cleared the substrate (imitates ≈ a strong MLP) and the percept (no
perception gap). The remaining suspect: both masters fall back to random actions
(`torch.randint`) in non-critical states — unlearnable by construction. This toggles that
fallback to a DETERMINISTIC default (fire_taming.EXPLORE_DETERMINISTIC) and re-measures,
side by side:

  - imitation accuracy (fire + water), via imitation_ceiling.run_master
  - the masters' OWN arena survival
  - the WG organism's survival when trained on this teacher

If accuracy AND organism survival jump under the deterministic teacher, the cap is the
teacher's randomness — confirmed.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import experiments.world.fire_taming as ft
from experiments.world.fire_taming import fire_oracle, evaluate
from experiments.world.quest import water_master
from experiments.world.consolidate_base import run_arm, build_battery
from experiments.world.wg_survival import policy_act
import experiments.world.imitation_ceiling as IC


def _p(*a):
    print(*a, flush=True)


def confirm(cond_name, det, *, seeds_org, eval_eps, cold, thirst):
    ft.EXPLORE_DETERMINISTIC = det
    _p(f"\n================== {cond_name} ==================")
    # imitation measured over ALL states (the fallback region — where teacher noise lives —
    # is exactly the part the organism must also imitate; band-filtering would hide it).
    args = types.SimpleNamespace(seeds=40, cap=900)
    _p("-- imitation accuracy (all states) --")
    rf = IC.run_master("FIRE-ORACLE", fire_oracle, "all", args)
    rw = IC.run_master("WATER-MASTER", water_master, "all", args)
    _p("-- masters' OWN survival --")
    evaluate(lambda w, p: fire_oracle(w), "fire-oracle")
    evaluate(lambda w, p: water_master(w), "water-master")
    _p("-- WG organism survival (trained on THIS teacher) --")
    survs = []
    for s in range(seeds_org):
        torch.manual_seed(s)
        r = run_arm("lambda_wg", s, solo_ep=120, fire_ep=150, water_ep=150,
                    cold_bat=cold, warm_bat=None, thirst_bat=thirst,
                    ewc_strength=1.0, keep_sub=True)
        sv, _, _ = evaluate(policy_act(r["_sub"]), f"WG organism s{s}", seeds=eval_eps)
        survs.append(sv)
    m = statistics.mean(survs)
    sd = statistics.pstdev(survs) if len(survs) > 1 else 0.0
    _p(f"  >> WG organism survival: {m:.1f} +/- {sd:.1f}  ({[round(x,1) for x in survs]})")
    return dict(fire_acc=rf["sub_p"], water_acc=rw["sub_p"], org_surv=m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds-org", type=int, default=3)
    ap.add_argument("--eval-eps", type=int, default=40)
    args = ap.parse_args()
    _p("=== DETERMINISTIC-TEACHER CONFIRMATION ===")
    # fixed metric batteries, built ONCE under the stochastic teacher (reused both arms)
    ft.EXPLORE_DETERMINISTIC = False
    cold = build_battery(fire_oracle, which="cold")
    thirst = build_battery(water_master, which="thirsty")
    s = confirm("STOCHASTIC teacher (baseline)", False,
                seeds_org=args.seeds_org, eval_eps=args.eval_eps, cold=cold, thirst=thirst)
    d = confirm("DETERMINISTIC teacher (random fallback removed)", True,
                seeds_org=args.seeds_org, eval_eps=args.eval_eps, cold=cold, thirst=thirst)
    _p("\n=== CONFIRMATION SUMMARY (stochastic -> deterministic teacher) ===")
    _p(f"  fire imitation (trioron) : {s['fire_acc']:.3f} -> {d['fire_acc']:.3f}  "
       f"(Δ {d['fire_acc']-s['fire_acc']:+.3f})")
    _p(f"  water imitation (trioron): {s['water_acc']:.3f} -> {d['water_acc']:.3f}  "
       f"(Δ {d['water_acc']-s['water_acc']:+.3f})")
    _p(f"  WG organism survival     : {s['org_surv']:.1f} -> {d['org_surv']:.1f}  "
       f"(Δ {d['org_surv']-s['org_surv']:+.1f})")
    _p("  both rising = teacher stochasticity WAS the cap (substrate & percept cleared).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
