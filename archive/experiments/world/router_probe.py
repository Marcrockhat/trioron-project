"""Session 018 — the cheap router probe (handoff next-up #1, lever b).

The overheat ceiling is ROUTING-priority, not the donor (s017): the best WARM
donor leaves vocabulary overheat at exactly 26/40 because heat-danger ramps
LINEARLY in temp while heating is fast (0.04/step). Milder drives outrank WARM
until temp is near-lethal, then flee-lag guarantees overshoot.

Cheap test of the fix: steepen the heat-danger ramp (vocabulary.HEAT_GAMMA<1) so
WARM wins the argmax earlier, and see whether earlier engagement alone breaks 26.

We sweep gamma over two arms:
  ARBITER  — privileged argmax-danger router (NO manifold). The clean upper
             bound: if steepening can't help this, the learned router can't
             either. Cheap (just donors), so we sweep it densely first.
  ROUTED   — the deployable manifold organism (rebuilt per gamma, since the
             labels that fit the manifolds shift with gamma). Run only for the
             gammas that move the arbiter.

Usage:
    python3 -m experiments.world.router_probe --smoke
    python3 -m experiments.world.router_probe              # arbiter sweep + routed
    python3 -m experiments.world.router_probe --gammas 1.0 0.5 0.35
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import experiments.world.vocabulary as voc                  # noqa: E402
from experiments.world.fire_taming import evaluate          # noqa: E402
from experiments.world.primitives import load_donor, PRIM_DIR  # noqa: E402
from experiments.world.vocabulary import (                  # noqa: E402
    PRIM_ORDER, ArbiterOrganism, build_vocabulary,
)


def _p(*a):
    print(*a, flush=True)


def _load_donors():
    donors = {}
    for name in PRIM_ORDER:
        donors[name], _ = load_donor(PRIM_DIR / f"{name}.pt")
    return donors


def _route_usage(hist):
    tot = max(sum(hist), 1)
    return "  ".join(f"{PRIM_ORDER[i]}={100*hist[i]/tot:.0f}%" for i in range(len(hist)))


def arbiter_sweep(donors, gammas, *, seeds):
    """Privileged argmax-danger router across heat-ramp steepness."""
    _p("\n[ARBITER] privileged argmax-danger router (upper bound, no manifold):")
    rows = []
    for g in gammas:
        voc.HEAT_GAMMA = g
        arb = ArbiterOrganism(donors)
        surv, causes, _ = evaluate(lambda w, p: arb.act(w, p),
                                   f"arbiter gamma={g:.2f}", seeds=seeds)
        rows.append((g, surv, causes.get("overheat", 0), _route_usage(arb.route_hist)))
    voc.HEAT_GAMMA = 1.0
    _p("\n  gamma | survival | overheat | route usage")
    for g, surv, oh, usage in rows:
        _p(f"  {g:5.2f} | {surv:7.1f}  | {oh:5d}/{seeds} | {usage}")
    return rows


def routed_sweep(gammas, *, seeds, router_seeds, per_prim_cap):
    """Deployable manifold organism; rebuilt per gamma (labels shift with gamma)."""
    _p("\n[ROUTED] deployable manifold organism (rebuilt per gamma):")
    rows = []
    for g in gammas:
        voc.HEAT_GAMMA = g
        org, _ = build_vocabulary(router_seeds=router_seeds, per_prim_cap=per_prim_cap)
        surv, causes, _ = evaluate(lambda w, p: org.act(w, p),
                                   f"routed gamma={g:.2f}", seeds=seeds)
        rows.append((g, surv, causes.get("overheat", 0), _route_usage(org.route_hist)))
    voc.HEAT_GAMMA = 1.0
    _p("\n  gamma | survival | overheat | route usage")
    for g, surv, oh, usage in rows:
        _p(f"  {g:5.2f} | {surv:7.1f}  | {oh:5d}/{seeds} | {usage}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--gammas", type=float, nargs="+", default=None)
    ap.add_argument("--eval-seeds", type=int, default=40)
    ap.add_argument("--router-seeds", type=int, default=120)
    ap.add_argument("--per-prim-cap", type=int, default=1500)
    ap.add_argument("--arbiter-only", action="store_true")
    args = ap.parse_args()

    gammas = args.gammas or [1.0, 0.7, 0.5, 0.35, 0.25]
    if args.smoke:
        args.eval_seeds, args.router_seeds, args.per_prim_cap = 6, 30, 300
        gammas = [1.0, 0.5]

    _p("=== s018 ROUTER PROBE: heat-ramp steepness vs overheat ceiling ===")
    _p(f"gammas={gammas}  eval_seeds={args.eval_seeds}  (baseline gamma=1.0 = 26/40)")

    donors = _load_donors()
    arbiter_sweep(donors, gammas, seeds=args.eval_seeds)
    if not args.arbiter_only:
        routed_sweep(gammas, seeds=args.eval_seeds,
                     router_seeds=args.router_seeds, per_prim_cap=args.per_prim_cap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
