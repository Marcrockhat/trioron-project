"""Phase 2.5 of the Mode-E survival recipe — router HYSTERESIS.

The dream loop (Phase 3) confirmed at n=40 that donor-dreaming is a one-shot
rescue: it lifts survival 72.5 -> 92.1 and the keep-best rollback makes that
stick, but it cannot climb past it. Overheat stays the dominant death cause
(21-26/40) through every iteration regardless of how much the WARM donor is
sharpened. The ceiling is therefore NOT donor quality — it is the ARBITRATION.

The memoryless router (vocabulary.VocabularyOrganism) re-routes every step by
argmax manifold log-likelihood. Near the fire this flip-flops: as the organism
warms, WARM's cold-danger collapses through the temp~0.5 valley, routing hands
off to a primitive that does not manage temperature, that primitive lingers by
the fire, and temp spikes to overheat — the WARM donor never gets to execute its
(widened, Phase-2 item 6) leave-before-overheat behaviour because routing leaves
WARM the instant temp dips out of its manifold.

HYSTERESIS fixes the arbitration, not the donor: once a primitive is engaged,
HOLD it until a challenger's routing score beats the incumbent by `margin` AND a
minimum dwell has elapsed. This keeps WARM engaged across the danger valley so it
can finish regulating temperature and walk away from the fire.

The deployable router uses only the percept (manifold LL margin). We also report
the privileged danger-margin arbiter as the upper bound, exactly as Phase 2 used
ArbiterOrganism.

Usage:
    python3 -m experiments.world.hysteresis --smoke
    python3 -m experiments.world.hysteresis              # baseline + sweep
"""
from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import experiments.world.fire_taming as _ft               # noqa: E402
from experiments.world.tile_world import TileWorld         # noqa: E402
from experiments.world.mirror_cells import _solo           # noqa: E402
from experiments.world.vocabulary import (                 # noqa: E402
    build_vocabulary, VocabularyOrganism, PRIM_ORDER, _ctx, danger,
)


def _p(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------
# Hysteretic organism — routing inertia over the manifold scores
# ----------------------------------------------------------------------
class HysteresisOrganism:
    """VocabularyOrganism with routing inertia. Keeps the engaged primitive until
    a challenger's manifold log-likelihood beats it by `margin` AND at least
    `min_dwell` steps have elapsed since the last switch. margin=0, min_dwell=0
    reduces to the memoryless VocabularyOrganism."""

    def __init__(self, donors, router, *, margin=0.0, min_dwell=0):
        self.donors = donors
        self.router = router
        self.margin = margin
        self.min_dwell = min_dwell
        self.current = None
        self.dwell = 0
        self.route_hist = [0, 0, 0, 0]
        self.switches = 0

    @torch.no_grad()
    def act(self, w, p):
        scores = self.router._scores(_ctx(p.unsqueeze(0)))[0]      # [n_prim] LL
        challenger = int(scores.argmax())
        if self.current is None:
            self.current = challenger
        elif (self.dwell >= self.min_dwell
              and scores[challenger] > scores[self.current] + self.margin):
            self.current = challenger
            self.dwell = 0
            self.switches += 1
        else:
            self.dwell += 1
        i = self.current
        self.route_hist[i] += 1
        sub = self.donors[PRIM_ORDER[i]]
        return int(sub(_solo(p.unsqueeze(0)))[0].argmax())


class DangerHysteresisOrganism:
    """Privileged upper bound: same inertia logic but over the hand-coded
    argmax-danger signal (world state) instead of the learned manifold. Shows the
    survival ceiling hysteretic arbitration can reach if routing were perfect."""

    def __init__(self, donors, *, margin=0.0, min_dwell=0):
        self.donors = donors
        self.margin = margin
        self.min_dwell = min_dwell
        self.current = None
        self.dwell = 0
        self.route_hist = [0, 0, 0, 0]
        self.switches = 0

    @torch.no_grad()
    def act(self, w, p):
        d = danger(w)
        order = sorted(PRIM_ORDER, key=lambda k: d[k], reverse=True)
        challenger = PRIM_ORDER.index(order[0])
        if self.current is None:
            self.current = challenger
        elif (self.dwell >= self.min_dwell
              and d[order[0]] > d[PRIM_ORDER[self.current]] + self.margin):
            self.current = challenger
            self.dwell = 0
            self.switches += 1
        else:
            self.dwell += 1
        i = self.current
        self.route_hist[i] += 1
        return int(self.donors[PRIM_ORDER[i]](_solo(p.unsqueeze(0)))[0].argmax())


# ----------------------------------------------------------------------
# Probe — survival + death-cause breakdown (overheat is the metric that matters)
# ----------------------------------------------------------------------
def _cause(w):
    if w.alive:
        return "timeout"
    if w.temp <= 0.05:
        return "cold"
    if w.temp >= 0.98:
        return "overheat"
    if w.thirst <= 0:
        return "thirst"
    if w.energy <= 0:
        return "energy"
    return "integrity"


@torch.no_grad()
def probe(org, *, seeds=40, max_steps=300):
    surv, causes = [], Counter()
    for ep in range(seeds):
        w = TileWorld(seed=ep * 7000 + 7, max_steps=max_steps)
        p = w.percept(); done = False
        while not done:
            a = org.act(w, p)
            p, _, done, info = w.step(a)
        surv.append(info["t"])
        causes[_cause(w)] += 1
    return statistics.mean(surv), causes


def _fmt(causes):
    return dict(causes.most_common())


# ----------------------------------------------------------------------
# Sweep
# ----------------------------------------------------------------------
def run(*, eval_seeds=40, router_seeds=120, smoke=False):
    _ft.EXPLORE_DETERMINISTIC = False
    org, _ = build_vocabulary(router_seeds=router_seeds)
    donors = org.donors
    router = org.router

    # baseline: memoryless router (margin 0, dwell 0)
    base = VocabularyOrganism(donors, router)
    b_surv, b_causes = probe(base, seeds=eval_seeds)
    tot = max(sum(base.route_hist), 1)
    usage = "  ".join(f"{PRIM_ORDER[i]}={100*base.route_hist[i]/tot:.0f}%"
                      for i in range(4))
    _p(f"[baseline memoryless] survival {b_surv:.1f}   overheat "
       f"{b_causes['overheat']}/{eval_seeds}")
    _p(f"                      deaths {_fmt(b_causes)}")
    _p(f"                      route usage {usage}\n")

    # learned-router hysteresis sweep
    if smoke:
        margins = [10.0, 50.0]
        dwells = [3]
    else:
        margins = [0.0, 5.0, 20.0, 50.0, 150.0]
        dwells = [1, 3, 8]

    _p("[learned-router hysteresis] margin x min_dwell -> survival (overheat)")
    best = (b_surv, 0.0, 0, b_causes)
    for m in margins:
        row = []
        for dw in dwells:
            h = HysteresisOrganism(donors, router, margin=m, min_dwell=dw)
            s, c = probe(h, seeds=eval_seeds)
            row.append(f"dw{dw}:{s:.1f}({c['overheat']})")
            if s > best[0]:
                best = (s, m, dw, c)
        _p(f"   margin {m:6.1f}   " + "   ".join(row))
    _p(f"\n   best learned-router: survival {best[0]:.1f}  "
       f"(margin {best[1]}, dwell {best[2]})  overheat {best[3]['overheat']}/{eval_seeds}")
    _p(f"                        deaths {_fmt(best[3])}")

    # privileged danger-hysteresis upper bound (one sane config)
    dh = DangerHysteresisOrganism(donors, margin=0.05, min_dwell=3)
    d_surv, d_causes = probe(dh, seeds=eval_seeds)
    _p(f"\n[danger-hysteresis upper bound] margin 0.05, dwell 3: survival "
       f"{d_surv:.1f}   overheat {d_causes['overheat']}/{eval_seeds}")
    _p(f"                                deaths {_fmt(d_causes)}")

    _p(f"\n=== VERDICT ===")
    _p(f"    baseline {b_surv:.1f}  ->  best-hysteresis {best[0]:.1f}  "
       f"(lift {best[0]-b_surv:+.1f})   bars: fire-master 87.4 | reactive 141.8")
    return b_surv, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--eval-seeds", type=int, default=40)
    ap.add_argument("--router-seeds", type=int, default=120)
    args = ap.parse_args()
    if args.smoke:
        args.eval_seeds, args.router_seeds = 8, 40

    _p("=== PHASE 2.5: router HYSTERESIS (attack the overheat ceiling) ===")
    _p("    routing inertia: hold the engaged primitive across the danger valley\n")
    run(eval_seeds=args.eval_seeds, router_seeds=args.router_seeds,
        smoke=args.smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
