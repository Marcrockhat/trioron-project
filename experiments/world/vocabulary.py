"""Phase 2 of the Mode-E survival recipe — the vocabulary organism + router.

Phase 1 (primitives.py) built four clean single-drive donors. Each is an expert
at ONE drive and dies of the drives it ignores (WARM overheats; HYDRATE freezes
and gets caught by the predator; ...). The quest thesis (quest.py): a learner
that INTEGRATES many specialists should outlast every one of them.

This module composes them with a manifold ROUTER (the v2.0-core z2 mechanism,
the same one world_routing.py validated for fire-vs-water): per primitive, fit a
full-covariance Gaussian over the interoceptive CONTEXT slice of its band
percepts; at each step, route the live percept to the primitive whose manifold
finds it most likely, and act with that donor. The router is a learned task
selector — it infers which drive is pressing from the homeostatic state (now
fully observable: drives + predator scent are all in the 77-d percept).

  context code = percept[63:77]  (scent 6 + drives 4 + night 1 + predator 3 = 14-d)

Two results decide Phase 2:
  1. ROUTING accuracy on held-out band percepts (does the router pick the right
     primitive?) — full-cov vs diagonal, as in world_routing.
  2. SURVIVAL of the routed organism vs the single donors, the masters, and the
     random / reactive floors. Win = outlast the best single skill (and aim for
     the masters).

Usage:
    python3 -m experiments.world.vocabulary --smoke
    python3 -m experiments.world.vocabulary               # full: routing + survival
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import experiments.world.fire_taming as _ft               # noqa: E402
from experiments.world.fire_taming import evaluate, fire_oracle  # noqa: E402
from experiments.world.quest import water_master, food_master    # noqa: E402
from experiments.world.tile_world import (                 # noqa: E402
    TileWorld, run_random, run_reactive,
)
from experiments.world.mirror_cells import _solo           # noqa: E402
from experiments.world.primitives import (                 # noqa: E402
    PRIMITIVES, collect, load_donor, evade_master, _pred_delta, PRIM_DIR,
)
from experiments.world.tile_world import FIRE, POISON       # noqa: E402
from trioron.learning.manifold import ManifoldArchive      # noqa: E402

# the homeostatic context slice of the 77-d percept (scent+drives+night+predator)
CTX_LO, CTX_HI = 63, 77
# s018 split: WARM (seek-fire-when-cold) and FLEE (cool-when-hot) are now separate
# primitives, so the router's cold->WARM / hot->FLEE choice is explicit instead of
# bundled into one dual-role donor that defaulted to seeking fire (the 26/40
# overheat ceiling). Order must match primitives.PRIMITIVES keys.
PRIM_ORDER = ["WARM", "FLEE", "HYDRATE", "FORAGE", "EVADE"]


# Routing is an URGENCY arbitration: at each step, address the drive nearest to
# killing you. We score every drive's danger as "fraction of the way to death"
# (so they are comparable), and the routing label for a state is its argmax-
# danger drive. This makes the per-primitive manifolds DISJOINT (one class per
# state) and therefore discriminative — unlike overlapping critical bands, where
# a cold-and-thirsty state feeds every manifold and nothing separates.
def _pred_prox(w):
    dx, dy = _pred_delta(w)
    return 1.0 / (1.0 + abs(dx) + abs(dy))         # 1 adjacent, ->0 far


# Heat ramps EARLIER than its raw fraction-to-lethal: heating is fast (0.04/step)
# and lethal-overshoot-prone, while cooling is slow (~0.008-0.015/step), so the
# router must engage WARM (flee mode) well before temp is near-lethal or flee-lag
# guarantees overshoot. HEAT_GAMMA<1 makes heat-danger CONVEX-early (concave in
# temp): at gamma=0.5, temp 0.7 reads danger 0.64 instead of 0.41, so WARM wins
# the argmax sooner. gamma=1.0 recovers the original linear ramp. (s018 probe.)
HEAT_GAMMA = 1.0


def danger(w):
    """Per-drive danger in [0,1+]; fraction of the way from safe to lethal."""
    cold = max(0.0, (0.5 - w.temp) / 0.48)         # lethal cold at temp~0.02
    heat = max(0.0, (w.temp - 0.5) / 0.49)         # lethal heat at temp~0.99
    if HEAT_GAMMA != 1.0 and heat > 0.0:
        heat = heat ** HEAT_GAMMA                  # earlier FLEE engagement
    return {
        "WARM": cold,                              # too cold -> seek fire
        "FLEE": heat,                              # too hot  -> leave fire
        "HYDRATE": 1.0 - w.thirst,                 # lethal at thirst 0
        "FORAGE": 1.0 - w.energy,                  # lethal at energy 0
        "EVADE": _pred_prox(w),                    # predator closing in
    }


def argmax_danger(w):
    d = danger(w)
    return max(PRIM_ORDER, key=lambda k: d[k])


def _ctx(percepts):
    """[N,77] percepts -> [N,14] interoceptive context code for routing."""
    if percepts.dim() == 1:
        percepts = percepts.unsqueeze(0)
    return percepts[:, CTX_LO:CTX_HI]


# ----------------------------------------------------------------------
# The router — a full-cov manifold per primitive over its context code
# ----------------------------------------------------------------------
class VocabularyRouter:
    """Per-primitive full-covariance Gaussian over the context code; route a
    query to argmax log-likelihood. Mirrors world_routing.route_eval but the
    classes are the four primitives and the code is the percept context slice."""

    def __init__(self, arena, *, full_cov=True):
        self.full_cov = full_cov
        self.archive = ManifoldArchive(arena, full_cov=full_cov)
        self.classes = []          # primitive index per archive class

    def fit(self, codes_by_prim):
        for i, name in enumerate(PRIM_ORDER):
            codes = codes_by_prim[name]
            self.archive.update_class(i, codes)
            self.classes.append(i)
        return self

    def _scores(self, codes):
        """[N,14] -> [N, n_prim] log-likelihood under each primitive's manifold."""
        lls = []
        for i in self.classes:
            astro = self.archive.get(i)
            ll = (astro.log_likelihood_full(codes) if self.full_cov
                  else astro.log_likelihood(codes))
            lls.append(ll)
        return torch.stack(lls, dim=1)

    @torch.no_grad()
    def route(self, percept):
        """A single 77-d percept -> primitive index (argmax LL)."""
        return int(self._scores(_ctx(percept)).argmax(dim=1)[0])

    @torch.no_grad()
    def route_batch(self, percepts):
        return self._scores(_ctx(percepts)).argmax(dim=1)


# ----------------------------------------------------------------------
# The vocabulary organism — router + four donors, one action per step
# ----------------------------------------------------------------------
class VocabularyOrganism:
    """Manifold-routed: picks the primitive from the PERCEPT alone (deployable)."""

    def __init__(self, donors, router):
        self.donors = donors           # {name: substrate}
        self.router = router
        self.route_hist = [0] * len(PRIM_ORDER)

    @torch.no_grad()
    def act(self, w, p):
        i = self.router.route(p)
        self.route_hist[i] += 1
        sub = self.donors[PRIM_ORDER[i]]
        return int(sub(_solo(p.unsqueeze(0)))[0].argmax())


class ArbiterOrganism:
    """Upper bound: routes by the hand-coded argmax-danger teacher (privileged
    world state). Shows the survival ceiling the manifold router is chasing — if
    the router matches this, the learned selector has captured the arbitration."""

    def __init__(self, donors):
        self.donors = donors
        self.route_hist = [0] * len(PRIM_ORDER)

    @torch.no_grad()
    def act(self, w, p):
        name = argmax_danger(w)
        self.route_hist[PRIM_ORDER.index(name)] += 1
        return int(self.donors[name](_solo(p.unsqueeze(0)))[0].argmax())


# ----------------------------------------------------------------------
# Router data — argmax-danger-labelled states from exploratory rollouts.
# Random rollouts die of every cause and so visit every drive's danger range;
# we label each state by its single most-urgent drive. min_danger excludes the
# all-safe states (full drives, no predator) where there is no urgency signal.
# ----------------------------------------------------------------------
@torch.no_grad()
def collect_router_states(*, seeds=120, per_prim_cap=1500, max_steps=300,
                          min_danger=0.15):
    buckets = {name: [] for name in PRIM_ORDER}
    g = torch.Generator().manual_seed(12345)
    for ep in range(seeds):
        w = TileWorld(seed=ep * 17 + 5, max_steps=max_steps)
        p = w.percept(); done = False
        while not done:
            d = danger(w)
            lbl = max(PRIM_ORDER, key=lambda k: d[k])
            if d[lbl] >= min_danger and len(buckets[lbl]) < per_prim_cap:
                buckets[lbl].append(p)
            a = int(torch.randint(0, 6, (1,), generator=g))
            p, _, done, _ = w.step(a)
        if all(len(buckets[n]) >= per_prim_cap for n in PRIM_ORDER):
            break
    return {name: _ctx(torch.stack(v)) for name, v in buckets.items() if v}


# ----------------------------------------------------------------------
# Build: load donors, collect critical-state codes, fit router
# ----------------------------------------------------------------------
def build_vocabulary(*, full_cov=True, router_seeds=120, per_prim_cap=1500):
    donors = {}
    for name in PRIM_ORDER:
        path = PRIM_DIR / f"{name}.pt"
        if not path.exists():
            raise FileNotFoundError(f"missing donor {path}; run primitives.py first")
        donors[name], _ = load_donor(path)
    codes_by_prim = collect_router_states(seeds=router_seeds,
                                          per_prim_cap=per_prim_cap)
    missing = [n for n in PRIM_ORDER if n not in codes_by_prim]
    if missing:
        raise ValueError(f"no critical states collected for {missing}; raise seeds")
    # the router's astrocytes can ride any donor's arena (non-forward, harmless)
    arena = next(iter(donors.values())).arena
    router = VocabularyRouter(arena, full_cov=full_cov).fit(codes_by_prim)
    return VocabularyOrganism(donors, router), codes_by_prim


# ----------------------------------------------------------------------
# Diagnostic 1 — routing accuracy on held-out band codes
# ----------------------------------------------------------------------
def routing_accuracy(codes_by_prim, *, full_cov, split=0.7, arena):
    router = VocabularyRouter(arena, full_cov=full_cov)
    test = {}
    train = {}
    for name in PRIM_ORDER:
        codes = codes_by_prim[name]
        ntr = int(split * codes.shape[0])
        train[name] = codes[:ntr]
        test[name] = codes[ntr:]
    router.fit(train)
    correct = 0; total = 0; recall = {}
    for i, name in enumerate(PRIM_ORDER):
        tc = test[name]
        if tc.shape[0] == 0:
            recall[name] = float("nan"); continue
        pred = router._scores(tc).argmax(dim=1)
        hit = int((pred == i).sum())
        correct += hit; total += tc.shape[0]
        recall[name] = hit / tc.shape[0]
    return correct / max(total, 1), recall


# ----------------------------------------------------------------------
# Full run
# ----------------------------------------------------------------------
def _p(*a):
    print(*a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--router-seeds", type=int, default=120)
    ap.add_argument("--per-prim-cap", type=int, default=1500)
    ap.add_argument("--eval-seeds", type=int, default=40)
    ap.add_argument("--diagonal", action="store_true", help="diagonal Gaussian router")
    args = ap.parse_args()

    full_cov = not args.diagonal
    if args.smoke:
        args.router_seeds, args.per_prim_cap, args.eval_seeds = 30, 300, 6

    _p("=== PHASE 2: vocabulary organism + manifold router (v2.0 core) ===")
    _p(f"router = {'full-cov' if full_cov else 'diagonal'} over 14-d context code\n")

    org, codes = build_vocabulary(full_cov=full_cov, router_seeds=args.router_seeds,
                                  per_prim_cap=args.per_prim_cap)
    arena = next(iter(org.donors.values())).arena

    # [1] routing accuracy (held-out band codes)
    n_prim = len(PRIM_ORDER)
    acc_f, rec_f = routing_accuracy(codes, full_cov=True, arena=arena)
    acc_d, rec_d = routing_accuracy(codes, full_cov=False, arena=arena)
    _p(f"[1] ROUTING accuracy (chance {1/n_prim:.2f}):")
    _p(f"      full-cov {acc_f:.3f}   diagonal {acc_d:.3f}")
    _p("      per-primitive recall (full-cov): " +
       "  ".join(f"{k}={rec_f[k]:.2f}" for k in PRIM_ORDER))

    # [2] survival: manifold-routed organism vs the argmax-danger arbiter (the
    # routing upper bound) vs floors / single donors / masters
    _p("\n[2] SURVIVAL (steps alive, max 300):")
    surv, _, _ = evaluate(lambda w, p: org.act(w, p), "VOCABULARY (manifold-route)",
                          seeds=args.eval_seeds)
    tot = max(sum(org.route_hist), 1)
    _p("      route usage: " +
       "  ".join(f"{PRIM_ORDER[i]}={100*org.route_hist[i]/tot:.0f}%"
                 for i in range(n_prim)))
    arb = ArbiterOrganism(org.donors)
    arb_surv, _, _ = evaluate(lambda w, p: arb.act(w, p),
                              "ARBITER (argmax-danger)", seeds=args.eval_seeds)
    tota = max(sum(arb.route_hist), 1)
    _p("      route usage: " +
       "  ".join(f"{PRIM_ORDER[i]}={100*arb.route_hist[i]/tota:.0f}%"
                 for i in range(n_prim)))

    # floors + best single donor + a master, for context
    rnd = statistics.mean(run_random(0, episodes=args.eval_seeds))
    rea = statistics.mean(run_reactive(0, episodes=args.eval_seeds))
    _p(f"      floors: random {rnd:.1f}   reactive {rea:.1f}")
    for name in PRIM_ORDER:
        d = org.donors[name]
        evaluate(lambda w, p, m=d: int(m(_solo(p.unsqueeze(0)))[0].argmax()),
                 f"{name}-donor solo", seeds=args.eval_seeds)
    for mname, mfn in (("fire", fire_oracle), ("water", water_master),
                       ("food", food_master), ("evade", evade_master)):
        evaluate(lambda w, p, f=mfn: f(w), f"{mname}-master", seeds=args.eval_seeds)

    _p(f"\n  WIN = vocabulary survival {surv:.1f} beats the best single donor "
       f"(and aims at the masters).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
