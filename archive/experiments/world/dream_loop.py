"""Phase 3 of the Mode-E survival recipe — frustration -> dream self-improvement.

Phase 2 built a vocabulary organism that routes four primitives and outlasts
every single skill (72.5) and 3/4 masters. Its ceiling is one weak primitive:
WARM overheats (26/40 deaths) — its band omitted enough leave-before-overheat
practice, and the dominant death cause is overheat.

This is the loop that closes the gap, the same shape as the Pong dream loop
(-20 -> +1, above oracle): DEPLOY -> detect FRUSTRATION (the dominant death
cause) -> map it to the primitive at fault -> collect the organism's OWN failure
trajectories (the steps it took into that death) -> DREAM: relabel them with the
primitive's master (the correction it should have made) and consolidate them into
the donor, INTERLEAVED with replay of the original skill demos so the navigation
is not forgotten -> re-deploy. Repeat.

The thesis (quest.py): the integrated organism has no single blind spot — it
carries EVADE, which no resource-master has (fire-master dies of the predator) —
so once its weak primitive is sharpened it should OUTLAST every master.

Usage:
    python3 -m experiments.world.dream_loop --smoke
    python3 -m experiments.world.dream_loop --iters 4
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
from experiments.world.fire_taming import evaluate        # noqa: E402
from experiments.world.tile_world import TileWorld        # noqa: E402
from experiments.world.mirror_cells import _solo          # noqa: E402
from experiments.world.primitives import (                # noqa: E402
    PRIMITIVES, collect, save_donor,
)
from experiments.world.vocabulary import (                # noqa: E402
    build_vocabulary, PRIM_ORDER, ArbiterOrganism,
)

# which primitive a death-cause implicates (the frustration -> skill map)
DEATH_TO_PRIM = {
    "overheat": "WARM", "cold": "WARM",
    "thirst": "HYDRATE", "energy": "FORAGE", "integrity": "EVADE",
}


def _p(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------
# Frustration probe — survival + dominant death cause
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# Collect the organism's OWN failures, relabelled by the master (DAgger)
# ----------------------------------------------------------------------
@torch.no_grad()
def collect_failures(org, prim_name, *, cause, seeds=60, window=20,
                     max_steps=300, cap=900):
    """Run the organism; on episodes that die of `cause`, take the last `window`
    states and relabel with the failing primitive's master — the action it should
    have taken. These are the corrections the dream consolidates."""
    master = PRIMITIVES[prim_name]["master"]
    band = PRIMITIVES[prim_name]["band"]
    P, Y = [], []
    for ep in range(seeds):
        w = TileWorld(seed=ep * 131 + 17, max_steps=max_steps)
        p = w.percept(); done = False
        # record (percept, master-action, in-band) live — the master action and
        # band membership must be read at the state, before stepping
        steps = []
        while not done:
            steps.append((p, master(w), band(w)))
            a = org.act(w, p)
            p, _, done, info = w.step(a)
        if _cause(w) != cause:
            continue
        for pp, ma, inb in steps[-window:]:
            if inb:
                P.append(pp); Y.append(ma)
            if len(P) >= cap:
                break
        if len(P) >= cap:
            break
    if not P:
        return None, None
    return torch.stack(P), torch.tensor(Y)


# ----------------------------------------------------------------------
# The dream — consolidate corrections into the donor, replay the skill
# ----------------------------------------------------------------------
def dream_correct(donor, prim_name, corrections, *, replay_seeds=30,
                  replay_cap=600, epochs=120, lr=2e-3, batch=128,
                  corr_weight=3):
    """Fine-tune the failing donor on its master corrections, INTERLEAVED with
    replay of the original skill demos (anti-forgetting). corr_weight oversamples
    the corrections so the rare failure-mode decisions actually move the policy."""
    spec = PRIMITIVES[prim_name]
    Pc, Yc = corrections
    # skill replay (the original navigation demos) so the dream doesn't forget it
    Pr, Yr = collect(spec["master"], spec["band"], seeds=replay_seeds, cap=replay_cap)
    P = torch.cat([Pr] + [Pc] * corr_weight, dim=0)
    Y = torch.cat([Yr] + [Yc] * corr_weight, dim=0)

    opt = torch.optim.Adam(donor.trainable_tensors(), lr=lr)
    ce = torch.nn.functional.cross_entropy
    n = P.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            ce(donor(_solo(P[idx])), Y[idx]).backward()
            donor.zero_dormant_grads()
            opt.step()
    return donor


# ----------------------------------------------------------------------
# Donor snapshot / restore — for keep-best rollback
# ----------------------------------------------------------------------
def _snapshot(donor):
    return (donor.arena.bias.detach().clone(),
            donor.arena.edge_weight.detach().clone())


def _restore(donor, snap):
    with torch.no_grad():
        donor.arena.bias.copy_(snap[0])
        donor.arena.edge_weight.copy_(snap[1])


def _ranked_causes(causes):
    """Death causes with a mapped primitive, most-common first."""
    return [(c, n) for c, n in causes.most_common() if c in DEATH_TO_PRIM]


# ----------------------------------------------------------------------
# The loop — DAgger aggregation + keep-best rollback + plateau escalation
# ----------------------------------------------------------------------
def dream_loop(*, iters=6, eval_seeds=40, router_seeds=120, fail_seeds=60,
               window=20, accept_margin=2.0, plateau_patience=2, save=False):
    _ft.EXPLORE_DETERMINISTIC = False
    org, _ = build_vocabulary(router_seeds=router_seeds)

    # per-primitive AGGREGATED correction buffers (DAgger: never discard) and a
    # rejection counter that drives escalation to the next death cause.
    agg = {name: ([], []) for name in PRIM_ORDER}
    rejects = {name: 0 for name in PRIM_ORDER}

    best_surv, causes = probe(org, seeds=eval_seeds)
    best_snaps = {name: _snapshot(org.donors[name]) for name in PRIM_ORDER}
    curve = [best_surv]
    _p(f"[iter 0] survival {best_surv:.1f}   deaths {dict(causes.most_common())}")

    for it in range(1, iters + 1):
        # pick the dominant death cause whose primitive hasn't plateaued
        cand = [(c, n) for c, n in _ranked_causes(causes)
                if rejects[DEATH_TO_PRIM[c]] < plateau_patience]
        if not cand:
            _p(f"[iter {it}] every implicated primitive has plateaued — stop.")
            break
        dom, n = cand[0]
        prim = DEATH_TO_PRIM[dom]
        tag = "" if dom == _ranked_causes(causes)[0][0] else "  (escalated)"
        _p(f"[iter {it}] frustration: '{dom}' x{n} -> sharpen {prim}{tag}")

        Pc, Yc = collect_failures(org, prim, cause=dom, seeds=fail_seeds,
                                  window=window)
        if Pc is None:
            _p(f"           no '{dom}' failures captured — skip.")
            rejects[prim] += 1
            continue
        agg[prim][0].append(Pc); agg[prim][1].append(Yc)        # AGGREGATE
        Pa = torch.cat(agg[prim][0], dim=0); Ya = torch.cat(agg[prim][1], dim=0)
        _p(f"           +{Pc.shape[0]} frames -> {Pa.shape[0]} aggregated corrections")

        snap = _snapshot(org.donors[prim])                       # for rollback
        dream_correct(org.donors[prim], prim, (Pa, Ya))
        surv, new_causes = probe(org, seeds=eval_seeds)

        if surv >= best_surv - accept_margin:                    # ACCEPT
            verdict = "accept"
            if surv > best_surv:
                best_surv = surv
                best_snaps[prim] = _snapshot(org.donors[prim])
            rejects[prim] = 0
            causes = new_causes
        else:                                                    # ROLLBACK
            verdict = "reject->rollback"
            _restore(org.donors[prim], snap)
            rejects[prim] += 1
            # causes unchanged: same state, may re-try (bigger buffer) or escalate
        curve.append(surv)
        _p(f"           -> survival {surv:.1f}  [{verdict}, best {best_surv:.1f}]  "
           f"deaths {dict(new_causes.most_common())}")

    # restore the best donor set seen
    for name in PRIM_ORDER:
        _restore(org.donors[name], best_snaps[name])
    if save:
        for name in PRIM_ORDER:
            save_donor(org.donors[name], name, seed=0, n_mirror=8,
                       acc=-1.0, chance=-1.0, drive=PRIMITIVES[name]["drive"])
    _p(f"\n[final] restored best donor set — survival {best_surv:.1f}")
    return org, curve, best_surv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--eval-seeds", type=int, default=40)
    args = ap.parse_args()
    if args.smoke:
        args.iters, args.eval_seeds = 2, 8

    _p("=== PHASE 3: frustration -> dream self-improvement (v2.0 core) ===")
    _p("    DAgger aggregation + keep-best rollback + plateau escalation\n")
    org, curve, best = dream_loop(iters=args.iters, eval_seeds=args.eval_seeds,
                                  router_seeds=60 if args.smoke else 120,
                                  fail_seeds=20 if args.smoke else 60)

    _p(f"\n=== SURVIVAL TRAJECTORY ===")
    _p("    " + "  ->  ".join(f"{s:.1f}" for s in curve))
    _p(f"    start {curve[0]:.1f}  best {best:.1f}  lift {best - curve[0]:+.1f}")
    _p("    bars: best single donor 58.3 | fire-master 87.4 | reactive 141.8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
