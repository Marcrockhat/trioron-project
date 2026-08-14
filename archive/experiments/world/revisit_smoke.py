"""Revisit smoke (2026-08-14) — does ONE organism on the CURRENT core still
absorb skills and outlive its baseline (and ideally its masters)?

Not a verdict run: single seed, reduced episodes. The question is validity on
the current `conscience-core` substrate (the world arc predates two months of
core evolution — the 0.76→0.71 era-drift showed old claims need re-running).

Protocol: masters' bars → train solo → eval → DAgger fire → eval → DAgger
water → eval. Read: survival trend + cause-of-death shifts (cold falls after
fire, thirst falls after water, neither regresses after the other).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# importing fire_taming sets the tamed-fire physics
from experiments.world.fire_taming import evaluate
from experiments.world.render_organism import train_solo
from experiments.world.mirror_cells import _solo
from experiments.world.quest import MASTERS as _QUEST_MASTERS
from experiments.world.quest_chapter import dagger_resume
from experiments.world.primitives import evade_master, cool_flee_master, _pred_delta
from experiments.world.fire_taming import _near_fire, _toward, _away_from_fire
from experiments.world.primitives import _flee_pred
from experiments.world.tile_world import ACTIONS, FIRE, WATER, FOOD, BERRY


def perfect_master(w):
    """The complete master — every specialist's wisdom under one urgency
    arbiter, fully deterministic (no random walk, so every state is learnable):
      1. never stand on the flame;
      2. flee a close predator (unless freezing is the nearer death);
      3. tap the stove: soak only while truly cold, leave EARLY (temp 0.35,
         not the fire-oracle's 0.3+overshoot band);
      4. drink/eat what you stand on when below brim;
      5. seek the most urgent resource (temp, then thirst-vs-energy);
      6. fallback: top up water (the renewable tile), else rest."""
    here = int(w.grid[w.py, w.px])
    if here == FIRE:
        return _away_from_fire(w)
    dx, dy = _pred_delta(w)
    if max(abs(dx), abs(dy)) <= 2 and w.temp > 0.15:
        return _flee_pred(w)
    if _near_fire(w):
        if w.temp < 0.35:
            return ACTIONS.index("rest")
        return _away_from_fire(w)
    if here == WATER and w.thirst < 0.85:
        return ACTIONS.index("consume")
    if here in (FOOD, BERRY) and w.energy < 0.85:
        return ACTIONS.index("consume")
    if w.temp < 0.45 and w.thirst > 0.25 and w.energy > 0.25:
        t = _toward(w, FIRE)
        if t is not None:
            return t
    if w.thirst <= w.energy and w.thirst < 0.7:
        t = _toward(w, WATER)
        if t is not None:
            return t
    if w.energy < 0.7:
        t = _toward(w, FOOD)
        if t is not None:
            return t
    t = _toward(w, WATER)                    # deterministic fallback: stage at water
    return t if t is not None else ACTIONS.index("rest")


MASTERS = dict(_QUEST_MASTERS, evade=evade_master, flee=cool_flee_master,
               perfect=perfect_master)


def _eval(sub, label, seeds):
    return evaluate(lambda w, p: int(sub(_solo(p.unsqueeze(0)))[0].argmax()),
                    label, seeds=seeds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--solo-eps", type=int, default=120)
    ap.add_argument("--chap-eps", type=int, default=120)
    ap.add_argument("--eval-seeds", type=int, default=40)
    ap.add_argument("--n-mirror", type=int, default=8)
    ap.add_argument("--chain", type=str, default="fire,water",
                    help="comma list of masters to apprentice to, in order")
    args = ap.parse_args()
    chain = [c.strip() for c in args.chain.split(",") if c.strip()]
    assert all(c in MASTERS for c in chain), f"unknown master in {chain}"

    print("=== REVISIT SMOKE — skill absorption on the current core ===")
    print(f"seed={args.seed} solo={args.solo_eps} chap={args.chap_eps} "
          f"eval_seeds={args.eval_seeds} n_mirror={args.n_mirror} "
          f"chain={chain}\n")

    print("The masters (the bars):")
    bars = {}
    for name in dict.fromkeys(["fire", "water", "food"] + chain):
        s, _, _ = evaluate(lambda w, p, f=MASTERS[name]: f(w), f"{name}-master",
                           seeds=args.eval_seeds)
        bars[name] = s
    best = max(bars, key=bars.get)
    print(f"  → best master: {best} at {bars[best]:.1f}\n")

    print(f"training solo ({args.solo_eps} eps)...")
    sub = train_solo(args.seed, args.solo_eps, n_mirror=args.n_mirror)
    print(f"  organism: {sub.n_cells} cells, {sub.n_edges} edges")
    s, c, _ = _eval(sub, "solo baseline", args.eval_seeds)
    stages = [("solo", s, c)]

    for name in chain:
        print(f"\napprenticing to {name.upper()} master ({args.chap_eps} eps)...")
        sub = dagger_resume(sub, MASTERS[name], seed=args.seed,
                            episodes=args.chap_eps)
        s, c, _ = _eval(sub, f"after {name.upper():<6s}", args.eval_seeds)
        stages.append((f"+{name}", s, c))

    print("\n=== VERDICT (single seed — direction, not proof) ===")
    print("  survival: " + " → ".join(f"{n} {v:.1f}" for n, v, _ in stages)
          + f"   (best master bar: {bars[best]:.1f})")
    for cause in ("cold", "overheat", "thirst", "energy", "integrity"):
        row = " → ".join(f"{c.get(cause, 0):>2d}" for _, _, c in stages)
        print(f"  {cause:>9s} deaths: {row}")
    grew = stages[-1][1] > stages[0][1]
    beat = stages[-1][1] > bars[best]
    print(f"\n  absorbs skills (survival grew): {'YES' if grew else 'NO'}"
          f"   outlasts best master: {'YES' if beat else 'NO'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
