"""Autopsy — what kills the organism? Predator, or its own foraging failure?

Instruments greedy rollouts: cause of death (which drive crossed its lethal
band), how many times the predator actually touched it and how much integrity
that cost, and how often it chose to consume. Compares the mirror-solo organism
against the hand-coded reactive floor (which DOES drink/eat) and v1.
"""
from __future__ import annotations

import sys
import statistics
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.world.tile_world import TileWorld, N_ACTION, ACTIONS
from experiments.world.mirror_cells import build_mirror, _solo
from experiments.world.render_organism import train_solo
from experiments.world.organism_v1 import train_organism, eval_greedy as v1_eval


def cause_of_death(w):
    if not w.alive:
        if w.thirst <= 0: return "thirst"
        if w.energy <= 0: return "energy"
        if w.integrity <= 0: return "integrity(predator/poison/fire)"
        if w.temp <= 0.05: return "cold"
        if w.temp >= 0.98: return "overheat"
        return "unknown"
    return "timeout(survived max)"


@torch.no_grad()
def autopsy(act_fn, label, seeds=30, max_steps=300):
    causes = Counter()
    survivals, pred_hits, integ_lost, consumes = [], [], [], []
    for ep in range(seeds):
        w = TileWorld(seed=ep * 7000 + 7, max_steps=max_steps)
        p = w.percept(); done = False
        hits = 0; lost = 0.0; eats = 0
        while not done:
            a = act_fn(w, p)
            if ACTIONS[a] == "consume": eats += 1
            integ_before = w.integrity
            on_pred = (w.pred[0] == w.px and w.pred[1] == w.py)
            p, _, done, info = w.step(a)
            if on_pred:                      # predator was on us this step
                hits += 1
                lost += max(0.0, integ_before - w.integrity)
        causes[cause_of_death(w)] += 1
        survivals.append(info["t"]); pred_hits.append(hits)
        integ_lost.append(lost); consumes.append(eats)
    print(f"\n=== {label} (n={seeds}) ===")
    print(f"  survival       {statistics.mean(survivals):5.1f} ± "
          f"{statistics.stdev(survivals):.1f}")
    print(f"  consume actions{statistics.mean(consumes):5.1f}/episode")
    print(f"  predator hits  {statistics.mean(pred_hits):5.1f}/episode  "
          f"(integrity lost to predator {statistics.mean(integ_lost):.2f}, "
          f"lethal at 1.0)")
    print(f"  cause of death: " +
          ", ".join(f"{k}={v}" for k, v in causes.most_common()))


def main():
    print("Training organisms for autopsy (mirror-solo 250ep, v1 250ep)...")
    mirror = build_mirror(0, n_mirror=8)
    mirror = train_solo(seed=0, episodes=250)
    _, v1 = train_organism(0, 250)

    def mirror_act(w, p):
        return int(mirror(_solo(p.unsqueeze(0)))[0].argmax())

    def v1_act(w, p):
        return int(v1(p.unsqueeze(0))[0].argmax())

    # reactive baseline: reuse the world's hand-coded policy inline
    from experiments.world.tile_world import _DXY, FOOD, WATER, BERRY, FIRE
    def reactive_act(w, p):
        s = w.size
        def toward(tile):
            d = w._scent(tile)
            if float(d.abs().sum()) == 0: return None
            if abs(float(d[0])) >= abs(float(d[1])):
                return ACTIONS.index("E" if d[0] > 0 else "W")
            return ACTIONS.index("S" if d[1] > 0 else "N")
        here = int(w.grid[w.py, w.px])
        pred_adj = abs(w.pred[0]-w.px) <= 1 and abs(w.pred[1]-w.py) <= 1
        a = None
        if pred_adj and w.integrity < 0.8:
            a = ACTIONS.index("E" if w.pred[0] <= w.px else "W")
        elif w.temp < 0.3:
            a = toward(FIRE)
        elif here == WATER and w.thirst < 0.7:
            a = ACTIONS.index("consume")
        elif here in (FOOD, BERRY) and w.energy < 0.7:
            a = ACTIONS.index("consume")
        if a is None:
            if w.thirst <= w.energy and w.thirst < 0.6: a = toward(WATER)
            elif w.energy < 0.6: a = toward(FOOD)
        if a is None: a = int(torch.randint(0, 4, (1,)))
        return a

    autopsy(reactive_act, "reactive (hand-coded, drinks/eats)")
    autopsy(v1_act, "organism_v1 (the 43.8 workhorse)")
    autopsy(mirror_act, "mirror-solo (the goldfish in the GIF)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
