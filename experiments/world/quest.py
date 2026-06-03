"""The Quest for Immortality — a chosen protagonist apprentices to many masters.

Premise (Rocky 2026-06-03): no single master is complete. The fire-oracle is a
temperature virtuoso but dies of thirst (~83). A water-virtuoso dies of cold.
Each master is a SPECIALIST. The protagonist apprentices to all of them and
INTEGRATES their skills — and so should outlast every one of its masters. It need
not be immortal; it must last longer than those who taught it.

This file (chapter 1) builds the masters and CROWNS the protagonist — the best
learner among the candidates, saved with a persistent identity so it can carry
its accumulated skills across the apprenticeship chapters that follow.
"""
from __future__ import annotations

import sys
import statistics
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# importing fire_taming sets the tamed-fire physics + gives us the fire master
from experiments.world.fire_taming import fire_oracle, evaluate, TileWorld
from experiments.world.tile_world import (
    ACTIONS, WATER, FOOD, BERRY, FIRE,
)
from experiments.world.render_organism import train_solo
from experiments.world.mirror_cells import build_mirror, _solo

RUNS = Path(__file__).resolve().parent.parent.parent / "runs"


# ----------------------------------------------------------------------
# The masters — each a virtuoso of ONE drive, naive about the rest
# ----------------------------------------------------------------------
def _toward(w, tile):
    d = w._scent(tile)
    if float(d.abs().sum()) == 0:
        return None
    if abs(float(d[0])) >= abs(float(d[1])):
        return ACTIONS.index("E" if d[0] > 0 else "W")
    return ACTIONS.index("S" if d[1] > 0 else "N")


def water_master(w):
    """Hydration virtuoso: keep thirst brimming; ignore temp/predator."""
    here = int(w.grid[w.py, w.px])
    if here == WATER and w.thirst < 0.95:
        return ACTIONS.index("consume")
    if w.thirst < 0.85:
        t = _toward(w, WATER)
        if t is not None:
            return t
    if here in (FOOD, BERRY) and w.energy < 0.7:   # cheap opportunistic bite
        return ACTIONS.index("consume")
    return int(torch.randint(0, 4, (1,)))


def food_master(w):
    """Foraging virtuoso: keep energy brimming; ignore temp/predator."""
    here = int(w.grid[w.py, w.px])
    if here in (FOOD, BERRY) and w.energy < 0.95:
        return ACTIONS.index("consume")
    if w.energy < 0.85:
        t = _toward(w, FOOD)
        if t is not None:
            return t
    if here == WATER and w.thirst < 0.7:           # cheap opportunistic drink
        return ACTIONS.index("consume")
    return int(torch.randint(0, 4, (1,)))


MASTERS = {"fire": fire_oracle, "water": water_master, "food": food_master}


# ----------------------------------------------------------------------
# Crown the protagonist — the best LEARNER among the candidates
# ----------------------------------------------------------------------
def crown_protagonist(seeds=range(5), episodes=150):
    print("Selecting the chosen one — screening candidates by learning aptitude")
    print("(solo TD, tamed-fire world; the best survivor earns the crown):\n")
    results = []
    for seed in seeds:
        sub = train_solo(seed=seed, episodes=episodes)
        s, _, _ = evaluate(lambda w, p, m=sub: int(m(_solo(p.unsqueeze(0)))[0].argmax()),
                           f"candidate seed={seed}", seeds=30)
        results.append((seed, s, sub))
    best_seed, best_surv, best_sub = max(results, key=lambda r: r[1])
    torch.save({"seed": best_seed,
                "bias": best_sub.arena.bias.detach().clone(),
                "edge_weight": best_sub.arena.edge_weight.detach().clone(),
                "skills": [], "name": None,
                "baseline_survival": best_surv},
               RUNS / "protagonist.pt")
    return best_seed, best_surv, results


def main():
    print("=== THE QUEST FOR IMMORTALITY — chapter 1: the masters & the crown ===\n")
    print("The masters (each a specialist; note how each dies of a DIFFERENT cause):")
    bars = {}
    for name, fn in MASTERS.items():
        s, causes, _ = evaluate(lambda w, p, f=fn: f(w), f"{name}-master", seeds=40)
        bars[name] = s
    best_master = max(bars, key=bars.get)
    print(f"\n  → the bar to beat ('outlast your masters'): "
          f"{best_master}-master at {bars[best_master]:.1f} steps\n")

    best_seed, best_surv, _ = crown_protagonist()
    print(f"\n=== THE CHOSEN ONE ===")
    print(f"  crowned: organism seed={best_seed}  (raw solo survival {best_surv:.1f})")
    print(f"  saved → runs/protagonist.pt  (skills=[], unnamed)")
    print(f"  the quest: apprentice it to fire/water/food masters (DAgger) and")
    print(f"  integrate, until it outlasts the best master ({bars[best_master]:.1f}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
