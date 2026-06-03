"""Capacity control — is "one skill fills the mirror cells" a LOW-cell artifact?

Rocky (2026-06-03): the saturation we saw at 8 mirror cells is EXPECTED when
cells are few; it would be UNEXPECTED if it persisted with many cells. So sweep
mirror-cell count over the SAME protocol (solo warmup → fire DAgger → water
DAgger) and ask: at high capacity, does the second skill (water) finally take?

  - water TAKES at high N  → it was capacity; growth is the principled fix.
  - water STILL FAILS at high N → not capacity; deeper (shared-pathway
    interference / credit gating / real fire-water conflict). The surprising case.

Signals per capacity:
  water learned?  → thirst-deaths fall  (after-fire → after-water)
  fire retained?  → cold-deaths stay low (after-fire → after-water)
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.world.fire_taming import evaluate, TileWorld     # sets tamed-fire physics
from experiments.world.render_organism import train_solo
from experiments.world.quest_chapter import dagger_resume
from experiments.world.quest import MASTERS
from experiments.world.mirror_cells import _solo, mirror_ids

SEED = 2   # Adam's seed — these are "Adam with a bigger brain"


def _eval(sub, label):
    return evaluate(lambda w, p: int(sub(_solo(p.unsqueeze(0)))[0].argmax()),
                    label, seeds=40)


def run_capacity(n_mirror, solo_eps=120, chap_eps=200):
    print(f"\n########## mirror cells = {n_mirror} ##########")
    sub = train_solo(SEED, solo_eps, n_mirror=n_mirror)
    print(f"  ({len(mirror_ids(sub))} mirror cells, {sub.n_edges} edges)")
    _eval(sub, f"n={n_mirror} baseline")
    sub = dagger_resume(sub, MASTERS["fire"], seed=SEED, episodes=chap_eps)
    _, fc, ffire = _eval(sub, f"n={n_mirror} after-FIRE ")
    sub = dagger_resume(sub, MASTERS["water"], seed=SEED, episodes=chap_eps)
    _, wc, wfire = _eval(sub, f"n={n_mirror} after-WATER")
    return {"n": n_mirror,
            "fire_cold": fc["cold"], "fire_thirst": fc["thirst"], "fire_occ": ffire,
            "water_cold": wc["cold"], "water_thirst": wc["thirst"], "water_occ": wfire}


def main():
    print("=== CAPACITY CONTROL: does more mirror cells let a 2nd skill fit? ===")
    rows = [run_capacity(n) for n in (8, 64, 128)]
    print("\n=== SUMMARY ===")
    print(f"  {'n_mirror':>8s} | {'thirst: fire→water':>20s} | "
          f"{'cold(fire retained?)':>22s} | {'water learned?':>14s}")
    for r in rows:
        learned = "YES" if r["water_thirst"] < r["fire_thirst"] - 2 else "no"
        retained = "yes" if r["water_cold"] <= r["fire_cold"] + 3 else "REGRESSED"
        print(f"  {r['n']:>8d} | {r['fire_thirst']:>8d} → {r['water_thirst']:<9d} | "
              f"{r['fire_cold']:>3d} → {r['water_cold']:<3d} {retained:>13s} | "
              f"{learned:>14s}")
    print("\n  water learned at high N → capacity was the wall (grow per skill).")
    print("  water still failing at high N → deeper than capacity.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
