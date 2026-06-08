"""Is the organism "worried" by temperature? — approach-avoidance conflict probe.

Rocky's hypothesis (2026-06-02): the maneuvering needed to avoid cold/heat
stresses the organism. Fire is the only warmth, but it burns (-integrity on the
tile) and overheats (warming +0.15/step vs cooling -0.008..-0.015 — asymmetric),
so when cold the policy is pulled toward fire AND away from it. That conflict
should show as INDECISION near fire: small action-value margins (top1 - top2)
and dithering (reversing direction step to step).

Measures Q-margin and direction-flip rate, binned by temp state × near-fire,
for a trained organism vs the random floor. A "worried" signature = margin
collapses / flips spike specifically when COLD and NEAR FIRE.

Saves/reuses runs/mirror_organism.pt so we stop retraining for every probe.
"""
from __future__ import annotations

import sys
import statistics
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.world.tile_world import TileWorld, N_ACTION, ACTIONS, FIRE
from experiments.world.mirror_cells import build_mirror, _solo
from experiments.world.render_organism import train_solo

CKPT = Path(__file__).resolve().parent.parent.parent / "runs" / "mirror_organism.pt"
_OPP = {0: 1, 1: 0, 2: 3, 3: 2}   # N<->S, E<->W


def get_organism(seed=0, episodes=250):
    sub = build_mirror(seed, n_mirror=8)
    if CKPT.exists():
        sd = torch.load(CKPT)
        with torch.no_grad():
            sub.arena.bias.copy_(sd["bias"])
            sub.arena.edge_weight.copy_(sd["edge_weight"])
        print(f"loaded organism from {CKPT.name}")
    else:
        print(f"training organism ({episodes} ep) → {CKPT.name} ...")
        sub = train_solo(seed=seed, episodes=episodes)
        torch.save({"bias": sub.arena.bias.detach().clone(),
                    "edge_weight": sub.arena.edge_weight.detach().clone()}, CKPT)
    return sub


def near_fire(w):
    s = w.size
    return any(int(w.grid[(w.py + dy) % s, (w.px + dx) % s]) == FIRE
               for dy in (-1, 0, 1) for dx in (-1, 0, 1))


def temp_state(t):
    return "cold" if t < 0.3 else ("hot" if t > 0.7 else "ok")


@torch.no_grad()
def probe(sub, label, seeds=40, max_steps=300):
    # bins keyed by (temp_state, near_fire) → lists
    margins = {}
    flips = {}
    counts = {}
    for ts in ("cold", "ok", "hot"):
        for nf in (True, False):
            margins[(ts, nf)] = []
            flips[(ts, nf)] = []
            counts[(ts, nf)] = 0
    for ep in range(seeds):
        w = TileWorld(seed=ep * 7000 + 7, max_steps=max_steps)
        p = w.percept(); done = False; prev_move = None
        while not done:
            q = sub(_solo(p.unsqueeze(0)))[0]
            top = torch.topk(q, 2).values
            margin = float(top[0] - top[1])
            a = int(q.argmax())
            key = (temp_state(w.temp), near_fire(w))
            counts[key] += 1
            margins[key].append(margin)
            move = a if a < 4 else None
            is_flip = (prev_move is not None and move is not None
                       and move == _OPP.get(prev_move))
            flips[key].append(1.0 if is_flip else 0.0)
            if move is not None:
                prev_move = move
            p, _, done, info = w.step(a)
    print(f"\n=== {label} ===")
    print(f"  {'state':>14s} | {'n':>6s} | {'Q-margin':>9s} | {'dither%':>7s}")
    for ts in ("cold", "ok", "hot"):
        for nf in (True, False):
            k = (ts, nf); n = counts[k]
            if n == 0:
                continue
            m = statistics.mean(margins[k])
            fl = 100 * statistics.mean(flips[k])
            tag = f"{ts}/{'near-fire' if nf else 'no-fire'}"
            print(f"  {tag:>14s} | {n:6d} | {m:9.3f} | {fl:6.1f}%")


def main():
    sub = get_organism()
    probe(sub, "trained organism (margin small + dither high near fire = 'worried')")

    class RandPolicy:
        def __call__(self, x):
            return torch.randn(1, N_ACTION)
    probe(RandPolicy(), "random floor (reference: no structure)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
