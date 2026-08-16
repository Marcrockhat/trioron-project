"""Optimized-row showcase GIF: TD trioron nest (checkpoint) vs the
10x-budget DQN (dqn3000.pt). Both sides at their best-trained state —
the visual companion to the page's 'optimized budget' row.

Run (any cwd), after dqn3000.pt exists:
  python3 <abs>/render_optimized_duel.py [--map-seed N]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse

import torch

from experiments.world import fire_taming as _ft          # tamed physics
from experiments.world.dqn_baseline import QNet
from experiments.world.mirror_cells import _solo
from experiments.world.vocabulary import PRIM_ORDER
from experiments.world.watch_duel import (DUEL, RUNS, get_agents, record,
                                          render)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map-seed", type=int, default=987654)
    args = ap.parse_args()

    router, donors, _ = get_agents(seed=0)      # router_td.pt fast path
    q = QNet()
    q.load_state_dict(torch.load(DUEL / "dqn3000.pt"))

    def act_nest(w, p):
        i = int(router(p.unsqueeze(0))[0].argmax())
        leaf = donors[PRIM_ORDER[i]]
        return (int(leaf(_solo(p.unsqueeze(0)))[0].argmax()),
                PRIM_ORDER[i])

    frames_a = record(act_nest, args.map_seed)
    frames_b = record(
        lambda w, p: (int(q(p.unsqueeze(0))[0].argmax()), "-"),
        args.map_seed)
    for fps, tag in ((5, "_slow"), (10, ""), (20, "_fast")):
        out = RUNS / f"optimized_vs_dqn3000_map{args.map_seed}{tag}.gif"
        n = render(frames_a, frames_b, out, fps=fps,
                   title_a="TD TRIORON NEST (consequence-taught router)",
                   title_b="FLAT DQN (3000 training episodes, 10x)")
        print(f"[gif] {out} ({n} frames, {fps} fps)  "
              f"nest t={frames_a[-1]['t']} dqn t={frames_b[-1]['t']}")


if __name__ == "__main__":
    main()
