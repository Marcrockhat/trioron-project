"""Render the s049 showcase GIF: phasecyte wake-nest vs the recorded DQN.

The honest artifact (n=3 verdict): in-world the dream lift is a wash
(+2.0±3.8), but the WAKE nest beats the DQN control on every seed
(47.4±4.6 vs 37.9) — one gradient-free pass per skill vs 300 training
episodes. This renders that duel on the same unseen map. The DQN is the
s048 checkpoint (archive/runs/duel/dqn.pt), NOT retrained (Rocky's
ruling: the control is recorded).

Run (any cwd): python3 <abs>/render_phasecyte_duel.py [--map-seed N]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse

import torch

from experiments.world import fire_taming as _ft          # tamed physics
from experiments.world.watch_duel import RUNS, DUEL, record, render
from experiments.world.dqn_baseline import QNet
from experiments.world.world_phasecyte import (
    NestOrganism, WakeLeafPolicy, build_wake_nest)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map-seed", type=int, default=987654)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    nest = build_wake_nest(args.seed)
    wake = NestOrganism(nest, [WakeLeafPolicy(nest.leaves[g])
                               for g in sorted(nest.leaves)])
    q = QNet()
    q.load_state_dict(torch.load(DUEL / "dqn.pt"))

    frames_a = record(lambda w, p: wake.act(w, p), args.map_seed)
    frames_b = record(
        lambda w, p: (int(q(p.unsqueeze(0))[0].argmax()), "-"),
        args.map_seed)
    out = RUNS / f"phasecyte_vs_dqn_map{args.map_seed}.gif"
    n = render(frames_a, frames_b, out,
               title_a="PHASECYTE NEST (5 skills, one gradient-free pass)",
               title_b="FLAT DQN (300 training episodes)")
    print(f"[gif] {out} ({n} frames)  nest t={frames_a[-1]['t']} "
          f"dqn t={frames_b[-1]['t']}")


if __name__ == "__main__":
    main()
