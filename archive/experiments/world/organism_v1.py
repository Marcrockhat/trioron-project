"""Rung 1 — the organism survives the world. (2026-06-01)

The first grounded "is it alive at all" test (docs/design/organism_roadmap.md).
A trioron substrate acts in the mini-Minecraft world and must beat the honest
baselines (random 15.1, reactive 20.4 steps).

Learning = predict-then-act on VALUE. A pure 1-step reward predictor can't
navigate (moving toward food has no immediate reward — only consuming does), so
the organism predicts the discounted future drive-satisfaction of each action
(a small Q-learner; the trioron is the value net) and acts greedily on it. Still
label-free: the only signal is the drive-change the world returns.

v1 keeps the organism SIMPLE — a plain linear trioron value net, no satellites/
quad/growth yet. If survival demands memory or comparison, we ADD them and show
they earn their keep (the heterogeneous-substrate discipline). Start simple,
make it survive, then complicate only on evidence.
"""
from __future__ import annotations

import argparse
import statistics
import sys
from collections import deque
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from trioron.core import Envelope, construct
from trioron.bases import seeded
from trioron.phenotype import default_dispatch_table
from experiments.world.tile_world import (
    TileWorld, N_ACTION, run_random, run_reactive,
)

PERCEPT_DIM = 77   # matches TileWorld.percept() (scent + predator scent)


def build(seed, nonlinear=False):
    torch.manual_seed(seed)
    sub = construct(base=seeded(PERCEPT_DIM, N_ACTION, interior_cells=32,
                                nonlinear=nonlinear),
                    envelope=Envelope(max_parameter_bytes=400_000),
                    dispatch_table=default_dispatch_table(), capacity=2048, sparsity_k=0)
    sub.compile(); sub.prepare_training()
    return sub


def train_organism(seed, episodes, *, gamma=0.95, lr=3e-3, batch=64,
                   eps_start=1.0, eps_end=0.1, nonlinear=False, max_steps=300,
                   verbose=False):
    sub = build(seed, nonlinear=nonlinear)
    opt = torch.optim.Adam(sub.trainable_tensors(), lr=lr)
    buf = deque(maxlen=20000)
    g = torch.Generator().manual_seed(seed + 31)
    lengths = []

    for ep in range(episodes):
        w = TileWorld(seed=seed * 1000 + ep, max_steps=max_steps)
        p = w.percept()
        eps = eps_end + (eps_start - eps_end) * max(0.0, 1 - ep / (0.7 * episodes))
        done = False
        while not done:
            if torch.rand(1, generator=g).item() < eps:
                a = int(torch.randint(0, N_ACTION, (1,), generator=g))
            else:
                with torch.no_grad():
                    a = int(sub(p.unsqueeze(0))[0].argmax())
            p2, r, done, info = w.step(a)
            buf.append((p, a, r, p2, float(done)))
            p = p2
            # learn (TD on discounted value)
            if len(buf) >= batch:
                idx = torch.randint(0, len(buf), (batch,), generator=g)
                bp = torch.stack([buf[i][0] for i in idx])
                ba = torch.tensor([buf[i][1] for i in idx])
                br = torch.tensor([buf[i][2] for i in idx])
                bp2 = torch.stack([buf[i][3] for i in idx])
                bd = torch.tensor([buf[i][4] for i in idx])
                q = sub(bp)[torch.arange(batch), ba]
                with torch.no_grad():
                    q_next = sub(bp2).max(dim=1).values
                    target = br + gamma * q_next * (1 - bd)
                loss = torch.nn.functional.mse_loss(q, target)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(sub.trainable_tensors(), 1.0)
                sub.zero_dormant_grads(); opt.step()
        lengths.append(info["t"])
        if verbose and (ep % max(1, episodes // 10) == 0 or ep == episodes - 1):
            recent = statistics.mean(lengths[-20:])
            print(f"    ep {ep:4d} eps={eps:.2f} survival(last20)={recent:.1f}")
    return lengths, sub


@torch.no_grad()
def eval_greedy(sub, seed, episodes=30, max_steps=300):
    """Survival with the trained policy, NO exploration — the fair number."""
    lengths = []
    for ep in range(episodes):
        w = TileWorld(seed=seed * 7000 + ep, max_steps=max_steps)
        p = w.percept(); done = False
        while not done:
            a = int(sub(p.unsqueeze(0))[0].argmax())
            p, _, done, info = w.step(a)
        lengths.append(info["t"])
    return lengths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--nonlinear", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.episodes = 1, 120

    print(f"organism_v1: rung 1 — survive the world. seeds={args.seeds} "
          f"episodes={args.episodes} nonlinear={args.nonlinear}")
    print("baselines (difficulty-reduced world, n=60): random≈27.9, reactive≈38.5")

    final = []
    for seed in range(args.seeds):
        L, sub = train_organism(seed, args.episodes, nonlinear=args.nonlinear,
                                verbose=args.smoke)
        greedy = eval_greedy(sub, seed, episodes=30)
        gm = statistics.mean(greedy)
        early = statistics.mean(L[:30])
        final.append(gm)
        print(f"  [seed {seed}] train: early(first30)={early:.1f} → "
              f"late(last30)={statistics.mean(L[-30:]):.1f}  |  "
              f"GREEDY eval={gm:.1f} (max {max(greedy)})")
    m = statistics.mean(final); s = statistics.stdev(final) if len(final) > 1 else 0.0
    verdict = "ALIVE (beats reactive)" if m > 38.5 else (
        "beats random, not reactive" if m > 27.9 else "fails (≈random)")
    print(f"\n  === organism late survival: {m:.1f}±{s:.1f}  [{verdict}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
