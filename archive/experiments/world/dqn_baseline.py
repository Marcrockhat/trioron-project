"""The control we owe ourselves (2026-08-14): vanilla flat DQN vs the nest.

Same world (tamed-fire physics), same 77-d percept, same reward (drive delta),
same budget as the trioron router runs (300 episodes, gamma .95, lr 3e-3,
batch 64, buffer 20k, eps 0.1+0.9 decayed over 70%), same eval protocol.
The only difference: a plain 2x128 ReLU MLP trained end-to-end — no leaves, no
router, no masters, no bands.

Bars: (b) TD trioron-router 163.2, hand-coded arbiter 167.6, Gaussian nest
148.1, reactive 141.8, flat linear trioron solo 52.4.

If DQN >= 163 at the same budget: nesting is how TRIORON gets competitive; the
product claim rests on modularity/CL/footprint, not raw sample efficiency.
If DQN < 163: the nest's sample-efficiency claim has teeth.
"""
from __future__ import annotations

import argparse
import statistics
import sys
from collections import deque
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.world.fire_taming import evaluate       # tamed physics
from experiments.world.tile_world import TileWorld, N_ACTION

PERCEPT_DIM = 77


class QNet(torch.nn.Module):
    def __init__(self, h=128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(PERCEPT_DIM, h), torch.nn.ReLU(),
            torch.nn.Linear(h, h), torch.nn.ReLU(),
            torch.nn.Linear(h, N_ACTION))

    def forward(self, x):
        return self.net(x)


def train_dqn(seed, *, episodes=300, gamma=0.95, lr=3e-3, batch=64,
              max_steps=300, hidden=128):
    torch.manual_seed(seed)
    q = QNet(hidden)
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    buf = deque(maxlen=20000)
    g = torch.Generator().manual_seed(seed + 31)
    curve = []
    for ep in range(episodes):
        eps = 0.1 + 0.9 * max(0.0, 1 - ep / (0.7 * episodes))
        w = TileWorld(seed=seed * 1000 + ep, max_steps=max_steps)
        p = w.percept(); done = False
        while not done:
            if torch.rand(1, generator=g).item() < eps:
                a = int(torch.randint(0, N_ACTION, (1,), generator=g))
            else:
                with torch.no_grad():
                    a = int(q(p.unsqueeze(0))[0].argmax())
            p2, r, done, info = w.step(a)
            buf.append((p, a, r, p2, float(done))); p = p2
            if len(buf) >= batch:
                idx = torch.randint(0, len(buf), (batch,), generator=g)
                bs = [buf[j] for j in idx]
                bp = torch.stack([b[0] for b in bs])
                ba = torch.tensor([b[1] for b in bs])
                br = torch.tensor([b[2] for b in bs])
                bp2 = torch.stack([b[3] for b in bs])
                bd = torch.tensor([b[4] for b in bs])
                qv = q(bp)[torch.arange(batch), ba]
                with torch.no_grad():
                    tgt = br + gamma * q(bp2).max(dim=1).values * (1 - bd)
                loss = torch.nn.functional.mse_loss(qv, tgt)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(q.parameters(), 1.0)
                opt.step()
        curve.append(info["t"])
    return q, curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--eval-seeds", type=int, default=40)
    ap.add_argument("--hidden", type=int, default=128)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    n_par = sum(p.numel() for p in QNet(args.hidden).parameters())
    print(f"=== FLAT DQN BASELINE (2x{args.hidden} ReLU MLP, {n_par} params) ===")
    print(f"episodes={args.episodes} (same budget as trioron router) "
          f"seeds={seeds} eval_seeds={args.eval_seeds}\n", flush=True)

    survs = []
    for seed in seeds:
        q, curve = train_dqn(seed, episodes=args.episodes, hidden=args.hidden)
        tail = statistics.mean(curve[-30:])
        s, _, _ = evaluate(
            lambda w, p: int(q(p.unsqueeze(0))[0].argmax()),
            f"DQN seed={seed}", seeds=args.eval_seeds)
        print(f"      train tail(30)={tail:.1f}", flush=True)
        survs.append(s)

    print(f"\n=== VERDICT (n={len(seeds)}) ===")
    print(f"  flat DQN: mean {statistics.mean(survs):.1f}  "
          f"per-seed {['%.1f' % s for s in survs]}")
    print("  bars: TD trioron-router 163.2 | arbiter 167.6 | Gaussian nest 148.1"
          " | reactive 141.8 | flat linear trioron 52.4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
