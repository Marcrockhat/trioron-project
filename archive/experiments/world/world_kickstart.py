"""Kickstarted DQN — the nest as TEACHER. s049.

Rocky's question: the DQN was tabula-rasa; how fast does it learn if
OUR organism trains it? Three arms, all the same QNet/TD machinery,
differing only in where knowledge enters:

  rasa   — tabula-rasa DQN (the recorded control, re-run to capture
           the learning CURVE at eval checkpoints);
  guide  — teacher-guided exploration (kickstarting): epsilon-actions
           come from the TD nest instead of randint. Not cloning — the
           student still learns its own Q by TD; the teacher just walks
           it into the states (night fire camps, water runs) that
           random exploration never reaches;
  bc+td  — behavior-clone the nest's rollouts first (the classic
           demonstration warm start), then TD fine-tune. s048 predicts
           the clone half chokes (~0.73 action-acc ceiling on
           arbitrated behavior; perfect-master student 35.1).

Teacher = the validated TD nest from checkpoints (router_td.pt +
donors, 163.2-class). Metric = 40-map eval survival at episode
checkpoints {25, 50, 100, 200, 300} — the sample-efficiency curve.

Run (any cwd): python3 <abs>/world_kickstart.py [--seed 0]
               [--arms rasa,guide,bctd] [--episodes 300]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse
import statistics
import time
from collections import deque

import torch

from experiments.world import fire_taming as _ft          # tamed physics
from experiments.world.dqn_baseline import QNet
from experiments.world.mirror_cells import _solo
from experiments.world.tile_world import N_ACTION, TileWorld
from experiments.world.vocabulary import PRIM_ORDER
from experiments.world.watch_duel import get_agents

CHECKPOINTS = (25, 50, 100, 200, 300)


def make_teacher():
    router, donors, _ = get_agents(seed=0)

    @torch.no_grad()
    def teach(p: torch.Tensor) -> int:
        i = int(router(p.unsqueeze(0))[0].argmax())
        return int(donors[PRIM_ORDER[i]](_solo(p.unsqueeze(0)))[0].argmax())
    return teach


@torch.no_grad()
def eval_q(q, seeds=40, max_steps=300) -> float:
    surv = []
    for ep in range(seeds):
        w = TileWorld(seed=ep * 7000 + 7, max_steps=max_steps)
        p, done = w.percept(), False
        while not done:
            p, _, done, info = w.step(int(q(p.unsqueeze(0))[0].argmax()))
        surv.append(info["t"])
    return statistics.mean(surv)


def bc_pretrain(q, teach, seed, *, episodes=40, epochs=50, lr=3e-3,
                batch=128):
    """Clone the teacher's rollouts (the teacher drives, states come
    from ITS distribution — plain BC, no DAgger)."""
    P, Y = [], []
    for ep in range(episodes):
        w = TileWorld(seed=seed * 1000 + 80000 + ep, max_steps=300)
        p, done = w.percept(), False
        while not done:
            a = teach(p)
            P.append(p)
            Y.append(a)
            p, _, done, _ = w.step(a)
    P, Y = torch.stack(P), torch.tensor(Y)
    g = torch.Generator().manual_seed(seed + 81)
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    for _ in range(epochs):
        perm = torch.randperm(len(P), generator=g)
        for i in range(0, len(P), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            torch.nn.functional.cross_entropy(q(P[idx]), Y[idx]).backward()
            opt.step()
    with torch.no_grad():
        acc = float((q(P).argmax(1) == Y).float().mean())
    print(f"    [bc] {len(P)} pairs, clone acc {acc:.3f}", flush=True)
    return q


def train_dqn_arm(seed, arm, teach, *, episodes=300, gamma=0.95, lr=3e-3,
                  batch=64, max_steps=300):
    torch.manual_seed(seed)
    q = QNet()
    if arm == "bctd":
        q = bc_pretrain(q, teach, seed)
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    buf = deque(maxlen=20000)
    g = torch.Generator().manual_seed(seed + 31)
    marks = {}
    for ep in range(episodes):
        eps = 0.1 + 0.9 * max(0.0, 1 - ep / (0.7 * episodes))
        w = TileWorld(seed=seed * 1000 + ep, max_steps=max_steps)
        p, done = w.percept(), False
        while not done:
            if torch.rand(1, generator=g).item() < eps:
                if arm == "guide":
                    a = teach(p)          # teacher explores for it
                else:
                    a = int(torch.randint(0, N_ACTION, (1,), generator=g))
            else:
                with torch.no_grad():
                    a = int(q(p.unsqueeze(0))[0].argmax())
            p2, r, done, info = w.step(a)
            buf.append((p, a, r, p2, float(done)))
            p = p2
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
        if ep + 1 in CHECKPOINTS:
            marks[ep + 1] = eval_q(q)
            print(f"    [{arm} seed {seed}] ep {ep + 1:>3d}: "
                  f"eval {marks[ep + 1]:.1f}", flush=True)
    return q, marks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", type=str, default="rasa,guide,bctd")
    ap.add_argument("--episodes", type=int, default=300)
    args = ap.parse_args()
    t0 = time.time()
    teach = make_teacher()
    print(f"teacher = TD nest (ckpt); arms={args.arms} seed={args.seed}")
    results = {}
    for arm in args.arms.split(","):
        _, marks = train_dqn_arm(args.seed, arm, teach,
                                 episodes=args.episodes)
        results[arm] = marks
        print(f"  [{arm}] curve: " + "  ".join(
            f"ep{k}={v:.1f}" for k, v in sorted(marks.items())), flush=True)
    print(f"\n[seed {args.seed}] final ({args.episodes} eps): " + "  ".join(
        f"{arm}={marks.get(args.episodes, 0):.1f}"
        for arm, marks in results.items()) +
        f"  (bars: nest one-pass wake ~47, TD nest 148.5)  "
        f"elapsed={time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
