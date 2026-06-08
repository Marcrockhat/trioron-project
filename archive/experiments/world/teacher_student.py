"""Teacher–student imitation — the fair test of mirroring (2026-06-02).

The pop=8-from-scratch test failed because there was no one worth copying (the
"fittest" was just the luckiest mediocre peer → premature conformity). The
tournament finding is "copy the SUCCESSFUL." So: take a *fully-trained*
organism (the ~50-survivor) as a TEACHER, and compare fresh students who
imitate it vs students who learn from scratch.

Win condition for imitation: students that copy a competent teacher reach
competence FASTER (fewer episodes to beat reactive ≈38) and at least as high.
The imitation weight anneals to 0 over the first half so the student weans off
the teacher and can surpass it on its own RL.
"""
from __future__ import annotations

import argparse
import statistics
import sys
from collections import deque
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.world.tile_world import TileWorld, N_ACTION
from experiments.world.organism_v1 import build, train_organism


@torch.no_grad()
def collect_demos(teacher, n_ep, seed, max_steps=300):
    """Greedy teacher rollouts → (state, action) demonstrations."""
    demos = []
    for ep in range(n_ep):
        w = TileWorld(seed=seed * 13000 + ep, max_steps=max_steps)
        p = w.percept(); done = False
        while not done:
            a = int(teacher(p.unsqueeze(0))[0].argmax())
            demos.append((p, a))
            p, _, done, _ = w.step(a)
    return demos


def train_student(imitate, demos, *, seed, episodes, gamma=0.95, lr=3e-3,
                  batch=64, imit_w0=1.0, max_steps=300):
    sub = build(seed)
    opt = torch.optim.Adam(sub.trainable_tensors(), lr=lr)
    buf = deque(maxlen=20000)
    g = torch.Generator().manual_seed(seed + 31)
    curve = []
    for ep in range(episodes):
        # imitation weight anneals to 0 over the first half (then pure self-RL)
        imit_w = imit_w0 * max(0.0, 1 - ep / (0.5 * episodes)) if imitate else 0.0
        eps = 0.1 + 0.9 * max(0.0, 1 - ep / (0.7 * episodes))
        w = TileWorld(seed=seed * 1000 + ep, max_steps=max_steps)
        p = w.percept(); done = False
        while not done:
            if torch.rand(1, generator=g).item() < eps:
                a = int(torch.randint(0, N_ACTION, (1,), generator=g))
            else:
                with torch.no_grad():
                    a = int(sub(p.unsqueeze(0))[0].argmax())
            p2, r, done, info = w.step(a)
            buf.append((p, a, r, p2, float(done))); p = p2
            if len(buf) >= batch:
                idx = torch.randint(0, len(buf), (batch,), generator=g)
                bs = [buf[i] for i in idx]
                bp = torch.stack([b[0] for b in bs]); ba = torch.tensor([b[1] for b in bs])
                br = torch.tensor([b[2] for b in bs]); bp2 = torch.stack([b[3] for b in bs])
                bd = torch.tensor([b[4] for b in bs])
                q = sub(bp)[torch.arange(batch), ba]
                with torch.no_grad():
                    tgt = br + gamma * sub(bp2).max(dim=1).values * (1 - bd)
                loss = torch.nn.functional.mse_loss(q, tgt)
                if imit_w > 0 and len(demos) >= batch:
                    didx = torch.randint(0, len(demos), (batch,), generator=g)
                    ds = torch.stack([demos[j][0] for j in didx])
                    da = torch.tensor([demos[j][1] for j in didx])
                    loss = loss + imit_w * torch.nn.functional.cross_entropy(sub(ds), da)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(sub.trainable_tensors(), 1.0)
                sub.zero_dormant_grads(); opt.step()
        curve.append(info["t"])
    return curve, sub


@torch.no_grad()
def eval_greedy(sub, seed, episodes=30, max_steps=300):
    out = []
    for ep in range(episodes):
        w = TileWorld(seed=seed * 7000 + ep, max_steps=max_steps)
        p = w.percept(); done = False
        while not done:
            a = int(sub(p.unsqueeze(0))[0].argmax()); p, _, done, info = w.step(a)
        out.append(info["t"])
    return statistics.mean(out)


def episodes_to(curve, thresh, window=20):
    """First episode where the trailing-window mean survival ≥ thresh."""
    for e in range(window, len(curve)):
        if statistics.mean(curve[e - window:e]) >= thresh:
            return e
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--teacher-episodes", type=int, default=400)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.episodes, args.teacher_episodes = 1, 150, 250

    print(f"teacher_student: does imitating a competent teacher beat scratch? "
          f"seeds={args.seeds} student_eps={args.episodes}")
    print("training teacher...")
    _, teacher = train_organism(99, args.teacher_episodes)
    t_surv = eval_greedy(teacher, 99)
    demos = collect_demos(teacher, 40, 99)
    print(f"  teacher greedy survival = {t_surv:.1f}  ({len(demos)} demo steps)")
    print("reactive≈38.5, random≈27.9 | win = imitate reaches competence FASTER")

    # "unaided" = learn alone from scratch; "apprentice" = learn by APPRENTICING
    # to a competent master (Abbeel & Ng apprenticeship learning). The mechanism
    # is named *apprenticing* (substrate = mirror cells) to avoid colliding with
    # Mima (counterfeit learning) and shared-replay.
    for mode in ("unaided", "apprentice"):
        e2c, finals, earlies = [], [], []
        for seed in range(args.seeds):
            curve, sub = train_student(mode == "apprentice", demos, seed=seed,
                                       episodes=args.episodes)
            e2c.append(episodes_to(curve, 38.5))
            earlies.append(statistics.mean(curve[:30]))
            finals.append(eval_greedy(sub, seed))
        e2c_clean = [x for x in e2c if x is not None]
        e2c_str = f"{statistics.mean(e2c_clean):.0f}" if e2c_clean else "never"
        print(f"  {mode:>8s}: early(first30)={statistics.mean(earlies):.1f}  "
              f"eps→beat-reactive={e2c_str} ({len(e2c_clean)}/{args.seeds})  "
              f"final greedy={statistics.mean(finals):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
