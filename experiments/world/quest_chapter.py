"""The Quest — apprenticeship chapters. Adam learns one master's skill per chapter,
RESUMING his own substrate so skills accumulate (and old ones must survive).

Each chapter: load Adam → DAgger-apprentice to one master (continuing his weights,
keeping his own TD self-RL so he can SURPASS the master) → evaluate survival, full
cause-of-death (retention of prior skills shows here), and the skill metric → save
Adam back with the new skill appended.

Usage:  python3 quest_chapter.py --master fire    # then water, then food
        python3 quest_chapter.py --status          # where Adam stands
"""
from __future__ import annotations

import sys
import argparse
import statistics
from collections import deque
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.world.fire_taming import evaluate, TileWorld     # sets tamed-fire physics
from experiments.world.tile_world import N_ACTION
from experiments.world.mirror_cells import (
    build_mirror, _solo, keep_only_mirror_grads, obs_onehot,
)
from experiments.world.quest import MASTERS

RUNS = Path(__file__).resolve().parent.parent.parent / "runs"
CKPT = RUNS / "protagonist.pt"


def load_adam():
    meta = torch.load(CKPT)
    sub = build_mirror(meta["seed"], n_mirror=8)
    with torch.no_grad():
        sub.arena.bias.copy_(meta["bias"])
        sub.arena.edge_weight.copy_(meta["edge_weight"])
    return sub, meta


def save_adam(sub, meta):
    meta["bias"] = sub.arena.bias.detach().clone()
    meta["edge_weight"] = sub.arena.edge_weight.detach().clone()
    torch.save(meta, CKPT)


@torch.no_grad()
def collect_master_demos(master_fn, n_ep, seed=99, max_steps=300):
    demos = []
    for ep in range(n_ep):
        w = TileWorld(seed=seed * 13000 + ep, max_steps=max_steps)
        p = w.percept(); done = False
        while not done:
            a = master_fn(w); demos.append((p, a))
            p, _, done, _ = w.step(a)
    return demos


def dagger_resume(sub, master_fn, *, seed, episodes, gamma=0.95, lr=2e-3,
                  batch=64, imit_w=0.5, max_steps=300):
    """Continue training Adam: DAgger imitation (master labels HIS states) + his
    own TD self-RL. Lower lr / exploration than from-scratch — he's already alive."""
    opt = torch.optim.Adam(sub.trainable_tensors(), lr=lr)
    a = sub.arena
    buf = deque(maxlen=20000)
    demos = deque(collect_master_demos(master_fn, 20, seed=seed), maxlen=20000)
    g = torch.Generator().manual_seed(seed + 71)
    ce = torch.nn.functional.cross_entropy
    for ep in range(episodes):
        eps = 0.05 + 0.25 * max(0.0, 1 - ep / (0.7 * episodes))   # gentle, he's competent
        w = TileWorld(seed=seed * 1000 + ep + 5000, max_steps=max_steps)
        p = w.percept(); done = False
        while not done:
            demos.append((p, master_fn(w)))            # DAgger: master labels his state
            if torch.rand(1, generator=g).item() < eps:
                act = int(torch.randint(0, N_ACTION, (1,), generator=g))
            else:
                with torch.no_grad():
                    act = int(sub(_solo(p.unsqueeze(0)))[0].argmax())
            p2, r, done, info = w.step(act)
            buf.append((p, act, r, p2, float(done))); p = p2
            if len(buf) >= batch and len(demos) >= batch:
                idx = torch.randint(0, len(buf), (batch,), generator=g)
                bs = [buf[i] for i in idx]
                bp = _solo(torch.stack([b[0] for b in bs]))
                ba = torch.tensor([b[1] for b in bs])
                br = torch.tensor([b[2] for b in bs])
                bp2 = _solo(torch.stack([b[3] for b in bs]))
                bd = torch.tensor([b[4] for b in bs])
                q = sub(bp)[torch.arange(batch), ba]
                with torch.no_grad():
                    tgt = br + gamma * sub(bp2).max(dim=1).values * (1 - bd)
                opt.zero_grad()
                torch.nn.functional.mse_loss(q, tgt).backward()
                td_b = a.bias.grad.clone(); td_w = a.edge_weight.grad.clone()
                didx = torch.randint(0, len(demos), (batch,), generator=g)
                ds = torch.stack([demos[i][0] for i in didx])
                da = torch.tensor([demos[i][1] for i in didx])
                avatar = torch.cat([ds, obs_onehot(da, batch)], dim=1)
                opt.zero_grad()
                (imit_w * (ce(sub(avatar), da) + ce(sub(_solo(ds)), da))).backward()
                keep_only_mirror_grads(sub)
                a.bias.grad.add_(td_b); a.edge_weight.grad.add_(td_w)
                torch.nn.utils.clip_grad_norm_(sub.trainable_tensors(), 1.0)
                sub.zero_dormant_grads(); opt.step()
    return sub


def report_state(sub, label):
    return evaluate(lambda w, p: int(sub(_solo(p.unsqueeze(0)))[0].argmax()),
                    label, seeds=40)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", choices=list(MASTERS), help="skill to apprentice")
    ap.add_argument("--episodes", type=int, default=250)
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    sub, meta = load_adam()
    name = meta.get("name") or "Adam"
    print(f"=== {name} — skills so far: {meta.get('skills') or '[]'} ===")
    print(f"baseline at crowning: {meta.get('baseline_survival', '?')}\n")

    print("BEFORE this chapter:")
    before, before_c, before_fire = report_state(sub, f"{name} (before)")
    if args.status or not args.master:
        return 0

    print(f"\napprenticing to the {args.master}-master (DAgger, {args.episodes} ep)...\n")
    sub = dagger_resume(sub, MASTERS[args.master], seed=meta["seed"],
                        episodes=args.episodes)
    after, after_c, after_fire = report_state(sub, f"{name} (after {args.master})")

    if args.master not in (meta["skills"]):
        meta["skills"].append(args.master)
    meta["name"] = name
    save_adam(sub, meta)
    print(f"\n  Δsurvival {after - before:+.1f}  "
          f"(retention check: read the cause-of-death breakdown above —")
    print(f"   skills learned in earlier chapters should NOT regress)")
    print(f"  {name} now holds skills: {meta['skills']}  → saved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
