"""Structural dreaming — the nest births a NEW primitive from its own
failures and breaks past its taught repertoire. s049/s050 test.

Rocky's ask: "add new primitives during dreaming, self-reflect, break
from its limit." The s048 dreaming design's unbuilt third stage
(structural dreams = new-leaf enrollment), now testable end to end:

  1. BASELINE — the validated TD nest (5 master-taught leaves +
     consequence-taught trioron router; n=3 148.5±12.9).
  2. SELF-REFLECT — the organism rolls ITSELF on diagnosis maps
     (disjoint from eval), reads its own cause-of-death table, and
     names the drive that kills it most. No human in the loop.
  3. NEW PRIMITIVE — a sixth trioron leaf is TD-trained from scratch
     with reward = the implicated drive's delta ONLY. Taught by
     consequence, not by any master — the skill the masters never had.
  4. ENROLL + RE-ARBITRATE — the router retrains over 6 leaves
     (cold-start TD; width sweep says 6 is inside the safe zone).
  5. EVAL — same 40-map protocol; survival + cause shift vs baseline.

Success = extended nest > 148.5-era baseline on the same seed AND the
implicated death cause shrinks. Failure modes stated in advance: the
new leaf may rediscover WARM (redundant → router ignores it, no harm),
or 6-way cold-start TD may lose more than the leaf adds.

Run (any cwd): python3 <abs>/world_dream_newleaf.py [--seed N]
             [--td-episodes 300] [--eval-seeds 40]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse
import time
from collections import Counter, deque

import torch

from trioron.bases import seeded
from trioron.core import Envelope, construct
from trioron.phenotype import default_dispatch_table

from experiments.world import fire_taming as _ft          # tamed physics
from experiments.world.fire_taming import evaluate, _cause
from experiments.world.mirror_cells import _solo
from experiments.world.router_trioron import (PERCEPT_DIM, load_leaves,
                                              train_router_td)
from experiments.world.tile_world import N_ACTION, TileWorld
from experiments.world.vocabulary import PRIM_ORDER

# implicated drive per death cause, and its contribution to drive_sum
DRIVE_OF = {"cold": "temperature", "overheat": "temperature",
            "thirst": "thirst", "energy": "energy",
            "integrity": "integrity"}


def drive_val(w, drive: str) -> float:
    if drive == "temperature":
        return -2.0 * abs(w.temp - 0.5)      # drive_sum's temp term
    return float(getattr(w, drive))


class Organism:
    """Router substrate over per-leaf policies; leaf i answers via
    fns[i](p[1,77]) -> logits[1,6]."""

    def __init__(self, router_sub, fns, names):
        self.router, self.fns, self.names = router_sub, fns, names
        self.route_hist = [0] * len(fns)

    @torch.no_grad()
    def act(self, w, p):
        i = int(self.router(p.unsqueeze(0))[0].argmax())
        self.route_hist[i] += 1
        return int(self.fns[i](p.unsqueeze(0))[0].argmax()), self.names[i]


def build_leaf_sub(seed):
    torch.manual_seed(seed)
    sub = construct(
        base=seeded(PERCEPT_DIM, N_ACTION, interior_cells=32,
                    nonlinear=True),
        envelope=Envelope(max_parameter_bytes=400_000),
        dispatch_table=default_dispatch_table(), capacity=2048,
        sparsity_k=0)
    sub.compile()
    sub.prepare_training()
    return sub


def train_drive_leaf(seed, drive, *, episodes=300, gamma=0.95, lr=3e-3,
                     batch=64, max_steps=300):
    """TD over ACTIONS, reward = the implicated drive's delta only —
    consequence-taught, no master (mirrors train_router_td)."""
    sub = build_leaf_sub(seed + 900)
    opt = torch.optim.Adam(sub.trainable_tensors(), lr=lr)
    buf = deque(maxlen=20000)
    g = torch.Generator().manual_seed(seed + 931)
    for ep in range(episodes):
        eps = 0.1 + 0.9 * max(0.0, 1 - ep / (0.7 * episodes))
        w = TileWorld(seed=seed * 1000 + 70000 + ep, max_steps=max_steps)
        p, done = w.percept(), False
        while not done:
            if torch.rand(1, generator=g).item() < eps:
                a = int(torch.randint(0, N_ACTION, (1,), generator=g))
            else:
                with torch.no_grad():
                    a = int(sub(p.unsqueeze(0))[0].argmax())
            v0 = drive_val(w, drive)
            p2, _, done, info = w.step(a)
            r = drive_val(w, drive) - v0
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
                q = sub(bp)[torch.arange(batch), ba]
                with torch.no_grad():
                    tgt = br + gamma * sub(bp2).max(dim=1).values * (1 - bd)
                loss = torch.nn.functional.mse_loss(q, tgt)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(sub.trainable_tensors(), 1.0)
                sub.zero_dormant_grads(); opt.step()
    return sub


def train_router_td_n(seed, fns, *, episodes=300, gamma=0.95, lr=3e-3,
                      batch=64, max_steps=300):
    """train_router_td generalized to N leaf policies (fns list)."""
    n = len(fns)
    torch.manual_seed(seed)
    sub = construct(
        base=seeded(PERCEPT_DIM, n, interior_cells=32, nonlinear=True),
        envelope=Envelope(max_parameter_bytes=400_000),
        dispatch_table=default_dispatch_table(), capacity=2048,
        sparsity_k=0)
    sub.compile()
    sub.prepare_training()
    opt = torch.optim.Adam(sub.trainable_tensors(), lr=lr)
    buf = deque(maxlen=20000)
    g = torch.Generator().manual_seed(seed + 31)
    for ep in range(episodes):
        eps = 0.1 + 0.9 * max(0.0, 1 - ep / (0.7 * episodes))
        w = TileWorld(seed=seed * 1000 + ep, max_steps=max_steps)
        p, done = w.percept(), False
        while not done:
            if torch.rand(1, generator=g).item() < eps:
                i = int(torch.randint(0, n, (1,), generator=g))
            else:
                with torch.no_grad():
                    i = int(sub(p.unsqueeze(0))[0].argmax())
            with torch.no_grad():
                a = int(fns[i](p.unsqueeze(0))[0].argmax())
            p2, r, done, info = w.step(a)
            buf.append((p, i, r, p2, float(done)))
            p = p2
            if len(buf) >= batch:
                idx = torch.randint(0, len(buf), (batch,), generator=g)
                bs = [buf[j] for j in idx]
                bp = torch.stack([b[0] for b in bs])
                bi = torch.tensor([b[1] for b in bs])
                br = torch.tensor([b[2] for b in bs])
                bp2 = torch.stack([b[3] for b in bs])
                bd = torch.tensor([b[4] for b in bs])
                q = sub(bp)[torch.arange(batch), bi]
                with torch.no_grad():
                    tgt = br + gamma * sub(bp2).max(dim=1).values * (1 - bd)
                loss = torch.nn.functional.mse_loss(q, tgt)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(sub.trainable_tensors(), 1.0)
                sub.zero_dormant_grads(); opt.step()
    return sub


@torch.no_grad()
def self_reflect(org, seed, *, episodes=40, max_steps=300):
    """The organism reads its own death statistics on diagnosis maps
    (disjoint seed line from the eval protocol)."""
    causes = Counter()
    for ep in range(episodes):
        w = TileWorld(seed=seed * 1000 + 50000 + ep, max_steps=max_steps)
        p, done = w.percept(), False
        while not done:
            p, _, done, info = w.step(org.act(w, p)[0])
        if info["t"] < max_steps:
            causes[_cause(w)] += 1
    return causes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--td-episodes", type=int, default=300)
    ap.add_argument("--eval-seeds", type=int, default=40)
    args = ap.parse_args()
    t0 = time.time()

    donors = load_leaves()
    leaf_fns = [
        (lambda p1, d=donors[n]: d(_solo(p1))) for n in PRIM_ORDER]

    print(f"[1/5] baseline TD nest (seed {args.seed})", flush=True)
    router5, curve = train_router_td(args.seed, donors,
                                     episodes=args.td_episodes)
    base = Organism(router5, leaf_fns, PRIM_ORDER)
    s_base, base_causes, _ = evaluate(
        lambda w, p: base.act(w, p)[0], "base 5-leaf nest   ",
        seeds=args.eval_seeds)

    print(f"[2/5] self-reflection on diagnosis maps  "
          f"t={time.time() - t0:.0f}s", flush=True)
    causes = self_reflect(base, args.seed)
    top = causes.most_common(1)[0][0] if causes else "cold"
    drive = DRIVE_OF.get(top, "temperature")
    print(f"  organism's own diagnosis: deaths={dict(causes)} -> "
          f"top cause '{top}' -> implicated drive '{drive}'", flush=True)

    print(f"[3/5] dreaming a new primitive: TD on '{drive}' delta only "
          f"(no master)  t={time.time() - t0:.0f}s", flush=True)
    new_leaf = train_drive_leaf(args.seed, drive,
                                episodes=args.td_episodes)
    new_name = f"SELF_{top.upper()}"

    print(f"[4/5] enrollment: 6-leaf router TD  "
          f"t={time.time() - t0:.0f}s", flush=True)
    fns6 = leaf_fns + [lambda p1: new_leaf(p1)]
    names6 = PRIM_ORDER + [new_name]
    router6 = train_router_td_n(args.seed, fns6,
                                episodes=args.td_episodes)
    ext = Organism(router6, fns6, names6)

    print(f"[5/5] eval  t={time.time() - t0:.0f}s", flush=True)
    s_ext, ext_causes, _ = evaluate(
        lambda w, p: ext.act(w, p)[0], f"6-leaf nest (+{new_name})",
        seeds=args.eval_seeds)
    share = ext.route_hist[-1] / max(1, sum(ext.route_hist))
    print(f"\n[seed {args.seed}] base={s_base:.1f} -> extended={s_ext:.1f} "
          f"(delta {s_ext - s_base:+.1f})")
    print(f"  '{top}' deaths: {base_causes.get(top, 0)}/{args.eval_seeds}"
          f" -> {ext_causes.get(top, 0)}/{args.eval_seeds}   "
          f"new-leaf route share {share:.2f}")
    print(f"  route_hist={dict(zip(names6, ext.route_hist))}  "
          f"elapsed={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
