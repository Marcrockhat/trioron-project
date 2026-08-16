"""The CHANGE TEST — adaptation cost + retention after the world shifts.

Rocky's question made concrete: a well-curriculated DQN may match the
nest on a FROZEN world — so measure what happens when the world moves.
Change: the map's fires burn out — fire_n 4 -> 1. Warming is the same
skill but heat becomes scarce; both agents' learned behavior is stale.

Protocol (both agents start from their validated checkpoints):
  zero-shot   — eval on the changed world, no adaptation;
  adapt       — matched 50-episode budget in the changed world:
                nest fine-tunes ONLY the router (leaves frozen — the
                architecture's claim is that skills are separable);
                DQN fine-tunes everything (it has no other choice);
  retention   — after adapting, re-eval on the ORIGINAL world.
                This is the forgetting axis: frozen leaves cannot
                forget; a flat net that adapts must overwrite.

Reference (original world, ckpts): nest 163.25, DQN 34.9.

Run (any cwd): python3 <abs>/world_change_test.py [--seed 0]
               [--adapt-episodes 50] [--eval-seeds 40]
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
from experiments.world.tile_world import TileWorld
from experiments.world.vocabulary import PRIM_ORDER
from experiments.world.watch_duel import DUEL, get_agents

CHANGED = dict(fire_n=1)          # the world shift: fires burn out
N_PRIM = len(PRIM_ORDER)


@torch.no_grad()
def eval_policy(act, *, world_kw=None, seeds=40, max_steps=300) -> float:
    kw = world_kw or {}
    surv = []
    for ep in range(seeds):
        w = TileWorld(seed=ep * 7000 + 7, max_steps=max_steps, **kw)
        p, done = w.percept(), False
        while not done:
            p, _, done, info = w.step(act(p))
        surv.append(info["t"])
    return statistics.mean(surv)


def nest_act_fn(router, donors):
    @torch.no_grad()
    def act(p):
        i = int(router(p.unsqueeze(0))[0].argmax())
        return int(donors[PRIM_ORDER[i]](_solo(p.unsqueeze(0)))[0].argmax())
    return act


def dqn_act_fn(q):
    @torch.no_grad()
    def act(p):
        return int(q(p.unsqueeze(0))[0].argmax())
    return act


def arm_lock(router, donors, seed, *, saliency_batches=60):
    """Prime the λ epigenetic lock BEFORE adapting (manual §5.6): replay
    original-world experience through the router, accumulate |w·g|
    saliency, refresh per-cell λ, and anchor the checkpoint weights.
    The fine-tune then adds strength·ewc_penalty to every TD loss —
    the soft pull that lets the router adapt where it is plastic and
    hold where the original arbitration lives."""
    from trioron.learning.epigenetic_lock import (accumulate_saliency,
                                                  anchor, refresh_lambda)
    g = torch.Generator().manual_seed(seed + 61)
    for ep in range(saliency_batches):
        w = TileWorld(seed=ep * 7000 + 7, max_steps=60)
        p, done = w.percept(), False
        P, I, R = [], [], []
        while not done:
            with torch.no_grad():
                i = int(router(p.unsqueeze(0))[0].argmax())
                a = int(donors[PRIM_ORDER[i]](
                    _solo(p.unsqueeze(0)))[0].argmax())
            p2, r, done, _ = w.step(a)
            P.append(p); I.append(i); R.append(r)
            p = p2
        bp = torch.stack(P)
        bi = torch.tensor(I)
        qv = router(bp)[torch.arange(len(P)), bi]
        # reward-weighted sensitivity as the saliency signal: the
        # gradient magnitude is ∝ |r|·|∂q/∂w| on the visited choices
        loss = torch.nn.functional.mse_loss(
            qv, torch.tensor(R, dtype=torch.float32) + qv.detach())
        loss.backward()
        accumulate_saliency(router.arena)
        for t in router.trainable_tensors():
            if t.grad is not None:
                t.grad = None
    refresh_lambda(router.arena)
    anchor(router.arena)


def finetune_router(router, donors, seed, *, episodes=50, gamma=0.95,
                    lr=3e-3, batch=64, max_steps=300, lock_strength=0.0):
    """Adapt the NEST: TD on the router only; leaves stay frozen.
    lock_strength>0 adds the λ EWC pull toward the anchored weights AND
    freezes branch_alpha — the dendrite gains are outside the anchor's
    coverage (the s049 lobotomy lesson: α drift alone reshapes the
    policy), so under the lock they are structural, not plastic."""
    from trioron.learning.epigenetic_lock import ewc_penalty
    params = router.trainable_tensors()
    if lock_strength > 0:
        params = [t for t in params
                  if t is not router.arena.branch_alpha]
    opt = torch.optim.Adam(params, lr=lr)
    buf = deque(maxlen=20000)
    g = torch.Generator().manual_seed(seed + 41)
    for ep in range(episodes):
        eps = 0.1 + 0.2 * max(0.0, 1 - ep / (0.7 * episodes))
        w = TileWorld(seed=seed * 1000 + 90000 + ep, max_steps=max_steps,
                      **CHANGED)
        p, done = w.percept(), False
        while not done:
            if torch.rand(1, generator=g).item() < eps:
                i = int(torch.randint(0, N_PRIM, (1,), generator=g))
            else:
                with torch.no_grad():
                    i = int(router(p.unsqueeze(0))[0].argmax())
            with torch.no_grad():
                a = int(donors[PRIM_ORDER[i]](
                    _solo(p.unsqueeze(0)))[0].argmax())
            p2, r, done, _ = w.step(a)
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
                qv = router(bp)[torch.arange(batch), bi]
                with torch.no_grad():
                    tgt = br + gamma * router(bp2).max(dim=1).values * (1 - bd)
                loss = torch.nn.functional.mse_loss(qv, tgt)
                if lock_strength > 0:
                    loss = loss + lock_strength * ewc_penalty(router.arena)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(router.trainable_tensors(),
                                               1.0)
                router.zero_dormant_grads(); opt.step()
    return router


def finetune_dqn(q, seed, *, episodes=50, gamma=0.95, lr=3e-3, batch=64,
                 max_steps=300):
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    buf = deque(maxlen=20000)
    g = torch.Generator().manual_seed(seed + 51)
    for ep in range(episodes):
        eps = 0.1 + 0.2 * max(0.0, 1 - ep / (0.7 * episodes))
        w = TileWorld(seed=seed * 1000 + 90000 + ep, max_steps=max_steps,
                      **CHANGED)
        p, done = w.percept(), False
        while not done:
            if torch.rand(1, generator=g).item() < eps:
                a = int(torch.randint(0, q.net[-1].out_features, (1,),
                                      generator=g))
            else:
                with torch.no_grad():
                    a = int(q(p.unsqueeze(0))[0].argmax())
            p2, r, done, _ = w.step(a)
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
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--adapt-episodes", type=int, default=50)
    ap.add_argument("--eval-seeds", type=int, default=40)
    ap.add_argument("--lock-strength", type=float, default=300.0)
    args = ap.parse_args()
    t0 = time.time()

    router, donors, q = get_agents(seed=0)     # both from checkpoints
    nest_act = nest_act_fn(router, donors)
    dqn_act = dqn_act_fn(q)
    E = args.eval_seeds

    n0o = eval_policy(nest_act, seeds=E)
    d0o = eval_policy(dqn_act, seeds=E)
    n0c = eval_policy(nest_act, world_kw=CHANGED, seeds=E)
    d0c = eval_policy(dqn_act, world_kw=CHANGED, seeds=E)
    print(f"[zero-shot] original: nest {n0o:.1f}  dqn {d0o:.1f}   "
          f"CHANGED: nest {n0c:.1f}  dqn {d0c:.1f}  "
          f"t={time.time() - t0:.0f}s", flush=True)

    def report(tag, act, oc, oo):
        c = eval_policy(act, world_kw=CHANGED, seeds=E)
        o = eval_policy(act, seeds=E)
        print(f"[seed {args.seed}] {tag}: CHANGED {c:.1f} (Δ{c - oc:+.1f})"
              f"   RETENTION {o:.1f} (Δ{o - oo:+.1f})  "
              f"t={time.time() - t0:.0f}s", flush=True)

    # arm 1: nest, naive router fine-tune (fresh reload)
    r1, d1, _ = get_agents(seed=0)
    finetune_router(r1, d1, args.seed, episodes=args.adapt_episodes)
    report("nest naive-adapt ", nest_act_fn(r1, d1), n0c, n0o)

    # arm 2: nest, λ-locked router fine-tune (the native machinery)
    r2, d2, _ = get_agents(seed=0)
    arm_lock(r2, d2, args.seed)
    finetune_router(r2, d2, args.seed, episodes=args.adapt_episodes,
                    lock_strength=args.lock_strength)
    report(f"nest λ-lock s={args.lock_strength:<4.0f}",
           nest_act_fn(r2, d2), n0c, n0o)

    # arm 3: DQN fine-tune (it has no partial-freeze option)
    q3 = QNet()
    q3.load_state_dict(torch.load(DUEL / "dqn.pt"))
    finetune_dqn(q3, args.seed, episodes=args.adapt_episodes)
    report("dqn fine-tune    ", dqn_act_fn(q3), d0c, d0o)


if __name__ == "__main__":
    main()
