"""Nest-of-nests vs absorption (s051): combining three seed-different
drive organisms into ONE organism.

Rocky's question: we have the same organism grown from different seeds
(s050 band-masked drive nests, seeds 0/1/2 = 112.6/113.0/159.0 on the
fixed 40-map eval). An organism should be able to ABSORB its siblings.
Is a nest of nests (route between whole organisms) comparable to
absorbing them (one substrate per drive carrying every sibling's cells)?

Arms (all evaluated on the identical 40 maps, fire_taming.evaluate):
  single   — each s050 band nest alone (reproduces s050; bar = best 159.0)
  vote     — majority action over the three nests, tie -> seed 0
             (zero-training committee control)
  nest     — NEST OF NESTS: outer trioron router (77 -> 32 quad -> 3)
             picks WHICH seed-nest acts this tick; each sub-nest keeps its
             own band router + leaves untouched. Outer router TD, cold,
             300 eps, world reward (train_router_td_n recipe).
  absorb   — ABSORB: per drive, seed-0 leaf <- graft(seed-1 leaf, seed-2
             leaf; merge_output=True, wiring="none", Protocol B) — one fat
             leaf per drive = exact Q-head sum of the three siblings (spec
             §5.3, lifecycle.graft). Band router retrained cold over the 4
             fat leaves, 300 eps (leaves untouched).
  absorb_settle — as absorb, plus each fat leaf settled 50 eps of TD on
             its own drive reward (eps=0.1, lr=1e-3) BEFORE the router.

Only three organisms exist, so combination is n=1 on organisms; n=3 is
on the ROUTER seed (--rseed 0/1/2) for the learned arms.

Run:  OMP_NUM_THREADS=1 python3 archive/experiments/world/world_nest_of_nests.py \
        --arm {single,vote,nest,absorb,absorb_settle} --rseed N
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse
import time
from collections import deque

import torch

from experiments.world import fire_taming as _ft  # noqa: F401  tamed physics
from experiments.world.fire_taming import evaluate
from experiments.world.drive_common import (build_router_sub, _load_into,
                                            load_drive_leaves)
from experiments.world.world_dream_newleaf import N_ACTION, drive_val
from experiments.world.world_drive_band import (BandOrganism, eligible,
                                                train_router_band)
from experiments.world.world_drive_vocab import DRIVES, save_sub
from experiments.world.tile_world import TileWorld
from trioron.lifecycle.graft import graft

ROOT = Path(__file__).resolve().parents[3]
BAND_RUNS = ROOT / "runs" / "drive_band"
OUT = ROOT / "runs" / "nest_of_nests"
SEEDS = (0, 1, 2)
S050 = {0: 112.6, 1: 113.0, 2: 159.0}


# ── loaders ──────────────────────────────────────────────────────────

def load_band_nest(seed):
    leaves, names = load_drive_leaves(seed)
    router = _load_into(build_router_sub(seed, len(DRIVES)),
                        BAND_RUNS / f"router_band_seed{seed}.pt")
    fns = [(lambda p1, l=l: l(p1)) for l in leaves]
    return BandOrganism(router, fns, names), leaves, router


def live_params(sub):
    a = sub.arena
    return int(a.alive.sum()) + int(a.edge_cursor)


def tick_us(fn, p, w, n=200):
    with torch.no_grad():
        fn(w, p)
        t = time.time()
        for _ in range(n):
            fn(w, p)
    return (time.time() - t) / n * 1e6


# ── arm: vote ────────────────────────────────────────────────────────

class VoteOrganism:
    def __init__(self, orgs):
        self.orgs = orgs

    @torch.no_grad()
    def act(self, w, p):
        acts = [o.act(w, p)[0] for o in self.orgs]
        counts = torch.bincount(torch.tensor(acts), minlength=N_ACTION)
        best = int(counts.max())
        if best == 1:                       # all differ -> seed 0
            return acts[0]
        return int(counts.argmax())


# ── arm: nest of nests ───────────────────────────────────────────────

class NestOfNests:
    def __init__(self, router, orgs):
        self.router, self.orgs = router, orgs
        self.route_hist = [0] * len(orgs)

    @torch.no_grad()
    def act(self, w, p):
        i = int(self.router(p.unsqueeze(0))[0].argmax())
        self.route_hist[i] += 1
        return self.orgs[i].act(w, p)[0]


def train_outer_router(seed, orgs, *, episodes=300, gamma=0.95, lr=3e-3,
                       batch=64, max_steps=300):
    """train_router_td_n over WHOLE organisms (act(w, p)); world reward."""
    n = len(orgs)
    sub = build_router_sub(seed, n)
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
            a = orgs[i].act(w, p)[0]
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


# ── arm: absorb ──────────────────────────────────────────────────────

def absorb_leaves():
    """Per drive: seed-0 leaf absorbs seed-1 and seed-2 leaves (head-merged
    graft, Protocol B). Returns fat leaves (exact Q-sum of the siblings)."""
    per_seed = {s: load_drive_leaves(s)[0] for s in SEEDS}
    fat = []
    for k, drive in enumerate(DRIVES):
        rec = per_seed[0][k]
        for s in SEEDS[1:]:
            graft(rec, per_seed[s][k], freeze=False, wiring="none",
                  merge_output=True)
        rec.prepare_training()
        fat.append(rec)
    return fat


def settle_leaf(sub, seed, drive, *, episodes=50, gamma=0.95, lr=1e-3,
                batch=64, max_steps=300, eps=0.1):
    """Short greedy-ish TD settle of an absorbed leaf on its OWN drive
    reward (train_drive_leaf recipe, fixed eps, lower lr)."""
    opt = torch.optim.Adam(sub.trainable_tensors(), lr=lr)
    buf = deque(maxlen=20000)
    g = torch.Generator().manual_seed(seed + 977)
    for ep in range(episodes):
        w = TileWorld(seed=seed * 1000 + 80000 + ep, max_steps=max_steps)
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


# ── main ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["single", "vote", "nest", "absorb",
                             "absorb_settle"])
    ap.add_argument("--rseed", type=int, default=0)
    ap.add_argument("--td-episodes", type=int, default=300)
    ap.add_argument("--settle-episodes", type=int, default=50)
    ap.add_argument("--eval-seeds", type=int, default=40)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rs = args.rseed + 700          # router seed line, disjoint from s050
    probe_w = TileWorld(seed=7, max_steps=300); probe_p = probe_w.percept()
    print(f"NEST-OF-NESTS vs ABSORB (s051) arm={args.arm} rseed={args.rseed}"
          f" td={args.td_episodes}", flush=True)

    if args.arm == "single":
        for s in SEEDS:
            org, leaves, router = load_band_nest(s)
            sc, causes, _ = evaluate(lambda w, p: org.act(w, p)[0],
                                     f"band nest seed {s}", seeds=args.eval_seeds)
            print(f"[single seed {s}] {sc:.1f} (s050 {S050[s]})  params="
                  f"{live_params(router) + sum(map(live_params, leaves))}  "
                  f"tick={tick_us(lambda w, p: org.act(w, p), probe_p, probe_w):.0f}us  "
                  f"causes={dict(causes)}", flush=True)
        return

    if args.arm == "vote":
        orgs = [load_band_nest(s)[0] for s in SEEDS]
        vo = VoteOrganism(orgs)
        sc, causes, _ = evaluate(vo.act, "majority vote x3", seeds=args.eval_seeds)
        print(f"[vote] {sc:.1f}  tick={tick_us(vo.act, probe_p, probe_w):.0f}us"
              f"  causes={dict(causes)}  elapsed={time.time() - t0:.0f}s", flush=True)
        return

    if args.arm == "nest":
        nests = [load_band_nest(s) for s in SEEDS]
        orgs = [n[0] for n in nests]
        print(f"[1/2] outer router TD  t={time.time() - t0:.0f}s", flush=True)
        router = train_outer_router(rs, orgs, episodes=args.td_episodes)
        save_sub(router, OUT / f"outer_router_rseed{args.rseed}.pt")
        non = NestOfNests(router, orgs)
        print(f"[2/2] eval  t={time.time() - t0:.0f}s", flush=True)
        sc, causes, _ = evaluate(non.act, "nest of nests", seeds=args.eval_seeds)
        params = live_params(router) + sum(
            live_params(n[2]) + sum(map(live_params, n[1])) for n in nests)
        print(f"[nest rseed {args.rseed}] {sc:.1f}  best-single 159.0  "
              f"outer_route_hist={dict(zip(SEEDS, non.route_hist))}  "
              f"params={params}  tick={tick_us(non.act, probe_p, probe_w):.0f}us  "
              f"causes={dict(causes)}  elapsed={time.time() - t0:.0f}s", flush=True)
        return

    # absorb / absorb_settle
    print(f"[1/3] absorb (head-merged graft x2 per drive)  t={time.time() - t0:.0f}s",
          flush=True)
    fat = absorb_leaves()
    print(f"  fat leaves: cells={[int(l.arena.alive.sum()) for l in fat]}  "
          f"params={[live_params(l) for l in fat]}", flush=True)
    if args.arm == "absorb_settle":
        print(f"[1b/3] settle {args.settle_episodes} eps/leaf  t={time.time() - t0:.0f}s",
              flush=True)
        for k, drive in enumerate(DRIVES):
            settle_leaf(fat[k], rs + 10 * k, drive, episodes=args.settle_episodes)
    fns = [(lambda p1, l=l: l(p1)) for l in fat]
    names = [f"DRIVE_{d.upper()}" for d in DRIVES]
    print(f"[2/3] band router TD over fat leaves  t={time.time() - t0:.0f}s", flush=True)
    router = train_router_band(rs, fns, episodes=args.td_episodes)
    save_sub(router, OUT / f"router_{args.arm}_rseed{args.rseed}.pt")
    org = BandOrganism(router, fns, names)
    print(f"[3/3] eval  t={time.time() - t0:.0f}s", flush=True)
    sc, causes, _ = evaluate(lambda w, p: org.act(w, p)[0], args.arm,
                             seeds=args.eval_seeds)
    params = live_params(router) + sum(map(live_params, fat))
    print(f"[{args.arm} rseed {args.rseed}] {sc:.1f}  best-single 159.0  "
          f"route_hist={dict(zip(names, org.route_hist))}  params={params}  "
          f"tick={tick_us(lambda w, p: org.act(w, p), probe_p, probe_w):.0f}us  "
          f"causes={dict(causes)}  elapsed={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
