"""Arm 2 (s050): band-masked arbitration over the zero-master drive nest.

Hypothesis from arm (a): the gap to the master nest is ARBITRATION, not
leaf skill (solo drive leaves >= solo masters; nest cold-deaths 9-12/40
while the temperature leaf alone dies cold 2-3/40). Masters were
band-cleaned; drive leaves have flat Q outside their band, so a router
mistake turns into near-random action.

The accessible fix keeps zero masters: the user declares one STRESS
THRESHOLD per drive (the same numbers the masters' bands used) and the
router is masked to leaves whose drive is currently in stress (all
eligible when none is). Router TD is retrained under the mask (cold
start, same budget); leaves are the saved drive leaves, untouched.

Bands (declared thresholds, mirroring primitives.py):
  temperature: always eligible (setpoint drive; WARM+FLEE covered R)
  thirst:      thirst < 0.9
  energy:      energy < 0.85
  integrity:   predator within Chebyshev 2 (a THREAT-SENSE threshold,
               not a drive level — the one band that needs a percept,
               stated as such) or standing on FIRE/POISON

Run: python3 <abs>/world_drive_band.py [--seed N]
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
from experiments.world.primitives import (_band_evade, _band_forage,
                                          _band_hydrate)
from experiments.world.drive_common import build_router_sub, load_drive_leaves
from experiments.world.tile_world import TileWorld
from experiments.world.world_drive_vocab import BASELINE, save_sub

BANDS = [lambda w: True, _band_hydrate, _band_forage, _band_evade]


def eligible(w):
    m = torch.tensor([bool(b(w)) for b in BANDS])
    return m if m.any() else torch.ones_like(m)


class BandOrganism:
    def __init__(self, router, fns, names):
        self.router, self.fns, self.names = router, fns, names
        self.route_hist = [0] * len(fns)

    @torch.no_grad()
    def act(self, w, p):
        q = self.router(p.unsqueeze(0))[0].clone()
        q[~eligible(w)] = -1e9
        i = int(q.argmax())
        self.route_hist[i] += 1
        return int(self.fns[i](p.unsqueeze(0))[0].argmax()), self.names[i]


def train_router_band(seed, fns, *, episodes=300, gamma=0.95, lr=3e-3,
                      batch=64, max_steps=300):
    """train_router_td_n with the eligibility mask applied to both the
    behaviour policy (explore among eligible) and the bootstrap max."""
    n = len(fns)
    sub = build_router_sub(seed, n)
    opt = torch.optim.Adam(sub.trainable_tensors(), lr=lr)
    buf = deque(maxlen=20000)
    g = torch.Generator().manual_seed(seed + 31)
    for ep in range(episodes):
        eps = 0.1 + 0.9 * max(0.0, 1 - ep / (0.7 * episodes))
        w = TileWorld(seed=seed * 1000 + ep, max_steps=max_steps)
        p, done = w.percept(), False
        m = eligible(w)
        while not done:
            if torch.rand(1, generator=g).item() < eps:
                cand = torch.nonzero(m).flatten()
                i = int(cand[torch.randint(0, len(cand), (1,), generator=g)])
            else:
                with torch.no_grad():
                    q = sub(p.unsqueeze(0))[0].clone(); q[~m] = -1e9
                    i = int(q.argmax())
            with torch.no_grad():
                a = int(fns[i](p.unsqueeze(0))[0].argmax())
            p2, r, done, info = w.step(a)
            m2 = eligible(w)
            buf.append((p, i, r, p2, float(done), m2))
            p, m = p2, m2
            if len(buf) >= batch:
                idx = torch.randint(0, len(buf), (batch,), generator=g)
                bs = [buf[j] for j in idx]
                bp = torch.stack([b[0] for b in bs])
                bi = torch.tensor([b[1] for b in bs])
                br = torch.tensor([b[2] for b in bs])
                bp2 = torch.stack([b[3] for b in bs])
                bd = torch.tensor([b[4] for b in bs])
                bm2 = torch.stack([b[5] for b in bs])
                q = sub(bp)[torch.arange(batch), bi]
                with torch.no_grad():
                    q2 = sub(bp2).masked_fill(~bm2, -1e9)
                    tgt = br + gamma * q2.max(dim=1).values * (1 - bd)
                loss = torch.nn.functional.mse_loss(q, tgt)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(sub.trainable_tensors(), 1.0)
                sub.zero_dormant_grads(); opt.step()
    return sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--td-episodes", type=int, default=300)
    ap.add_argument("--eval-seeds", type=int, default=40)
    args = ap.parse_args()
    out = Path(__file__).resolve().parents[3] / "runs" / "drive_band"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    leaves, names = load_drive_leaves(args.seed)
    fns = [(lambda p1, l=l: l(p1)) for l in leaves]
    print(f"[1/2] band-masked router TD  t={time.time() - t0:.0f}s", flush=True)
    router = train_router_band(args.seed, fns, episodes=args.td_episodes)
    save_sub(router, out / f"router_band_seed{args.seed}.pt")
    org = BandOrganism(router, fns, names)
    print(f"[2/2] eval  t={time.time() - t0:.0f}s", flush=True)
    s, causes, _ = evaluate(lambda w, p: org.act(w, p)[0],
                            "band-masked drive nest", seeds=args.eval_seeds)
    base = BASELINE.get(args.seed)
    print(f"\n[seed {args.seed}] band-masked drive nest={s:.1f}"
          + (f"  master-built {base:.1f}  delta {s - base:+.1f}"
             if base is not None else "")
          + f"\n  route_hist={dict(zip(names, org.route_hist))}  "
          f"causes={dict(causes)}  elapsed={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
