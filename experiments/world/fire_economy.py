"""Fire economy — is pyrophobia a move-COST problem, not a learning-capacity one?

Rocky's hypothesis (2026-06-03): the solo organism never learns to use fire because
the trip doesn't pay. Two things make fire's net return negative for a naive policy:

  1. SPARSITY/DISTANCE. Fire is the rarest resource (max(2, s//3)=4 tiles in 144 cells
     vs 12 water/food), so it's the farthest. Warming is gradual (+0.04/step tamed,
     not a one-shot bank like consume), and BOTH temp extremes are lethal — so a long
     cold trek to a sparse fire can freeze you before you arrive.
  2. ZERO-REWARD WALK. Reward is the per-step drive delta (tile_world.step). Walking
     toward fire improves no drive en route (temp keeps dropping), so the +warmth only
     registers on arrival. With gamma=0.95 that payoff is discounted gamma^K over a
     K-step trip — distant fire is under-credited and the value learner reads "fire =
     net-negative" -> emergent pyrophobia (session 013).

Two orthogonal levers, in a 2x2:
  - DENSE fire: fire_n 4 -> 12 (as dense as water/food) shortens every trip.
  - SHAPING  : potential-based fire-approach reward, active only when cold
               (temp<0.5). F = gamma*phi(s') - phi(s), phi = closeness-to-fire.
               Potential-based => provably preserves the optimal policy (Ng 1999);
               it only fills the zero-reward walk with a gradient, it does NOT pay
               the organism to sit in the fire.

Arms (solo TD scratch learner, NO teacher — this is about the world, not apprenticing):
  baseline (4, no-shape)  dense (12, no-shape)  shape (4, shape)  both (12, shape)

Win = denser and/or shaped fire lifts a SOLO learner's fire-use (occupancy up,
cold-deaths down, survival up) toward the oracle — confirming the trip-economy, not
a capacity ceiling, was the blocker. (If neither moves it, pyrophobia is deeper than
geometry and apprenticeship really is required.)
"""
from __future__ import annotations

import argparse
import statistics
import sys
from collections import deque, Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# importing fire_taming sets the tamed-fire physics (WARM_RATE=0.04 etc.)
from experiments.world.fire_taming import fire_oracle, _near_fire, _cause
from experiments.world.tile_world import TileWorld, N_ACTION, FIRE
from experiments.world.mirror_cells import build_mirror, _solo


def _p(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------
# Potential-based fire-approach shaping (cold-gated, policy-preserving)
# ----------------------------------------------------------------------
def fire_potential(w, k=0.3):
    """phi(s) = k * closeness-to-nearest-fire, but ONLY when cold (temp<0.5).
    Closeness = ||scent(FIRE)|| in [0,~1] (1 when adjacent). When warm, phi=0 so
    the shaping never rewards lingering at the fire once you're no longer cold."""
    if w.temp >= 0.5:
        return 0.0
    d = w._scent(FIRE)
    return k * float((d[0] * d[0] + d[1] * d[1]) ** 0.5)


# ----------------------------------------------------------------------
# Solo TD scratch learner (optionally fire-density + shaping; no teacher)
# ----------------------------------------------------------------------
def train_solo_eco(seed, episodes, *, fire_n=None, shaping=False, gamma=0.95,
                   lr=3e-3, batch=64, max_steps=300):
    sub = build_mirror(seed, n_mirror=8)
    opt = torch.optim.Adam(sub.trainable_tensors(), lr=lr)
    buf = deque(maxlen=20000)
    g = torch.Generator().manual_seed(seed + 31)
    for ep in range(episodes):
        eps = 0.1 + 0.9 * max(0.0, 1 - ep / (0.7 * episodes))
        w = TileWorld(seed=seed * 1000 + ep, max_steps=max_steps, fire_n=fire_n)
        p = w.percept(); done = False
        while not done:
            if torch.rand(1, generator=g).item() < eps:
                a = int(torch.randint(0, N_ACTION, (1,), generator=g))
            else:
                with torch.no_grad():
                    a = int(sub(_solo(p.unsqueeze(0)))[0].argmax())
            phi_s = fire_potential(w) if shaping else 0.0
            p2, r, done, info = w.step(a)
            if shaping:                                  # potential-based: F = g*phi' - phi
                phi_s2 = 0.0 if done else fire_potential(w)
                r = r + gamma * phi_s2 - phi_s
            buf.append((p, a, r, p2, float(done))); p = p2
            if len(buf) >= batch:
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
                torch.nn.utils.clip_grad_norm_(sub.trainable_tensors(), 1.0)
                sub.zero_dormant_grads(); opt.step()
    return sub


# ----------------------------------------------------------------------
# Evaluation in a world with the MATCHING fire density
# ----------------------------------------------------------------------
@torch.no_grad()
def evaluate_eco(sub, label, *, fire_n=None, seeds=40, max_steps=300):
    surv, near, tot = [], 0, 0
    causes = Counter()
    for ep in range(seeds):
        w = TileWorld(seed=ep * 7000 + 7, max_steps=max_steps, fire_n=fire_n)
        p = w.percept(); done = False
        while not done:
            a = int(sub(_solo(p.unsqueeze(0)))[0].argmax())
            tot += 1
            if _near_fire(w):
                near += 1
            p, _, done, info = w.step(a)
        surv.append(info["t"]); causes[_cause(w)] += 1
    fire_use = (100 * near / tot) if tot else 0.0
    _p(f"  {label:>22s}: survival {statistics.mean(surv):5.1f}  "
       f"cold-deaths {causes['cold']:2d}/{seeds}  fire-occupancy {fire_use:4.1f}%")
    return statistics.mean(surv), causes["cold"], fire_use


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.episodes = 1, 25

    _p("=== FIRE ECONOMY: is pyrophobia a move-COST problem? (solo TD, no teacher) ===")
    _p("    levers: DENSE fire (4->12 tiles) and/or potential-based approach SHAPING\n")
    _p("oracle reference (tamed physics, 4 fire) for the ceiling:")
    from experiments.world.fire_taming import evaluate as ft_eval
    ft_eval(lambda w, p: fire_oracle(w), "fire-oracle(4)", seeds=20)
    _p("")

    arms = {
        "baseline(4,plain)": dict(fire_n=4, shaping=False),
        "dense(12,plain)":   dict(fire_n=12, shaping=False),
        "shape(4,shaped)":   dict(fire_n=4, shaping=True),
        "both(12,shaped)":   dict(fire_n=12, shaping=True),
    }
    out = {}
    for name, cfg in arms.items():
        _p(f"########## {name} ##########")
        rows = []
        for s in range(args.seeds):
            sub = train_solo_eco(s, args.episodes, fire_n=cfg["fire_n"],
                                 shaping=cfg["shaping"])
            surv, cold, fire = evaluate_eco(sub, f"  seed{s}", fire_n=cfg["fire_n"])
            rows.append((surv, cold, fire))
        out[name] = rows

    _p("\n=== SUMMARY (means, n=%d) ===" % args.seeds)
    _p(f"  {'arm':>18s} | {'survival':>9s} | {'cold-deaths/40':>14s} | {'fire-occ%':>9s}")
    for name, rows in out.items():
        sv = statistics.mean(r[0] for r in rows)
        cd = statistics.mean(r[1] for r in rows)
        fo = statistics.mean(r[2] for r in rows)
        _p(f"  {name:>18s} | {sv:9.1f} | {cd:14.1f} | {fo:9.1f}")
    _p("\n  win = dense/shaped lifts fire-occ UP + cold-deaths DOWN vs baseline,")
    _p("  i.e. the SOLO learner overcomes pyrophobia once the trip pays -> it was")
    _p("  the move-economy, not a learning ceiling.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
