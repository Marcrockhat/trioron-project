"""How many leaves can the consequence-taught trioron router arbitrate? (2026-08-14)

router_trioron.py showed a TD-taught trioron router over 5 leaves discovers an
arbitration policy (163.2) near the hand-coded ceiling (167.6). This sweep asks
the WIDTH question: hold the training budget fixed and inflate the leaf roster —
does discovery still find the 5 real skills among distractors?

Roster at width N: the 5 real donors + (N-5) distractors, alternating
  noisy copies  — real donor weights + gaussian noise (a plausible-but-worse
                  sibling; the hard case: partially useful, must be out-ranked)
  random inits  — untrained substrates (useless; must be ignored)

Fixed: TD budget (episodes), eval protocol, seed. Measured per N:
  eval survival, last-30 training mean (speed), and the share of routing mass
  the 5 REAL leaves keep (did discovery find the needles?).
"""
from __future__ import annotations

import argparse
import statistics
import sys
from collections import deque
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.world.vocabulary import PRIM_ORDER, _p
from experiments.world.mirror_cells import build_mirror, _solo
from experiments.world.fire_taming import evaluate
from experiments.world.tile_world import TileWorld
from experiments.world.router_trioron import build_router_sub, load_leaves

PERCEPT_DIM = 77


def make_roster(width, *, noise=0.3, seed=0):
    """5 real donors + (width-5) distractors. Returns (list of substrates,
    names, is_real mask)."""
    donors = load_leaves()
    leaves = [donors[n] for n in PRIM_ORDER]
    names = list(PRIM_ORDER)
    real = [True] * len(leaves)
    g = torch.Generator().manual_seed(seed + 555)
    k = 0
    while len(leaves) < width:
        if k % 2 == 0:                       # noisy copy of a real donor
            src = donors[PRIM_ORDER[k // 2 % len(PRIM_ORDER)]]
            sub = build_mirror(seed + 100 + k, n_mirror=8)
            with torch.no_grad():
                sub.arena.bias.copy_(src.arena.bias
                                     + noise * torch.randn(src.arena.bias.shape,
                                                           generator=g))
                sub.arena.edge_weight.copy_(
                    src.arena.edge_weight
                    + noise * torch.randn(src.arena.edge_weight.shape,
                                          generator=g))
            names.append(f"noisy-{PRIM_ORDER[k // 2 % len(PRIM_ORDER)]}")
        else:                                # useless random-init substrate
            sub = build_mirror(seed + 200 + k, n_mirror=8)
            names.append(f"random-{k}")
        leaves.append(sub)
        real.append(False)
        k += 1
    return leaves, names, real


def train_td_router(seed, leaves, *, episodes, gamma=0.95, lr=3e-3, batch=64,
                    max_steps=300):
    n_leaf = len(leaves)
    sub = build_router_sub(seed)             # built for 5 outputs; rebuild below
    if n_leaf != 5:
        from trioron.core import Envelope, construct
        from trioron.bases import seeded
        from trioron.phenotype import default_dispatch_table
        torch.manual_seed(seed)
        sub = construct(
            base=seeded(PERCEPT_DIM, n_leaf, interior_cells=32, nonlinear=True),
            envelope=Envelope(max_parameter_bytes=800_000),
            dispatch_table=default_dispatch_table(), capacity=4096, sparsity_k=0,
        )
        sub.compile()
        sub.prepare_training()
    opt = torch.optim.Adam(sub.trainable_tensors(), lr=lr)
    buf = deque(maxlen=20000)
    g = torch.Generator().manual_seed(seed + 31)
    curve = []
    for ep in range(episodes):
        eps = 0.1 + 0.9 * max(0.0, 1 - ep / (0.7 * episodes))
        w = TileWorld(seed=seed * 1000 + ep, max_steps=max_steps)
        p = w.percept(); done = False
        while not done:
            if torch.rand(1, generator=g).item() < eps:
                i = int(torch.randint(0, n_leaf, (1,), generator=g))
            else:
                with torch.no_grad():
                    i = int(sub(p.unsqueeze(0))[0].argmax())
            with torch.no_grad():
                a = int(leaves[i](_solo(p.unsqueeze(0)))[0].argmax())
            p2, r, done, info = w.step(a)
            buf.append((p, i, r, p2, float(done))); p = p2
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
        curve.append(info["t"])
    return sub, curve


def run_width(width, *, seed, episodes, eval_seeds):
    leaves, names, real = make_roster(width, seed=seed)
    sub, curve = train_td_router(seed, leaves, episodes=episodes)
    hist = [0] * len(leaves)

    @torch.no_grad()
    def act(w, p):
        i = int(sub(p.unsqueeze(0))[0].argmax())
        hist[i] += 1
        return int(leaves[i](_solo(p.unsqueeze(0)))[0].argmax())

    surv, _, _ = evaluate(act, f"N={width:>2d} routed", seeds=eval_seeds)
    tot = max(sum(hist), 1)
    real_mass = sum(h for h, r in zip(hist, real) if r) / tot
    top = sorted(zip(names, hist), key=lambda x: -x[1])[:6]
    _p(f"      real-leaf routing mass: {real_mass:.2f}   top used: "
       + "  ".join(f"{n}={100*h/tot:.0f}%" for n, h in top))
    return dict(width=width, surv=surv, tail=statistics.mean(curve[-30:]),
                real_mass=real_mass)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", type=str, default="5,10,20,40")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--eval-seeds", type=int, default=40)
    args = ap.parse_args()
    widths = [int(x) for x in args.widths.split(",")]

    _p("=== ROUTER WIDTH SWEEP — TD discovery vs roster size (fixed budget) ===")
    _p(f"seed={args.seed} episodes={args.episodes} eval_seeds={args.eval_seeds} "
       f"widths={widths}\n")
    rows = [run_width(wd, seed=args.seed, episodes=args.episodes,
                      eval_seeds=args.eval_seeds) for wd in widths]

    _p("\n=== SUMMARY (bars: 5-leaf TD router 163.2, arbiter 167.6, "
       "reactive 141.8) ===")
    _p(f"  {'N':>4s} | {'eval surv':>9s} | {'train tail':>10s} | {'real mass':>9s}")
    for r in rows:
        _p(f"  {r['width']:>4d} | {r['surv']:>9.1f} | {r['tail']:>10.1f} | "
           f"{r['real_mass']:>9.2f}")
    _p("\n  read: survival flat as N grows → routing scales; falling survival +"
       " falling real-mass → exploration cost is the wall at this budget.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
