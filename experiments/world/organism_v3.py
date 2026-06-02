"""Rung 1, loop closed — the organism ACTS on its learned world-model.

v2 learned a world-model (accrued Numa) but acted via a separate value head, so
learning and survival were decoupled — survival stayed flat (~18). v3 closes the
loop: the trioron IS a transition model M(state, action) → next state, and the
organism PLANS — a short greedy rollout through M — to choose the action that
keeps its drives healthiest. Now better learning → better rollouts → (if the
loop is real) longer survival. This is the test of whether Numa-style learning
is USEFUL, not just present.

State for planning = the organism's own interoception + scent (derivable from
its percept; no privileged world access): drives (4) + scent gradients (6) +
night (1) = 11-d. Rollout horizon H with discount; health = drive balance.
"""
from __future__ import annotations

import argparse
import statistics
import sys
from collections import deque
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from trioron.core import Envelope, construct
from trioron.bases import seeded
from trioron.phenotype import default_dispatch_table
from experiments.world.tile_world import TileWorld, N_ACTION

C_DIM = 11            # compact state: drives(4) + scent(6) + night(1)
M_IN = C_DIM + N_ACTION


def compact(percept):
    """Extract the planning state from the percept (layout: 63 local | 6 scent
    | 4 drives | 1 night). Drives first so health() can read them."""
    scent = percept[63:69]; drives = percept[69:73]; night = percept[73:74]
    return torch.cat([drives, scent, night])


def health(c):
    """Drive balance: energy + thirst + integrity − 2|temp−0.5| (c[:4])."""
    e, th, integ, temp = c[..., 0], c[..., 1], c[..., 2], c[..., 3]
    return e + th + integ - 2 * (temp - 0.5).abs()


def build(seed, nonlinear=False):
    torch.manual_seed(seed)
    sub = construct(base=seeded(M_IN, C_DIM, interior_cells=48, nonlinear=nonlinear),
                    envelope=Envelope(max_parameter_bytes=500_000),
                    dispatch_table=default_dispatch_table(), capacity=2048, sparsity_k=0)
    sub.compile(); sub.prepare_training()
    return sub


def _onehot(a, n=N_ACTION):
    v = torch.zeros(n); v[a] = 1.0; return v


@torch.no_grad()
def plan(sub, c, horizon, gamma):
    """Greedy rollout through M: for each first action, roll the model forward
    picking the locally-best action, accumulate discounted health. Return argmax
    first action."""
    best_a, best_val = 0, -1e9
    for a0 in range(N_ACTION):
        c1 = sub(torch.cat([c, _onehot(a0)]).unsqueeze(0))[0]
        val = float(health(c1))
        cc = c1
        for t in range(1, horizon):
            # locally-best next action under the model
            cand = sub(torch.stack([torch.cat([cc, _onehot(a)]) for a in range(N_ACTION)]))
            hh = health(cand)
            bi = int(hh.argmax())
            cc = cand[bi]
            val += (gamma ** t) * float(hh[bi])
        if val > best_val:
            best_val, best_a = val, a0
    return best_a


def train_organism(seed, episodes, *, gamma=0.9, lr=3e-3, batch=64, horizon=5,
                   eps_start=1.0, eps_end=0.05, nonlinear=False, max_steps=300):
    sub = build(seed, nonlinear=nonlinear)
    opt = torch.optim.Adam(sub.trainable_tensors(), lr=lr)
    buf = deque(maxlen=20000)
    g = torch.Generator().manual_seed(seed + 31)
    lengths = []
    for ep in range(episodes):
        w = TileWorld(seed=seed * 1000 + ep, max_steps=max_steps)
        c = compact(w.percept())
        eps = eps_end + (eps_start - eps_end) * max(0.0, 1 - ep / (0.7 * episodes))
        done = False
        while not done:
            if torch.rand(1, generator=g).item() < eps:
                a = int(torch.randint(0, N_ACTION, (1,), generator=g))
            else:
                a = plan(sub, c, horizon, gamma)
            p2, r, done, info = w.step(a)
            c2 = compact(p2)
            buf.append((c, a, c2))
            c = c2
            if len(buf) >= batch:                  # train the transition model M
                idx = torch.randint(0, len(buf), (batch,), generator=g)
                bs = [buf[i] for i in idx]
                bin_ = torch.stack([torch.cat([b[0], _onehot(b[1])]) for b in bs])
                btgt = torch.stack([b[2] for b in bs])
                pred = sub(bin_)
                loss = torch.nn.functional.mse_loss(pred, btgt)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(sub.trainable_tensors(), 1.0)
                sub.zero_dormant_grads(); opt.step()
        lengths.append(info["t"])
    return lengths, sub


@torch.no_grad()
def eval_greedy(sub, seed, horizon, gamma, episodes=30, max_steps=300):
    lengths = []
    for ep in range(episodes):
        w = TileWorld(seed=seed * 7000 + ep, max_steps=max_steps)
        c = compact(w.percept()); done = False
        while not done:
            a = plan(sub, c, horizon, gamma)
            p2, _, done, info = w.step(a); c = compact(p2)
        lengths.append(info["t"])
    return lengths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--nonlinear", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.episodes = 1, 120

    print(f"organism_v3: loop closed — model-based planning (horizon={args.horizon}). "
          f"seeds={args.seeds} episodes={args.episodes} nonlinear={args.nonlinear}")
    print("baselines: random≈15.1, reactive≈21.9 | v1(model-free Q)≈18.4")
    final = []
    for seed in range(args.seeds):
        L, sub = train_organism(seed, args.episodes, horizon=args.horizon,
                                nonlinear=args.nonlinear)
        greedy = eval_greedy(sub, seed, args.horizon, 0.9)
        gm = statistics.mean(greedy)
        final.append(gm)
        print(f"  [seed {seed}] train early={statistics.mean(L[:30]):.1f} → "
              f"late={statistics.mean(L[-30:]):.1f} | GREEDY={gm:.1f} (max {max(greedy)})")
    m = statistics.mean(final); s = statistics.stdev(final) if len(final) > 1 else 0.0
    verdict = "ALIVE (beats reactive)" if m > 21.9 else (
        "beats v1/random, not reactive" if m > 18.4 else "no better than v1")
    print(f"\n  === model-based survival: {m:.1f}±{s:.1f}  [{verdict}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
