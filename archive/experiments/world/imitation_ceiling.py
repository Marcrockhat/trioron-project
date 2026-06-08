"""Conclusive adjudication: WHY does the consolidated organism survive ≈ random while the
masters survive 75-95? Three hypotheses, one test.

We measure IMITATION ACCURACY (predict the master's action on held-out decision states),
not survival — accuracy is the clean signal. Four arms, same data, same split:

  chance          majority-class action — the floor.
  MLP(percept)    a strong simple learner on the 74-d PERCEPT the organism actually sees.
  MLP(state)      the SAME MLP on the full privileged world state the master reads
                  (drives + dx,dy,proximity to nearest fire/water/food/berry).
  trioron(percept) the actual substrate (mirror cells) trained supervised on the percept.

Decision matrix:
  MLP(state) high, MLP(percept) low   → PERCEPTION GAP: the env's percept discards the
                                        decision info (Rocky's "env doesn't allow learning",
                                        in the partial-observability sense). Fix = upstream state.
  MLP(percept) high, trioron(percept) low → SUBSTRATE/TRAINING is the bottleneck. Fix = substrate.
  MLP(percept) high, trioron(percept) high → imitation is fine; the loss is elsewhere
                                        (DAgger/TD mixing in consolidate_base, or survival≠skill).
  MLP(state) ALSO low                 → the master uses info we didn't featurize
                                        (history/memory) → POMDP needs recurrence.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.world.fire_taming import fire_oracle, TileWorld
from experiments.world.quest import water_master
from experiments.world.mirror_cells import build_mirror, _solo
from experiments.world.tile_world import (
    N_ACTION, N_TILE, FIRE, WATER, FOOD, BERRY,
)


def _p(*a):
    print(*a, flush=True)


def _nearest_delta(w, tile):
    """Toroidal nearest cell of `tile`: signed (dx,dy)/size + proximity 1/(1+dist)."""
    s = w.size
    cells = (w.grid == tile).nonzero(as_tuple=False)
    if cells.numel() == 0:
        return [0.0, 0.0, 0.0]
    ys = cells[:, 0].float(); xs = cells[:, 1].float()
    dx = ((xs - w.px + s / 2) % s) - s / 2
    dy = ((ys - w.py + s / 2) % s) - s / 2
    dist = dx.abs() + dy.abs()
    i = int(dist.argmin())
    return [float(dx[i]) / s, float(dy[i]) / s, 1.0 / (1.0 + float(dist[i]))]


def state_features(w):
    """Full privileged world state the master conditions on (no master-internal logic)."""
    here = torch.zeros(N_TILE); here[int(w.grid[w.py, w.px])] = 1.0
    drives = [w.temp, w.thirst, w.energy, w.integrity, 1.0 if w.is_night else 0.0]
    spatial = []
    for t in (FIRE, WATER, FOOD, BERRY):
        spatial += _nearest_delta(w, t)
    return torch.tensor(drives + here.tolist() + spatial, dtype=torch.float32)


@torch.no_grad()
def collect(master_fn, which, *, seeds=30, cap=900, max_steps=300):
    P, S, Y = [], [], []
    for ep in range(seeds):
        w = TileWorld(seed=ep * 31 + 3, max_steps=max_steps)
        p = w.percept(); done = False
        while not done:
            if which == "all":
                keep = True
            elif which == "cold":
                keep = (w.temp < 0.5)
            else:
                keep = (w.thirst < 0.9)
            a = master_fn(w)
            if keep:
                P.append(p); S.append(state_features(w)); Y.append(a)
            p, _, done, _ = w.step(a)
            if len(P) >= cap:
                break
        if len(P) >= cap:
            break
    return torch.stack(P), torch.stack(S), torch.tensor(Y)


class MLP(torch.nn.Module):
    def __init__(self, d_in, h=128, d_out=N_ACTION):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, h), torch.nn.ReLU(),
            torch.nn.Linear(h, h), torch.nn.ReLU(),
            torch.nn.Linear(h, d_out))

    def forward(self, x):
        return self.net(x)


def train_eval_mlp(Xtr, Ytr, Xte, Yte, *, epochs=300, lr=3e-3, seed=0):
    torch.manual_seed(seed)
    m = MLP(Xtr.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    ce = torch.nn.functional.cross_entropy
    for _ in range(epochs):
        opt.zero_grad(); ce(m(Xtr), Ytr).backward(); opt.step()
    with torch.no_grad():
        return (m(Xte).argmax(1) == Yte).float().mean().item()


def train_eval_substrate(Ptr, Ytr, Pte, Yte, *, epochs=300, lr=3e-3, seed=0, batch=128):
    torch.manual_seed(seed)
    sub = build_mirror(seed, n_mirror=8)
    opt = torch.optim.Adam(sub.trainable_tensors(), lr=lr)
    ce = torch.nn.functional.cross_entropy
    n = Ptr.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            ce(sub(_solo(Ptr[idx])), Ytr[idx]).backward()
            sub.zero_dormant_grads()
            opt.step()
    with torch.no_grad():
        return (sub(_solo(Pte)).argmax(1) == Yte).float().mean().item()


def run_master(name, master_fn, which, args):
    P, S, Y = collect(master_fn, which, seeds=args.seeds, cap=args.cap)
    n = P.shape[0]
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(n, generator=g)
    ntr = int(n * 0.8)
    tr, te = perm[:ntr], perm[ntr:]
    # chance = majority-class action on the test set
    maj = int(torch.bincount(Y[tr], minlength=N_ACTION).argmax())
    chance = (Y[te] == maj).float().mean().item()

    acc_p = train_eval_mlp(P[tr], Y[tr], P[te], Y[te])
    acc_s = train_eval_mlp(S[tr], Y[tr], S[te], Y[te])
    acc_sub = train_eval_substrate(P[tr], Y[tr], P[te], Y[te])

    _p(f"\n### {name}  (n={n}, {N_ACTION} actions, majority-class={maj}) ###")
    _p(f"  chance (majority-class)     : {chance:.3f}")
    _p(f"  MLP(percept 74-d)           : {acc_p:.3f}")
    _p(f"  MLP(full privileged state)  : {acc_s:.3f}")
    _p(f"  trioron substrate (percept) : {acc_sub:.3f}")
    return dict(chance=chance, mlp_p=acc_p, mlp_s=acc_s, sub_p=acc_sub)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--cap", type=int, default=900)
    args = ap.parse_args()
    _p("=== IMITATION CEILING: substrate vs perception vs teacher ===")
    r_fire = run_master("FIRE-ORACLE (cold states)", fire_oracle, "cold", args)
    r_water = run_master("WATER-MASTER (thirsty states)", water_master, "thirsty", args)

    _p("\n=== VERDICT ===")
    for nm, r in (("fire", r_fire), ("water", r_water)):
        gap_perc = r["mlp_s"] - r["mlp_p"]      # state advantage = perception gap
        gap_sub = r["mlp_p"] - r["sub_p"]       # MLP advantage over substrate = substrate gap
        _p(f"  [{nm}] perception-gap (state−percept)={gap_perc:+.3f}   "
           f"substrate-gap (MLP−trioron)={gap_sub:+.3f}")
    _p("  big perception-gap → fix upstream state; big substrate-gap → fix substrate; "
       "both small & high → imitation fine, loss is in DAgger/TD or survival≠skill.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
