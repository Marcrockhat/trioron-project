"""A TRIORON as the router — closing the honesty gap (2026-08-14).

The vocabulary organism's 148.1 came from trioron leaves under a GAUSSIAN
dispatcher fitted to hand-coded danger labels — not "a trioron utilizing
triorons". This experiment puts a real substrate at the top of the nest:

  arm (a) IMITATION-taught router — seeded(77 -> 5) substrate, nonlinear,
          full-credit CE on the same argmax-danger labels the Gaussian used.
          Same supervision, trioron computation. The fallback.
  arm (b) CONSEQUENCE-taught router — same substrate, Q-learning over the
          5 leaf-choices; reward = the world's drive-delta signal. The
          arbitration policy is DISCOVERED from dying, never shown the
          danger formula. The real test.

Both use frozen leaf donors (runs/primitives/*.pt). Bars: the Gaussian
vocabulary router (148.1), the hand-coded argmax-danger arbiter (167.6),
reactive floor (141.8). Diagnosis lessons baked in: nonlinear=True (linear
caps at 0.60 on arbitration), no mirror-gating (full credit).
"""
from __future__ import annotations

import argparse
import statistics
import sys
from collections import deque
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.world.vocabulary import (
    PRIM_ORDER, danger, build_vocabulary, ArbiterOrganism, _p,
)
from experiments.world.primitives import PRIM_DIR, load_donor
from experiments.world.mirror_cells import _solo
from experiments.world.fire_taming import evaluate
from experiments.world.tile_world import TileWorld, run_reactive
from trioron.core import Envelope, construct
from trioron.bases import seeded
from trioron.phenotype import default_dispatch_table

PERCEPT_DIM = 77
N_PRIM = len(PRIM_ORDER)


def build_router_sub(seed, *, nonlinear=True):
    torch.manual_seed(seed)
    sub = construct(
        base=seeded(PERCEPT_DIM, N_PRIM, interior_cells=32, nonlinear=nonlinear),
        envelope=Envelope(max_parameter_bytes=400_000),
        dispatch_table=default_dispatch_table(), capacity=2048, sparsity_k=0,
    )
    sub.compile()
    sub.prepare_training()
    return sub


def load_leaves():
    donors = {}
    for name in PRIM_ORDER:
        donors[name], _ = load_donor(PRIM_DIR / f"{name}.pt")
    return donors


class TrioronRoutedOrganism:
    """percept -> router substrate -> leaf substrate -> action."""

    def __init__(self, router_sub, donors):
        self.router = router_sub
        self.donors = donors
        self.route_hist = [0] * N_PRIM

    @torch.no_grad()
    def act(self, w, p):
        i = int(self.router(p.unsqueeze(0))[0].argmax())
        self.route_hist[i] += 1
        leaf = self.donors[PRIM_ORDER[i]]
        return int(leaf(_solo(p.unsqueeze(0)))[0].argmax())


# ----------------------------------------------------------------------
# Arm (a): imitation-taught — CE on argmax-danger labels
# ----------------------------------------------------------------------
@torch.no_grad()
def collect_labeled(*, seeds=120, cap=7000, min_danger=0.15, max_steps=300):
    P, Y = [], []
    g = torch.Generator().manual_seed(12345)
    for ep in range(seeds):
        w = TileWorld(seed=ep * 17 + 5, max_steps=max_steps)
        p = w.percept(); done = False
        while not done:
            d = danger(w)
            lbl = max(PRIM_ORDER, key=lambda k: d[k])
            if d[lbl] >= min_danger:
                P.append(p); Y.append(PRIM_ORDER.index(lbl))
            a = int(torch.randint(0, 6, (1,), generator=g))
            p, _, done, _ = w.step(a)
        if len(P) >= cap:
            break
    return torch.stack(P), torch.tensor(Y)


def train_router_imitation(seed, *, epochs=300, lr=3e-3, batch=128):
    P, Y = collect_labeled()
    n = P.shape[0]
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(n, generator=g)
    ntr = int(n * 0.8)
    tr, te = perm[:ntr], perm[ntr:]
    sub = build_router_sub(seed)
    opt = torch.optim.Adam(sub.trainable_tensors(), lr=lr)
    ce = torch.nn.functional.cross_entropy
    for _ in range(epochs):
        pm = torch.randperm(ntr, generator=g)
        for i in range(0, ntr, batch):
            idx = tr[pm[i:i + batch]]
            opt.zero_grad()
            ce(sub(P[idx]), Y[idx]).backward()
            sub.zero_dormant_grads()
            opt.step()
    with torch.no_grad():
        acc = (sub(P[te]).argmax(1) == Y[te]).float().mean().item()
    return sub, acc


# ----------------------------------------------------------------------
# Arm (b): consequence-taught — Q-learning over leaf-choices
# ----------------------------------------------------------------------
def train_router_td(seed, donors, *, episodes=300, gamma=0.95, lr=3e-3,
                    batch=64, max_steps=300):
    sub = build_router_sub(seed)
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
                i = int(torch.randint(0, N_PRIM, (1,), generator=g))
            else:
                with torch.no_grad():
                    i = int(sub(p.unsqueeze(0))[0].argmax())
            with torch.no_grad():
                a = int(donors[PRIM_ORDER[i]](_solo(p.unsqueeze(0)))[0].argmax())
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


@torch.no_grad()
def agreement_with_formula(router_sub, *, seeds=20, max_steps=300):
    """How often does the discovered policy pick the formula's choice?"""
    hit = tot = 0
    g = torch.Generator().manual_seed(999)
    for ep in range(seeds):
        w = TileWorld(seed=ep * 41 + 11, max_steps=max_steps)
        p = w.percept(); done = False
        while not done:
            d = danger(w)
            lbl = max(PRIM_ORDER, key=lambda k: d[k])
            if d[lbl] >= 0.15:
                i = int(router_sub(p.unsqueeze(0))[0].argmax())
                hit += int(PRIM_ORDER[i] == lbl); tot += 1
            a = int(torch.randint(0, 6, (1,), generator=g))
            p, _, done, _ = w.step(a)
    return hit / max(tot, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--td-episodes", type=int, default=300)
    ap.add_argument("--imit-epochs", type=int, default=300)
    ap.add_argument("--eval-seeds", type=int, default=40)
    args = ap.parse_args()

    _p("=== TRIORON ROUTER: (b) consequence-taught vs (a) imitation fallback ===")
    _p(f"seed={args.seed} td_eps={args.td_episodes} imit_epochs={args.imit_epochs} "
       f"eval_seeds={args.eval_seeds} nonlinear=True full-credit\n")
    donors = load_leaves()

    _p("arm (a): imitation-taught trioron router (danger labels)...")
    sub_a, acc_a = train_router_imitation(args.seed, epochs=args.imit_epochs)
    _p(f"  held-out label accuracy: {acc_a:.3f}")
    org_a = TrioronRoutedOrganism(sub_a, donors)
    surv_a, _, _ = evaluate(lambda w, p: org_a.act(w, p),
                            "(a) imit trioron-router", seeds=args.eval_seeds)
    tot = max(sum(org_a.route_hist), 1)
    _p("      route usage: " + "  ".join(
        f"{PRIM_ORDER[i]}={100*org_a.route_hist[i]/tot:.0f}%" for i in range(N_PRIM)))

    _p("\narm (b): consequence-taught trioron router (Q over leaf-choices)...")
    sub_b, curve = train_router_td(args.seed, donors, episodes=args.td_episodes)
    tail = statistics.mean(curve[-30:])
    _p(f"  training survival, last-30-episode mean: {tail:.1f}")
    org_b = TrioronRoutedOrganism(sub_b, donors)
    surv_b, _, _ = evaluate(lambda w, p: org_b.act(w, p),
                            "(b) TD trioron-router  ", seeds=args.eval_seeds)
    tot = max(sum(org_b.route_hist), 1)
    _p("      route usage: " + "  ".join(
        f"{PRIM_ORDER[i]}={100*org_b.route_hist[i]/tot:.0f}%" for i in range(N_PRIM)))
    agree = agreement_with_formula(sub_b)
    _p(f"      agreement with the hand-coded danger formula: {agree:.3f} "
       "(NOT a target — just: did discovery converge to our design or elsewhere?)")

    _p("\nbars on the same eval:")
    org_g, _ = build_vocabulary()
    surv_g, _, _ = evaluate(lambda w, p: org_g.act(w, p),
                            "Gaussian vocab router  ", seeds=args.eval_seeds)
    arb = ArbiterOrganism(donors)
    surv_arb, _, _ = evaluate(lambda w, p: arb.act(w, p),
                              "hand-coded arbiter     ", seeds=args.eval_seeds)
    rea = statistics.mean(run_reactive(0, episodes=args.eval_seeds))

    _p("\n=== VERDICT (single seed) ===")
    _p(f"  (b) discovered trioron router : {surv_b:.1f}")
    _p(f"  (a) imitation trioron router  : {surv_a:.1f}")
    _p(f"  Gaussian router (prev nest)   : {surv_g:.1f}")
    _p(f"  hand-coded arbiter (ceiling)  : {surv_arb:.1f}")
    _p(f"  reactive floor                : {rea:.1f}")
    _p("\n  100%-trioron claim holds if (b) ≥ Gaussian router; "
       "(a) is the fallback if TD credit fails.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
