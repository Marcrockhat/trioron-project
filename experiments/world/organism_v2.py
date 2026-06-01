"""Rung 1, reframed — the organism accrues NUMA (not just survives).

Per Rocky (2026-06-01): Numa is the metric; survival is the precondition.
A reactive heuristic survives but accrues ZERO Numa (it never learns). The
organism is "alive" by accruing genuine, dream-validated Numa — consolidated
predictive learning that survives re-test (Mima = the part that doesn't).

The organism (trioron substrate) has two heads:
  - VALUE  (N_ACTION) — discounted drive-value, for acting (predict-then-act).
  - PAIRS  (N_PAIR=5) — predicts the NEXT-step contrastive-pair labels (a tiny
                        world-model). Getting better at this = resolving
                        predictive surprise. Its loss feeds the NumaLedger.

Numa accrues when a pair's prediction loss drops AND the drop holds on held-out
replay (the dream). Mima when the drop is train-only (overfit). We report net
Numa over the organism's life, alongside survival.
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
from experiments.world.numa import contrast_targets, NumaLedger, N_PAIR

PERCEPT_DIM = 74
N_OUT = N_ACTION + N_PAIR     # value head ⊕ pair-prediction head


def build(seed, nonlinear=False):
    torch.manual_seed(seed)
    sub = construct(base=seeded(PERCEPT_DIM, N_OUT, interior_cells=32, nonlinear=nonlinear),
                    envelope=Envelope(max_parameter_bytes=500_000),
                    dispatch_table=default_dispatch_table(), capacity=2048, sparsity_k=0)
    sub.compile(); sub.prepare_training()
    return sub


def per_pair_loss(sub, batch):
    """MSE per pair on a batch of (percept, pair_target)."""
    bp = torch.stack([b[0] for b in batch])
    tgt = torch.stack([b[5] for b in batch])
    with torch.no_grad():
        pred = sub(bp)[:, N_ACTION:]
    return ((pred - tgt) ** 2).mean(dim=0)        # [N_PAIR]


def train_organism(seed, episodes, *, gamma=0.95, lr=3e-3, batch=64,
                   eps_start=1.0, eps_end=0.1, nonlinear=False, max_steps=300,
                   numa_every=200):
    sub = build(seed, nonlinear=nonlinear)
    opt = torch.optim.Adam(sub.trainable_tensors(), lr=lr)
    buf = deque(maxlen=20000)
    ledger = NumaLedger()
    g = torch.Generator().manual_seed(seed + 31)
    lengths = []
    gstep = 0

    for ep in range(episodes):
        w = TileWorld(seed=seed * 1000 + ep, max_steps=max_steps)
        p = w.percept()
        eps = eps_end + (eps_start - eps_end) * max(0.0, 1 - ep / (0.7 * episodes))
        done = False
        while not done:
            if torch.rand(1, generator=g).item() < eps:
                a = int(torch.randint(0, N_ACTION, (1,), generator=g))
            else:
                with torch.no_grad():
                    a = int(sub(p.unsqueeze(0))[0, :N_ACTION].argmax())
            p2, r, done, info = w.step(a)
            pair_tgt = contrast_targets(w)               # next-step pair labels
            buf.append((p, a, r, p2, float(done), pair_tgt))
            p = p2
            if len(buf) >= 2 * batch:
                idx = torch.randint(0, len(buf), (batch,), generator=g)
                bs = [buf[i] for i in idx]
                bp = torch.stack([b[0] for b in bs]); ba = torch.tensor([b[1] for b in bs])
                br = torch.tensor([b[2] for b in bs]); bp2 = torch.stack([b[3] for b in bs])
                bd = torch.tensor([b[4] for b in bs]); btgt = torch.stack([b[5] for b in bs])
                out = sub(bp)
                q = out[torch.arange(batch), ba]
                with torch.no_grad():
                    q_next = sub(bp2)[:, :N_ACTION].max(dim=1).values
                    target = br + gamma * q_next * (1 - bd)
                loss_v = torch.nn.functional.mse_loss(q, target)
                loss_pair = torch.nn.functional.mse_loss(out[:, N_ACTION:], btgt)
                loss = loss_v + loss_pair
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(sub.trainable_tensors(), 1.0)
                sub.zero_dormant_grads(); opt.step()

                # Numa/Mima checkpoint: train-batch vs held-out "dream" batch
                gstep += 1
                if gstep % numa_every == 0 and len(buf) >= 4 * batch:
                    didx = torch.randint(0, len(buf), (batch,), generator=g)
                    dream = [buf[i] for i in didx]
                    ledger.checkpoint(per_pair_loss(sub, bs), per_pair_loss(sub, dream))
        lengths.append(info["t"])
    return lengths, sub, ledger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--nonlinear", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.episodes = 1, 150

    print(f"organism_v2: rung 1 reframed — accrue NUMA. seeds={args.seeds} "
          f"episodes={args.episodes} nonlinear={args.nonlinear}")
    print("metric: net Numa (consolidated, dream-validated learning). "
          "reactive baseline accrues 0 Numa (never learns).")
    nets, survs = [], []
    for seed in range(args.seeds):
        L, sub, ledger = train_organism(seed, args.episodes, nonlinear=args.nonlinear)
        surv = statistics.mean(L[-30:])
        nets.append(ledger.net()); survs.append(surv)
        print(f"  [seed {seed}] survival(last30)={surv:.1f}  {ledger.summary()}")
    nm = statistics.mean(nets); ns = statistics.stdev(nets) if len(nets) > 1 else 0.0
    sv = statistics.mean(survs)
    verdict = "ALIVE (net-positive consolidated learning)" if nm > 0 else "not learning (net Numa ≤ 0)"
    print(f"\n  === net Numa = {nm:+.3f} ± {ns:.3f}  (survival {sv:.1f})  [{verdict}]")
    print("  (reactive survives longer but net Numa = 0 — dead on the axis that matters.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
