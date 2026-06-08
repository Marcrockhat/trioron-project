"""Unison v1 — triorons + satellites + quad in ONE architecture (2026-06-01).

The session built three pieces separately:
  - linear triorons      — sensing (base substrate).
  - satellites           — MEMORY (persistent leaky trace; experiments/satellites_v1).
  - quad cells           — COMPARISON (σ(z)=z+z²; trioron/phenotype/dendrite.py).

This wires them into a single substrate that solves a task needing ALL of them:
full delayed match-to-sample (DMS). Data flow over the sequence:

    percept → interior(linear) → satellites(memory of the sample, persists)
    at the test step:  quad cells read [satellite memory ⊕ current interior]
                       → products → match signal → output

So satellites supply the remembered sample, quad cells compute the relation
between it and the current test, and the head reads the comparison. Neither
piece alone suffices: memory-only carries the sample but can't compare (linear);
quad-only can compare but has nothing to compare against across the delay.

Arms:
  memory-only  satellites → output (linear). Recalls, can't compare → fails DMS.
  unison       satellites → quad ← interior, quad → output. Memory + comparison.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trioron.core import Envelope, construct
from trioron.core.epigenome import LINEAR, DENDRITE, RECURRENT, OUTPUT, has_gene
from trioron.bases import seeded
from trioron.phenotype import default_dispatch_table
from trioron.learning.manifold import get_interior_ids
from experiments.bench_grounded_world import N_SENSE
from experiments.bench_fine_temporal import make_prototypes, dms_batch
from experiments.satellites_v1 import SatelliteOp


def _alloc_cell(a, gene):
    cid = int(a.alloc(1)[0].item())
    a.parent[cid] = -1
    a.lineage_root[cid] = cid
    a.output_dim[cid] = 1
    a.epigenome[cid] = (1 << gene)              # clear LINEAR; express `gene`
    a.refresh_phenotype(cid)
    return cid


def _edges(a, src_ids, dst, w=0.3):
    s = torch.as_tensor(src_ids, dtype=torch.int32)
    a.add_edges(s, torch.full_like(s, dst), w * torch.randn(s.numel()))


def build(arm, n_sat, n_quad, *, h_init, cap, seed, lam):
    torch.manual_seed(seed)
    op = SatelliteOp(lam=lam)
    table = default_dispatch_table()            # DENDRITE → real quad already
    table[RECURRENT] = op                        # satellites
    sub = construct(base=seeded(N_SENSE, 2, interior_cells=h_init),
                    envelope=Envelope(max_parameter_bytes=cap),
                    dispatch_table=table, capacity=2048, sparsity_k=0)
    sub.compile()
    a = sub.arena
    interior = get_interior_ids(a).long().tolist()
    output = [c for c in a.alive_ids().tolist()
              if has_gene(int(a.epigenome[c].item()), OUTPUT)]

    # satellites (memory): read interior, persist a leaky trace
    sat_ids = []
    for _ in range(n_sat):
        cid = _alloc_cell(a, RECURRENT)
        _edges(a, interior, cid)                  # interior → satellite (write memory)
        for o in output:                          # satellites → output (linear path)
            _edges(a, [cid], o, w=0.1)
        sat_ids.append(cid)

    # quad cells (comparison): read satellites ⊕ interior, write output
    quad_ids = []
    if arm == "unison":
        for _ in range(n_quad):
            cid = _alloc_cell(a, DENDRITE)
            _edges(a, sat_ids + interior, cid)    # compare memory vs current
            for o in output:
                _edges(a, [cid], o, w=0.1)
            quad_ids.append(cid)

    a.rank_dirty = True
    sub.prepare_training()
    return sub, op, sorted(sat_ids), quad_ids


def run(arm, *, n_sat, n_quad, delay, h_init, cap, seed, epochs, batch, lr, lam, noise):
    g = torch.Generator().manual_seed(seed)
    protos = make_prototypes(4, g)
    sub, op, sat_ids, quad_ids = build(arm, n_sat, n_quad, h_init=h_init, cap=cap,
                                       seed=seed, lam=lam)
    nsat = len(sat_ids)
    opt = torch.optim.Adam(sub.trainable_tensors(), lr=lr)

    def run_seq(sample, fill, test):
        b = sample.shape[0]
        op.reset(b, nsat)
        steps = [sample] + [fill[:, t] for t in range(fill.shape[1])] + [test]
        logits = None
        for percept in steps:
            logits = sub(percept)
        return logits

    for ep in range(epochs):
        for _ in range(40):
            s, f, t, y = dms_batch(batch, protos, delay, noise, g)
            logits = run_seq(s, f, t)
            loss = torch.nn.functional.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(sub.trainable_tensors(), 1.0)
            sub.zero_dormant_grads(); opt.step()
    with torch.no_grad():
        s, f, t, y = dms_batch(2000, protos, delay, noise, g)
        correct = 0
        for i in range(0, 2000, 250):
            sl = slice(i, i + 250)
            correct += int((run_seq(s[sl], f[sl], t[sl]).argmax(-1) == y[sl]).sum())
    return correct / 2000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--h-init", type=int, default=8)
    ap.add_argument("--n-sat", type=int, default=12)
    ap.add_argument("--n-quad", type=int, default=12)
    ap.add_argument("--delay", type=int, default=1)
    ap.add_argument("--lam", type=float, default=0.6)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--cap", type=int, default=500_000)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.epochs = 1, 20

    print(f"unison_v1: full DMS (memory+comparison) in one substrate. "
          f"n_sat={args.n_sat} n_quad={args.n_quad} delay={args.delay} "
          f"seeds={args.seeds} epochs={args.epochs}  (chance=0.500)")
    for arm, name in [("memory-only", "satellites only (recall, can't compare)"),
                      ("unison", "satellites + quad (memory + comparison)")]:
        accs = []
        for seed in range(args.seeds):
            t0 = time.time()
            acc = run(arm, n_sat=args.n_sat, n_quad=args.n_quad, delay=args.delay,
                      h_init=args.h_init, cap=args.cap, seed=seed, epochs=args.epochs,
                      batch=args.batch, lr=args.lr, lam=args.lam, noise=args.noise)
            accs.append(acc)
            print(f"  [{arm:>11s} s{seed}] acc={acc:.3f} ({time.time()-t0:.0f}s)")
        m = statistics.mean(accs); s = statistics.stdev(accs) if len(accs) > 1 else 0.0
        v = "SOLVES" if m > 0.65 else "fails"
        print(f"  === {arm}: {m:.3f}±{s:.3f}  [{v}]  — {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
