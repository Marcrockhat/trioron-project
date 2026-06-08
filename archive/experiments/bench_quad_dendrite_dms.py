"""Quad-dendrite hypothesis on DMS: do multiplicative interactions make the
match/compare RELATION learnable where linear+ReLU cannot?

Finding (bench_fine_temporal): the substrate recalls perfectly but cannot learn
delayed match-to-sample — loss frozen at chance. Hypothesis: interior cells are
linear (y = Σ w·a), which can't compute products a_i·a_j, and equality/matching
IS a product. The spec's dendrite phenotype uses the quad nonlinearity
σ(z)=z+z² (spec §3.6); z² = (Σ w·a)² expands to all pairwise products a_i·a_j —
the multiplicative term comparison needs.

But the shipped phenotype table registers DENDRITE → linear.forward_batch (a
stub — every phenotype is currently linear). So this bench IMPLEMENTS the quad
forward and injects it via a custom dispatch table, sets the dendrite phenotype
on interior cells, and re-runs DMS. Linear vs quad-dendrite, same task/harness.

Kept in experiments/ (custom dispatch, not core) pending spec review — per the
spec-is-source-of-truth discipline.
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
from trioron.core.epigenome import LINEAR, DENDRITE, PERCEPTION, OUTPUT, has_gene
from trioron.bases import seeded
from trioron.phenotype import default_dispatch_table, linear
from trioron.learning.manifold import get_interior_ids
from experiments.bench_grounded_world import N_SENSE
from experiments.bench_fine_temporal import make_prototypes, dms_batch, pregrow


def make_quad_forward(coef: float):
    """Dendrite forward: σ(z) = z + coef·z², z = linear combine. The z² term
    is Σ_i Σ_j w_i w_j a_i a_j — pairwise input products (multiplicative)."""
    def quad_forward_batch(act, bucket, arena):
        z = linear.forward_batch(act, bucket, arena)
        return z + coef * z * z
    return quad_forward_batch


def build(arm_topology, phenotype, *, h_init, budget, cap, seed, coef):
    H = h_init + budget
    table = default_dispatch_table()
    if phenotype == "quad":
        table[DENDRITE] = make_quad_forward(coef)
    sub = construct(
        base=seeded(N_SENSE + H, 2, interior_cells=h_init),
        envelope=Envelope(max_parameter_bytes=cap),
        dispatch_table=table, capacity=2048, sparsity_k=0,
    )
    sub.compile()
    torch.manual_seed(seed + 500)
    pregrow(arm_topology, sub, budget)
    if phenotype == "quad":
        # interior cells → dendrite phenotype: clear LINEAR (bit 0, always set),
        # set DENDRITE so primary_phenotype (lowest set gene) resolves to it.
        a = sub.arena
        for cid in a.alive_ids().tolist():
            epi = int(a.epigenome[cid].item())
            if has_gene(epi, PERCEPTION) or has_gene(epi, OUTPUT):
                continue
            a.epigenome[cid] = (epi & ~(1 << LINEAR)) | (1 << DENDRITE)
            a.refresh_phenotype(cid)
        a.rank_dirty = True
    sub.prepare_training()
    return sub


def run(arm_topology, phenotype, *, k, delay, h_init, budget, cap, seed,
        epochs, batch, lr, lam, noise, coef):
    g = torch.Generator().manual_seed(seed)
    protos = make_prototypes(k, g)
    sub = build(arm_topology, phenotype, h_init=h_init, budget=budget, cap=cap,
                seed=seed, coef=coef)
    iids = get_interior_ids(sub.arena).long(); Htr = iids.numel()
    opt = torch.optim.Adam(sub.trainable_tensors(), lr=lr)

    def run_seq(s, f, t):
        b = s.shape[0]; m = torch.zeros(b, Htr)
        steps = [s] + [f[:, i] for i in range(f.shape[1])] + [t]
        lg = None
        for p in steps:
            lg = sub(torch.cat([p, m], 1))
            h = sub.live_activations[:, iids]
            m = lam * m + (1 - lam) * h
            # Smooth RMS normalization: bounds the z² magnitude (no blowup) while
            # preserving the trace's *pattern* (the comparison signal) and keeping
            # gradient nonzero everywhere — unlike a hard clamp, which saturates to
            # zero gradient and strands some seeds.
            rms = m.pow(2).mean(dim=1, keepdim=True).sqrt() + 1e-5
            m = m / rms
        return lg

    for ep in range(epochs):
        for _ in range(40):
            s, f, t, y = dms_batch(batch, protos, delay, noise, g)
            lg = run_seq(s, f, t)
            loss = torch.nn.functional.cross_entropy(lg, y)
            opt.zero_grad(); loss.backward()
            # The quad z² term produces large gradients; clipping is required
            # for stable training (without it, training diverges to chance).
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
    ap.add_argument("--budget", type=int, default=20)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--delay", type=int, default=1)
    ap.add_argument("--lam", type=float, default=0.6)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--coef", type=float, default=1.0, help="quad coefficient")
    ap.add_argument("--cap", type=int, default=400_000)
    ap.add_argument("--topology", default="self-arrange")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.epochs = 1, 20

    print(f"bench_quad_dendrite_dms: DMS K={args.k} delay={args.delay} "
          f"topology={args.topology} coef={args.coef} seeds={args.seeds} "
          f"epochs={args.epochs}  (chance=0.500)")
    arms = [("linear", "linear (control — known to fail)"),
            ("quad", "quad-dendrite (hypothesis)")]
    for phen, label in arms:
        accs = []
        for seed in range(args.seeds):
            t0 = time.time()
            acc = run(args.topology, phen, k=args.k, delay=args.delay,
                      h_init=args.h_init, budget=args.budget, cap=args.cap,
                      seed=seed, epochs=args.epochs, batch=args.batch, lr=args.lr,
                      lam=args.lam, noise=args.noise, coef=args.coef)
            accs.append(acc)
            print(f"  [{phen:>5s} seed={seed}] acc={acc:.3f} ({time.time()-t0:.0f}s)")
        m = statistics.mean(accs); s = statistics.stdev(accs) if len(accs) > 1 else 0.0
        verdict = "LEARNS" if m > 0.65 else ("partial" if m > 0.55 else "fails")
        print(f"  === {phen}: {m:.3f}±{s:.3f}  [{verdict}]  — {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
