"""Satellites v1 — a new cell type alongside the triorons (Rocky, 2026-06-01).

Biological analogue: satellite glial cells — support cells that surround neurons
and regulate their resource environment. Here, satellites are the substrate's
memory + homeostasis organ. They have feats the triorons lack:

  v1 (this file):
    1. MEMORY        — persistent leaky state across forward passes (triorons are
                       stateless). Held intrinsically in the satellite cell, with
                       BPTT through it — NO external Python trace.
    2. RESOURCE SENSE — reads the substrate's own budget (arena.pressure()); the
                       triorons are blind to it.
    3. DIVISION      — satellites spawn (here: gated by sensed resource — they
                       refuse to divide when the envelope is tight).
  v2 (later): modulation of trioron gain/plasticity; phase (wake/dream) timescale;
             quad relational compute.

Implementation: satellites occupy the `recurrent` phenotype slot (epigenome bit
3) — they ARE the stateful/recurrent cell type, which ships today as a linear
stub. A custom dispatch table registers RECURRENT → SatelliteOp; no core change.

Test: delayed grounded recall. Memory lives in satellite cells wired
interior→satellite→output. temporal-on (satellites persist) should recover the
cue at the probe step; temporal-off (satellites reset every step) cannot.
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
from trioron.core.epigenome import LINEAR, RECURRENT, OUTPUT, PERCEPTION, has_gene
from trioron.bases import seeded
from trioron.phenotype import default_dispatch_table, linear
from trioron.learning.manifold import get_interior_ids
from experiments.bench_grounded_world import N_SENSE, N_CONSEQ
from experiments.bench_temporal_gate import balanced_cues


# ----------------------------------------------------------------------
# The satellite forward: memory (leaky trace) + resource sensing
# ----------------------------------------------------------------------
class SatelliteOp:
    """Stateful forward for satellite cells (RECURRENT phenotype slot).

    Holds a per-satellite leaky trace persisting across forward() calls (the
    feat triorons lack). Reads arena.pressure() as a resource signal. The trace
    is a live tensor (not detached) → BPTT flows through the sequence.
    """
    def __init__(self, lam: float = 0.6, w_res: float = 0.0):
        self.lam = lam
        self.w_res = w_res
        self.trace = None          # [batch, n_sat] in ascending sat-id order
        self.hold = True           # False ⇒ reset every step (temporal-off)
        self.last_pressure = 0.0

    def reset(self, batch: int, n_sat: int):
        self.trace = torch.zeros(batch, n_sat)

    def __call__(self, act, bucket, arena):
        z = linear.forward_batch(act, bucket, arena)     # [batch, n_bucket]
        order = torch.argsort(bucket.cell_ids)           # → ascending cell-id order
        z_sorted = z[:, order]
        r = float(arena.pressure())                      # RESOURCE SENSING
        self.last_pressure = r
        if self.trace is None or self.trace.shape != z_sorted.shape:
            self.trace = torch.zeros_like(z_sorted)
        if self.hold:
            self.trace = self.lam * self.trace + (1.0 - self.lam) * z_sorted
        else:
            self.trace = z_sorted                        # memoryless
        out_sorted = self.trace + self.w_res * r
        inv = torch.argsort(order)
        return out_sorted[:, inv]


# ----------------------------------------------------------------------
# Build a substrate with satellites wired interior → satellite → output
# ----------------------------------------------------------------------
def add_satellite(arena, src_ids, dst_ids, seed_w=0.3):
    """Allocate one satellite cell: RECURRENT phenotype, edges from src_ids
    (interior) and to dst_ids (output)."""
    cid = int(arena.alloc(1)[0].item())
    arena.parent[cid] = -1
    arena.lineage_root[cid] = cid
    arena.output_dim[cid] = 1
    # RECURRENT phenotype: clear LINEAR (bit 0 always set; primary_phenotype
    # returns the lowest set expression gene) so it resolves to RECURRENT.
    arena.epigenome[cid] = (1 << RECURRENT)
    arena.refresh_phenotype(cid)
    s = src_ids.to(torch.int32)
    arena.add_edges(s, torch.full_like(s, cid),
                    seed_w * torch.randn(s.numel()))
    d = dst_ids.to(torch.int32)
    arena.add_edges(torch.full_like(d, cid), d,
                    seed_w * torch.randn(d.numel()))
    return cid


def build(n_sat, *, h_init, cap, seed, lam, w_res):
    torch.manual_seed(seed)
    op = SatelliteOp(lam=lam, w_res=w_res)
    table = default_dispatch_table()
    table[RECURRENT] = op
    sub = construct(
        base=seeded(N_SENSE, N_CONSEQ, interior_cells=h_init),
        envelope=Envelope(max_parameter_bytes=cap),
        dispatch_table=table, capacity=2048, sparsity_k=0,
    )
    sub.compile()
    a = sub.arena
    interior = get_interior_ids(a).long()
    output = torch.tensor([c for c in a.alive_ids().tolist()
                           if has_gene(int(a.epigenome[c].item()), OUTPUT)], dtype=torch.long)
    sat_ids = [add_satellite(a, interior, output) for _ in range(n_sat)]
    a.rank_dirty = True
    sub.prepare_training()
    return sub, op, sorted(sat_ids), interior, output


def divide_satellite(sub, op, sat_ids, interior, output, resource_ceiling=0.95):
    """Satellite division — gated by SENSED resource (refuses when tight)."""
    a = sub.arena
    if a.pressure() > resource_ceiling:
        return None, "refused (resource pressure too high)"
    cid = add_satellite(a, interior, output)
    a.rank_dirty = True
    sub.compile()
    return cid, "divided"


# ----------------------------------------------------------------------
# Delayed grounded recall — intrinsic memory test
# ----------------------------------------------------------------------
def run(hold, *, n_sat, delay, h_init, cap, seed, epochs, batch, lr, lam, noise):
    g = torch.Generator().manual_seed(seed)
    cue_tr, y_tr = balanced_cues(700, noise, g)
    cue_te, y_te = balanced_cues(200, noise, g)
    sub, op, sat_ids, interior, output = build(n_sat, h_init=h_init, cap=cap,
                                               seed=seed, lam=lam, w_res=0.0)
    op.hold = hold
    n = len(sat_ids)
    opt = torch.optim.Adam(sub.trainable_tensors(), lr=lr)
    zero = lambda b: torch.zeros(b, N_SENSE)

    def run_seq(cue, b):
        op.reset(b, n)
        steps = [cue] + [zero(b) for _ in range(delay)]   # cue then probe(s)
        logits = None
        for percept in steps:
            logits = sub(percept)
        return logits

    n_tr = cue_tr.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n_tr, generator=g)
        for bi in range(max(1, n_tr // batch)):
            idx = perm[bi * batch:(bi + 1) * batch]
            cue = cue_tr[idx]
            logits = run_seq(cue, cue.shape[0])
            loss = torch.nn.functional.cross_entropy(logits, y_tr[idx])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(sub.trainable_tensors(), 1.0)
            sub.zero_dormant_grads(); opt.step()
    with torch.no_grad():
        correct = 0
        for s in range(0, cue_te.shape[0], 200):
            e = min(s + 200, cue_te.shape[0])
            preds = run_seq(cue_te[s:e], e - s).argmax(-1)
            correct += int((preds == y_te[s:e]).sum())
    return correct / cue_te.shape[0], op.last_pressure, sub, op, sat_ids, interior, output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--h-init", type=int, default=8)
    ap.add_argument("--n-sat", type=int, default=12)
    ap.add_argument("--delay", type=int, default=1)
    ap.add_argument("--lam", type=float, default=0.6)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--cap", type=int, default=400_000)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.epochs = 1, 15

    print(f"satellites_v1: delayed recall, intrinsic memory in satellites "
          f"(recurrent slot). n_sat={args.n_sat} delay={args.delay} "
          f"seeds={args.seeds} epochs={args.epochs}  (chance={1/N_CONSEQ:.3f})")
    for hold, name in [(False, "satellites-OFF (reset each step)"),
                       (True,  "satellites-ON  (intrinsic memory)")]:
        accs = []; last = None
        for seed in range(args.seeds):
            t0 = time.time()
            acc, pressure, sub, op, sat_ids, interior, output = run(
                hold, n_sat=args.n_sat, delay=args.delay, h_init=args.h_init,
                cap=args.cap, seed=seed, epochs=args.epochs, batch=args.batch,
                lr=args.lr, lam=args.lam, noise=args.noise)
            accs.append(acc); last = (sub, op, sat_ids, interior, output, pressure)
            print(f"  [{name[:13]} seed={seed}] acc={acc:.3f} "
                  f"sensed_pressure={pressure:.4f} ({time.time()-t0:.0f}s)")
        m = statistics.mean(accs); s = statistics.stdev(accs) if len(accs) > 1 else 0.0
        verdict = "RECALLS" if m > 0.6 else "fails"
        print(f"  === {name}: {m:.3f}±{s:.3f}  [{verdict}]")

    # demonstrate division gated by sensed resource
    sub, op, sat_ids, interior, output, pressure = last
    print(f"\nResource-sensing + division demo (n_sat={len(sat_ids)}, "
          f"pressure={sub.arena.pressure():.4f}):")
    for trial, ceil in [("normal ceiling 0.95", 0.95), ("tight ceiling 0.0001", 0.0001)]:
        cid, msg = divide_satellite(sub, op, sat_ids, interior, output, resource_ceiling=ceil)
        print(f"  divide({trial}): {msg}" + (f" → new satellite {cid}" if cid else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
