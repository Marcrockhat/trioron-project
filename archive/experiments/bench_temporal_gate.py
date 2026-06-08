"""Axis 7 falsification gate — does carrying the substrate's own hidden state
across time let it solve a task that is unsolvable without memory?

Design: docs/design/temporal_axis_v7.md §7. The minimal test of the temporal
axis before any per-cell `mnemonic` gene lands in core.

Task — DELAYED GROUNDED RECALL
------------------------------
Each trial is a sequence:  [cue] → [filler]×(delay-1) → [probe].
  - cue   : a grounded percept (has an innate consequence class).
  - filler: neutral noise percepts (distractors; the memory must survive them).
  - probe : zero senses — no sensory info. The target (scored ONLY here) is the
            *cue's* consequence. To answer, the substrate must have carried the
            cue forward in time.

Arms
----
  temporal-off : memoryless. Trace fed as zeros every step. At the probe step
                 senses=0 and trace=0 → no information → pinned at the marginal
                 (majority-class rate).
  temporal-on  : the substrate's OWN interior activations are carried as a leaky
                 trace m_t = λ·m_{t-1} + (1-λ)·h_t and fed back as input. The
                 final-step loss backpropagates through the whole sequence
                 (BPTT), so the substrate LEARNS what to keep in memory. This is
                 internal recurrence (not a handed-in echo of the past input).

Both arms share architecture (input = senses ⊕ trace-slots); only whether the
trace carries signal differs. Gate: temporal-on beats temporal-off by Δ ≥ 0.30
abs at n=3. A delay sweep shows the memory horizon.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trioron.learning.manifold import get_interior_ids
from experiments.bench_arena_hierarchical import build_substrate
from experiments.bench_grounded_world import (
    SENSES, N_SENSE, CONSEQUENCES, N_CONSEQ, innate_consequence,
)


# ----------------------------------------------------------------------
# Balanced cue sampling (so the majority-class baseline is ~1/N_CONSEQ)
# ----------------------------------------------------------------------
def balanced_cues(n_per_class: int, noise: float, gen):
    """Return (cue_percepts, labels) with ~equal items per consequence class."""
    buckets = {c: [] for c in range(N_CONSEQ)}
    need = n_per_class * N_CONSEQ
    while sum(len(v) for v in buckets.values()) < need:
        p = (torch.rand(512, N_SENSE, generator=gen) < 0.35).float()
        for i in range(p.shape[0]):
            c = innate_consequence(p[i], nourish_cue="sweet")
            if len(buckets[c]) < n_per_class:
                buckets[c].append(p[i])
    cues, labels = [], []
    for c in range(N_CONSEQ):
        for v in buckets[c]:
            cues.append(v); labels.append(c)
    cues = torch.stack(cues)
    labels = torch.tensor(labels, dtype=torch.long)
    x = cues + noise * torch.randn_like(cues)
    perm = torch.randperm(x.shape[0], generator=gen)
    return x[perm], labels[perm]


def filler_batch(b, delay, noise, gen):
    """(delay-1) neutral filler percepts per trial: low-density noise."""
    if delay <= 1:
        return torch.zeros(b, 0, N_SENSE)
    f = (torch.rand(b, delay - 1, N_SENSE, generator=gen) < 0.2).float()
    return f + noise * torch.randn_like(f)


# ----------------------------------------------------------------------
# One arm
# ----------------------------------------------------------------------
def run_arm(temporal_on, *, delay, h_init, cap_bytes, seed, epochs, batch, lr,
            lam, noise, verbose):
    g = torch.Generator().manual_seed(seed)
    cue_tr, y_tr = balanced_cues(700, noise, g)
    cue_te, y_te = balanced_cues(200, noise, g)
    H = h_init
    sub = build_substrate(N_SENSE + H, N_CONSEQ, h_init, cap_bytes, seed)
    sub.compile(); sub.prepare_training()
    iids = get_interior_ids(sub.arena).long()
    assert iids.numel() == H, f"trace dim {iids.numel()} != H {H}"
    opt = torch.optim.Adam(sub.trainable_tensors(), lr=lr)
    zero_sense = lambda b: torch.zeros(b, N_SENSE)

    def run_sequence(cue, fill, train):
        """Forward the [cue]→fillers→[probe] sequence; return probe logits.
        Carries the leaky interior trace when temporal_on."""
        b = cue.shape[0]
        m = torch.zeros(b, H)
        steps = [cue] + [fill[:, t] for t in range(fill.shape[1])] + [zero_sense(b)]
        logits = None
        for t, percept in enumerate(steps):
            tin = m if temporal_on else torch.zeros(b, H)
            logits = sub(torch.cat([percept, tin], dim=1))
            if temporal_on:
                h = sub.live_activations[:, iids]
                m = lam * m + (1.0 - lam) * h   # keep graph → BPTT
        return logits

    n_tr = cue_tr.shape[0]
    for epoch in range(epochs):
        perm = torch.randperm(n_tr, generator=g)
        for bi in range(max(1, n_tr // batch)):
            idx = perm[bi * batch:(bi + 1) * batch]
            cue = cue_tr[idx]; y = y_tr[idx]
            fill = filler_batch(cue.shape[0], delay, noise, g)
            logits = run_sequence(cue, fill, train=True)
            loss = torch.nn.functional.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); sub.zero_dormant_grads(); opt.step()
        if verbose and epoch % 10 == 0:
            print(f"      ep{epoch} loss={loss.item():.3f}")

    # eval
    with torch.no_grad():
        correct = 0
        for s in range(0, cue_te.shape[0], 256):
            e = min(s + 256, cue_te.shape[0])
            cue = cue_te[s:e]
            fill = filler_batch(cue.shape[0], delay, noise, g)
            preds = run_sequence(cue, fill, train=False).argmax(dim=-1)
            correct += int((preds == y_te[s:e]).sum().item())
        acc = correct / cue_te.shape[0]
    majority = float(torch.bincount(y_te, minlength=N_CONSEQ).max().item()) / y_te.shape[0]
    return acc, majority


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--h-init", type=int, default=12)
    ap.add_argument("--lam", type=float, default=0.6, help="trace decay λ")
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--cap-bytes", type=int, default=300_000)
    ap.add_argument("--delays", default="0,1,2,4")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.epochs, args.delays = 1, 20, "1,2"

    delays = [int(d) for d in args.delays.split(",")]
    print(f"bench_temporal_gate: seeds={args.seeds} epochs={args.epochs} "
          f"h_init={args.h_init} λ={args.lam} noise={args.noise} "
          f"senses={N_SENSE} conseq={N_CONSEQ} delays={delays}")
    print(f"  (gate: temporal-on − temporal-off ≥ 0.30 abs ⇒ Axis 7 PASS)")

    for delay in delays:
        offs, ons = [], []
        for seed in range(args.seeds):
            t0 = time.time()
            off_acc, maj = run_arm(False, delay=delay, h_init=args.h_init,
                                   cap_bytes=args.cap_bytes, seed=seed,
                                   epochs=args.epochs, batch=args.batch, lr=args.lr,
                                   lam=args.lam, noise=args.noise, verbose=args.smoke)
            on_acc, _ = run_arm(True, delay=delay, h_init=args.h_init,
                                cap_bytes=args.cap_bytes, seed=seed,
                                epochs=args.epochs, batch=args.batch, lr=args.lr,
                                lam=args.lam, noise=args.noise, verbose=args.smoke)
            offs.append(off_acc); ons.append(on_acc)
            print(f"  [delay={delay} seed={seed}] off={off_acc:.3f} on={on_acc:.3f} "
                  f"(majority={maj:.3f}) Δ={on_acc-off_acc:+.3f} ({time.time()-t0:.0f}s)")
        om, os_ = statistics.mean(offs), (statistics.stdev(offs) if len(offs) > 1 else 0.0)
        nm, ns = statistics.mean(ons), (statistics.stdev(ons) if len(ons) > 1 else 0.0)
        delta = nm - om
        verdict = "PASS" if delta >= 0.30 else ("partial" if delta >= 0.10 else "FAIL")
        print(f"  === delay={delay}: off={om:.3f}±{os_:.3f}  on={nm:.3f}±{ns:.3f}  "
              f"Δ={delta:+.3f}  [{verdict}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
