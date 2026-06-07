"""Calibrate the growth veto (K, P) on a task that GENUINELY needs capacity.

Unlike chicken-goat (where linear capacity is already sufficient and growth is
pure waste), this probe builds a deliberate *linear* capacity wall and lets the
substrate grow into it -- so we can MEASURE, not guess, the two knobs the
marginal-utility veto needs:

  K = how long after a division before its benefit shows (the integration
      window).  Driven by the post-divide loss transient: does loss RISE first
      (objection 2), how far, and how many steps until it beats the pre-divide
      best?
  P = how many consecutive "dry" divisions (no new best) we must tolerate before
      vetoing -- i.e. the widest capacity threshold a single division can fail
      to cross on its own (objection 1).

Task: low-rank blobs.  C classes whose means are orthonormal directions in
D-dim space.  A pure-linear hidden layer of width h realises an input->logit map
of rank <= h, so h < C-1 caps accuracy; each linear `divide` adds one rank.
Interior cells are LINEAR (seeded nonlinear=False) -- the same mechanism the
chicken-goat harness grows with, so the calibration transfers.

Growth here is driven by a running-min PLATEAU controller with P=inf (always
grow on plateau), NOT the frustration detector -- because we want to OBSERVE
every division's outcome and read off what the veto's K and P *should* be.

Run:  python3 -m experiments.growth_exercise.calibrate_growth --seeds 5
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from trioron.core import Envelope, construct
from trioron.bases import seeded
from trioron.phenotype import default_dispatch_table
from trioron.lifecycle import divide, GrowthConfig
from trioron.learning.manifold import get_interior_ids


def lowrank_blobs(D: int, C: int, sep: float, noise: float, n: int, seed: int,
                  geom_seed: int = 0):
    """C Gaussian blobs at orthonormal mean directions in R^D.

    Full-rank linear classifier separates them near-perfectly; a rank-h readout
    (hidden width h) can only resolve ~h of the C orthonormal directions.

    The class-mean GEOMETRY is fixed by ``geom_seed`` (NOT ``seed``) so train and
    test live in the SAME coordinate frame -- pass the same geom_seed for a split
    and different sample ``seed`` for the points.  (Earlier the geometry was keyed
    on ``seed``, so train and test had different means -> the test column was
    meaningless.)
    """
    gm = torch.Generator().manual_seed(geom_seed)
    # Orthonormal class directions (first C columns of a random orthogonal DxD).
    A = torch.randn(D, D, generator=gm)
    Q, _ = torch.linalg.qr(A)
    means = (Q[:, :C] * sep).T                      # (C, D)
    gs = torch.Generator().manual_seed(seed)
    y = torch.randint(0, C, (n,), generator=gs)
    x = means[y] + noise * torch.randn(n, D, generator=gs)
    return x, y, means


def count_cells(sub) -> int:
    return int(sub.arena.alive_ids().numel())


def run_seed(seed: int, *, D: int, C: int, sep: float, noise: float,
             total_steps: int, batch: int, lr: float, h_init: int,
             budget: int, probe_patience: int, eps: float, cap: int):
    torch.manual_seed(seed)
    # Same geometry (geom_seed=seed) for both splits; different sample seeds.
    xtr, ytr, _ = lowrank_blobs(D, C, sep, noise, 4096, seed, geom_seed=seed)
    xte, yte, _ = lowrank_blobs(D, C, sep, noise, 4096, seed + 1000, geom_seed=seed)

    sub = construct(base=seeded(D, C, interior_cells=h_init, nonlinear=False),
                    envelope=Envelope(max_parameter_bytes=cap),
                    dispatch_table=default_dispatch_table(),
                    capacity=2048, sparsity_k=0)
    sub.compile()
    sub.prepare_training()
    opt = torch.optim.Adam(sub.trainable_tensors(), lr=lr)

    n = len(xtr)
    loss_log: list[float] = []          # per-step training loss
    events: list[dict] = []             # one per division
    cells_log: list[tuple[int, int]] = []  # (step, n_cells) after each division

    running_min = float("inf")
    since_best = 0
    grown = 0

    g = torch.Generator().manual_seed(seed * 7919)
    for step in range(total_steps):
        idx = torch.randint(0, n, (batch,), generator=g)
        x, y = xtr[idx], ytr[idx]
        loss = F.cross_entropy(sub(x), y)
        lv = loss.item()
        loss_log.append(lv)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sub.trainable_tensors(), 1.0)
        sub.zero_dormant_grads()
        opt.step()
        opt.zero_grad()

        if lv < running_min - eps:
            running_min = lv
            since_best = 0
        else:
            since_best += 1

        # Plateau controller, P=inf: force ONE division on every plateau and
        # record its pre-divide state so we can score it afterwards.
        if since_best >= probe_patience and grown < budget:
            ip = get_interior_ids(sub.arena).long().tolist()
            if ip:
                parent = ip[torch.randint(0, len(ip), (1,), generator=g).item()]
                ev = divide(sub.arena, parent, GrowthConfig())
                if ev:
                    grown += 1
                    sub.arena.rank_dirty = True
                    sub.compile()
                    # Optimizer identity is stable (fixed-capacity tensors); do
                    # NOT rebuild (would reset Adam momentum -> spurious spike).
                    events.append({
                        "div_idx": grown,
                        "log_pos": len(loss_log) - 1,   # step index of pre-divide loss
                        "pre_best": running_min,
                        "pre_loss": lv,
                    })
                    cells_log.append((step, count_cells(sub)))
                    since_best = 0

    with torch.no_grad():
        tr = (sub(xtr).argmax(-1) == ytr).float().mean().item()
        te = (sub(xte).argmax(-1) == yte).float().mean().item()

    # --- score each division over a generous observation window ---
    W = probe_patience * 4
    for i, ev in enumerate(events):
        s = ev["log_pos"]
        # window runs until the next division (or W steps, whichever first)
        nxt = events[i + 1]["log_pos"] if i + 1 < len(events) else len(loss_log)
        end = min(s + 1 + W, nxt)
        win = loss_log[s + 1:end]
        if not win:
            ev.update(recovered=False, steps_to_recover=None, peak_rise=0.0, win_min=ev["pre_best"])
            continue
        pre_best = ev["pre_best"]
        pre_loss = ev["pre_loss"]
        peak_rise = max(win) - pre_loss            # >0 => loss rose after divide
        win_min = min(win)
        steps_to_recover = None
        for j, v in enumerate(win):
            if v < pre_best - eps:
                steps_to_recover = j + 1
                break
        ev.update(recovered=steps_to_recover is not None,
                  steps_to_recover=steps_to_recover,
                  peak_rise=peak_rise, win_min=win_min)

    return {
        "seed": seed, "train_acc": tr, "test_acc": te,
        "grown": grown, "cells_end": count_cells(sub),
        "events": events,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--D", type=int, default=12)
    ap.add_argument("--C", type=int, default=8)
    ap.add_argument("--sep", type=float, default=4.0)
    ap.add_argument("--noise", type=float, default=1.0)
    ap.add_argument("--total-steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--h-init", type=int, default=1)
    ap.add_argument("--budget", type=int, default=12)
    ap.add_argument("--probe-patience", type=int, default=120)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--cap", type=int, default=200_000)
    args = ap.parse_args()

    print(f"calibrate_growth | low-rank blobs D={args.D} C={args.C} sep={args.sep} "
          f"noise={args.noise} | h_init={args.h_init} budget={args.budget} "
          f"probe_patience={args.probe_patience} lr={args.lr}")
    print(f"linear ceiling target: full rank h>={args.C-1} should separate {args.C} classes\n")

    rows = []
    all_events = []
    t0 = time.time()
    for seed in range(args.seeds):
        r = run_seed(seed, D=args.D, C=args.C, sep=args.sep, noise=args.noise,
                     total_steps=args.total_steps, batch=args.batch, lr=args.lr,
                     h_init=args.h_init, budget=args.budget,
                     probe_patience=args.probe_patience, eps=args.eps, cap=args.cap)
        rows.append(r)
        all_events.extend(r["events"])
        # per-seed division ladder
        rec = sum(1 for e in r["events"] if e["recovered"])
        print(f"  seed={seed}  grown={r['grown']:>2d}  cells->{r['cells_end']:>2d}  "
              f"train={r['train_acc']:.3f} test={r['test_acc']:.3f}  "
              f"productive_divs={rec}/{len(r['events'])}")

    elapsed = time.time() - t0
    print(f"\n--- calibration summary (n={len(rows)} seeds, {len(all_events)} divisions) ---  ({elapsed:.1f}s)")
    tr = statistics.mean(r["train_acc"] for r in rows)
    te = statistics.mean(r["test_acc"] for r in rows)
    gr = statistics.mean(r["grown"] for r in rows)
    print(f"  final  train {tr:.3f}  test {te:.3f}  cells grown {gr:.1f}")

    recov = [e for e in all_events if e["recovered"]]
    dry = [e for e in all_events if not e["recovered"]]
    print(f"\n  PRODUCTIVE divisions (set a new best): {len(recov)}/{len(all_events)}")
    if recov:
        kts = [e["steps_to_recover"] for e in recov]
        print(f"    steps-to-new-best (=> K):  mean {statistics.mean(kts):.0f}  "
              f"median {statistics.median(kts):.0f}  "
              f"p90 {sorted(kts)[int(0.9*len(kts))-1] if len(kts)>1 else kts[0]}  "
              f"max {max(kts)}")
    rises = [e["peak_rise"] for e in all_events]
    n_rose = sum(1 for v in rises if v > args.eps)
    print(f"\n  post-divide loss RISE (objection 2): {n_rose}/{len(all_events)} divisions rose above pre-divide loss")
    if rises:
        print(f"    peak_rise: mean {statistics.mean(rises):+.3f}  "
              f"median {statistics.median(rises):+.3f}  max {max(rises):+.3f}")
    print(f"\n  DRY divisions (no new best within window => what P must tolerate): {len(dry)}/{len(all_events)}")
    # longest run of consecutive dry divisions within a seed = minimum viable P
    worst_run = 0
    for r in rows:
        run = 0
        for e in r["events"]:
            run = run + 1 if not e["recovered"] else 0
            worst_run = max(worst_run, run)
    print(f"    longest consecutive-dry run across seeds (=> min viable P): {worst_run}")


if __name__ == "__main__":
    raise SystemExit(main())
