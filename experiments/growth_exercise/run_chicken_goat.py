"""Feed chicken-vs-goat to a v2.0 trioron substrate; report epoch-to-convergence.

A growth probe on a trivially (near-)separable 1-feature binary task.  A healthy
substrate should converge in a handful of epochs and grow ~0 cells (frustration
never sustains on a problem this easy).  Growth is ON (frustration-gated divide,
relying on the s020 forward-projection fix -- no manual re-wiring).

Run:  n=10 seeds, 30 epochs each, batch 64.
Reports per seed: convergence epoch, final train/test acc, cells grown.
Summary: epoch-to-convergence mean +/- std.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from trioron.core import Envelope, construct
from trioron.core.epigenome import OUTPUT, has_gene
from trioron.bases import seeded
from trioron.phenotype import default_dispatch_table
from trioron.lifecycle import divide, GrowthConfig
from trioron.learning import FrustrationDetector, FrustrationConfig
from trioron.learning.manifold import get_interior_ids

from experiments.growth_exercise.chicken_goat import make_animals, bayes_accuracy

INPUT_DIM = 1


def count_cells(sub) -> int:
    return int(sub.arena.alive_ids().numel())


def run_seed(seed: int, *, species: list[str], n_classes: int, converge_acc: float,
             epochs: int, batch: int, lr: float, h_init: int,
             cap: int, grow_budget: int, grow: bool, log: bool):
    torch.manual_seed(seed)

    train = make_animals(species, seed=seed, log=log)
    test = make_animals(species, seed=seed + 1000, log=log)

    sub = construct(base=seeded(INPUT_DIM, n_classes, interior_cells=h_init),
                    envelope=Envelope(max_parameter_bytes=cap),
                    dispatch_table=default_dispatch_table(),
                    capacity=2048, sparsity_k=0)
    sub.compile()
    sub.prepare_training()

    frust = FrustrationDetector(FrustrationConfig())
    opt = torch.optim.Adam(sub.trainable_tensors(), lr=lr)

    n = len(train)
    steps_per_epoch = (n + batch - 1) // batch
    grown = 0
    frust_steps = 0
    converged_epoch = None
    n0 = count_cells(sub)

    for ep in range(1, epochs + 1):
        g = torch.Generator().manual_seed(seed * 7919 + ep)
        order = torch.randperm(n, generator=g)
        for bi in range(steps_per_epoch):
            idx = order[bi * batch:(bi + 1) * batch]
            x, y = train.x[idx], train.y[idx]
            logits = sub(x)
            loss = F.cross_entropy(logits, y)
            m = frust.step(loss.item())
            if frust.is_frustrated:
                frust_steps += 1
            (loss * m).backward()
            torch.nn.utils.clip_grad_norm_(sub.trainable_tensors(), 1.0)
            sub.zero_dormant_grads()
            opt.step()
            opt.zero_grad()

            if grow and grown < grow_budget and frust.is_frustrated and frust_steps >= 25:
                ip = get_interior_ids(sub.arena).long().tolist()
                if ip:
                    parent = ip[torch.randint(0, len(ip), (1,)).item()]
                    ev = divide(sub.arena, parent, GrowthConfig())
                    if ev:
                        grown += 1
                        sub.arena.rank_dirty = True
                        sub.compile()
                        frust_steps = 0

        with torch.no_grad():
            train_acc = (sub(train.x).argmax(-1) == train.y).float().mean().item()
        if converged_epoch is None and train_acc >= converge_acc:
            converged_epoch = ep

    with torch.no_grad():
        train_acc = (sub(train.x).argmax(-1) == train.y).float().mean().item()
        test_acc = (sub(test.x).argmax(-1) == test.y).float().mean().item()

    return {
        "seed": seed,
        "converged_epoch": converged_epoch,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "grown": grown,
        "cells_start": n0,
        "cells_end": count_cells(sub),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--h-init", type=int, default=2)
    ap.add_argument("--cap", type=int, default=200_000)
    ap.add_argument("--grow-budget", type=int, default=40)
    ap.add_argument("--species", nargs="+", default=["chicken", "goat"],
                    help="species to classify (default: chicken goat)")
    ap.add_argument("--margin", type=float, default=0.02,
                    help="convergence target = Bayes ceiling - margin")
    ap.add_argument("--no-grow", action="store_true", help="disable growth (fixed substrate)")
    ap.add_argument("--log", action="store_true", help="log-transform weight before standardizing")
    args = ap.parse_args()

    grow = not args.no_grow
    species = args.species
    n_classes = len(species)
    # Honest convergence target: just under the weight-only Bayes ceiling.
    ceiling = bayes_accuracy(make_animals(species, seed=0))
    converge_acc = ceiling - args.margin

    print(f"animals-by-weight | species={species} ({n_classes} classes) | "
          f"seeds={args.seeds} epochs={args.epochs} batch={args.batch} "
          f"lr={args.lr} h_init={args.h_init} grow={grow} log={args.log}")
    print(f"Bayes ceiling (weight-only) = {ceiling:.3f}; "
          f"convergence = first epoch with train acc >= {converge_acc:.3f}\n")

    rows = []
    t0 = time.time()
    for seed in range(args.seeds):
        r = run_seed(seed, species=species, n_classes=n_classes, converge_acc=converge_acc,
                     epochs=args.epochs, batch=args.batch, lr=args.lr,
                     h_init=args.h_init, cap=args.cap, grow_budget=args.grow_budget,
                     grow=grow, log=args.log)
        rows.append(r)
        ce = r["converged_epoch"]
        ce_s = f"{ce:>2d}" if ce is not None else " -"
        print(f"  seed={seed}  converged@epoch={ce_s}  "
              f"train={r['train_acc']:.3f}  test={r['test_acc']:.3f}  "
              f"grown={r['grown']}  cells {r['cells_start']}->{r['cells_end']}")
    elapsed = time.time() - t0

    conv = [r["converged_epoch"] for r in rows if r["converged_epoch"] is not None]
    n_conv = len(conv)
    print(f"\n--- summary (n={len(rows)}) ---  ({elapsed:.1f}s)")
    print(f"  converged: {n_conv}/{len(rows)} seeds")
    if conv:
        m = statistics.mean(conv)
        s = statistics.stdev(conv) if len(conv) > 1 else 0.0
        print(f"  epochs-to-convergence: {m:.1f} +/- {s:.1f}  "
              f"(min {min(conv)}, max {max(conv)})")
    tr = statistics.mean(r["train_acc"] for r in rows)
    te = statistics.mean(r["test_acc"] for r in rows)
    gr = statistics.mean(r["grown"] for r in rows)
    print(f"  final train acc: {tr:.3f}   test acc: {te:.3f}")
    print(f"  cells grown (mean): {gr:.1f}")


if __name__ == "__main__":
    raise SystemExit(main())
