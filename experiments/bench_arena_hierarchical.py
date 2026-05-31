"""Does self-organized depth help on a compositional task? — Arena substrate.

Motivation
----------
Session 009 found the Arena substrate is structurally a 1-hidden-layer MLP
(`substrate-is-bipartite-mlp`): `divide()`'s `a.rank < child_rank` policy
forbids interior↔interior edges, so growth only widens. The `--self-arrange`
relaxation (`same_rank_edges`, grow.py) lets a new cell draw from same-rank
cells; Kahn's BFS then promotes it to rank+1, so depth self-organizes.

chained-15's binary MNIST/Fashion/EMNIST tasks are near-linearly-separable,
so depth may not be the lever there. This bench builds a task where depth IS
the lever — **hierarchical parity composition** — the textbook function for
which a depth-2 network needs O(n) units while depth-1 needs O(2^n).

Task
----
Input: N_BITS in {-1,+1}, split into N_GROUPS groups of BITS_PER_GROUP
informative bits, plus N_DISTRACTOR irrelevant bits. The label is the
per-group parity (product of signs) read as a binary number → 2**N_GROUPS
classes. Computing each group parity is a composition (XOR tree); combining
groups into a class is a second composition. A bipartite (width-only) substrate
must memorize; a deep (self-arranged) substrate can compose.

Arms (matched growth budget — same division schedule, so only topology differs)
----
  no-growth     seed substrate, no division (capacity floor)
  bipartite     divide() with same_rank_edges=False (widens only)
  self-arrange  divide() with same_rank_edges=True  (depth emerges)

Reading: bipartite vs self-arrange at identical cell/param count isolates the
effect of depth. If self-arrange wins, self-organized depth is the mechanism.

Env / flags: see argparse in main(). Smoke: --smoke (tiny, ~1 min).
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
from trioron.core.epigenome import OUTPUT, PERCEPTION, has_gene
from trioron.core.state import CellState
from trioron.bases import seeded
from trioron.phenotype import default_dispatch_table
from trioron.lifecycle import divide, GrowthConfig


# ----------------------------------------------------------------------
# Dataset: hierarchical parity composition
# ----------------------------------------------------------------------
def make_dataset(n_groups: int, bits_per_group: int, n_distractor: int,
                 seed: int):
    """Enumerate (or sample) all bit patterns; label = per-group parity
    read as a binary number. Returns (x_train, y_train, x_test, y_test).

    Inputs are {-1,+1} floats. Parity of a group = product of its signs
    mapped to {0,1}. Class = sum_g parity_g * 2**g  →  2**n_groups classes.
    """
    n_info = n_groups * bits_per_group
    n_bits = n_info + n_distractor
    g = torch.Generator().manual_seed(seed)

    total = 1 << n_bits
    if total <= 16384:
        # Full enumeration — clean, no sampling noise.
        idx = torch.arange(total)
        bits = ((idx.unsqueeze(1) >> torch.arange(n_bits)) & 1)  # (total, n_bits)
    else:
        n_samples = 16384
        bits = torch.randint(0, 2, (n_samples, n_bits), generator=g)

    signs = bits.float() * 2 - 1  # {0,1} -> {-1,+1}

    # Per-group parity over the informative bits (product of signs > 0).
    label = torch.zeros(signs.shape[0], dtype=torch.long)
    for grp in range(n_groups):
        lo = grp * bits_per_group
        hi = lo + bits_per_group
        prod = signs[:, lo:hi].prod(dim=1)        # +1 if even # of -1s
        parity = (prod < 0).long()                # 1 if odd # of -1s
        label = label + (parity << grp)

    # Shuffle and split 80/20.
    perm = torch.randperm(signs.shape[0], generator=g)
    signs, label = signs[perm], label[perm]
    n_tr = int(0.8 * signs.shape[0])
    return signs[:n_tr], label[:n_tr], signs[n_tr:], label[n_tr:]


# ----------------------------------------------------------------------
# Substrate construction
# ----------------------------------------------------------------------
def build_substrate(input_dim: int, n_classes: int, h_init: int,
                    cap_bytes: int, seed: int):
    torch.manual_seed(seed)
    sub = construct(
        base=seeded(input_dim, n_classes, interior_cells=h_init),
        envelope=Envelope(max_parameter_bytes=cap_bytes),
        dispatch_table=default_dispatch_table(),
        capacity=2048,
        sparsity_k=0,
    )
    return sub


def interior_parents(arena):
    return [cid for cid in arena.alive_ids().tolist()
            if not has_gene(int(arena.epigenome[cid].item()), PERCEPTION)
            and not has_gene(int(arena.epigenome[cid].item()), OUTPUT)
            and arena.state[cid] == CellState.ACTIVE]


def rank_stats(arena):
    """Return (max_interior_rank, n_interior_with_rank_gt_1, histogram)."""
    arena.rank_dirty = True
    ids = arena.alive_ids().tolist()
    hist = {}
    deep = 0
    max_r = 0
    for cid in ids:
        is_out = has_gene(int(arena.epigenome[cid].item()), OUTPUT)
        r = int(arena.rank[cid].item())
        hist[r] = hist.get(r, 0) + 1
        if not is_out:
            max_r = max(max_r, r)
            if r > 1:
                deep += 1
    return max_r, deep, dict(sorted(hist.items()))


def param_count(sub):
    return int(sum(p.numel() for p in sub.trainable_tensors()))


# ----------------------------------------------------------------------
# Train one arm
# ----------------------------------------------------------------------
def train_arm(arm: str, x_tr, y_tr, x_te, y_te, n_classes, *,
              h_init: int, cap_bytes: int, seed: int, epochs: int,
              batch: int, lr: float, growth_budget: int, grow_every: int,
              verbose: bool):
    sub = build_substrate(x_tr.shape[1], n_classes, h_init, cap_bytes, seed)
    sub.compile()
    sub.prepare_training()

    same_rank = (arm == "self-arrange")
    do_grow = (arm != "no-growth")
    cfg = GrowthConfig(same_rank_edges=same_rank)

    opt = torch.optim.Adam(sub.trainable_tensors(), lr=lr)
    torch.manual_seed(seed + 1000)  # division RNG, identical across grow arms
    n_tr = x_tr.shape[0]
    grown = 0
    step = 0

    for epoch in range(epochs):
        perm = torch.randperm(n_tr)
        for b in range(max(1, n_tr // batch)):
            idx = perm[b * batch:(b + 1) * batch]
            logits = sub(x_tr[idx])
            loss = torch.nn.functional.cross_entropy(logits, y_tr[idx])
            loss.backward()
            sub.zero_dormant_grads()
            opt.step()
            opt.zero_grad()

            # Deterministic, matched growth schedule across grow arms.
            if do_grow and grown < growth_budget and step > 0 and step % grow_every == 0:
                ip = interior_parents(sub.arena)
                if ip:
                    parent = ip[torch.randint(0, len(ip), (1,)).item()]
                    ev = divide(sub.arena, parent, cfg)
                    if ev:
                        grown += 1
                        sub.compile()
                        opt = torch.optim.Adam(sub.trainable_tensors(), lr=lr)
            step += 1
        if verbose:
            tr_acc = eval_acc(sub, x_tr, y_tr)
            print(f"    [{arm}] epoch {epoch+1}/{epochs} loss={loss.item():.3f} "
                  f"train_acc={tr_acc:.3f} grown={grown}")

    test_acc = eval_acc(sub, x_te, y_te)
    max_r, deep, hist = rank_stats(sub.arena)
    return {
        "arm": arm,
        "test_acc": test_acc,
        "grown": grown,
        "max_interior_rank": max_r,
        "n_deep": deep,
        "rank_hist": hist,
        "params": param_count(sub),
        "n_cells": int(sub.arena.alive_ids().numel()),
    }


@torch.no_grad()
def eval_acc(sub, x, y):
    correct = 0
    for s in range(0, x.shape[0], 1024):
        e = min(s + 1024, x.shape[0])
        preds = sub(x[s:e]).argmax(dim=-1)
        correct += int((preds == y[s:e]).sum().item())
    return correct / max(1, x.shape[0])


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--h-init", type=int, default=8)
    p.add_argument("--growth-budget", type=int, default=40)
    p.add_argument("--grow-every", type=int, default=20)
    p.add_argument("--cap-bytes", type=int, default=200_000)
    p.add_argument("--n-groups", type=int, default=2)
    p.add_argument("--bits-per-group", type=int, default=4)
    p.add_argument("--n-distractor", type=int, default=4)
    p.add_argument("--arms", default="no-growth,bipartite,self-arrange")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.seeds, args.epochs = 1, 12

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    n_classes = 1 << args.n_groups
    input_dim = args.n_groups * args.bits_per_group + args.n_distractor

    print(f"bench_arena_hierarchical: seeds={args.seeds} epochs={args.epochs} "
          f"h_init={args.h_init} growth={args.growth_budget}@{args.grow_every} "
          f"task={args.n_groups}grp×{args.bits_per_group}bit+{args.n_distractor}distractor "
          f"→ {input_dim}-d in, {n_classes} classes  arms={arms}")

    results = {a: [] for a in arms}
    for seed in range(args.seeds):
        x_tr, y_tr, x_te, y_te = make_dataset(
            args.n_groups, args.bits_per_group, args.n_distractor, seed)
        if seed == 0:
            print(f"  dataset: train={x_tr.shape[0]} test={x_te.shape[0]} "
                  f"class balance={torch.bincount(y_tr).tolist()}")
        for arm in arms:
            t0 = time.time()
            r = train_arm(
                arm, x_tr, y_tr, x_te, y_te, n_classes,
                h_init=args.h_init, cap_bytes=args.cap_bytes, seed=seed,
                epochs=args.epochs, batch=args.batch, lr=args.lr,
                growth_budget=args.growth_budget, grow_every=args.grow_every,
                verbose=args.smoke)
            r["t"] = time.time() - t0
            results[arm].append(r)
            print(f"  [seed {seed} {arm:>12s}] test_acc={r['test_acc']:.4f} "
                  f"cells={r['n_cells']} params={r['params']} grown={r['grown']} "
                  f"max_rank={r['max_interior_rank']} deep={r['n_deep']} "
                  f"rank_hist={r['rank_hist']} ({r['t']:.0f}s)")

    print("\n=== aggregate (mean ± std test_acc) ===")
    for arm in arms:
        accs = [r["test_acc"] for r in results[arm]]
        m = statistics.mean(accs)
        s = statistics.stdev(accs) if len(accs) > 1 else 0.0
        deep = statistics.mean([r["n_deep"] for r in results[arm]])
        mr = statistics.mean([r["max_interior_rank"] for r in results[arm]])
        prm = statistics.mean([r["params"] for r in results[arm]])
        print(f"  {arm:>12s}  test_acc={m:.4f} ± {s:.4f}  "
              f"params≈{prm:.0f}  mean_max_rank={mr:.1f}  mean_deep_cells={deep:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
