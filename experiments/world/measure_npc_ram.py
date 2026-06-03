"""Measure the REAL runtime RAM of one NPC brain (a mirror organism).

Two numbers, both honest:
  1. tensor bytes — sum of every persistent tensor in the arena + compiled plan
     (the structural footprint; what you'd serialise).
  2. process RSS delta — build K organisms in INFERENCE mode (no optimizer, no
     gradients) and divide the OS-level resident-memory growth by K. This is the
     true marginal cost of one more NPC, Python overhead and all.

Swept over capacity: 2048 (our training default — over-provisioned) vs right-
sized (256, 160). Adam uses ~126 cells / ~3,100 edges, so 160-256 is ample.
"""
from __future__ import annotations

import sys
import gc
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.world.mirror_cells import build_mirror, _solo, INPUT_DIM


def vmrss_kb():
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS"):
            return int(line.split()[1])
    return -1


def tensor_bytes(obj):
    """Sum bytes of all torch tensors held directly on an object's __dict__."""
    total = 0
    for v in vars(obj).values():
        if torch.is_tensor(v):
            total += v.element_size() * v.nelement()
    return total


def arena_bytes(sub):
    b = tensor_bytes(sub.arena)
    plan = getattr(sub.scheduler, "_plan", None)
    if plan is not None:
        b += tensor_bytes(plan)
        for bucket in getattr(plan, "buckets", []):
            b += tensor_bytes(bucket)
    return b


def build_inference_organism(seed, capacity):
    sub = build_mirror(seed, n_mirror=8, capacity=capacity)
    # strip training state: no grad tracking, no optimizer
    sub.arena.bias.requires_grad_(False)
    sub.arena.edge_weight.requires_grad_(False)
    return sub


def measure(capacity, K=200):
    # warm one up, report its tensor bytes + used cells/edges
    one = build_inference_organism(0, capacity)
    used_cells, used_edges = one.n_cells, one.n_edges
    tb = arena_bytes(one)
    # transient activation buffer for a batch-1 forward
    act_buf = 1 * capacity * 4
    del one; gc.collect()

    base = vmrss_kb()
    keep = [build_inference_organism(i, capacity) for i in range(K)]
    # one forward each so the plan/caches are fully realised
    x = torch.zeros(1, INPUT_DIM)
    with torch.no_grad():
        for s in keep:
            s(x)
    after = vmrss_kb()
    per_kb = (after - base) / K
    del keep; gc.collect()
    return {"cap": capacity, "cells": used_cells, "edges": used_edges,
            "tensor_kb": tb / 1024, "act_kb": act_buf / 1024, "rss_per_kb": per_kb}


def main():
    print("NPC brain RAM — mirror organism, INFERENCE mode (no optimizer/grads)\n")
    print(f"  (uses ~126 cells / ~3,100 edges; INPUT_DIM={INPUT_DIM})\n")
    print(f"  {'capacity':>8s} | {'tensors':>9s} | {'act buf':>8s} | "
          f"{'real RSS/NPC':>12s} | {'1k NPCs':>9s} | {'10k NPCs':>9s}")
    for cap in (2048, 256, 160):
        r = measure(cap)
        print(f"  {r['cap']:>8d} | {r['tensor_kb']:>7.1f}KB | {r['act_kb']:>6.1f}KB | "
              f"{r['rss_per_kb']:>10.1f}KB | {r['rss_per_kb']/1024:>6.1f}MB×1k→"
              f"{r['rss_per_kb']*1000/1024/1024:>4.2f}GB | "
              f"{r['rss_per_kb']*10000/1024/1024:>5.2f}GB")
    print("\n  tensors = serialisable brain size; RSS/NPC = true marginal OS cost.")
    print("  (deployed int8-compressed trioron from prior work ≈ 157 KB total.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
