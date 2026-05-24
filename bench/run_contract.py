"""Phase 1 performance contract gate.  See spec §6.

Exercises the bench-target operations at dev scale (50K params / ~1K cells)
and reports PASS/FAIL against the hard ceilings from spec §6.1.

Usage:
    python3 -m bench.run_contract
"""
from __future__ import annotations

import sys
import time
import tempfile
from pathlib import Path
from dataclasses import dataclass

import torch

from trioron.core import Envelope, construct
from trioron.bases import seeded
from trioron.phenotype import default_dispatch_table
from trioron.lifecycle import divide
from trioron.learning import CreditTracker, ManifoldArchive, dream_cycle, DreamConfig
from trioron.viz import capture_full


@dataclass
class Target:
    name: str
    soft_ms: float
    hard_ms: float
    actual_ms: float = 0.0

    @property
    def passed(self) -> bool:
        return self.actual_ms <= self.hard_ms

    @property
    def status(self) -> str:
        if self.actual_ms <= self.soft_ms:
            return "PASS"
        elif self.actual_ms <= self.hard_ms:
            return "WARN"
        return "FAIL"


def _build_dev_substrate():
    """~1K cells, ~50K edges at dev scale."""
    input_dim = 28
    n_classes = 10
    interior = 100
    cap = input_dim + interior + n_classes + 200
    edge_cap = input_dim * interior + interior * n_classes + 5000

    sub = construct(
        base=seeded(input_dim, n_classes, interior_cells=interior),
        envelope=Envelope(),
        dispatch_table=default_dispatch_table(),
        capacity=cap,
    )
    return sub


def _time_ms(fn, warmup: int = 2, repeats: int = 10) -> float:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return sorted(times)[len(times) // 2]


def bench_forward(sub) -> Target:
    sub.prepare_training()
    x = torch.randn(32, 28)
    ms = _time_ms(lambda: sub(x))
    return Target("forward (batch=32)", soft_ms=3.0, hard_ms=10.0, actual_ms=ms)


def bench_backward(sub) -> Target:
    sub.prepare_training()
    x = torch.randn(32, 28)
    opt = torch.optim.SGD(sub.trainable_tensors(), lr=0.01)

    def step():
        logits = sub(x)
        loss = logits.sum()
        loss.backward()
        sub.zero_dormant_grads()
        opt.step()
        opt.zero_grad()

    ms = _time_ms(step)
    return Target("backward + opt", soft_ms=8.0, hard_ms=25.0, actual_ms=ms)


def bench_growth(_sub) -> Target:
    fresh = _build_dev_substrate()
    fresh.prepare_training()
    out_ids = fresh.scheduler._plan.output_ids
    parent = int(out_ids[0].item())

    # Warm up
    divide(fresh.arena, parent)

    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        divide(fresh.arena, parent)
        times.append((time.perf_counter() - t0) * 1000)
    ms = sorted(times)[len(times) // 2]
    return Target("growth (single)", soft_ms=0.1, hard_ms=0.5, actual_ms=ms)


def bench_rank_recompute(sub) -> Target:
    def do_rank():
        sub.arena.rank_dirty = True
        sub.graph.recompute_ranks()

    ms = _time_ms(do_rank, warmup=3, repeats=20)
    return Target("rank recompute", soft_ms=0.5, hard_ms=2.0, actual_ms=ms)


def bench_dream_cycle(sub) -> Target:
    sub.prepare_training()
    credit = CreditTracker(sub.arena)
    archive = ManifoldArchive(sub.arena)

    for c in range(10):
        archive.update_class(c, torch.randn(50, 28))
    archive.finalize_all()

    cfg = DreamConfig(replay_batch_size=32, replay_lr=0.01)

    def do_dream():
        dream_cycle(sub, credit, archive, current_class=9, cfg=cfg)

    ms = _time_ms(do_dream, warmup=1, repeats=3)
    return Target("dream cycle (10 classes)", soft_ms=500.0, hard_ms=1500.0, actual_ms=ms)


def bench_snapshot(sub) -> Target:
    ms = _time_ms(lambda: capture_full(sub.arena, task_index=0))
    return Target("snapshot JSON", soft_ms=10.0, hard_ms=50.0, actual_ms=ms)


def main():
    print("=" * 60)
    print("Trioron v2.0 — Phase 1 Performance Contract")
    print("=" * 60)
    print()

    sub = _build_dev_substrate()
    print(f"Substrate: {sub.n_cells} cells, {sub.n_edges} edges")
    print()

    targets = [
        bench_forward(sub),
        bench_backward(sub),
        bench_growth(sub),
        bench_rank_recompute(sub),
        bench_dream_cycle(sub),
        bench_snapshot(sub),
    ]

    all_pass = True
    print(f"{'Operation':<30} {'Actual':>10} {'Soft':>10} {'Hard':>10} {'Status':>8}")
    print("-" * 72)
    for t in targets:
        unit = "ms" if t.hard_ms < 1000 else "s"
        if unit == "s":
            actual_str = f"{t.actual_ms / 1000:.3f} s"
            soft_str = f"{t.soft_ms / 1000:.1f} s"
            hard_str = f"{t.hard_ms / 1000:.1f} s"
        else:
            actual_str = f"{t.actual_ms:.2f} ms"
            soft_str = f"{t.soft_ms:.1f} ms"
            hard_str = f"{t.hard_ms:.1f} ms"
        print(f"{t.name:<30} {actual_str:>10} {soft_str:>10} {hard_str:>10} {t.status:>8}")
        if not t.passed:
            all_pass = False

    print()
    if all_pass:
        print("RESULT: ALL PASS")
    else:
        print("RESULT: FAIL — hard ceiling exceeded")
        sys.exit(1)


if __name__ == "__main__":
    main()
