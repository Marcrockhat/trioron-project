"""Behavior probe: random-projection fusion adapter (Phase C handoff
item 2, 2026-05-06).

Builds two donors on synthetic Gaussian-blob data at two scenarios:
  (A) same L0 seed (canonical path, no projection — paper baseline)
  (B) different L0 seeds (random-projection adapter — fallback path)

Both runs use identical training data. Reports per-branch task-aware
accuracy and the accuracy gap that random projection costs.

Run:    python3 -m experiments.probe_random_projection_fusion
"""
from __future__ import annotations
import os
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trioron.api import (
    TaskData, TrioronConfig, absorb, build_donor, evaluate,
)


def _make_synthetic_task(name, classes, *, seed, n_per_class=160, input_dim=64):
    g = torch.Generator().manual_seed(seed)
    Xs, ys = [], []
    for c in classes:
        center = torch.randn(input_dim, generator=g) * 4.0
        x = center + torch.randn(n_per_class, input_dim, generator=g)
        Xs.append(x)
        ys.append(torch.full((n_per_class,), c, dtype=torch.int64))
    X = torch.cat(Xs, dim=0)
    y = torch.cat(ys, dim=0)
    n_train = int(0.8 * X.shape[0])
    perm = torch.randperm(X.shape[0], generator=g)
    X = X[perm].float()
    y = y[perm]
    return TaskData(
        name=name,
        X_train=X[:n_train], y_train=y[:n_train],
        X_test=X[n_train:],  y_test=y[n_train:],
        classes=list(classes),
    )


def _train_donors(td: Path, seed_a: int, seed_b: int, label_suffix: str):
    cfg = TrioronConfig(cap_bytes=8_000)
    tasks_a = [_make_synthetic_task("blob_01", [0, 1], seed=11, input_dim=32)]
    tasks_b = [_make_synthetic_task("blob_23", [2, 3], seed=22, input_dim=32)]
    donor_a = build_donor(
        tasks=tasks_a, label=f"a{label_suffix}",
        out_path=td / f"donor_a{label_suffix}.pt",
        seed=seed_a, epochs_per_task=3, config=cfg,
    )
    donor_b = build_donor(
        tasks=tasks_b, label=f"b{label_suffix}",
        out_path=td / f"donor_b{label_suffix}.pt",
        seed=seed_b, epochs_per_task=3, config=cfg,
    )
    eval_tasks = tasks_a + tasks_b
    return donor_a, donor_b, eval_tasks


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        print("\n========================================")
        print("Scenario A: SHARED L0 (seeds 42 & 42)")
        print("========================================")
        a1, b1, eval_tasks_1 = _train_donors(td, seed_a=42, seed_b=42, label_suffix="_same")
        org_same = absorb(donor_paths=[a1, b1], out_path=td / "org_same.pt")
        m_same = evaluate(organism_path=org_same, eval_tasks=eval_tasks_1)
        print(f"\n[scenario A] task-aware mean: {m_same['task_aware_mean']:.4f}")
        print(f"[scenario A] full-union mean: {m_same['full_union_mean']:.4f}")
        for r in m_same["per_task"]:
            print(f"    {r['task']:>10}  task-aware {r['task_aware']:.4f}  "
                  f"full-union {r['full_union']:.4f}")

        print("\n========================================")
        print("Scenario B: MISMATCHED L0 (seeds 42 & 7)")
        print("========================================")
        a2, b2, eval_tasks_2 = _train_donors(td, seed_a=42, seed_b=7, label_suffix="_mix")
        org_mix = absorb(donor_paths=[a2, b2], out_path=td / "org_mix.pt")
        m_mix = evaluate(organism_path=org_mix, eval_tasks=eval_tasks_2)
        print(f"\n[scenario B] task-aware mean: {m_mix['task_aware_mean']:.4f}")
        print(f"[scenario B] full-union mean: {m_mix['full_union_mean']:.4f}")
        for r in m_mix["per_task"]:
            print(f"    {r['task']:>10}  task-aware {r['task_aware']:.4f}  "
                  f"full-union {r['full_union']:.4f}")

        # Compare per-task — donor A is canonical in both scenarios
        # (seed=42 wins majority OR ties → first appearance; identical
        # across runs). Donor B is the one whose L0 differs in B.
        gap_task = m_same["task_aware_mean"] - m_mix["task_aware_mean"]
        gap_full = m_same["full_union_mean"] - m_mix["full_union_mean"]
        print("\n========================================")
        print("GAP (shared - mismatched, positive = projection costs accuracy)")
        print("========================================")
        print(f"  task-aware  Δ = {gap_task:+.4f}  "
              f"({m_same['task_aware_mean']:.4f} → {m_mix['task_aware_mean']:.4f})")
        print(f"  full-union  Δ = {gap_full:+.4f}  "
              f"({m_same['full_union_mean']:.4f} → {m_mix['full_union_mean']:.4f})")
        # Per-task drilldown — line up by task name.
        same_by = {r["task"]: r for r in m_same["per_task"]}
        mix_by = {r["task"]: r for r in m_mix["per_task"]}
        for name in same_by:
            sa = same_by[name]["task_aware"]
            mx = mix_by[name]["task_aware"]
            d = sa - mx
            print(f"  {name:>10}  task-aware Δ = {d:+.4f}  "
                  f"({sa:.4f} → {mx:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
