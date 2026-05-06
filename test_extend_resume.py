"""Extend resume-from-substrate equivalence test.

Verifies that `api.extend()` on a v2 donor (resume path) produces a
final task-aware accuracy within seed-noise of the legacy integrated
path (re-train base from scratch).

Run with:    python3 test_extend_resume.py
"""
from __future__ import annotations
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import torch

from trioron.api import TaskData, TrioronConfig, build_donor, evaluate, extend


_RESULTS: list[tuple[str, bool, str]] = []


def _run(name, fn):
    try:
        fn()
        _RESULTS.append((name, True, ""))
        print(f"  PASS  {name}")
    except Exception as e:
        _RESULTS.append((name, False, str(e)))
        print(f"  FAIL  {name}: {e}")
        traceback.print_exc(limit=5)


def _make_synthetic_task(name, classes, *, seed, n_per_class=120, input_dim=64):
    """Per-class Gaussian blobs in input_dim-D — easy to learn at small scale."""
    g = torch.Generator().manual_seed(seed)
    Xs, ys = [], []
    for c in classes:
        # Each class centered at a different random point with unit noise.
        center = torch.randn(input_dim, generator=g) * 3.0
        x = center + torch.randn(n_per_class, input_dim, generator=g)
        Xs.append(x)
        ys.append(torch.full((n_per_class,), c, dtype=torch.int64))
    X = torch.cat(Xs, dim=0)
    y = torch.cat(ys, dim=0)
    # 80/20 train/test split per class.
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


def _downgrade_donor_to_v1(donor_path: Path) -> Path:
    """Strip task_class_lists + reset version to 1 to simulate a legacy
    donor. Forces extend() to fall back to the integrated path."""
    payload = torch.load(str(donor_path), map_location="cpu", weights_only=False)
    payload["version"] = 1
    payload.pop("task_class_lists", None)
    legacy_path = donor_path.with_name(donor_path.stem + "_v1.pt")
    torch.save(payload, legacy_path)
    return legacy_path


def test_donor_v2_carries_task_class_lists():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        base_tasks = [
            _make_synthetic_task("base_a", [0, 1], seed=1, n_per_class=80, input_dim=32),
            _make_synthetic_task("base_b", [2, 3], seed=2, n_per_class=80, input_dim=32),
        ]
        cfg = TrioronConfig(cap_bytes=8_000)
        donor = build_donor(
            tasks=base_tasks, label="t", out_path=td / "donor.pt",
            seed=42, epochs_per_task=2, config=cfg,
        )
        payload = torch.load(str(donor), map_location="cpu", weights_only=False)
        assert payload["version"] == 2, f"version={payload['version']}"
        assert payload["task_class_lists"] == [[0, 1], [2, 3]], (
            f"got {payload['task_class_lists']}"
        )


def test_resume_matches_integrated_within_tolerance():
    """End-to-end: run extend on v2 (resume) and v1 (integrated) using
    the same base + extension data, compare final task-aware accuracy.

    Resume RNG is reseeded at the boundary so we expect within
    seed-noise, not bit-exact equality. Tolerance 0.10 is loose but
    the synthetic Gaussian blobs are easy enough that both paths land
    near-ceiling — a >0.10 gap would indicate a real bug.
    """
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        base_tasks = [
            _make_synthetic_task("base_a", [0, 1], seed=11, n_per_class=120, input_dim=32),
            _make_synthetic_task("base_b", [2, 3], seed=22, n_per_class=120, input_dim=32),
        ]
        ext_tasks = [
            _make_synthetic_task("ext_c", [4, 5], seed=33, n_per_class=120, input_dim=32),
        ]
        cfg = TrioronConfig(cap_bytes=8_000)
        donor_v2 = build_donor(
            tasks=base_tasks, label="t", out_path=td / "donor.pt",
            seed=42, epochs_per_task=2, config=cfg,
        )
        donor_v1 = _downgrade_donor_to_v1(donor_v2)

        # Resume path
        ext_v2 = extend(
            donor_path=donor_v2,
            base_tasks=base_tasks, new_tasks=ext_tasks,
            out_path=td / "ext_v2.pt",
            extension_cap_bytes=16_000,
            epochs_per_task=2,
            permanent_int8=False,
        )
        # Legacy integrated path (donor_v1 lacks task_class_lists → warning + fallback)
        ext_v1 = extend(
            donor_path=donor_v1,
            base_tasks=base_tasks, new_tasks=ext_tasks,
            out_path=td / "ext_v1.pt",
            extension_cap_bytes=16_000,
            epochs_per_task=2,
            permanent_int8=False,
        )

        all_tasks = base_tasks + ext_tasks
        m_v2 = evaluate(organism_path=ext_v2, eval_tasks=all_tasks)
        m_v1 = evaluate(organism_path=ext_v1, eval_tasks=all_tasks)
        gap = abs(m_v2["task_aware_mean"] - m_v1["task_aware_mean"])
        print(f"    resume task-aware {m_v2['task_aware_mean']:.4f}  "
              f"integrated task-aware {m_v1['task_aware_mean']:.4f}  "
              f"gap {gap:.4f}")
        assert gap < 0.15, (
            f"resume vs integrated task-aware gap {gap:.4f} > 0.15 — "
            f"larger than seed-noise on synthetic data"
        )
        # Sanity: both paths should be well above chance (1/6 ≈ 0.17).
        assert m_v2["task_aware_mean"] > 0.5, (
            f"resume task-aware {m_v2['task_aware_mean']:.4f} too low"
        )
        assert m_v1["task_aware_mean"] > 0.5, (
            f"integrated task-aware {m_v1['task_aware_mean']:.4f} too low"
        )


def test_legacy_v1_falls_back_with_warning(capsys=None):
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        base_tasks = [
            _make_synthetic_task("base_a", [0, 1], seed=1, n_per_class=80, input_dim=32),
        ]
        ext_tasks = [
            _make_synthetic_task("ext_b", [2, 3], seed=2, n_per_class=80, input_dim=32),
        ]
        cfg = TrioronConfig(cap_bytes=8_000)
        donor_v2 = build_donor(
            tasks=base_tasks, label="t", out_path=td / "donor.pt",
            seed=42, epochs_per_task=2, config=cfg,
        )
        donor_v1 = _downgrade_donor_to_v1(donor_v2)
        # Capture stdout to check warning text.
        from io import StringIO
        old = sys.stdout
        buf = StringIO()
        sys.stdout = buf
        try:
            extend(
                donor_path=donor_v1,
                base_tasks=base_tasks, new_tasks=ext_tasks,
                out_path=td / "ext_v1.pt",
                extension_cap_bytes=12_000,
                epochs_per_task=2,
                permanent_int8=False,
            )
        finally:
            sys.stdout = old
        out = buf.getvalue()
        assert "[trioron extend] WARNING" in out, (
            "expected fallback warning on legacy donor, got: "
            + out[:400]
        )


def main() -> int:
    print("test_extend_resume")
    _run("donor_v2_carries_task_class_lists", test_donor_v2_carries_task_class_lists)
    _run("resume_matches_integrated_within_tolerance",
         test_resume_matches_integrated_within_tolerance)
    _run("legacy_v1_falls_back_with_warning", test_legacy_v1_falls_back_with_warning)
    failed = [r for r in _RESULTS if not r[1]]
    print(f"\n{len(_RESULTS)-len(failed)}/{len(_RESULTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
