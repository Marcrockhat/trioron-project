"""Self-contained tests for trioron.packnet.

Run with:    python test_packnet.py

Verifies the PackNet controller's contract:
  - Frozen weights stay frozen across training steps.
  - Per-task masks are disjoint (no overlap).
  - Cumulative frozen count grows monotonically and never exceeds total weights.
  - Inference mask zeros out non-task weights and restore() round-trips.
  - Re-init at begin_task touches only free weights.
"""
from __future__ import annotations
import sys
import traceback

import torch

from trioron.network import TrioronNetwork
from trioron.packnet import PackNetController


_RESULTS: list[tuple[str, bool, str]] = []


def _run(name: str, fn) -> None:
    try:
        fn()
        _RESULTS.append((name, True, ""))
        print(f"  PASS  {name}")
    except Exception as e:
        _RESULTS.append((name, False, str(e)))
        print(f"  FAIL  {name}: {e}")
        traceback.print_exc(limit=3)


def _make_net():
    return TrioronNetwork(
        [
            (8, 6, "relu"),
            (6, 6, "relu"),
            (6, 3, "tanh"),
        ]
    )


def _total_weights(net):
    n = 0
    for layer in net.layers:
        n += layer.W.numel() + layer.b.numel()
    return n


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #


def test_init_no_frozen_no_masks():
    torch.manual_seed(0)
    net = _make_net()
    ctrl = PackNetController(net, n_total_tasks=4)
    assert ctrl.tasks_done == 0
    for Wm, bm in ctrl.frozen:
        assert not Wm.any().item(), "no frozen W bits at init"
        assert not bm.any().item(), "no frozen b bits at init"
    assert ctrl.task_masks == {}


def test_end_task_grows_frozen_monotonically():
    torch.manual_seed(0)
    net = _make_net()
    total = _total_weights(net)
    ctrl = PackNetController(net, n_total_tasks=4)

    counts = []
    for t in range(1, 5):
        ctrl.begin_task(t)
        ctrl.end_task(t)
        counts.append(ctrl.cumulative_frozen_count())

    # Strictly non-decreasing (and each step adds something for non-empty layer).
    for i in range(1, len(counts)):
        assert counts[i] >= counts[i - 1], (
            f"frozen count went backwards: {counts}"
        )
    assert counts[-1] <= total, f"frozen count {counts[-1]} > total {total}"


def test_per_task_masks_disjoint():
    torch.manual_seed(0)
    net = _make_net()
    ctrl = PackNetController(net, n_total_tasks=4)
    for t in range(1, 5):
        ctrl.begin_task(t)
        ctrl.end_task(t)

    # For each layer, check that all task masks are pairwise disjoint.
    for li in range(len(net.layers)):
        accumulated_W = torch.zeros_like(ctrl.task_masks[1][li][0])
        for tid in range(1, 5):
            kW, _ = ctrl.task_masks[tid][li]
            overlap = accumulated_W & kW
            assert not overlap.any().item(), (
                f"layer {li} task {tid}: mask overlaps with prior tasks"
            )
            accumulated_W = accumulated_W | kW


def test_freeze_grads_zeros_frozen_gradient():
    torch.manual_seed(0)
    net = _make_net()
    ctrl = PackNetController(net, n_total_tasks=2)
    ctrl.begin_task(1)
    ctrl.end_task(1)

    # Run a forward+backward and verify frozen entries get zero gradient.
    x = torch.randn(4, 8)
    y = net(x).sum()
    y.backward()
    ctrl.freeze_grads()

    for li, layer in enumerate(net.layers):
        Wm, bm = ctrl.frozen[li]
        if layer.W.grad is not None:
            frozen_grad = layer.W.grad[Wm]
            assert (frozen_grad == 0).all().item(), (
                f"layer {li}: frozen W has non-zero grad after freeze_grads"
            )
        if layer.b.grad is not None:
            frozen_b_grad = layer.b.grad[bm]
            assert (frozen_b_grad == 0).all().item(), (
                f"layer {li}: frozen b has non-zero grad after freeze_grads"
            )


def test_frozen_weights_unchanged_under_optimizer_step():
    torch.manual_seed(0)
    net = _make_net()
    ctrl = PackNetController(net, n_total_tasks=2)
    ctrl.begin_task(1)
    ctrl.end_task(1)

    # Snapshot frozen weight values
    pre_W = [layer.W.data.clone() for layer in net.layers]
    pre_b = [layer.b.data.clone() for layer in net.layers]

    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(5):
        x = torch.randn(8, 8)
        loss = (net(x) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        ctrl.freeze_grads()
        opt.step()

    for li, layer in enumerate(net.layers):
        Wm, bm = ctrl.frozen[li]
        # Frozen entries must be exactly equal to pre_W[Wm]
        assert torch.allclose(
            layer.W.data[Wm], pre_W[li][Wm]
        ), f"layer {li}: frozen W changed under optimizer.step()"
        assert torch.allclose(
            layer.b.data[bm], pre_b[li][bm]
        ), f"layer {li}: frozen b changed under optimizer.step()"


def test_begin_task_reinitializes_only_free_weights():
    torch.manual_seed(0)
    net = _make_net()
    ctrl = PackNetController(net, n_total_tasks=3)
    ctrl.begin_task(1)
    ctrl.end_task(1)

    pre_W = [layer.W.data.clone() for layer in net.layers]

    ctrl.begin_task(2)

    for li, layer in enumerate(net.layers):
        Wm, _ = ctrl.frozen[li]
        # Frozen entries unchanged
        assert torch.allclose(
            layer.W.data[Wm], pre_W[li][Wm]
        ), f"layer {li}: frozen W changed after begin_task"
        # At least one free entry should differ from pre (re-init occurred).
        # (If frozen covers everything it can't differ; check before asserting.)
        if (~Wm).any():
            free_pre = pre_W[li][~Wm]
            free_post = layer.W.data[~Wm]
            assert not torch.allclose(free_pre, free_post), (
                f"layer {li}: free weights NOT re-initialized at begin_task"
            )


def test_inference_mask_zeros_non_task_weights_and_restore_round_trips():
    torch.manual_seed(0)
    net = _make_net()
    ctrl = PackNetController(net, n_total_tasks=3)
    for t in range(1, 4):
        ctrl.begin_task(t)
        ctrl.end_task(t)

    pre_W = [layer.W.data.clone() for layer in net.layers]
    pre_b = [layer.b.data.clone() for layer in net.layers]

    snap = ctrl.apply_inference_mask(eval_task_id=2)

    for li, layer in enumerate(net.layers):
        union_W = torch.zeros_like(layer.W, dtype=torch.bool)
        union_b = torch.zeros_like(layer.b, dtype=torch.bool)
        for tid in (1, 2):
            kW, kb = ctrl.task_masks[tid][li]
            union_W |= kW
            union_b |= kb
        # All weights NOT in union should be zero
        assert (layer.W.data[~union_W] == 0).all().item(), (
            f"layer {li}: non-task W not zeroed under inference mask"
        )
        assert (layer.b.data[~union_b] == 0).all().item(), (
            f"layer {li}: non-task b not zeroed under inference mask"
        )

    ctrl.restore(snap)
    for li, layer in enumerate(net.layers):
        assert torch.allclose(layer.W.data, pre_W[li]), (
            f"layer {li}: W not restored after restore()"
        )
        assert torch.allclose(layer.b.data, pre_b[li]), (
            f"layer {li}: b not restored after restore()"
        )


def test_apply_inference_mask_rejects_invalid_task():
    torch.manual_seed(0)
    net = _make_net()
    ctrl = PackNetController(net, n_total_tasks=3)
    ctrl.begin_task(1)
    ctrl.end_task(1)
    try:
        ctrl.apply_inference_mask(eval_task_id=2)
        raise RuntimeError("expected ValueError for unfinished task")
    except ValueError:
        pass
    try:
        ctrl.apply_inference_mask(eval_task_id=0)
        raise RuntimeError("expected ValueError for task_id 0")
    except ValueError:
        pass


def test_end_task_must_be_in_order():
    torch.manual_seed(0)
    net = _make_net()
    ctrl = PackNetController(net, n_total_tasks=3)
    ctrl.begin_task(1)
    try:
        ctrl.end_task(2)
        raise RuntimeError("expected ValueError when skipping a task")
    except ValueError:
        pass


def test_end_to_end_with_4_tasks():
    """Smoke: 4 tasks, train each on a different synthetic target,
    end with disjoint masks covering most of the network."""
    torch.manual_seed(0)
    net = _make_net()
    n_tasks = 4
    ctrl = PackNetController(net, n_total_tasks=n_tasks)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)

    for t in range(1, n_tasks + 1):
        ctrl.begin_task(t)
        # Re-create optimizer (state stale across re-init)
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)
        for _ in range(20):
            x = torch.randn(16, 8)
            target = torch.zeros(16, 3)
            target[:, (t - 1) % 3] = 1.0  # cycle target dim across tasks
            pred = net(x)
            loss = (pred - target).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            ctrl.freeze_grads()
            opt.step()
        ctrl.end_task(t)

    # All 4 tasks done, masks recorded.
    assert ctrl.tasks_done == 4
    capacities = ctrl.per_task_capacity()
    assert all(c > 0 for c in capacities), (
        f"some task got zero capacity: {capacities}"
    )
    # Per-task inference round-trip.
    for t in range(1, 5):
        snap = ctrl.apply_inference_mask(t)
        x = torch.randn(4, 8)
        with torch.no_grad():
            _ = net(x)  # must not crash
        ctrl.restore(snap)


# --------------------------------------------------------------------------- #
# Run                                                                         #
# --------------------------------------------------------------------------- #


def main():
    print("Running PackNetController tests")
    print("-" * 60)
    _run("init: no frozen, no task masks", test_init_no_frozen_no_masks)
    _run("end_task grows frozen monotonically", test_end_task_grows_frozen_monotonically)
    _run("per-task masks are disjoint", test_per_task_masks_disjoint)
    _run("freeze_grads zeros frozen gradient", test_freeze_grads_zeros_frozen_gradient)
    _run("frozen weights unchanged under optimizer step",
         test_frozen_weights_unchanged_under_optimizer_step)
    _run("begin_task re-inits free weights only",
         test_begin_task_reinitializes_only_free_weights)
    _run("inference mask zeros non-task; restore round-trips",
         test_inference_mask_zeros_non_task_weights_and_restore_round_trips)
    _run("apply_inference_mask rejects invalid task",
         test_apply_inference_mask_rejects_invalid_task)
    _run("end_task must be in order", test_end_task_must_be_in_order)
    _run("end-to-end with 4 tasks", test_end_to_end_with_4_tasks)

    n_pass = sum(1 for _, p, _ in _RESULTS if p)
    n_fail = len(_RESULTS) - n_pass
    print("-" * 60)
    print(f"{n_pass} PASS, {n_fail} FAIL")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
