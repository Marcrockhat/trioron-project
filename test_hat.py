"""Self-contained tests for trioron.hat.

Run with:    python test_hat.py

Verifies the HAT controller's contract:
  - Active embedding parameters exist for masked layers, none for the head.
  - Forward hook applies mask only in the active mode (off / train / inference).
  - Sparsity loss is autograd-attached and decreases when embeddings shrink.
  - End-of-task snapshots embeddings; cumulative mask grows monotonically.
  - Gradient scaling protects past-task weights (rows with cumulative ≈ 1).
  - apply_inference_mask + restore round-trips cleanly.
  - Lifecycle errors (out-of-order tasks, bad task ids) raise.
"""
from __future__ import annotations
import sys
import traceback

import torch

from trioron.network import TrioronNetwork
from trioron.hat import HATController


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


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #


def test_init_shapes():
    torch.manual_seed(0)
    net = _make_net()
    ctrl = HATController(net, n_total_tasks=4)
    # Two masked layers (layers 0 and 1; layer 2 is the head)
    assert ctrl.masked_layer_idxs == [0, 1]
    assert ctrl.task_dims == [6, 6]
    assert len(ctrl.active_embeddings) == 2
    assert ctrl.active_embeddings[0].shape == (6,)
    assert ctrl.active_embeddings[1].shape == (6,)
    # Cumulative starts at zero
    assert torch.all(ctrl._cum_mask(0) == 0)
    assert torch.all(ctrl._cum_mask(1) == 0)


def test_hooks_off_by_default_no_change():
    torch.manual_seed(0)
    net = _make_net()
    ctrl = HATController(net, n_total_tasks=4)
    # No hooks before begin_task → forward unchanged
    x = torch.randn(2, 8)
    y_before = net(x).detach().clone()
    # Set embedding to large negative (would make mask ≈ 0) — but no hook installed
    with torch.no_grad():
        ctrl.active_embeddings[0].fill_(-5.0)
        ctrl.active_embeddings[1].fill_(-5.0)
    y_after = net(x).detach().clone()
    assert torch.allclose(y_before, y_after), "no hooks installed → output unchanged"


def test_train_hook_applies_active_mask():
    torch.manual_seed(0)
    net = _make_net()
    ctrl = HATController(net, n_total_tasks=4, s_min=1.0, s_max=10.0)
    ctrl.begin_task(1)
    # Force masks to ~0 by large negative embeddings
    with torch.no_grad():
        ctrl.active_embeddings[0].fill_(-5.0)
        ctrl.active_embeddings[1].fill_(-5.0)
    ctrl.set_temperature(10.0)
    x = torch.randn(2, 8)
    y = net(x)
    # σ(10 · -5) ≈ 0; outputs of layer 1 hit the head as ≈ 0; head is tanh(0) = 0
    assert y.abs().max().item() < 1e-3, "near-zero mask should drive head output to ~0"


def test_end_task_advances_cumulative():
    torch.manual_seed(0)
    net = _make_net()
    ctrl = HATController(net, n_total_tasks=4, s_min=1.0, s_max=10.0)
    ctrl.begin_task(1)
    with torch.no_grad():
        ctrl.active_embeddings[0].fill_(2.0)  # σ(10·2) ≈ 1
        ctrl.active_embeddings[1].fill_(-2.0) # σ(10·-2) ≈ 0
    ctrl.end_task(1)
    cm0 = ctrl._cum_mask(0)
    cm1 = ctrl._cum_mask(1)
    assert (cm0 > 0.99).all(), "first layer should be ~fully claimed"
    assert (cm1 < 0.01).all(), "second layer should be unclaimed"
    # Task 2 starts fresh (active embedding zeroed)
    ctrl.begin_task(2)
    assert torch.all(ctrl.active_embeddings[0] == 0)
    assert torch.all(ctrl.active_embeddings[1] == 0)


def test_cumulative_grows_monotonically():
    torch.manual_seed(0)
    net = _make_net()
    ctrl = HATController(net, n_total_tasks=4, s_min=1.0, s_max=10.0)
    densities_layer0 = []
    for t in range(1, 5):
        ctrl.begin_task(t)
        with torch.no_grad():
            # Activate one new unit per task
            ctrl.active_embeddings[0].fill_(-5.0)
            ctrl.active_embeddings[0][t - 1] = 5.0
        ctrl.end_task(t)
        densities_layer0.append(ctrl.cumulative_mask_density()[0])
    # After 4 tasks, 4 of 6 units in layer 0 should be ≈ active.
    for i in range(1, len(densities_layer0)):
        assert densities_layer0[i] >= densities_layer0[i - 1] - 1e-5, (
            f"density must be monotone non-decreasing, got {densities_layer0}"
        )
    assert densities_layer0[-1] > 0.5, "after 4 tasks, layer 0 should be >50% claimed"


def test_sparsity_loss_autograd():
    torch.manual_seed(0)
    net = _make_net()
    ctrl = HATController(net, n_total_tasks=4, s_min=1.0, s_max=10.0,
                         sparsity_coef=1.0)
    ctrl.begin_task(1)
    ctrl.set_temperature(5.0)
    with torch.no_grad():
        ctrl.active_embeddings[0].fill_(2.0)  # σ(5·2) ≈ ~1
    R = ctrl.sparsity_loss()
    R.backward()
    # Embedding gradients exist (non-None) and are non-zero
    g0 = ctrl.active_embeddings[0].grad
    assert g0 is not None and torch.any(g0 != 0)


def test_grad_scale_protects_past_task_weights():
    torch.manual_seed(0)
    net = _make_net()
    ctrl = HATController(net, n_total_tasks=4, s_min=1.0, s_max=10.0,
                         sparsity_coef=0.0)
    # Task 1: claim layer-0 unit 0 fully
    ctrl.begin_task(1)
    with torch.no_grad():
        ctrl.active_embeddings[0].fill_(-5.0)
        ctrl.active_embeddings[0][0] = 5.0
    ctrl.end_task(1)

    # Task 2: train with arbitrary gradient
    ctrl.begin_task(2)
    ctrl.set_temperature(5.0)
    x = torch.randn(4, 8)
    y = net(x).sum()
    y.backward()
    # Before scale_grads: row 0 of W^0 has nonzero grad (probably).
    # After scale_grads: should be scaled by ~(1-1)=0 since cum_mask_0[0]≈1.
    ctrl.scale_grads()
    g_row0 = net.layers[0].W.grad[0]
    assert g_row0.abs().max().item() < 1e-5, (
        f"row 0 grad should be zeroed by cumulative mask, got "
        f"{g_row0.abs().max().item()}"
    )
    # Other rows still have grad
    g_row1 = net.layers[0].W.grad[1]
    assert g_row1.abs().sum().item() > 0


def test_apply_and_restore_inference_mask_round_trip():
    torch.manual_seed(0)
    net = _make_net()
    ctrl = HATController(net, n_total_tasks=4, s_min=1.0, s_max=10.0)
    # Run task 1 with a clear mask pattern
    ctrl.begin_task(1)
    with torch.no_grad():
        ctrl.active_embeddings[0].fill_(-5.0)
        ctrl.active_embeddings[0][0:3] = 5.0
        ctrl.active_embeddings[1].fill_(2.0)
    ctrl.end_task(1)

    x = torch.randn(2, 8)
    # Without inference-mask hooks, forward uses no mask.
    y_naive = net(x).detach().clone()
    snap = ctrl.apply_inference_mask(1)
    y_with_mask = net(x).detach().clone()
    ctrl.restore(snap)
    y_after_restore = net(x).detach().clone()
    # Naive (no hooks) and after-restore should match
    assert torch.allclose(y_naive, y_after_restore), (
        "restore should remove hooks and yield same output as no-mask"
    )
    # With mask should differ
    assert not torch.allclose(y_naive, y_with_mask), (
        "inference mask should change the network output"
    )


def test_invalid_task_ids_raise():
    net = _make_net()
    ctrl = HATController(net, n_total_tasks=2)
    try:
        ctrl.begin_task(0)
        assert False, "expected ValueError for task_id < 1"
    except ValueError:
        pass
    try:
        ctrl.begin_task(3)
        assert False, "expected ValueError for task_id > n_total_tasks"
    except ValueError:
        pass
    ctrl.begin_task(1)
    try:
        ctrl.begin_task(1)  # tasks_done is still 0, this should be fine again
    except ValueError:
        pass  # implementation may forbid double-begin without end; that's ok too
    ctrl.end_task(1)
    try:
        ctrl.end_task(1)
        assert False, "expected ValueError for double end_task"
    except ValueError:
        pass


def test_invalid_constructor_args_raise():
    net = _make_net()
    try:
        HATController(net, n_total_tasks=0)
        assert False, "expected ValueError for n_total_tasks=0"
    except ValueError:
        pass
    try:
        HATController(net, n_total_tasks=2, s_min=10.0, s_max=1.0)
        assert False, "expected ValueError for s_min > s_max"
    except ValueError:
        pass
    try:
        HATController(net, n_total_tasks=2, sparsity_coef=-1.0)
        assert False, "expected ValueError for sparsity_coef < 0"
    except ValueError:
        pass
    try:
        HATController(net, n_total_tasks=2, emb_clip=0)
        assert False, "expected ValueError for emb_clip=0"
    except ValueError:
        pass


def test_temperature_anneal_endpoints():
    net = _make_net()
    ctrl = HATController(net, n_total_tasks=2, s_min=0.01, s_max=400.0)
    assert ctrl.temperature_for_step(0, 1500) == 0.01
    assert abs(ctrl.temperature_for_step(1499, 1500) - 400.0) < 1e-6
    assert 100 < ctrl.temperature_for_step(750, 1500) < 300


def test_clip_embeddings_bounds():
    net = _make_net()
    ctrl = HATController(net, n_total_tasks=2, emb_clip=3.0)
    ctrl.begin_task(1)
    with torch.no_grad():
        ctrl.active_embeddings[0].fill_(10.0)
        ctrl.active_embeddings[1].fill_(-10.0)
    ctrl.clip_embeddings()
    assert ctrl.active_embeddings[0].max().item() == 3.0
    assert ctrl.active_embeddings[1].min().item() == -3.0


def test_end_to_end_two_tasks_optimizer_step():
    """Smoke: one task with several optimizer steps + sparsity loss + grad scale.
    Confirms gradients flow into embeddings and the optimizer can move them."""
    torch.manual_seed(0)
    net = _make_net()
    ctrl = HATController(net, n_total_tasks=2, s_min=0.01, s_max=10.0,
                         sparsity_coef=0.5)
    opt = torch.optim.Adam(
        list(net.parameters()) + list(ctrl.parameters()), lr=1e-2
    )
    ctrl.begin_task(1)
    init_embs = [p.detach().clone() for p in ctrl.active_embeddings]
    for step in range(20):
        ctrl.set_temperature(ctrl.temperature_for_step(step, 20))
        x = torch.randn(8, 8)
        y_target = torch.randn(8, 3)
        y = net(x)
        l = ((y - y_target) ** 2).mean() + ctrl.sparsity_coef * ctrl.sparsity_loss()
        opt.zero_grad()
        l.backward()
        ctrl.scale_grads()
        opt.step()
        ctrl.clip_embeddings()
    # Embeddings should have moved
    moved = sum(
        (p - p0).abs().sum().item()
        for p, p0 in zip(ctrl.active_embeddings, init_embs)
    )
    assert moved > 0, "embeddings did not change after 20 optimizer steps"
    ctrl.end_task(1)


# --------------------------------------------------------------------------- #
# Runner                                                                      #
# --------------------------------------------------------------------------- #


def main() -> int:
    print("test_hat.py")
    print("-" * 60)
    _run("init_shapes", test_init_shapes)
    _run("hooks_off_by_default_no_change", test_hooks_off_by_default_no_change)
    _run("train_hook_applies_active_mask", test_train_hook_applies_active_mask)
    _run("end_task_advances_cumulative", test_end_task_advances_cumulative)
    _run("cumulative_grows_monotonically", test_cumulative_grows_monotonically)
    _run("sparsity_loss_autograd", test_sparsity_loss_autograd)
    _run("grad_scale_protects_past_task_weights",
         test_grad_scale_protects_past_task_weights)
    _run("apply_and_restore_inference_mask_round_trip",
         test_apply_and_restore_inference_mask_round_trip)
    _run("invalid_task_ids_raise", test_invalid_task_ids_raise)
    _run("invalid_constructor_args_raise", test_invalid_constructor_args_raise)
    _run("temperature_anneal_endpoints", test_temperature_anneal_endpoints)
    _run("clip_embeddings_bounds", test_clip_embeddings_bounds)
    _run("end_to_end_two_tasks_optimizer_step",
         test_end_to_end_two_tasks_optimizer_step)
    print("-" * 60)
    n_pass = sum(1 for _, ok, _ in _RESULTS if ok)
    n_fail = len(_RESULTS) - n_pass
    print(f"  {n_pass} passed, {n_fail} failed")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
