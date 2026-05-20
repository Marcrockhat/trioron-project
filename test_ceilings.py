"""Self-contained tests for trioron.ceilings.

Run with:    python3 test_ceilings.py
"""
from __future__ import annotations
import sys
import traceback
import torch

from trioron.network import TrioronNetwork
from trioron.ceilings import (
    CeilingsController,
    REASON_ARRESTED,
    REASON_INVALID_LAYER,
    REASON_MEMORY_CEILING,
    REASON_OK,
    REASON_TIME_CEILING,
    division_param_delta,
)


_RESULTS: list[tuple[str, bool, str]] = []


def _run(name, fn):
    try:
        fn()
        _RESULTS.append((name, True, ""))
        print(f"  PASS  {name}")
    except Exception as e:
        _RESULTS.append((name, False, str(e)))
        print(f"  FAIL  {name}: {e}")
        traceback.print_exc(limit=3)


def _make_net():
    """3-layer network used across tests: [(4 → 6 relu), (6 → 5 relu), (5 → 3 tanh)]."""
    return TrioronNetwork(
        [
            (4, 6, "relu"),
            (6, 5, "relu"),
            (5, 3, "tanh"),
        ]
    )


# --------------------------------------------------------------------------- #
# division_param_delta — math
# --------------------------------------------------------------------------- #


def test_delta_middle_layer_matches_actual_growth():
    """The estimator's float count must equal the actual diff in
    sum(numel) over params + buffers after grow_layer."""
    net = _make_net()
    layer_idx = 1   # middle: fan_in=6, has next layer with n_nodes=3

    def total_numel():
        params = sum(p.numel() for p in net.parameters())
        buffers = 0
        for layer in net.layers:
            buffers += layer.lam.numel() + layer.u.numel()
            buffers += layer.W_anchor.numel() + layer.b_anchor.numel()
            buffers += layer.fisher_W.numel() + layer.fisher_b.numel()
        return params, buffers

    p_before, b_before = total_numel()
    delta = division_param_delta(net, layer_idx, optimizer_state_per_param=2)
    net.grow_layer(layer_idx, init_vec=None, peer_init_for_next=None)
    p_after, b_after = total_numel()

    actual_params = p_after - p_before
    actual_buffers = b_after - b_before
    assert delta.params_floats == actual_params, (
        f"predicted params Δ {delta.params_floats}, actual {actual_params}"
    )
    assert delta.buffers_floats == actual_buffers, (
        f"predicted buffers Δ {delta.buffers_floats}, actual {actual_buffers}"
    )
    # Adam optimizer state = 2x params.
    assert delta.optimizer_floats == 2 * actual_params


def test_delta_last_layer_no_cross_layer_term():
    """Growing the LAST layer must not include the next-layer column terms."""
    net = _make_net()
    last_idx = len(net.layers) - 1
    delta = division_param_delta(net, last_idx, optimizer_state_per_param=2)
    assert not delta.has_next_layer
    assert delta.next_layer_n_nodes == 0
    # Only the new node row+bias+branch_weight on this layer:
    fan_in = net.layers[last_idx].fan_in
    B_max = net.layers[last_idx].B_max
    assert delta.params_floats == fan_in + 1 + B_max
    assert delta.buffers_floats == 4 + 2 * fan_in
    assert delta.optimizer_floats == 2 * (fan_in + 1 + B_max)


def test_delta_first_layer_includes_next():
    net = _make_net()
    delta = division_param_delta(net, 0, optimizer_state_per_param=2)
    assert delta.has_next_layer
    assert delta.next_layer_n_nodes == net.layers[1].n_nodes
    fan_in = net.layers[0].fan_in
    B_max = net.layers[0].B_max
    next_n = net.layers[1].n_nodes
    assert delta.params_floats == (fan_in + 1 + B_max) + next_n
    assert delta.buffers_floats == (4 + 2 * fan_in) + 2 * next_n


def test_delta_invalid_layer_raises():
    net = _make_net()
    try:
        division_param_delta(net, 99)
    except IndexError:
        return
    raise AssertionError("expected IndexError on out-of-range layer_idx")


def test_delta_bytes_uses_dtype():
    net = _make_net()
    delta = division_param_delta(net, 0, optimizer_state_per_param=0)
    assert delta.bytes(dtype_bytes=4) == delta.total_floats * 4
    assert delta.bytes(dtype_bytes=2) == delta.total_floats * 2


def test_delta_optimizer_state_zero_for_sgd():
    net = _make_net()
    delta = division_param_delta(net, 0, optimizer_state_per_param=0)
    assert delta.optimizer_floats == 0
    assert delta.total_floats == delta.params_floats + delta.buffers_floats


# --------------------------------------------------------------------------- #
# CeilingsController — allow path
# --------------------------------------------------------------------------- #


def test_preflight_allows_with_huge_budget():
    net = _make_net()
    c = CeilingsController(
        M_max_bytes=1024 ** 3,           # 1 GB
        T_div_max_seconds=60.0,
        memory_provider=lambda: 0,        # pretend zero current allocation
    )
    decision = c.preflight(net, layer_idx=0)
    assert decision.allowed
    assert decision.reason == REASON_OK
    assert not c.arrested
    assert decision.delta_bytes > 0
    assert decision.projected_bytes == decision.delta_bytes


# --------------------------------------------------------------------------- #
# CeilingsController — memory ceiling
# --------------------------------------------------------------------------- #


def test_memory_ceiling_blocks_and_arrests():
    net = _make_net()
    # M_max set so any non-zero delta will overflow.
    c = CeilingsController(
        M_max_bytes=10,                   # 10 bytes — laughably small
        T_div_max_seconds=60.0,
        memory_provider=lambda: 0,
    )
    decision = c.preflight(net, layer_idx=0)
    assert not decision.allowed
    assert decision.reason == REASON_MEMORY_CEILING
    assert c.arrested
    assert c.arrest_reason == REASON_MEMORY_CEILING


def test_memory_ceiling_uses_current_alloc():
    """A growing process should be denied even if the delta alone fits."""
    net = _make_net()
    delta = division_param_delta(net, 0)
    # Budget is exactly delta_bytes, but current allocation is 1 byte — over.
    M_max = delta.bytes(dtype_bytes=4)
    c = CeilingsController(
        M_max_bytes=M_max,
        T_div_max_seconds=60.0,
        memory_provider=lambda: 1,
    )
    decision = c.preflight(net, layer_idx=0)
    assert not decision.allowed
    assert decision.reason == REASON_MEMORY_CEILING


# --------------------------------------------------------------------------- #
# CeilingsController — sticky arrest
# --------------------------------------------------------------------------- #


def test_arrest_is_sticky_across_subsequent_calls():
    net = _make_net()
    c = CeilingsController(
        M_max_bytes=10,
        T_div_max_seconds=60.0,
        memory_provider=lambda: 0,
    )
    first = c.preflight(net, layer_idx=0)
    assert first.reason == REASON_MEMORY_CEILING
    # Even with an effectively unlimited budget swap-in, arrest stays.
    c.M_max_bytes = 1024 ** 4
    second = c.preflight(net, layer_idx=0)
    assert not second.allowed
    assert second.reason == REASON_ARRESTED


# --------------------------------------------------------------------------- #
# CeilingsController — time ceiling
# --------------------------------------------------------------------------- #


class _FakeClock:
    def __init__(self, start: float = 0.0):
        self.t = float(start)

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += float(seconds)


def test_time_ceiling_arrests_after_overlong_stabilization():
    net = _make_net()
    clock = _FakeClock()
    c = CeilingsController(
        M_max_bytes=1024 ** 3,
        T_div_max_seconds=10.0,
        memory_provider=lambda: 0,
        time_provider=clock,
    )

    first = c.preflight(net, layer_idx=0)
    assert first.allowed

    c.mark_stabilization_start()
    clock.advance(11.0)                  # > 10s budget
    elapsed = c.mark_stabilization_end()
    assert elapsed == 11.0
    assert c.last_stab_seconds == 11.0

    # Next preflight must arrest with TIME reason.
    second = c.preflight(net, layer_idx=0)
    assert not second.allowed
    assert second.reason == REASON_TIME_CEILING
    assert c.arrested


def test_time_within_budget_does_not_arrest():
    net = _make_net()
    clock = _FakeClock()
    c = CeilingsController(
        M_max_bytes=1024 ** 3,
        T_div_max_seconds=10.0,
        memory_provider=lambda: 0,
        time_provider=clock,
    )

    c.preflight(net, layer_idx=0)
    c.mark_stabilization_start()
    clock.advance(5.0)
    c.mark_stabilization_end()
    second = c.preflight(net, layer_idx=0)
    assert second.allowed
    assert second.reason == REASON_OK
    assert not c.arrested


def test_mark_end_without_start_raises():
    c = CeilingsController(
        M_max_bytes=1024 ** 3,
        T_div_max_seconds=10.0,
        memory_provider=lambda: 0,
    )
    try:
        c.mark_stabilization_end()
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError when end called without start")


# --------------------------------------------------------------------------- #
# CeilingsController — invalid layer
# --------------------------------------------------------------------------- #


def test_invalid_layer_does_not_arrest():
    """An out-of-range layer_idx is a programming bug, not a physical
    constraint — surface it as a non-allowing decision but don't trip
    permanent arrest."""
    net = _make_net()
    c = CeilingsController(
        M_max_bytes=1024 ** 3,
        T_div_max_seconds=60.0,
        memory_provider=lambda: 0,
    )
    decision = c.preflight(net, layer_idx=99)
    assert not decision.allowed
    assert decision.reason == REASON_INVALID_LAYER
    assert not c.arrested
    # A subsequent valid call should still be allowed.
    again = c.preflight(net, layer_idx=0)
    assert again.allowed


# --------------------------------------------------------------------------- #
# CeilingsController — persistence
# --------------------------------------------------------------------------- #


def test_state_dict_round_trip_preserves_arrest():
    net = _make_net()
    c = CeilingsController(
        M_max_bytes=10,
        T_div_max_seconds=60.0,
        memory_provider=lambda: 0,
    )
    c.preflight(net, layer_idx=0)
    assert c.arrested

    state = c.state_dict()

    c2 = CeilingsController(
        M_max_bytes=1024 ** 3,
        T_div_max_seconds=60.0,
        memory_provider=lambda: 0,
    )
    c2.load_state_dict(state)
    assert c2.arrested
    assert c2.arrest_reason == REASON_MEMORY_CEILING
    decision = c2.preflight(net, layer_idx=0)
    assert decision.reason == REASON_ARRESTED


# --------------------------------------------------------------------------- #
# Integration: actual grow_layer after preflight allows
# --------------------------------------------------------------------------- #


def test_allowed_preflight_followed_by_grow_layer_works():
    """Preflight ALLOW → grow_layer → bookkeeping flow is integrated end-
    to-end. This is the orchestrator contract from §8 step 7."""
    torch.manual_seed(0)
    net = _make_net()
    n_nodes_before = list(net.n_nodes_per_layer())

    c = CeilingsController(
        M_max_bytes=1024 ** 3,
        T_div_max_seconds=60.0,
        memory_provider=lambda: 0,
        time_provider=_FakeClock(),
    )

    decision = c.preflight(net, layer_idx=1)
    assert decision.allowed
    net.grow_layer(layer_idx=1)
    n_nodes_after = list(net.n_nodes_per_layer())
    assert n_nodes_after[1] == n_nodes_before[1] + 1
    # Forward still works.
    out = net(torch.randn(2, 4))
    assert out.shape == (2, n_nodes_after[-1])


# --------------------------------------------------------------------------- #


def main():
    print("Running ceilings tests...")
    print(f"  torch version: {torch.__version__}")

    tests = [
        ("delta_middle_layer_matches_actual_growth",  test_delta_middle_layer_matches_actual_growth),
        ("delta_last_layer_no_cross_layer_term",      test_delta_last_layer_no_cross_layer_term),
        ("delta_first_layer_includes_next",           test_delta_first_layer_includes_next),
        ("delta_invalid_layer_raises",                test_delta_invalid_layer_raises),
        ("delta_bytes_uses_dtype",                    test_delta_bytes_uses_dtype),
        ("delta_optimizer_state_zero_for_sgd",        test_delta_optimizer_state_zero_for_sgd),
        ("preflight_allows_with_huge_budget",         test_preflight_allows_with_huge_budget),
        ("memory_ceiling_blocks_and_arrests",         test_memory_ceiling_blocks_and_arrests),
        ("memory_ceiling_uses_current_alloc",         test_memory_ceiling_uses_current_alloc),
        ("arrest_is_sticky_across_subsequent_calls",  test_arrest_is_sticky_across_subsequent_calls),
        ("time_ceiling_arrests_after_overlong_stab",  test_time_ceiling_arrests_after_overlong_stabilization),
        ("time_within_budget_does_not_arrest",        test_time_within_budget_does_not_arrest),
        ("mark_end_without_start_raises",             test_mark_end_without_start_raises),
        ("invalid_layer_does_not_arrest",             test_invalid_layer_does_not_arrest),
        ("state_dict_round_trip_preserves_arrest",    test_state_dict_round_trip_preserves_arrest),
        ("allowed_preflight_followed_by_grow_layer",  test_allowed_preflight_followed_by_grow_layer_works),
    ]

    for name, fn in tests:
        _run(name, fn)

    n_pass = sum(1 for _, ok, _ in _RESULTS if ok)
    n_fail = len(_RESULTS) - n_pass
    print()
    print(f"  Total: {len(_RESULTS)}    PASS: {n_pass}    FAIL: {n_fail}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
