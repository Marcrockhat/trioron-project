"""Self-contained tests for trioron.pruner.

Run with:    python3 test_pruner.py
"""
from __future__ import annotations
import sys
import traceback
import torch

from trioron.network import TrioronNetwork
from trioron.pruner import PruningController, utility_capture


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


def _force_low_utility(layer, idxs, value=0.0):
    """Helper: stamp specific node indices' utility to ~0."""
    with torch.no_grad():
        for i in idxs:
            layer.u[i] = value


def _force_high_utility(layer, idxs, value=1.0):
    with torch.no_grad():
        for i in idxs:
            layer.u[i] = value


# --------------------------------------------------------------------------- #
# utility_capture
# --------------------------------------------------------------------------- #


def test_utility_capture_updates_u_after_backward():
    torch.manual_seed(0)
    net = TrioronNetwork([(4, 6, "relu"), (6, 2, "linear")])
    pre = net.layers[0].u.clone()
    x = torch.randn(8, 4, requires_grad=False)
    y = torch.randn(8, 2)

    with utility_capture(net) as cap:
        pred = net(x)
        loss = (pred - y).pow(2).mean()
        loss.backward()
        cap.update_layer_utilities()

    post = net.layers[0].u
    assert not torch.allclose(pre, post), "u should have moved after capture+update"
    assert (post >= 0).all(), "contribution magnitude should be non-negative"


def test_utility_capture_act_grad_no_backward_noop():
    """In act_grad mode, no backward → .grad is None → contributions are
    zero. With u_decay=0.9 and contrib=0, u stays at 0 starting from 0."""
    net = TrioronNetwork([(4, 4, "relu"), (4, 2, "linear")])
    pre = [layer.u.clone() for layer in net.layers]
    x = torch.randn(2, 4, requires_grad=False)
    with utility_capture(net, mode="act_grad") as cap:
        _ = net(x)
        cap.update_layer_utilities()
    for pre_u, layer in zip(pre, net.layers):
        assert torch.allclose(pre_u, layer.u), "u should not change in act_grad without backward"


def test_utility_capture_act_var_works_without_backward():
    """In act_var mode, the signal is forward-only — no backward needed."""
    torch.manual_seed(0)
    net = TrioronNetwork([(4, 4, "relu"), (4, 2, "linear")])
    pre = net.layers[0].u.clone()
    x = torch.randn(16, 4)
    with utility_capture(net, mode="act_var") as cap:
        _ = net(x)
        cap.update_layer_utilities()
    post = net.layers[0].u
    assert not torch.allclose(pre, post), "act_var mode should update u from variance alone"
    assert (post >= 0).all()


def test_utility_capture_combined_at_convergence():
    """At convergence, gradients vanish — combined mode should still pick
    up the act_var component so used nodes don't get pruned."""
    torch.manual_seed(0)
    net = TrioronNetwork([(4, 4, "relu"), (4, 2, "linear")])
    x = torch.randn(16, 4)
    with utility_capture(net, mode="combined") as cap:
        pred = net(x)
        # Loss zero at the start (deliberate): backward yields tiny grads.
        loss = (pred * 0).sum()
        loss.backward()
        cap.update_layer_utilities()
    # Layer 0 has variance (random input → varying ReLU output → high var on
    # some nodes). u should reflect that, not be uniformly zero.
    u0 = net.layers[0].u
    assert (u0 > 0).any(), f"combined mode should pick up variance signal; u0={u0}"


def test_utility_capture_invalid_mode_raises():
    net = TrioronNetwork([(4, 4, "relu"), (4, 2, "linear")])
    try:
        with utility_capture(net, mode="not_a_mode"):
            pass
    except ValueError:
        return
    raise AssertionError("expected ValueError on bad mode")


# --------------------------------------------------------------------------- #
# PruningController
# --------------------------------------------------------------------------- #


def test_controller_no_prune_below_T_prune():
    net = TrioronNetwork([(4, 6, "relu"), (6, 2, "linear")])
    _force_low_utility(net.layers[0], [0, 1])
    pc = PruningController(u_threshold=0.01, T_prune=10, prune_clock=5)
    # Walk 9 steps — streak reaches 9 at clock tick (step 5), still < T_prune=10.
    pruned_total = []
    for s in range(1, 10):
        pruned_total.extend(pc.step(net, s))
        # Keep utility low.
        _force_low_utility(net.layers[0], [0, 1])
    assert net.layers[0].n_nodes == 6, "should not have pruned below T_prune"


def test_controller_prunes_at_T_prune_on_clock_tick():
    net = TrioronNetwork([(4, 6, "relu"), (6, 2, "linear")])
    pc = PruningController(u_threshold=0.01, T_prune=5, prune_clock=5)
    # Step up to and including the first clock tick at step 5. Streaks hit
    # T_prune=5 simultaneously for nodes 0 and 1, so both get pruned.
    for s in range(1, 6):
        _force_low_utility(net.layers[0], [0, 1])
        _force_high_utility(net.layers[0], [2, 3, 4, 5])
        pc.step(net, s)
    assert net.layers[0].n_nodes == 4, f"expected 4 nodes, got {net.layers[0].n_nodes}"
    assert net.layers[1].fan_in == 4, f"next-layer fan_in {net.layers[1].fan_in}"


def test_controller_keeps_high_utility_nodes():
    net = TrioronNetwork([(4, 5, "relu"), (5, 2, "linear")])
    _force_high_utility(net.layers[0], [0, 1, 2, 3, 4])
    pc = PruningController(u_threshold=0.01, T_prune=5, prune_clock=5)
    for s in range(1, 25):
        _force_high_utility(net.layers[0], [0, 1, 2, 3, 4])
        pc.step(net, s)
    assert net.layers[0].n_nodes == 5, "high-utility nodes must not be pruned"


def test_controller_protect_layers_skips_layer():
    net = TrioronNetwork([(4, 5, "relu"), (5, 5, "relu"), (5, 2, "linear")])
    pc = PruningController(u_threshold=0.01, T_prune=3, prune_clock=3, protect_layers=[0])
    # Run up to the first clock tick at step 3. All of layer 1's nodes have
    # streak == 3, all are candidates, but the last-node safety leaves one.
    for s in range(1, 4):
        _force_low_utility(net.layers[0], list(range(net.layers[0].n_nodes)))
        _force_low_utility(net.layers[1], list(range(net.layers[1].n_nodes)))
        pc.step(net, s)
    assert net.layers[0].n_nodes == 5, "protected layer must be untouched"
    assert net.layers[1].n_nodes == 1, f"expected 1 node remaining, got {net.layers[1].n_nodes}"


def test_controller_refuses_to_prune_last_node():
    net = TrioronNetwork([(4, 1, "relu"), (1, 2, "linear")])
    _force_low_utility(net.layers[0], [0])
    pc = PruningController(u_threshold=0.01, T_prune=2, prune_clock=2)
    for s in range(1, 10):
        _force_low_utility(net.layers[0], [0])
        pc.step(net, s)
    assert net.layers[0].n_nodes == 1, "must not prune the only node"


def test_controller_clears_streaks_after_prune():
    net = TrioronNetwork([(4, 4, "relu"), (4, 2, "linear")])
    pc = PruningController(u_threshold=0.01, T_prune=3, prune_clock=3)
    for s in range(1, 4):
        _force_low_utility(net.layers[0], [0])
        _force_high_utility(net.layers[0], [1, 2, 3])
        pc.step(net, s)
    assert net.layers[0].n_nodes == 3, "node 0 should have been pruned"
    assert pc.streak_snapshot() == {}, "streaks should be cleared after a prune"


def test_controller_invalid_params_raise():
    try:
        PruningController(T_prune=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on T_prune=0")
    try:
        PruningController(prune_clock=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on prune_clock=0")


# --------------------------------------------------------------------------- #


def main():
    print("Running pruner tests...")
    print(f"  torch version: {torch.__version__}")

    tests = [
        ("utility_capture_updates_u",        test_utility_capture_updates_u_after_backward),
        ("utility_capture_act_grad_no_back", test_utility_capture_act_grad_no_backward_noop),
        ("utility_capture_act_var_no_back",  test_utility_capture_act_var_works_without_backward),
        ("utility_capture_combined_conv",    test_utility_capture_combined_at_convergence),
        ("utility_capture_invalid_mode",     test_utility_capture_invalid_mode_raises),
        ("controller_no_prune_below_T",      test_controller_no_prune_below_T_prune),
        ("controller_prunes_at_T_on_tick",   test_controller_prunes_at_T_prune_on_clock_tick),
        ("controller_keeps_high_utility",    test_controller_keeps_high_utility_nodes),
        ("controller_protect_layers_skips",  test_controller_protect_layers_skips_layer),
        ("controller_refuses_last_node",     test_controller_refuses_to_prune_last_node),
        ("controller_clears_streaks",        test_controller_clears_streaks_after_prune),
        ("controller_invalid_params_raise",  test_controller_invalid_params_raise),
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
