"""Self-contained tests for trioron.network.

Run with:    python3 test_network.py
"""
from __future__ import annotations
import sys
import traceback
import torch

from trioron.network import TrioronNetwork


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


# --------------------------------------------------------------------------- #


def test_construction_and_forward():
    net = TrioronNetwork([(4, 8, "relu"), (8, 3, "linear")])
    x = torch.randn(2, 4)
    y = net(x)
    assert y.shape == (2, 3), f"got {tuple(y.shape)}"


def test_three_layer_network():
    net = TrioronNetwork([(8, 16, "relu"), (16, 16, "relu"), (16, 4, "linear")])
    x = torch.randn(5, 8)
    y = net(x)
    assert y.shape == (5, 4)


def test_dimension_mismatch_raises():
    raised = False
    try:
        TrioronNetwork([(4, 8, "relu"), (16, 3, "linear")])  # 8 != 16
    except ValueError:
        raised = True
    assert raised, "expected ValueError on dim mismatch"


def test_empty_specs_raises():
    raised = False
    try:
        TrioronNetwork([])
    except ValueError:
        raised = True
    assert raised


def test_n_parameters_correct():
    net = TrioronNetwork([(4, 8, "relu"), (8, 3, "linear")])
    # layer 1: W (8x4) + b (8) = 32 + 8 = 40
    # layer 2: W (3x8) + b (3) = 24 + 3 = 27
    expected = 40 + 27
    got = net.n_parameters()
    assert got == expected, f"got {got}, expected {expected}"


def test_ewc_penalty_zero_at_init():
    net = TrioronNetwork([(4, 8, "relu"), (8, 3, "linear")])
    for layer in net.layers:
        layer.lam.fill_(1.0)
    # No drift from anchor → zero
    assert abs(net.ewc_penalty().item()) < 1e-6


def test_ewc_penalty_sums_layers():
    net = TrioronNetwork([(4, 8, "relu"), (8, 3, "linear")])
    for layer in net.layers:
        layer.lam.fill_(1.0)
    # Drift only first layer
    with torch.no_grad():
        net.layers[0].W.add_(0.1)
    p_only_layer0 = net.ewc_penalty().item()
    # Now also drift second layer — total penalty should grow
    with torch.no_grad():
        net.layers[1].W.add_(0.1)
    p_both = net.ewc_penalty().item()
    assert p_both > p_only_layer0


def test_anchor_all_resets_penalty():
    net = TrioronNetwork([(4, 8, "relu"), (8, 3, "linear")])
    for layer in net.layers:
        layer.lam.fill_(1.0)
    with torch.no_grad():
        for layer in net.layers:
            layer.W.add_(0.1)
    assert net.ewc_penalty().item() > 0
    net.anchor_all()
    assert abs(net.ewc_penalty().item()) < 1e-6


def test_estimate_fisher_resets_and_populates():
    torch.manual_seed(0)
    net = TrioronNetwork([(4, 8, "relu"), (8, 3, "linear")])
    # Pre-fill Fisher with garbage; estimate_fisher should clear this.
    for layer in net.layers:
        layer.fisher_W.fill_(99.0)

    X = torch.randn(64, 4)
    Y = torch.randn(64, 3)

    def make_batches():
        for _ in range(20):
            idx = torch.randperm(64)[:32]
            yield X[idx], Y[idx]

    net.estimate_fisher(
        make_batches(),
        loss_fn=lambda p, y: (p - y).pow(2).mean(),
        n_batches=20,
    )

    # Fisher should NOT still be 99 anywhere (was reset).
    for i, layer in enumerate(net.layers):
        assert (layer.fisher_W != 99.0).all(), f"layer {i} fisher not reset"

    # Fisher should be nonzero somewhere (signal accumulated).
    has_signal = any(
        (layer.fisher_W.abs() > 0).any().item() for layer in net.layers
    )
    assert has_signal, "fisher accumulated nothing"


def test_state_dict_roundtrip():
    net = TrioronNetwork([(4, 8, "relu"), (8, 3, "linear")])
    for layer in net.layers:
        layer.lam.fill_(0.5)
        layer.fisher_W.fill_(0.3)
    sd = net.state_dict()

    net2 = TrioronNetwork([(4, 8, "relu"), (8, 3, "linear")])
    net2.load_state_dict(sd)
    for l1, l2 in zip(net.layers, net2.layers):
        assert torch.allclose(l1.W, l2.W)
        assert torch.allclose(l1.lam, l2.lam)
        assert torch.allclose(l1.fisher_W, l2.fisher_W)


def test_n_nodes_per_layer():
    net = TrioronNetwork([(4, 8, "relu"), (8, 3, "linear")])
    assert net.n_nodes_per_layer() == [8, 3]


def test_gradient_flows_through_full_stack():
    """A 3-layer network should have grad on every parameter after backward."""
    torch.manual_seed(0)
    net = TrioronNetwork(
        [(4, 8, "relu"), (8, 8, "relu"), (8, 2, "linear")]
    )
    x = torch.randn(8, 4)
    y = torch.randn(8, 2)
    pred = net(x)
    loss = (pred - y).pow(2).mean()
    loss.backward()
    for i, layer in enumerate(net.layers):
        assert layer.W.grad is not None, f"layer {i} W has no grad"
        assert layer.b.grad is not None, f"layer {i} b has no grad"
        assert (layer.W.grad.abs() > 0).any(), f"layer {i} W grad is all zero"


# --------------------------------------------------------------------------- #


def main():
    print("Running TrioronNetwork tests...")
    print(f"  torch version: {torch.__version__}")

    tests = [
        ("construction_and_forward",          test_construction_and_forward),
        ("three_layer_network",               test_three_layer_network),
        ("dimension_mismatch_raises",         test_dimension_mismatch_raises),
        ("empty_specs_raises",                test_empty_specs_raises),
        ("n_parameters_correct",              test_n_parameters_correct),
        ("ewc_penalty_zero_at_init",          test_ewc_penalty_zero_at_init),
        ("ewc_penalty_sums_layers",           test_ewc_penalty_sums_layers),
        ("anchor_all_resets_penalty",         test_anchor_all_resets_penalty),
        ("estimate_fisher_resets_and_populates", test_estimate_fisher_resets_and_populates),
        ("state_dict_roundtrip",              test_state_dict_roundtrip),
        ("n_nodes_per_layer",                 test_n_nodes_per_layer),
        ("gradient_flows_through_full_stack", test_gradient_flows_through_full_stack),
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
