"""Self-contained tests for trioron.network.

Run with:    python3 test_network.py
"""
from __future__ import annotations
import sys
import traceback
import torch
import torch.optim as optim

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


def test_populate_lambda_traditional_loop():
    """populate_lambda must lift λ off zero, set anchors, clear grads,
    and silence the silent-zero RuntimeWarning emitted by ewc_penalty.
    Models the Aidos "joint-training donor never called the consolidation
    cycle" path.
    """
    import warnings as _w
    from trioron.node import _EwcZeroWarning

    torch.manual_seed(0)
    net = TrioronNetwork([(4, 8, "relu"), (8, 3, "linear")])

    # Train naively — no update_fisher, no update_lambda, no anchor.
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    X = torch.randn(128, 4)
    Y = torch.randint(0, 3, (128,))
    for _ in range(20):
        idx = torch.randperm(128)[:32]
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(net(X[idx]), Y[idx])
        loss.backward()
        opt.step()

    # Precondition: λ is still all zero everywhere.
    for i, layer in enumerate(net.layers):
        assert (layer.lam == 0).all(), f"layer {i} lam should be all zero pre-populate"

    def make_batches():
        for _ in range(20):
            idx = torch.randperm(128)[:32]
            yield X[idx], Y[idx]

    net.populate_lambda(
        make_batches(),
        loss_fn=lambda p, y: torch.nn.functional.cross_entropy(p, y),
        n_batches=20,
        rescale_mean=True,
    )

    # Postcondition: λ populated, anchors moved to current W, grads cleared.
    for i, layer in enumerate(net.layers):
        assert (layer.lam > 0).any(), f"layer {i} lam still all zero post-populate"
        assert torch.allclose(layer.W_anchor, layer.W.detach()), (
            f"layer {i} W_anchor not snapped to current W"
        )
    for p in net.parameters():
        assert p.grad is None, "populate_lambda must clear stale gradients"

    # ewc_penalty must no longer trip the silent-zero warning.
    _EwcZeroWarning.reset()
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        pen = net.ewc_penalty()
    silent_zero_warns = [
        rec for rec in caught
        if issubclass(rec.category, RuntimeWarning)
        and "silently zero" in str(rec.message)
    ]
    assert len(silent_zero_warns) == 0, (
        "ewc_penalty must not warn after populate_lambda has run"
    )
    assert pen.item() >= 0.0  # Identity at the anchor, but autograd-attached.


def test_populate_lambda_rescale_mean_normalizes():
    """rescale_mean=True must drive each layer's lam.mean() to 1.0 (or
    leave it at zero if Fisher accumulated no signal in that layer)."""
    torch.manual_seed(1)
    net = TrioronNetwork([(4, 8, "relu"), (8, 3, "linear")])
    X = torch.randn(64, 4)
    Y = torch.randint(0, 3, (64,))

    def make_batches():
        for _ in range(10):
            idx = torch.randperm(64)[:16]
            yield X[idx], Y[idx]

    net.populate_lambda(
        make_batches(),
        loss_fn=lambda p, y: torch.nn.functional.cross_entropy(p, y),
        n_batches=10,
        rescale_mean=True,
    )
    for i, layer in enumerate(net.layers):
        if (layer.lam > 0).any():
            m = layer.lam.mean().item()
            assert abs(m - 1.0) < 1e-5, f"layer {i} lam.mean()={m}, expected 1.0"


def test_set_lambda_all_writes_each_layer():
    """set_lambda_all writes the per-layer signal into each layer's λ.
    Models the 'environment sense' channel — sensors / reward / attention
    feeding λ instead of (or alongside) Fisher.
    """
    net = TrioronNetwork([(4, 5, "relu"), (5, 3, "linear")])
    signals = [
        torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]),
        torch.tensor([2.0, 2.0, 2.0]),
    ]
    net.set_lambda_all(signals)
    assert torch.allclose(net.layers[0].lam, signals[0])
    assert torch.allclose(net.layers[1].lam, signals[1])


def test_set_lambda_all_additive_mode_layers_on_fisher():
    """Additive mode lets callers stack an external signal on top of a
    Fisher-derived λ — e.g. boost λ on cells flagged by a sensor without
    losing the cognitive-importance baseline."""
    net = TrioronNetwork([(4, 5, "relu"), (5, 3, "linear")])
    for layer in net.layers:
        layer.lam.fill_(1.0)
    signals = [torch.full((5,), 0.5), torch.full((3,), 2.0)]
    net.set_lambda_all(signals, mode="additive")
    assert torch.allclose(net.layers[0].lam, torch.full((5,), 1.5))
    assert torch.allclose(net.layers[1].lam, torch.full((3,), 3.0))


def test_set_lambda_all_wrong_length_raises():
    net = TrioronNetwork([(4, 5, "relu"), (5, 3, "linear")])
    try:
        net.set_lambda_all([torch.zeros(5)])  # only 1 signal for 2 layers
    except ValueError as e:
        assert "length 1" in str(e) and "n_layers 2" in str(e)
    else:
        raise AssertionError("expected ValueError for length mismatch")


def test_populate_lambda_no_rescale_preserves_magnitude():
    """rescale_mean=False must leave raw Fisher magnitudes intact (mean
    almost certainly != 1.0)."""
    torch.manual_seed(2)
    net = TrioronNetwork([(4, 8, "relu"), (8, 3, "linear")])
    X = torch.randn(64, 4)
    Y = torch.randint(0, 3, (64,))

    def make_batches():
        for _ in range(10):
            idx = torch.randperm(64)[:16]
            yield X[idx], Y[idx]

    net.populate_lambda(
        make_batches(),
        loss_fn=lambda p, y: torch.nn.functional.cross_entropy(p, y),
        n_batches=10,
        rescale_mean=False,
    )
    # With raw Fisher (cross-entropy, untrained net) the per-layer mean
    # will be small but nonzero — definitely not 1.0.
    any_layer_not_unit = any(
        (layer.lam > 0).any() and abs(layer.lam.mean().item() - 1.0) > 1e-3
        for layer in net.layers
    )
    assert any_layer_not_unit, (
        "with rescale_mean=False at least one layer's lam.mean() should "
        "differ from 1.0 (raw Fisher magnitudes are not pre-normalized)"
    )


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


def test_grow_layer_last_layer_no_cross_update():
    """Growing the last layer increases its output dim; nothing else changes."""
    net = TrioronNetwork([(4, 8, "relu"), (8, 2, "tanh")])
    pre = [layer.n_nodes for layer in net.layers]
    new_idx = net.grow_layer(layer_idx=1)
    assert new_idx == 2, f"new idx {new_idx}"
    assert net.layers[0].n_nodes == pre[0], "first layer should be untouched"
    assert net.layers[1].n_nodes == pre[1] + 1
    # Forward still works.
    x = torch.randn(3, 4)
    y = net(x)
    assert y.shape == (3, 3)


def test_grow_layer_middle_layer_propagates_to_next():
    """Growing layer i extends layer i+1's fan_in by 1."""
    net = TrioronNetwork(
        [(4, 8, "relu"), (8, 8, "relu"), (8, 2, "linear")]
    )
    fan_in_before = net.layers[2].fan_in
    n_nodes_before = [layer.n_nodes for layer in net.layers]
    net.grow_layer(layer_idx=1)
    # Middle layer's output grows.
    assert net.layers[1].n_nodes == n_nodes_before[1] + 1
    # Last layer's input grows to match.
    assert net.layers[2].fan_in == fan_in_before + 1
    # First layer untouched.
    assert net.layers[0].n_nodes == n_nodes_before[0]
    # End-to-end forward still works.
    x = torch.randn(3, 4)
    y = net(x)
    assert y.shape == (3, 2)


def test_grow_layer_zero_peer_init_preserves_output():
    """With peer_init_for_next defaulting to zeros, the network's output on
    inputs that don't excite the new node should be unchanged from before."""
    torch.manual_seed(0)
    net = TrioronNetwork(
        [(4, 8, "relu"), (8, 8, "relu"), (8, 2, "linear")]
    )
    x = torch.randn(5, 4)
    with torch.no_grad():
        y_before = net(x).clone()
    # Grow with init_vec = zeros so the new node also outputs zero
    # (relu(0)=0 for any zero pre-activation).
    init = torch.zeros(net.layers[1].fan_in)
    net.grow_layer(layer_idx=1, init_vec=init)
    with torch.no_grad():
        y_after = net(x)
    assert torch.allclose(y_before, y_after, atol=1e-6), (
        f"output diverged: max diff {(y_before - y_after).abs().max().item()}"
    )


def test_grow_layer_invalid_idx_raises():
    net = TrioronNetwork([(4, 8, "relu"), (8, 2, "linear")])
    try:
        net.grow_layer(layer_idx=5)
    except IndexError:
        return
    raise AssertionError("expected IndexError for out-of-range layer_idx")


def test_grow_layer_peer_init_shape_check():
    net = TrioronNetwork(
        [(4, 8, "relu"), (8, 8, "relu"), (8, 2, "linear")]
    )
    bad = torch.zeros(99)  # next layer has 2 nodes, not 99
    try:
        net.grow_layer(layer_idx=1, peer_init_for_next=bad)
    except ValueError:
        return
    raise AssertionError("expected ValueError on peer_init shape mismatch")


def test_prune_layer_node_last_layer():
    """Pruning the last layer reduces its output dim; no cross-layer cleanup."""
    net = TrioronNetwork([(4, 8, "relu"), (8, 4, "tanh")])
    net.prune_layer_node(layer_idx=1, node_idx=2, redistribute=True)
    assert net.layers[0].n_nodes == 8
    assert net.layers[1].n_nodes == 3
    x = torch.randn(2, 4)
    y = net(x)
    assert y.shape == (2, 3)


def test_prune_layer_node_middle_layer_drops_next_input_col():
    net = TrioronNetwork(
        [(4, 8, "relu"), (8, 6, "relu"), (6, 2, "linear")]
    )
    net.prune_layer_node(layer_idx=1, node_idx=3, redistribute=False)
    assert net.layers[1].n_nodes == 5
    assert net.layers[2].fan_in == 5
    x = torch.randn(2, 4)
    y = net(x)
    assert y.shape == (2, 2)


def test_prune_layer_node_redistributes_to_nearest_peer():
    """With redistribute=True, the pruned node's outgoing weights should
    be added to the cosine-similarity-nearest peer's column."""
    torch.manual_seed(0)
    net = TrioronNetwork(
        [(4, 4, "linear"), (4, 2, "linear")]
    )
    # Construct layer-0 weights so node 0 and node 2 are very similar.
    with torch.no_grad():
        net.layers[0].W.copy_(torch.tensor([
            [1.0, 0.0, 0.0, 0.0],     # node 0
            [0.0, 1.0, 0.0, 0.0],     # node 1
            [0.99, 0.01, 0.05, 0.05], # node 2 — close to node 0
            [0.0, 0.0, 0.0, 1.0],     # node 3
        ]))
        # Layer 1: distinct outgoing columns so we can spot the merge.
        net.layers[1].W.copy_(torch.tensor([
            [0.5, 1.0, 7.0, 0.5],   # row 0; col 2 = 7 (the victim's outgoing)
            [0.5, 1.0, 3.0, 0.5],   # row 1
        ]))
        col_0_before = net.layers[1].W[:, 0].clone()  # peer should absorb col 2
    # Prune node 2 of layer 0 — should redistribute its outgoing column to node 0.
    net.prune_layer_node(layer_idx=0, node_idx=2, redistribute=True)
    # After prune, node 0's outgoing col on layer 1 should be (col_0_before + col_2_before).
    # Layer 1 fan_in is now 3; col 0 corresponds to original node 0.
    expected_col_0 = col_0_before + torch.tensor([7.0, 3.0])
    assert torch.allclose(net.layers[1].W[:, 0], expected_col_0), (
        f"peer did not absorb victim's column. got {net.layers[1].W[:, 0]}, "
        f"expected {expected_col_0}"
    )
    # Layer 0 has 3 nodes left; layer 1 fan_in is 3.
    assert net.layers[0].n_nodes == 3
    assert net.layers[1].fan_in == 3


def test_prune_layer_node_refuses_last_node():
    net = TrioronNetwork([(4, 1, "relu"), (1, 2, "linear")])
    try:
        net.prune_layer_node(layer_idx=0, node_idx=0, redistribute=True)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_prune_layer_node_invalid_idx_raises():
    net = TrioronNetwork([(4, 4, "relu"), (4, 2, "linear")])
    try:
        net.prune_layer_node(layer_idx=0, node_idx=99)
    except IndexError:
        return
    raise AssertionError("expected IndexError on bad node_idx")


def test_grow_layer_optimizer_can_be_rebuilt():
    """After grow_layer, a fresh Adam over net.parameters() takes a real step."""
    import torch.optim as optim
    net = TrioronNetwork([(4, 8, "relu"), (8, 2, "tanh")])
    net.grow_layer(layer_idx=1)
    opt = optim.Adam(net.parameters(), lr=1e-2)
    x = torch.randn(8, 4)
    y_target = torch.randn(8, 3)
    pred = net(x)
    loss = (pred - y_target).pow(2).mean()
    opt.zero_grad()
    loss.backward()
    # Snapshot a parameter, step, confirm it changed.
    w0 = net.layers[1].W.detach().clone()
    opt.step()
    w1 = net.layers[1].W.detach()
    assert not torch.allclose(w0, w1), "optimizer step should have moved W"


# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Mixed precision (FP16 weights / FP32 buffers)                               #
# --------------------------------------------------------------------------- #


def _mp_net():
    return TrioronNetwork([(4, 4, "relu"), (4, 4, "relu"), (4, 2, "tanh")])


def test_to_mixed_precision_converts_weights_only():
    net = _mp_net()
    net.to_mixed_precision(weights_dtype=torch.float16)
    for layer in net.layers:
        assert layer.W.dtype == torch.float16
        assert layer.b.dtype == torch.float16
        # Buffers stay FP32 for the consolidation math.
        assert layer.W_anchor.dtype == torch.float32
        assert layer.b_anchor.dtype == torch.float32
        assert layer.fisher_W.dtype == torch.float32
        assert layer.fisher_b.dtype == torch.float32
        assert layer.lam.dtype == torch.float32
        assert layer.routing_scale.dtype == torch.float32
        assert layer.apoptosis_pulse.dtype == torch.float32


def test_mixed_precision_forward_runs():
    torch.manual_seed(0)
    net = _mp_net()
    net.to_mixed_precision(torch.float16)
    x = torch.randn(8, 4, dtype=torch.float16)
    y = net(x)
    assert y.dtype == torch.float16
    assert y.shape == (8, 2)


def test_mixed_precision_ewc_penalty_matches_fp32_within_tolerance():
    """EWC penalty value should agree between FP16-weights and FP32-weights
    networks for the same drift, up to FP16 precision."""
    torch.manual_seed(0)
    net_fp32 = _mp_net()
    # Drift weights so penalty is non-trivial.
    with torch.no_grad():
        for layer in net_fp32.layers:
            layer.W.data += 0.1
            layer.lam.fill_(1.0)
    pen_fp32 = float(net_fp32.ewc_penalty().item())

    torch.manual_seed(0)
    net_fp16 = _mp_net()
    with torch.no_grad():
        for layer in net_fp16.layers:
            layer.W.data += 0.1
            layer.lam.fill_(1.0)
    net_fp16.to_mixed_precision(torch.float16)
    pen_fp16 = float(net_fp16.ewc_penalty().item())

    rel = abs(pen_fp16 - pen_fp32) / max(pen_fp32, 1e-8)
    assert rel < 1e-2, (
        f"FP16 EWC penalty too far from FP32: fp32={pen_fp32:.6f} "
        f"fp16={pen_fp16:.6f} rel={rel:.4f}"
    )


def test_mixed_precision_backward_and_fisher_accumulate():
    """Train one step at FP16; Fisher (FP32) gets non-zero accumulation."""
    torch.manual_seed(0)
    net = _mp_net()
    net.to_mixed_precision(torch.float16)
    x = torch.randn(8, 4, dtype=torch.float16)
    y_target = torch.randn(8, 2, dtype=torch.float16)

    out = net(x)
    loss = (out - y_target).pow(2).mean()
    loss.backward()
    # Grad on W is FP16; Fisher is FP32 — accumulation should upcast cleanly.
    for layer in net.layers:
        layer.update_fisher()
        assert layer.fisher_W.dtype == torch.float32
        assert (layer.fisher_W > 0).any(), (
            "Fisher should have positive entries after a non-trivial step"
        )


def test_mixed_precision_grow_layer_preserves_dtypes():
    torch.manual_seed(0)
    net = _mp_net()
    net.to_mixed_precision(torch.float16)
    new_idx = net.grow_layer(layer_idx=1, task_idx=2)
    assert new_idx == net.layers[1].n_nodes - 1
    layer = net.layers[1]
    assert layer.W.dtype == torch.float16
    assert layer.b.dtype == torch.float16
    assert layer.W_anchor.dtype == torch.float32
    assert layer.fisher_W.dtype == torch.float32
    assert layer.routing_scale.dtype == torch.float32
    assert layer.apoptosis_pulse.dtype == torch.float32
    # Forward still works after growth.
    x = torch.randn(4, 4, dtype=torch.float16)
    out = net(x)
    assert out.dtype == torch.float16


def test_mixed_precision_anchor_keeps_fp32_after_anchor_all():
    """After anchor_all(), W_anchor must still be FP32 even though W is FP16."""
    torch.manual_seed(0)
    net = _mp_net()
    net.to_mixed_precision(torch.float16)
    with torch.no_grad():
        for layer in net.layers:
            layer.W.data += 0.05
    net.anchor_all()
    for layer in net.layers:
        assert layer.W_anchor.dtype == torch.float32, (
            f"anchor_all replaced FP32 anchor with FP16 — got {layer.W_anchor.dtype}"
        )


def test_mixed_precision_forward_accepts_fp32_input():
    """Mixed-precision net should auto-cast FP32 inputs to its weight
    dtype so callers don't need to know the precision."""
    torch.manual_seed(0)
    net = _mp_net()
    net.to_mixed_precision(torch.bfloat16)
    x_fp32 = torch.randn(8, 4, dtype=torch.float32)
    out = net(x_fp32)
    assert out.dtype == torch.bfloat16
    assert out.shape == (8, 2)


def test_mixed_precision_bfloat16_path():
    torch.manual_seed(0)
    net = _mp_net()
    net.to_mixed_precision(torch.bfloat16)
    for layer in net.layers:
        assert layer.W.dtype == torch.bfloat16
        assert layer.W_anchor.dtype == torch.float32
    x = torch.randn(4, 4, dtype=torch.float32)
    y_target = torch.randn(4, 2, dtype=torch.float32)
    out = net(x)
    loss = (out.float() - y_target).pow(2).mean()
    loss.backward()
    for layer in net.layers:
        layer.update_fisher()
        assert (layer.fisher_W > 0).any()


def test_mixed_precision_optimizer_rebuild_works():
    torch.manual_seed(0)
    net = _mp_net()
    net.to_mixed_precision(torch.float16)
    opt = optim.Adam(net.parameters(), lr=1e-3)
    x = torch.randn(8, 4, dtype=torch.float16)
    y = torch.randn(8, 2, dtype=torch.float16)
    out = net(x)
    loss = (out - y).pow(2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    # Weights moved.
    for layer in net.layers:
        assert layer.W.dtype == torch.float16


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
        ("populate_lambda_traditional_loop",     test_populate_lambda_traditional_loop),
        ("populate_lambda_rescale_mean",         test_populate_lambda_rescale_mean_normalizes),
        ("set_lambda_all_writes_each_layer",     test_set_lambda_all_writes_each_layer),
        ("set_lambda_all_additive_mode",         test_set_lambda_all_additive_mode_layers_on_fisher),
        ("set_lambda_all_wrong_length",          test_set_lambda_all_wrong_length_raises),
        ("populate_lambda_no_rescale",           test_populate_lambda_no_rescale_preserves_magnitude),
        ("state_dict_roundtrip",              test_state_dict_roundtrip),
        ("n_nodes_per_layer",                 test_n_nodes_per_layer),
        ("gradient_flows_through_full_stack", test_gradient_flows_through_full_stack),
        ("grow_layer_last_layer",             test_grow_layer_last_layer_no_cross_update),
        ("grow_layer_middle_layer",           test_grow_layer_middle_layer_propagates_to_next),
        ("grow_layer_zero_peer_preserves",    test_grow_layer_zero_peer_init_preserves_output),
        ("grow_layer_invalid_idx_raises",     test_grow_layer_invalid_idx_raises),
        ("grow_layer_peer_init_shape_check",  test_grow_layer_peer_init_shape_check),
        ("grow_layer_optimizer_rebuild",      test_grow_layer_optimizer_can_be_rebuilt),
        ("prune_layer_node_last_layer",       test_prune_layer_node_last_layer),
        ("prune_layer_node_middle_layer",     test_prune_layer_node_middle_layer_drops_next_input_col),
        ("prune_layer_node_redistribute",     test_prune_layer_node_redistributes_to_nearest_peer),
        ("prune_layer_node_refuses_last",     test_prune_layer_node_refuses_last_node),
        ("prune_layer_node_invalid_idx",      test_prune_layer_node_invalid_idx_raises),
        ("mp_to_mixed_precision_converts_weights_only",
                                              test_to_mixed_precision_converts_weights_only),
        ("mp_forward_runs",                   test_mixed_precision_forward_runs),
        ("mp_ewc_penalty_matches_fp32",
                                              test_mixed_precision_ewc_penalty_matches_fp32_within_tolerance),
        ("mp_backward_and_fisher_accumulate",
                                              test_mixed_precision_backward_and_fisher_accumulate),
        ("mp_grow_layer_preserves_dtypes",
                                              test_mixed_precision_grow_layer_preserves_dtypes),
        ("mp_anchor_keeps_fp32_after_anchor_all",
                                              test_mixed_precision_anchor_keeps_fp32_after_anchor_all),
        ("mp_forward_accepts_fp32_input",
                                              test_mixed_precision_forward_accepts_fp32_input),
        ("mp_bfloat16_path",                  test_mixed_precision_bfloat16_path),
        ("mp_optimizer_rebuild_works",
                                              test_mixed_precision_optimizer_rebuild_works),
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
