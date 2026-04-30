"""Self-contained tests for trioron.node.

Run with:    python test_node.py

No pytest dependency. Each test prints PASS/FAIL with a short message.
A nonzero exit code signals at least one failure.
"""

from __future__ import annotations
import sys
import traceback
import torch

from trioron.node import TrioronLayer


# --------------------------------------------------------------------------- #
# Test runner — minimal, prints PASS / FAIL per test.                         #
# --------------------------------------------------------------------------- #

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


def _approx_eq(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) < tol


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #


def test_forward_shape() -> None:
    layer = TrioronLayer(fan_in=4, n_nodes=3)
    x = torch.randn(2, 4)
    y = layer(x)
    assert y.shape == (2, 3), f"expected (2,3), got {tuple(y.shape)}"


def test_forward_relu_nonnegative() -> None:
    layer = TrioronLayer(fan_in=4, n_nodes=3, activation="relu")
    x = torch.randn(8, 4)
    y = layer(x)
    assert (y >= 0).all(), "ReLU output should be non-negative"


def test_initial_state_consistent() -> None:
    layer = TrioronLayer(fan_in=4, n_nodes=3)
    assert torch.allclose(layer.lam, torch.zeros(3))
    assert torch.allclose(layer.u, torch.zeros(3))
    assert torch.allclose(layer.W_anchor, layer.W.detach())
    assert torch.allclose(layer.fisher_W, torch.zeros_like(layer.W))


def test_fisher_accumulates_after_backward() -> None:
    torch.manual_seed(0)
    layer = TrioronLayer(fan_in=4, n_nodes=3, activation="linear")
    x = torch.randn(8, 4)
    target = torch.randn(8, 3)
    y = layer(x)
    loss = (y - target).pow(2).mean()
    loss.backward()
    assert layer.W.grad is not None, "no grad on W"
    layer.update_fisher()
    assert (layer.fisher_W.abs() > 0).any(), "Fisher info should be nonzero after step"


def test_update_lambda_from_fisher() -> None:
    layer = TrioronLayer(fan_in=4, n_nodes=3)
    layer.fisher_W.copy_(torch.tensor([
        [1.0, 1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0, 2.0],
        [0.5, 0.5, 0.5, 0.5],
    ]))
    layer.update_lambda()
    expected = torch.tensor([1.0, 2.0, 0.5])
    assert torch.allclose(layer.lam, expected), f"got {layer.lam.tolist()}"


def test_update_utility_decay_only() -> None:
    layer = TrioronLayer(fan_in=4, n_nodes=3, u_decay=0.5)
    layer.u.fill_(1.0)
    layer.update_utility(torch.zeros(3))
    # u_new = 0.5*1.0 + 0.5*0 = 0.5
    assert torch.allclose(layer.u, torch.full((3,), 0.5))


def test_update_utility_with_signal() -> None:
    layer = TrioronLayer(fan_in=4, n_nodes=3, u_decay=0.5)
    layer.u.fill_(0.0)
    layer.update_utility(torch.tensor([2.0, 0.0, -1.0]))
    # u_new = 0.5*0 + 0.5*[2,0,-1] = [1.0, 0.0, -0.5]
    assert torch.allclose(layer.u, torch.tensor([1.0, 0.0, -0.5]))


def test_update_utility_wrong_shape_raises() -> None:
    layer = TrioronLayer(fan_in=4, n_nodes=3)
    raised = False
    try:
        layer.update_utility(torch.zeros(2))
    except ValueError:
        raised = True
    assert raised, "expected ValueError on wrong-shape contributions"


def test_ewc_penalty_zero_at_anchor() -> None:
    layer = TrioronLayer(fan_in=4, n_nodes=3)
    # Even with high lambda, if W == W_anchor the penalty is zero.
    layer.lam.fill_(10.0)
    p = layer.ewc_penalty().item()
    assert _approx_eq(p, 0.0), f"penalty at anchor should be 0, got {p}"


def test_ewc_penalty_positive_after_drift() -> None:
    layer = TrioronLayer(fan_in=4, n_nodes=3)
    layer.lam.fill_(1.0)
    with torch.no_grad():
        layer.W.add_(torch.randn_like(layer.W) * 0.1)
    p = layer.ewc_penalty().item()
    assert p > 0.0, f"penalty after drift should be positive, got {p}"


def test_anchor_resets_penalty() -> None:
    layer = TrioronLayer(fan_in=4, n_nodes=3)
    layer.lam.fill_(1.0)
    with torch.no_grad():
        layer.W.add_(torch.randn_like(layer.W) * 0.1)
    assert layer.ewc_penalty().item() > 0
    layer.anchor_weights()
    p = layer.ewc_penalty().item()
    assert _approx_eq(p, 0.0), f"penalty after re-anchor should be 0, got {p}"


def test_ewc_penalty_has_grad() -> None:
    """The EWC penalty must be autograd-attached so it can contribute to .backward()."""
    layer = TrioronLayer(fan_in=4, n_nodes=3)
    layer.lam.fill_(1.0)
    with torch.no_grad():
        layer.W.add_(0.1)
    p = layer.ewc_penalty()
    assert p.requires_grad, "ewc_penalty must require grad"
    p.backward()
    assert layer.W.grad is not None
    assert (layer.W.grad.abs() > 0).any()


def test_grow_node_shapes() -> None:
    layer = TrioronLayer(fan_in=4, n_nodes=3)
    new_idx = layer.grow_node()
    assert new_idx == 3
    assert layer.n_nodes == 4
    for buf_name in ["lam", "u", "W_anchor", "b_anchor", "fisher_W", "fisher_b"]:
        buf = getattr(layer, buf_name)
        assert buf.shape[0] == 4, f"{buf_name} not resized"
    x = torch.randn(2, 4)
    y = layer(x)
    assert y.shape == (2, 4)


def test_grow_node_with_init_vec() -> None:
    layer = TrioronLayer(fan_in=4, n_nodes=3)
    init = torch.tensor([0.1, 0.2, 0.3, 0.4])
    layer.grow_node(init_vec=init)
    assert torch.allclose(layer.W.data[-1], init)


def test_grow_node_fully_plastic() -> None:
    """A newly grown node should start with lam=0 and u=0 (fully plastic, neutral utility)."""
    layer = TrioronLayer(fan_in=4, n_nodes=3)
    layer.lam.fill_(5.0)  # existing nodes are stiff
    layer.u.fill_(2.0)
    layer.grow_node()
    assert layer.lam[-1].item() == 0.0
    assert layer.u[-1].item() == 0.0


def test_prune_node_shapes() -> None:
    layer = TrioronLayer(fan_in=4, n_nodes=3)
    layer.prune_node(1)
    assert layer.n_nodes == 2
    x = torch.randn(2, 4)
    y = layer(x)
    assert y.shape == (2, 2)


def test_prune_node_removes_correct_row() -> None:
    layer = TrioronLayer(fan_in=4, n_nodes=3)
    # Mark each node distinctly.
    layer.u.copy_(torch.tensor([10.0, 20.0, 30.0]))
    layer.prune_node(1)
    assert torch.allclose(layer.u, torch.tensor([10.0, 30.0]))


def test_prune_last_node_raises() -> None:
    layer = TrioronLayer(fan_in=4, n_nodes=1)
    raised = False
    try:
        layer.prune_node(0)
    except ValueError:
        raised = True
    assert raised, "pruning the last node should raise"


def test_optimizer_works_after_grow() -> None:
    """After grow_node + optimizer rebuild, gradient descent should still work."""
    layer = TrioronLayer(fan_in=4, n_nodes=3, activation="linear")
    layer.grow_node()
    opt = torch.optim.SGD(layer.parameters(), lr=0.01)

    x = torch.randn(8, 4)
    target = torch.randn(8, 4)

    losses = []
    for _ in range(50):
        opt.zero_grad()
        loss = (layer(x) - target).pow(2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], (
        f"loss should decrease after grow+rebuild; "
        f"start={losses[0]:.4f} end={losses[-1]:.4f}"
    )


def test_state_dict_roundtrip() -> None:
    layer = TrioronLayer(fan_in=4, n_nodes=3)
    layer.fisher_W.fill_(0.7)
    layer.update_lambda()
    layer.u.copy_(torch.tensor([0.5, -0.2, 1.1]))
    sd = layer.state_dict()

    layer2 = TrioronLayer(fan_in=4, n_nodes=3)
    layer2.load_state_dict(sd)
    assert torch.allclose(layer2.fisher_W, layer.fisher_W)
    assert torch.allclose(layer2.lam, layer.lam)
    assert torch.allclose(layer2.u, layer.u)


def test_continual_learning_smoke() -> None:
    """End-to-end smoke test: train task A, anchor, train task B with EWC,
    verify task A is not catastrophically forgotten.

    This is a tiny version of the §8 step 2 verification — full version will
    live in a separate experiments/ script.
    """
    torch.manual_seed(42)

    # Task A and Task B: two different linear regressions on the same input space.
    x_a = torch.randn(64, 4)
    A = torch.randn(4, 3)
    y_a = x_a @ A

    x_b = torch.randn(64, 4)
    B = torch.randn(4, 3)
    y_b = x_b @ B

    layer = TrioronLayer(fan_in=4, n_nodes=3, activation="linear")

    def loss_on(x, y, ewc_strength=0.0):
        pred = layer(x)
        l = (pred - y).pow(2).mean()
        if ewc_strength > 0:
            l = l + ewc_strength * layer.ewc_penalty()
        return l

    # ---- Train on Task A ----
    opt = torch.optim.SGD(layer.parameters(), lr=0.05)
    for _ in range(300):
        opt.zero_grad()
        l = loss_on(x_a, y_a)
        l.backward()
        opt.step()

    # Fresh Fisher estimate at the converged weights, post-training.
    # (Kirkpatrick 2017 computes Fisher on the converged model, not as an
    # EMA across training — the EMA captures large early gradients that
    # don't reflect importance at the final solution.)
    with torch.no_grad():
        layer.fisher_W.zero_()
        layer.fisher_b.zero_()
    # Use a temporary high-decay so the buffer accumulates as a proper mean.
    saved_decay = layer.fisher_decay
    layer.fisher_decay = 0.5
    for _ in range(20):
        layer.W.grad = None
        layer.b.grad = None
        l = loss_on(x_a, y_a)
        l.backward()
        layer.update_fisher()
    layer.fisher_decay = saved_decay
    layer.update_lambda()
    layer.anchor_weights()

    loss_a_after_a = loss_on(x_a, y_a).item()

    # Snapshot the post-A state so both control and experimental start identically.
    snapshot = {k: v.clone() for k, v in layer.state_dict().items()}

    # ---- Train on Task B WITHOUT EWC (control) ----
    layer_no_ewc = TrioronLayer(fan_in=4, n_nodes=3, activation="linear")
    layer_no_ewc.load_state_dict(snapshot)
    layer_no_ewc.lam.zero_()  # disable EWC via lambda
    opt2 = torch.optim.SGD(layer_no_ewc.parameters(), lr=0.05)
    for _ in range(300):
        opt2.zero_grad()
        pred = layer_no_ewc(x_b)
        l = (pred - y_b).pow(2).mean()
        l.backward()
        opt2.step()
    loss_a_no_ewc = ((layer_no_ewc(x_a) - y_a).pow(2).mean()).item()

    # ---- Train on Task B WITH EWC (experimental) ----
    layer.load_state_dict(snapshot)
    opt3 = torch.optim.SGD(layer.parameters(), lr=0.05)
    ewc_strength = 5000.0  # strong enough to dominate small-Fisher regime
    for _ in range(300):
        opt3.zero_grad()
        pred = layer(x_b)
        l = (pred - y_b).pow(2).mean() + ewc_strength * layer.ewc_penalty()
        l.backward()
        opt3.step()
    loss_a_with_ewc = loss_on(x_a, y_a).item()

    print(
        f"    [continual] task-A loss after A:           {loss_a_after_a:.4f}"
    )
    print(
        f"    [continual] task-A loss after B, no EWC:   {loss_a_no_ewc:.4f}"
    )
    print(
        f"    [continual] task-A loss after B, with EWC: {loss_a_with_ewc:.4f}"
    )

    # The EWC variant should retain Task A better than the no-EWC control.
    assert loss_a_with_ewc < loss_a_no_ewc, (
        "EWC should reduce catastrophic forgetting; "
        f"with_ewc={loss_a_with_ewc:.4f} no_ewc={loss_a_no_ewc:.4f}"
    )


def test_grow_input_shapes():
    """grow_input adds an input column without changing n_nodes."""
    layer = TrioronLayer(fan_in=4, n_nodes=3, activation="relu")
    layer.grow_input()
    assert layer.fan_in == 5, f"fan_in {layer.fan_in}"
    assert layer.W.shape == (3, 5), f"W shape {tuple(layer.W.shape)}"
    assert layer.W_anchor.shape == (3, 5)
    assert layer.fisher_W.shape == (3, 5)
    # n_nodes / b / lam / u must be unchanged.
    assert layer.n_nodes == 3
    assert layer.b.shape == (3,)
    assert layer.lam.shape == (3,)
    assert layer.u.shape == (3,)


def test_grow_input_zero_default():
    """Default init is zeros — the new input contributes nothing initially."""
    layer = TrioronLayer(fan_in=4, n_nodes=3, activation="linear")
    x_old = torch.randn(2, 4)
    y_old = layer(x_old)
    layer.grow_input()
    x_new = torch.cat([x_old, torch.randn(2, 1)], dim=1)
    y_new = layer(x_new)
    # Output should be identical because the new column is zeros.
    assert torch.allclose(y_old, y_new), "default zero col must not change output"


def test_grow_input_with_init_col():
    """Custom init_col is placed in the last column of W."""
    layer = TrioronLayer(fan_in=4, n_nodes=3, activation="relu")
    init = torch.tensor([0.1, 0.2, 0.3])
    layer.grow_input(init_col=init)
    assert torch.allclose(layer.W[:, -1], init), f"new col {layer.W[:, -1]}"


def test_grow_input_then_grow_node_consistent():
    """grow_input then grow_node leaves everything self-consistent."""
    layer = TrioronLayer(fan_in=4, n_nodes=3, activation="relu")
    layer.grow_input()        # fan_in: 4 → 5
    layer.grow_node()         # n_nodes: 3 → 4
    assert layer.W.shape == (4, 5)
    assert layer.b.shape == (4,)
    assert layer.lam.shape == (4,)
    assert layer.u.shape == (4,)
    x = torch.randn(2, 5)
    y = layer(x)
    assert y.shape == (2, 4)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


def main() -> int:
    print("Running TrioronLayer tests...")
    print(f"  torch version: {torch.__version__}")

    tests = [
        ("forward_shape",                    test_forward_shape),
        ("forward_relu_nonnegative",         test_forward_relu_nonnegative),
        ("initial_state_consistent",         test_initial_state_consistent),
        ("fisher_accumulates",               test_fisher_accumulates_after_backward),
        ("update_lambda_from_fisher",        test_update_lambda_from_fisher),
        ("utility_decay_only",               test_update_utility_decay_only),
        ("utility_with_signal",              test_update_utility_with_signal),
        ("utility_wrong_shape_raises",       test_update_utility_wrong_shape_raises),
        ("ewc_zero_at_anchor",               test_ewc_penalty_zero_at_anchor),
        ("ewc_positive_after_drift",         test_ewc_penalty_positive_after_drift),
        ("anchor_resets_penalty",            test_anchor_resets_penalty),
        ("ewc_penalty_has_grad",             test_ewc_penalty_has_grad),
        ("grow_node_shapes",                 test_grow_node_shapes),
        ("grow_node_with_init_vec",          test_grow_node_with_init_vec),
        ("grow_node_fully_plastic",          test_grow_node_fully_plastic),
        ("grow_input_shapes",                test_grow_input_shapes),
        ("grow_input_zero_default",          test_grow_input_zero_default),
        ("grow_input_with_init_col",         test_grow_input_with_init_col),
        ("grow_input_then_grow_node",        test_grow_input_then_grow_node_consistent),
        ("prune_node_shapes",                test_prune_node_shapes),
        ("prune_node_removes_correct_row",   test_prune_node_removes_correct_row),
        ("prune_last_node_raises",           test_prune_last_node_raises),
        ("optimizer_works_after_grow",       test_optimizer_works_after_grow),
        ("state_dict_roundtrip",             test_state_dict_roundtrip),
        ("continual_learning_smoke",         test_continual_learning_smoke),
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
