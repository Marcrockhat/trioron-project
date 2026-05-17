"""Self-contained tests for trioron.node.

Run with:    python test_node.py

No pytest dependency. Each test prints PASS/FAIL with a short message.
A nonzero exit code signals at least one failure.
"""

from __future__ import annotations
import sys
import traceback
import torch

from trioron.node import TrioronLayer, _EwcZeroWarning


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
    # update_lambda: λ = sum(fisher_W, dim=1). Switched from mean to sum
    # 2026-05-03 after the chained-15 Fisher probe showed the mean
    # collapse drowned typical fisher_W magnitudes (~1e-3) below any
    # usable λ_floor across realistic fan_in (32–128).
    layer = TrioronLayer(fan_in=4, n_nodes=3)
    layer.fisher_W.copy_(torch.tensor([
        [1.0, 1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0, 2.0],
        [0.5, 0.5, 0.5, 0.5],
    ]))
    layer.update_lambda()
    expected = torch.tensor([4.0, 8.0, 2.0])
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


def test_saliency_zero_before_any_forward() -> None:
    """Right after construction, saliency_utility returns zeros."""
    layer = TrioronLayer(fan_in=4, n_nodes=3)
    sal = layer.saliency_utility()
    assert sal.shape == (3,)
    assert torch.allclose(sal, torch.zeros(3))


def test_saliency_dead_relu_node_zero() -> None:
    """A dead-relu node (output always 0) should score 0 saliency, even
    if its incoming weights are large. This is the bug the |W|·|grad_W|
    heuristic missed: large W on a dead node was scored "important"."""
    torch.manual_seed(0)
    layer = TrioronLayer(fan_in=4, n_nodes=3, activation="relu")
    # Force node 1 to be permanently dead: large negative bias overpowers
    # any input. Other nodes left at random init (will have some activity).
    with torch.no_grad():
        layer.W[1].fill_(0.5)   # large incoming weights
        layer.b[1].fill_(-100.0)  # but always dead
    x = torch.randn(8, 4)
    y = layer(x)
    # Sanity: node 1 outputs are all zero.
    assert torch.allclose(y[:, 1], torch.zeros(8))
    loss = y.sum()
    loss.backward()
    sal = layer.saliency_utility()
    # Dead node has zero saliency.
    assert sal[1].item() == 0.0
    # At least one other node has non-zero saliency (otherwise the test
    # is vacuous).
    assert sal[0].item() > 0 or sal[2].item() > 0


def test_saliency_active_node_positive() -> None:
    """A node with non-zero output on at least some inputs and a non-zero
    upstream gradient has strictly-positive saliency averaged over the
    batch."""
    torch.manual_seed(0)
    layer = TrioronLayer(fan_in=4, n_nodes=2, activation="relu")
    # Bias positive so most inputs activate (some still hit dead-relu
    # for adversarial inputs — that's fine, average will still be > 0).
    with torch.no_grad():
        layer.b.fill_(1.0)
    x = torch.randn(8, 4)
    y = layer(x)
    # Sanity: at least some outputs are non-zero per node.
    assert (y > 0).any(dim=0).all(), (
        "test setup invalid — both nodes are dead on this input batch"
    )
    loss = y.sum()
    loss.backward()
    sal = layer.saliency_utility()
    assert sal.shape == (2,)
    assert (sal > 0).all(), f"expected positive saliency, got {sal.tolist()}"


def test_saliency_no_grad_forward_preserves_state() -> None:
    """A no_grad forward (e.g. eval) must NOT overwrite the cached
    saliency from a prior training forward. Otherwise running eval
    between train step and saliency read would zero everything."""
    torch.manual_seed(0)
    layer = TrioronLayer(fan_in=4, n_nodes=3, activation="relu")
    with torch.no_grad():
        layer.b.fill_(1.0)
    # Training forward + backward → populates _last_y / _last_upstream.
    x = torch.randn(8, 4)
    y = layer(x)
    loss = y.sum()
    loss.backward()
    sal_before = layer.saliency_utility().clone()
    # Eval-mode forward must not corrupt those buffers.
    with torch.no_grad():
        _ = layer(torch.randn(8, 4))
    sal_after = layer.saliency_utility()
    assert torch.allclose(sal_before, sal_after), (
        f"no_grad forward overwrote saliency cache: "
        f"before {sal_before.tolist()} after {sal_after.tolist()}"
    )


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


def test_ewc_penalty_warns_on_all_zero_lambda() -> None:
    """ewc_penalty() must emit a one-shot RuntimeWarning when λ is all zero
    (the silent-MLP failure mode: consolidation cycle skipped, β is moot).
    Second call within the same process must NOT re-warn.
    """
    import warnings as _w
    _EwcZeroWarning.reset()
    layer = TrioronLayer(fan_in=4, n_nodes=3)
    # Default lambda_init=0.0, so this layer starts with all-zero λ.
    assert (layer.lam == 0).all(), "precondition: fresh layer has λ == 0"

    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        p1 = layer.ewc_penalty()
    assert p1.item() == 0.0, f"penalty must be exactly zero, got {p1.item()}"
    silent_zero_warns = [
        rec for rec in caught
        if issubclass(rec.category, RuntimeWarning)
        and "silently zero" in str(rec.message)
    ]
    assert len(silent_zero_warns) == 1, (
        f"expected exactly one silent-zero RuntimeWarning, got "
        f"{len(silent_zero_warns)}: {[str(r.message) for r in caught]}"
    )

    # One-shot: a second call must not re-emit.
    with _w.catch_warnings(record=True) as caught2:
        _w.simplefilter("always")
        layer.ewc_penalty()
    repeat_warns = [
        rec for rec in caught2
        if issubclass(rec.category, RuntimeWarning)
        and "silently zero" in str(rec.message)
    ]
    assert len(repeat_warns) == 0, (
        f"warning must be one-shot; got {len(repeat_warns)} on second call"
    )


def test_ewc_penalty_no_warn_when_lambda_populated() -> None:
    """ewc_penalty() must NOT warn once λ has been populated (the normal
    post-consolidation case).
    """
    import warnings as _w
    _EwcZeroWarning.reset()
    layer = TrioronLayer(fan_in=4, n_nodes=3)
    layer.lam.fill_(1.0)
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        layer.ewc_penalty()
    silent_zero_warns = [
        rec for rec in caught
        if issubclass(rec.category, RuntimeWarning)
        and "silently zero" in str(rec.message)
    ]
    assert len(silent_zero_warns) == 0, (
        f"no warning expected with populated λ; got "
        f"{[str(r.message) for r in silent_zero_warns]}"
    )


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


def test_prune_input_shapes():
    layer = TrioronLayer(fan_in=5, n_nodes=3, activation="relu")
    layer.prune_input(col_idx=2)
    assert layer.fan_in == 4
    assert layer.W.shape == (3, 4)
    assert layer.W_anchor.shape == (3, 4)
    assert layer.fisher_W.shape == (3, 4)
    # Per-output-node buffers untouched.
    assert layer.b.shape == (3,)
    assert layer.lam.shape == (3,)
    assert layer.u.shape == (3,)


def test_prune_input_removes_correct_column():
    layer = TrioronLayer(fan_in=4, n_nodes=2, activation="linear")
    with torch.no_grad():
        layer.W.copy_(torch.tensor([[1.0, 2.0, 3.0, 4.0],
                                    [5.0, 6.0, 7.0, 8.0]]))
    layer.prune_input(col_idx=1)
    expected = torch.tensor([[1.0, 3.0, 4.0], [5.0, 7.0, 8.0]])
    assert torch.allclose(layer.W, expected), f"got {layer.W}"


def test_prune_input_invalid_idx_raises():
    layer = TrioronLayer(fan_in=4, n_nodes=2, activation="relu")
    try:
        layer.prune_input(col_idx=99)
    except IndexError:
        return
    raise AssertionError("expected IndexError")


def test_prune_input_last_column_raises():
    layer = TrioronLayer(fan_in=1, n_nodes=2, activation="relu")
    try:
        layer.prune_input(col_idx=0)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


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
        ("saliency_zero_before_forward",     test_saliency_zero_before_any_forward),
        ("saliency_dead_relu_zero",          test_saliency_dead_relu_node_zero),
        ("saliency_active_positive",         test_saliency_active_node_positive),
        ("saliency_no_grad_preserves",       test_saliency_no_grad_forward_preserves_state),
        ("ewc_zero_at_anchor",               test_ewc_penalty_zero_at_anchor),
        ("ewc_positive_after_drift",         test_ewc_penalty_positive_after_drift),
        ("anchor_resets_penalty",            test_anchor_resets_penalty),
        ("ewc_penalty_has_grad",             test_ewc_penalty_has_grad),
        ("ewc_warns_on_all_zero_lambda",     test_ewc_penalty_warns_on_all_zero_lambda),
        ("ewc_no_warn_with_lambda",          test_ewc_penalty_no_warn_when_lambda_populated),
        ("grow_node_shapes",                 test_grow_node_shapes),
        ("grow_node_with_init_vec",          test_grow_node_with_init_vec),
        ("grow_node_fully_plastic",          test_grow_node_fully_plastic),
        ("grow_input_shapes",                test_grow_input_shapes),
        ("grow_input_zero_default",          test_grow_input_zero_default),
        ("grow_input_with_init_col",         test_grow_input_with_init_col),
        ("grow_input_then_grow_node",        test_grow_input_then_grow_node_consistent),
        ("prune_input_shapes",               test_prune_input_shapes),
        ("prune_input_removes_correct_col",  test_prune_input_removes_correct_column),
        ("prune_input_invalid_idx_raises",   test_prune_input_invalid_idx_raises),
        ("prune_input_last_column_raises",   test_prune_input_last_column_raises),
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
