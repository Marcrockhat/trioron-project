"""Core substrate smoke tests — construct, forward, backward, gradient flow."""
from __future__ import annotations

import torch

from trioron.core import Envelope, Arena, construct
from trioron.bases import minimal
from trioron.phenotype import default_dispatch_table


def _make_substrate(input_dim: int = 4, n_classes: int = 3, capacity: int = 128):
    return construct(
        base=minimal(input_dim, n_classes),
        envelope=Envelope(),
        dispatch_table=default_dispatch_table(),
        capacity=capacity,
    )


def test_construct_counts():
    sub = _make_substrate(input_dim=4, n_classes=3)
    assert sub.n_cells == 7  # 4 perception + 3 output
    assert sub.n_edges == 12  # 4 inputs * 3 outputs


def test_forward_shape():
    sub = _make_substrate(input_dim=4, n_classes=3)
    sub.prepare_training()
    x = torch.randn(2, 4)
    logits = sub(x)
    assert logits.shape == (2, 3)


def test_backward_gradient_flow():
    sub = _make_substrate(input_dim=4, n_classes=3)
    sub.prepare_training()
    x = torch.randn(2, 4)
    logits = sub(x)
    loss = logits.sum()
    loss.backward()

    assert sub.arena.bias.grad is not None
    assert sub.arena.edge_weight.grad is not None

    out_ids = sub.scheduler._plan.output_ids
    assert (sub.arena.bias.grad[out_ids.long()].abs() > 0).all()


def test_output_depends_on_input():
    sub = _make_substrate(input_dim=4, n_classes=3)
    sub.prepare_training()

    x1 = torch.ones(1, 4)
    x2 = torch.ones(1, 4) * 2.0
    y1 = sub(x1)
    y2 = sub(x2)
    assert not torch.allclose(y1, y2), "Output should differ for different inputs"


def test_training_step_updates_weights():
    sub = _make_substrate(input_dim=4, n_classes=3)
    sub.prepare_training()
    opt = torch.optim.SGD(sub.trainable_tensors(), lr=0.1)

    bias_before = sub.arena.bias.clone()

    x = torch.randn(8, 4)
    target = torch.randint(0, 3, (8,))

    logits = sub(x)
    loss = torch.nn.functional.cross_entropy(logits, target)
    loss.backward()
    opt.step()
    opt.zero_grad()

    bias_after = sub.arena.bias
    out_ids = sub.scheduler._plan.output_ids.long()
    assert not torch.allclose(bias_before[out_ids], bias_after[out_ids]), \
        "Output biases should change after an optimizer step"


def test_dormant_grad_mask():
    sub = _make_substrate(input_dim=4, n_classes=3)
    sub.prepare_training()

    from trioron.core.state import CellState
    out_ids = sub.scheduler._plan.output_ids
    dormant_id = out_ids[0].item()
    sub.arena.state[dormant_id] = CellState.DORMANT

    x = torch.randn(2, 4)
    logits = sub(x)
    loss = logits.sum()
    loss.backward()
    sub.zero_dormant_grads()

    assert sub.arena.bias.grad[dormant_id].item() == 0.0


def test_pressure_uncapped():
    sub = _make_substrate()
    assert sub.pressure == 0.0
