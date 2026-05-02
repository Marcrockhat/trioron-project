"""Self-contained tests for trioron.classification.

Run with:    python3 test_classification.py
"""
from __future__ import annotations
import math
import sys
import traceback

import torch
import torch.optim as optim

from trioron.network import TrioronNetwork
from trioron.classification import (
    SplitClassificationTask,
    accuracy,
    extend_output_head,
    masked_cross_entropy,
    predict_full,
    split_cifar100_tasks,
    split_mnist_tasks,
    summarize,
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


def _classifier(fan_in=8, hidden=16, n_classes=2):
    """Linear-headed network — last layer is 'linear' so its outputs are logits."""
    return TrioronNetwork(
        [(fan_in, hidden, "relu"), (hidden, hidden, "relu"), (hidden, n_classes, "linear")]
    )


# --------------------------------------------------------------------------- #
# Task curricula
# --------------------------------------------------------------------------- #


def test_split_mnist_tasks_shape():
    tasks = split_mnist_tasks()
    assert len(tasks) == 5
    assert tasks[0].classes == [0, 1]
    assert tasks[2].classes == [4, 5]
    assert tasks[-1].classes == [8, 9]
    seen = sorted(c for t in tasks for c in t.classes)
    assert seen == list(range(10)), f"got {seen}"


def test_split_cifar100_default_shape():
    tasks = split_cifar100_tasks()
    assert len(tasks) == 10
    assert tasks[0].classes == list(range(10))
    assert tasks[-1].classes == list(range(90, 100))
    seen = sorted(c for t in tasks for c in t.classes)
    assert seen == list(range(100))


def test_split_cifar100_custom_shape():
    tasks = split_cifar100_tasks(n_tasks=5, classes_per_task=20)
    assert len(tasks) == 5
    assert all(len(t.classes) == 20 for t in tasks)
    assert tasks[2].classes == list(range(40, 60))


def test_split_cifar100_overflow_raises():
    raised = False
    try:
        split_cifar100_tasks(n_tasks=11, classes_per_task=10)
    except ValueError:
        raised = True
    assert raised


# --------------------------------------------------------------------------- #
# extend_output_head
# --------------------------------------------------------------------------- #


def test_extend_output_head_grows_last_layer():
    net = _classifier(n_classes=2)
    assert net.layers[-1].n_nodes == 2
    new_idx = extend_output_head(net, 2)
    assert new_idx == [2, 3]
    assert net.layers[-1].n_nodes == 4
    # Forward still works and produces 4-wide logits.
    x = torch.randn(3, 8)
    out = net(x)
    assert out.shape == (3, 4), f"got {tuple(out.shape)}"


def test_extend_output_head_preserves_existing_logits():
    """Adding new output nodes must not perturb the first 2 logits' values."""
    net = _classifier(n_classes=2)
    x = torch.randn(5, 8)
    with torch.no_grad():
        before = net(x).detach().clone()
    extend_output_head(net, 3)
    with torch.no_grad():
        after = net(x).detach()
    diff = (before - after[:, :2]).abs().max().item()
    assert diff < 1e-6, f"existing logits drifted after extend: max abs diff = {diff}"


def test_extend_output_head_optimizer_rebuild_works():
    """After grow, the new W is a fresh nn.Parameter — old optimizer state
    is stale. Ensure a freshly built optimizer can step on the new params."""
    net = _classifier(n_classes=2)
    extend_output_head(net, 2)
    opt = optim.Adam(net.parameters(), lr=1e-2)
    x = torch.randn(4, 8)
    y = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    logits = net(x)
    loss = masked_cross_entropy(logits, y, active_classes=[0, 1, 2, 3])
    opt.zero_grad()
    loss.backward()
    # Every parameter has a gradient — including the new head rows.
    for p in net.parameters():
        assert p.grad is not None
    opt.step()  # must not raise


def test_extend_output_head_invalid_args():
    net = _classifier(n_classes=2)
    raised = False
    try:
        extend_output_head(net, 0)
    except ValueError:
        raised = True
    assert raised


# --------------------------------------------------------------------------- #
# masked_cross_entropy
# --------------------------------------------------------------------------- #


def test_masked_cross_entropy_matches_explicit_subset_ce():
    """Slicing logits + remapping labels in the helper should equal computing
    cross_entropy over the same slice manually."""
    torch.manual_seed(0)
    logits = torch.randn(10, 6)  # 6-wide head
    labels = torch.tensor([2, 4, 4, 2, 2, 4, 2, 4, 4, 2], dtype=torch.long)
    active = [2, 4]

    got = masked_cross_entropy(logits, labels, active_classes=active)

    # Manual reference: slice and remap.
    sub = logits[:, active]
    local = torch.where(labels == 2, torch.zeros_like(labels), torch.ones_like(labels))
    expected = torch.nn.functional.cross_entropy(sub, local)
    assert math.isclose(got.item(), expected.item(), rel_tol=1e-6, abs_tol=1e-6), \
        f"got {got.item()}, expected {expected.item()}"


def test_masked_cross_entropy_ignores_inactive_logits():
    """Changing logits OUTSIDE the active subset must not change the loss
    value — that's the whole point of the active-class mask."""
    torch.manual_seed(1)
    logits = torch.randn(8, 6)
    labels = torch.full((8,), 3, dtype=torch.long)
    active = [3, 5]
    base = masked_cross_entropy(logits, labels, active_classes=active).item()

    # Perturb columns 0, 1, 2, 4 — none of them are active.
    perturbed = logits.clone()
    perturbed[:, [0, 1, 2, 4]] += 100.0
    got = masked_cross_entropy(perturbed, labels, active_classes=active).item()
    assert math.isclose(got, base, rel_tol=1e-6, abs_tol=1e-6), \
        f"loss changed when inactive logits changed: {base} → {got}"


def test_masked_cross_entropy_label_outside_active_raises():
    logits = torch.randn(4, 6)
    labels = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    raised = False
    try:
        masked_cross_entropy(logits, labels, active_classes=[2, 3])
    except ValueError:
        raised = True
    assert raised, "expected ValueError when labels include classes outside active_classes"


def test_masked_cross_entropy_active_class_oob_raises():
    """active_classes referencing an output index that doesn't exist on the
    head should raise — this would silently mask the wrong column otherwise."""
    logits = torch.randn(2, 4)
    labels = torch.tensor([0, 1], dtype=torch.long)
    raised = False
    try:
        masked_cross_entropy(logits, labels, active_classes=[0, 5])
    except ValueError:
        raised = True
    assert raised


def test_masked_cross_entropy_backprop_only_through_active_columns():
    """Gradients on inactive columns should be zero — the loss didn't see them."""
    torch.manual_seed(2)
    net = _classifier(n_classes=4)
    x = torch.randn(6, 8)
    labels = torch.tensor([0, 1, 0, 1, 1, 0], dtype=torch.long)
    logits = net(x)
    loss = masked_cross_entropy(logits, labels, active_classes=[0, 1])
    loss.backward()
    head = net.layers[-1]
    # Rows 2 and 3 of the head's W correspond to inactive classes.
    inactive_W_grad = head.W.grad[2:4]
    assert inactive_W_grad.abs().max().item() < 1e-9, \
        f"inactive head rows received grad: {inactive_W_grad.abs().max().item()}"
    inactive_b_grad = head.b.grad[2:4]
    assert inactive_b_grad.abs().max().item() < 1e-9


# --------------------------------------------------------------------------- #
# predict_full / accuracy
# --------------------------------------------------------------------------- #


def test_predict_full_argmax_over_all_classes():
    logits = torch.tensor([
        [0.0, 0.0, 5.0, 0.0],
        [3.0, 0.0, 0.0, 0.0],
        [0.0, 9.0, 0.0, 0.0],
    ])
    preds = predict_full(logits)
    assert preds.tolist() == [2, 0, 1]


def test_accuracy_full_softmax_admits_other_task_mistakes():
    """The standard CL metric is full-softmax accuracy. A logit on a
    'wrong-task' class can win the argmax — we measure that."""
    logits = torch.tensor([
        [0.0, 0.0, 100.0, 0.0],   # arg = 2; wrong
        [0.0, 100.0, 0.0, 0.0],   # arg = 1; correct
    ])
    labels = torch.tensor([0, 1], dtype=torch.long)
    a = accuracy(logits, labels)
    assert math.isclose(a, 0.5)


def test_accuracy_restrict_to_overrides_full_softmax():
    """Restricting argmax to the current task's classes can recover
    accuracy that the full-softmax metric would miss — useful for
    task-aware-inference diagnostics."""
    logits = torch.tensor([
        [0.0, 0.0, 100.0, 0.0],   # full argmax = 2 (wrong); restricted to {0,1} → 0
        [0.0, 100.0, 0.0, 0.0],
    ])
    labels = torch.tensor([0, 1], dtype=torch.long)
    a_full = accuracy(logits, labels)
    a_restricted = accuracy(logits, labels, restrict_to=[0, 1])
    assert math.isclose(a_full, 0.5)
    assert math.isclose(a_restricted, 1.0)


# --------------------------------------------------------------------------- #
# summarize
# --------------------------------------------------------------------------- #


def test_summarize_final_accuracy_and_forgetting():
    """3 tasks, scripted matrix to make the metrics check explicit."""
    M = [
        [1.0, float("nan"), float("nan")],
        [0.6, 0.9, float("nan")],
        [0.4, 0.7, 0.8],
    ]
    rep = summarize(M, ["t0", "t1", "t2"])
    assert math.isclose(rep.final_accuracy, (0.4 + 0.7 + 0.8) / 3)
    # forget per task: t0: 1.0 - 0.4 = 0.6; t1: 0.9 - 0.7 = 0.2; t2 not counted.
    assert math.isclose(rep.avg_forgetting, (0.6 + 0.2) / 2)
    assert rep.per_task_final() == [0.4, 0.7, 0.8]


def test_summarize_size_mismatch_raises():
    M = [[1.0, 0.5], [0.4, 0.7]]
    raised = False
    try:
        summarize(M, ["only_one"])
    except ValueError:
        raised = True
    assert raised


# --------------------------------------------------------------------------- #
# end-to-end: tiny 3-task split-classification can actually learn task 0
# --------------------------------------------------------------------------- #


def test_end_to_end_can_learn_one_task():
    """Sanity check: with no continual-learning machinery at all, the
    classifier can learn the first 2-way task to high accuracy. This
    tells us the head + loss + extend pieces fit together. Real CL
    experiments live in experiments/bench_split_mnist.py."""
    torch.manual_seed(0)
    net = _classifier(fan_in=4, hidden=16, n_classes=2)
    opt = optim.Adam(net.parameters(), lr=1e-2)

    # 2-class linearly-separable problem on R^4.
    def sample(batch=64):
        labels = torch.randint(0, 2, (batch,))
        x = torch.randn(batch, 4)
        # class 1 has mean (+2, 0, 0, 0), class 0 mean (-2, 0, 0, 0)
        x[:, 0] += torch.where(labels == 1, 2.0, -2.0)
        return x, labels

    for _ in range(200):
        x, y = sample()
        logits = net(x)
        loss = masked_cross_entropy(logits, y, active_classes=[0, 1])
        opt.zero_grad()
        loss.backward()
        opt.step()

    x_eval, y_eval = sample(batch=512)
    with torch.no_grad():
        a = accuracy(net(x_eval), y_eval)
    assert a > 0.9, f"failed to learn trivial 2-way classification: acc={a}"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    tests = [
        ("split_mnist_tasks_shape", test_split_mnist_tasks_shape),
        ("split_cifar100_default_shape", test_split_cifar100_default_shape),
        ("split_cifar100_custom_shape", test_split_cifar100_custom_shape),
        ("split_cifar100_overflow_raises", test_split_cifar100_overflow_raises),
        ("extend_output_head_grows_last_layer", test_extend_output_head_grows_last_layer),
        ("extend_output_head_preserves_existing_logits",
         test_extend_output_head_preserves_existing_logits),
        ("extend_output_head_optimizer_rebuild_works",
         test_extend_output_head_optimizer_rebuild_works),
        ("extend_output_head_invalid_args", test_extend_output_head_invalid_args),
        ("masked_cross_entropy_matches_explicit_subset_ce",
         test_masked_cross_entropy_matches_explicit_subset_ce),
        ("masked_cross_entropy_ignores_inactive_logits",
         test_masked_cross_entropy_ignores_inactive_logits),
        ("masked_cross_entropy_label_outside_active_raises",
         test_masked_cross_entropy_label_outside_active_raises),
        ("masked_cross_entropy_active_class_oob_raises",
         test_masked_cross_entropy_active_class_oob_raises),
        ("masked_cross_entropy_backprop_only_through_active_columns",
         test_masked_cross_entropy_backprop_only_through_active_columns),
        ("predict_full_argmax_over_all_classes", test_predict_full_argmax_over_all_classes),
        ("accuracy_full_softmax_admits_other_task_mistakes",
         test_accuracy_full_softmax_admits_other_task_mistakes),
        ("accuracy_restrict_to_overrides_full_softmax",
         test_accuracy_restrict_to_overrides_full_softmax),
        ("summarize_final_accuracy_and_forgetting",
         test_summarize_final_accuracy_and_forgetting),
        ("summarize_size_mismatch_raises", test_summarize_size_mismatch_raises),
        ("end_to_end_can_learn_one_task", test_end_to_end_can_learn_one_task),
    ]

    print("Running test_classification.py")
    print("-" * 60)
    for name, fn in tests:
        _run(name, fn)
    print("-" * 60)
    n_pass = sum(1 for _, ok, _ in _RESULTS if ok)
    n_total = len(_RESULTS)
    print(f"  {n_pass}/{n_total} passed")
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
