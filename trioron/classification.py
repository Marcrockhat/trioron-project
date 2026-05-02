"""Classification head + cross-entropy machinery for split-class continual benches.

Shared by split-MNIST (raw 784-dim pixels) and split-CIFAR-100 (frozen
ResNet-18 features). The trioron network operates as the body; this
module manages the growing output head and the active-class-subset CE
loss.

Convention: output index i corresponds to global class i. When task t
trains classes [c_0, c_1, ...], training-time CE is computed over those
columns only (logits sliced + labels remapped to local positions).
Evaluation uses the full unrestricted softmax over every output present
so far — that is the standard "single-head with task-incremental
training" continual setup, which exposes inter-task interference.

Caller responsibilities:

  - The TrioronNetwork's last layer must be `linear` (raw logits). This
    module takes the net's natural forward pass to be the logits.
  - After `extend_output_head` the optimizer must be rebuilt — same
    optimizer-rebuild caveat as `TrioronLayer.grow_node`.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .network import TrioronNetwork


@dataclass
class SplitClassificationTask:
    """A single task in a split-classification curriculum.

    `classes` is the list of GLOBAL class indices this task introduces.
    e.g. for split-MNIST task 1: classes=[0, 1].
    `name` is for logging only.
    """
    name: str
    classes: List[int]


def split_mnist_tasks() -> List[SplitClassificationTask]:
    """Standard split-MNIST curriculum: 5 binary tasks (0-1, 2-3, 4-5, 6-7, 8-9)."""
    return [
        SplitClassificationTask(name=f"mnist_{2*i}_{2*i+1}", classes=[2 * i, 2 * i + 1])
        for i in range(5)
    ]


def split_cifar100_tasks(n_tasks: int = 10, classes_per_task: int = 10) -> List[SplitClassificationTask]:
    """Standard split-CIFAR-100 curriculum: n_tasks sequential blocks of
    classes_per_task classes each. Default 10×10."""
    if n_tasks * classes_per_task > 100:
        raise ValueError(
            f"n_tasks*classes_per_task={n_tasks*classes_per_task} > 100 (CIFAR-100 size)"
        )
    out: List[SplitClassificationTask] = []
    for i in range(n_tasks):
        start = i * classes_per_task
        out.append(
            SplitClassificationTask(
                name=f"cifar_{start:03d}_{start+classes_per_task-1:03d}",
                classes=list(range(start, start + classes_per_task)),
            )
        )
    return out


def extend_output_head(net: TrioronNetwork, n_new_classes: int) -> List[int]:
    """Append `n_new_classes` output nodes to the network's last layer.

    Each new node is initialized with the layer's standard kaiming-style
    init (no init_vec is supplied — classification heads are not
    PCA-of-residuals seeded; gradient descent finds the new weights).
    `task_idx` is left at 0 for these head extensions; the dreaming
    routing-starvation machinery doesn't act on the output layer in any
    bench that runs classification.

    Returns the list of new node indices in the head.

    The optimizer holding references to the head's W/b parameters MUST
    be rebuilt by the caller afterwards.
    """
    if n_new_classes < 1:
        raise ValueError(f"n_new_classes must be >= 1, got {n_new_classes}")
    if not net.layers:
        raise ValueError("net has no layers")
    last_idx = len(net.layers) - 1
    head = net.layers[last_idx]
    new_indices: List[int] = []
    for _ in range(n_new_classes):
        idx = head.grow_node(init_vec=None, task_idx=0)
        new_indices.append(idx)
    return new_indices


def masked_cross_entropy(
    logits: torch.Tensor,
    labels_global: torch.Tensor,
    active_classes: Sequence[int],
) -> torch.Tensor:
    """Cross-entropy over the active class subset only.

    logits: (batch, n_outputs_so_far) — full head output.
    labels_global: (batch,) long tensor of GLOBAL class indices; every
        entry must be in `active_classes`.
    active_classes: list of global class indices currently being trained.

    The logits are sliced to `active_classes`, the labels are remapped
    to local positions [0, len(active_classes)), and standard
    cross_entropy is applied over the local subset. This is the standard
    "task-incremental" training-time loss for split-class benches.
    """
    if not active_classes:
        raise ValueError("active_classes must be non-empty")
    if logits.dim() != 2:
        raise ValueError(f"expected logits of shape (B, C), got {tuple(logits.shape)}")
    if labels_global.dim() != 1 or labels_global.shape[0] != logits.shape[0]:
        raise ValueError(
            f"labels_global shape {tuple(labels_global.shape)} "
            f"incompatible with logits {tuple(logits.shape)}"
        )

    active_t = torch.as_tensor(
        list(active_classes), dtype=torch.long, device=logits.device,
    )
    if int(active_t.max().item()) >= logits.shape[1]:
        raise ValueError(
            f"active_classes contains {int(active_t.max().item())} but head "
            f"only has {logits.shape[1]} outputs — extend_output_head first"
        )

    # Build a {global_class -> local_idx} map and remap labels. We do
    # this without a python loop so it stays cheap on big batches.
    global_to_local = torch.full(
        (logits.shape[1],), -1, dtype=torch.long, device=logits.device,
    )
    local_idx = torch.arange(len(active_classes), device=logits.device)
    global_to_local[active_t] = local_idx

    local_labels = global_to_local[labels_global]
    if int(local_labels.min().item()) < 0:
        bad = labels_global[local_labels < 0].unique().tolist()
        raise ValueError(
            f"labels {bad} are not in active_classes={list(active_classes)}"
        )

    logits_active = logits.index_select(1, active_t)
    return F.cross_entropy(logits_active, local_labels)


def predict_full(logits: torch.Tensor) -> torch.Tensor:
    """Argmax over the full (unrestricted) head — eval-time prediction.

    Returns a (batch,) long tensor of GLOBAL class indices. Use this at
    evaluation: it allows mistakes that fall on classes from any task,
    which is what the standard split-* metric measures.
    """
    return torch.argmax(logits, dim=1)


def accuracy(
    logits: torch.Tensor,
    labels_global: torch.Tensor,
    restrict_to: Optional[Sequence[int]] = None,
) -> float:
    """Top-1 accuracy.

    restrict_to: optional list of class indices. When None (default),
        argmax is over all logits — the proper continual-learning metric
        because it admits mistakes on classes from other tasks. When
        supplied, argmax is restricted to that subset (useful for
        "task-aware inference" diagnostics; not the headline metric).
    """
    if restrict_to is None:
        preds = predict_full(logits)
    else:
        active_t = torch.as_tensor(
            list(restrict_to), dtype=torch.long, device=logits.device,
        )
        sub = logits.index_select(1, active_t)
        local = torch.argmax(sub, dim=1)
        preds = active_t[local]
    return float((preds == labels_global).float().mean().item())


@dataclass
class SplitClassificationReport:
    """End-of-curriculum summary for a split-classification run.

    accuracy_matrix[i][j] = accuracy on task j after training task i,
    using the full unrestricted argmax (the proper continual metric).
    NaN where j > i (task j not yet trained).
    """
    task_names: List[str]
    accuracy_matrix: List[List[float]]
    final_accuracy: float
    avg_forgetting: float

    def per_task_final(self) -> List[float]:
        K = len(self.task_names)
        return [self.accuracy_matrix[K - 1][j] for j in range(K)]


def summarize(
    accuracy_matrix: List[List[float]],
    task_names: Sequence[str],
) -> SplitClassificationReport:
    """Build a SplitClassificationReport from the per-step accuracy matrix.

    accuracy_matrix shape: K x K. Entry [i][j] = accuracy on task j after
    completing task i. NaN allowed for j > i (not-yet-seen tasks).

    avg_forgetting = mean over j < K-1 of (acc[j][j] - acc[K-1][j]).
    Higher = more forgetting. Standard CL metric.
    """
    K = len(accuracy_matrix)
    if K != len(task_names):
        raise ValueError(
            f"accuracy_matrix has {K} rows but {len(task_names)} task names"
        )
    final_row = accuracy_matrix[K - 1]
    final_accuracy = sum(final_row) / K

    forget: List[float] = []
    for j in range(K - 1):
        diag = accuracy_matrix[j][j]
        end = accuracy_matrix[K - 1][j]
        forget.append(diag - end)
    avg_forgetting = sum(forget) / len(forget) if forget else float("nan")

    return SplitClassificationReport(
        task_names=list(task_names),
        accuracy_matrix=[list(row) for row in accuracy_matrix],
        final_accuracy=final_accuracy,
        avg_forgetting=avg_forgetting,
    )


__all__ = [
    "SplitClassificationTask",
    "SplitClassificationReport",
    "split_mnist_tasks",
    "split_cifar100_tasks",
    "extend_output_head",
    "masked_cross_entropy",
    "predict_full",
    "accuracy",
    "summarize",
]
