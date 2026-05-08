"""CIFAR-100 loader + sense-application pipeline.

Loads torchvision's CIFAR-100 once, applies a chosen sense to every
image (no pixel access for the donor — only sense readings), fits a
Standardizer on the training partition, and yields a list of TaskData
suitable for ``trioron.api.build_donor``.
"""
from __future__ import annotations
import os
from typing import List, Sequence, Tuple

import torch
from torchvision import datasets as tvd

from trioron.senses import apply_sense, Standardizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DATA_ROOT = os.path.join(PROJECT_ROOT, "outputs", "data")


def load_cifar100(
    root: str = DEFAULT_DATA_ROOT,
    train: bool = True,
    download: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load CIFAR-100 as (N, 3, 32, 32) float32 in [0, 1] + int64 labels."""
    os.makedirs(root, exist_ok=True)
    ds = tvd.CIFAR100(root=root, train=train, download=download)
    images = torch.from_numpy(ds.data).float().div_(255.0)   # (N, 32, 32, 3)
    images = images.permute(0, 3, 1, 2).contiguous()         # (N, 3, 32, 32)
    labels = torch.tensor(ds.targets, dtype=torch.long)
    return images, labels


def _filter_classes(
    images: torch.Tensor,
    labels: torch.Tensor,
    classes: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    keep = torch.zeros(labels.shape[0], dtype=torch.bool)
    for c in classes:
        keep |= labels == int(c)
    return images[keep], labels[keep]


def build_sense_tasks(
    sense_name: str,
    class_groups: Sequence[Sequence[int]],
    root: str = DEFAULT_DATA_ROOT,
) -> Tuple[List["TaskData"], Standardizer]:
    """Build a TaskData list for one sense + curriculum slice.

    Args:
        sense_name: which sense to apply (key into trioron.senses.SENSES).
        class_groups: ordered list of class subsets. Each subset becomes
            one task in the donor's curriculum, in order. Class indices
            are CIFAR-100 fine labels (0..99); they live directly in the
            donor's global class space.
        root: directory for the torchvision cache.

    Returns:
        (tasks, standardizer) — tasks for build_donor; standardizer must
        be saved alongside the donor and applied at inference time.
    """
    from trioron.api import TaskData

    all_classes = sorted({int(c) for g in class_groups for c in g})

    train_imgs, train_labs = load_cifar100(root, train=True)
    test_imgs,  test_labs  = load_cifar100(root, train=False)
    Xtr_raw, ytr = _filter_classes(train_imgs, train_labs, all_classes)
    Xte_raw, yte = _filter_classes(test_imgs,  test_labs,  all_classes)

    # Apply sense to every selected image once, then standardize using
    # train statistics only.
    Xtr_sensed = apply_sense(sense_name, Xtr_raw)
    Xte_sensed = apply_sense(sense_name, Xte_raw)
    std = Standardizer.fit(Xtr_sensed)
    Xtr_sensed = std.transform(Xtr_sensed).contiguous()
    Xte_sensed = std.transform(Xte_sensed).contiguous()

    tasks: List[TaskData] = []
    for i, group in enumerate(class_groups):
        group = [int(c) for c in group]
        m_tr = torch.zeros(ytr.shape[0], dtype=torch.bool)
        m_te = torch.zeros(yte.shape[0], dtype=torch.bool)
        for c in group:
            m_tr |= ytr == c
            m_te |= yte == c
        tasks.append(TaskData(
            name=f"cifar100_{sense_name}_task{i}",
            X_train=Xtr_sensed[m_tr],
            y_train=ytr[m_tr],
            X_test=Xte_sensed[m_te],
            y_test=yte[m_te],
            classes=group,
        ))
    return tasks, std


# ---------------------------------------------------------------------
# Default curriculum slices
# ---------------------------------------------------------------------

# First-slice validation: 25 sequential CIFAR-100 fine classes split
# into 5 tasks of 5 classes each. Sequential is fine for architecture
# validation; superclass-aligned slices come once fusion is proven.
FIRST_SLICE_CLASSES: List[List[int]] = [
    list(range(i, i + 5)) for i in range(0, 25, 5)
]

# Full CIFAR-100: all 100 fine classes split into 20 tasks of 5 each.
# Sequential by fine-class index (CIFAR-100's natural 20×5 superclass
# structure does not happen to align with sequential indices, so the
# task grouping here is by index, not by superclass).
FULL_100_CLASSES: List[List[int]] = [
    list(range(i, i + 5)) for i in range(0, 100, 5)
]


SLICES = {
    "first": FIRST_SLICE_CLASSES,
    "full":  FULL_100_CLASSES,
}
