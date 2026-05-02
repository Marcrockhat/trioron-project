"""Shared dataset loaders for split-classification continual benches.

Backs split-MNIST (the canary) and the chained-15 task headline (MNIST →
FashionMNIST → KMNIST). All three are 10-class 28×28 grayscale, so they
flatten to 784-dim with no preprocessing differences. Datasets are
downloaded once via torchvision and cached under outputs/data/, which
is gitignored.

Convention:
  - Images are returned as float tensors in [0, 1], flattened to 784.
  - Labels are returned in the GLOBAL class space (the network's head
    output index). For split-MNIST the global space is 0..9. For the
    chained bench, MNIST contributes 0..9, FashionMNIST 10..19, KMNIST
    20..29 — the local→global remap happens here in the sampler.

Per the next_session_plan, raw flattened pixels — no CNN. Architectural
commitment: MLP-only.
"""
from __future__ import annotations
from dataclasses import dataclass
import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torchvision import datasets as tvd


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_ROOT = os.path.join(PROJECT_ROOT, "outputs", "data")

_DATASET_CTORS = {
    "mnist": tvd.MNIST,
    "fashion_mnist": tvd.FashionMNIST,
    "kmnist": tvd.KMNIST,
    # emnist_letters: EMNIST 'letters' split (26 letters, labels 1..26).
    # Loaded via the constructor wrapper below because EMNIST takes an
    # extra `split` kwarg the others don't.
    "emnist_letters": "EMNIST_LETTERS_SENTINEL",
}

IMAGE_DIM = 28 * 28  # 784


# ---------------------------------------------------------------------------
# Per-dataset cached tensor loader
# ---------------------------------------------------------------------------


def _load_split(name: str, train: bool, root: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load a torchvision split as flat [0,1] float tensors.

    First call downloads to `root`. Returns (images_flat, labels) where
    images_flat is shape (N, 784) float32 and labels is shape (N,)
    int64.
    """
    if name not in _DATASET_CTORS:
        raise ValueError(
            f"unknown dataset {name!r}; supported: {list(_DATASET_CTORS)}"
        )
    os.makedirs(root, exist_ok=True)

    if name == "emnist_letters":
        ds = tvd.EMNIST(root=root, split="letters", train=train, download=True)
        # EMNIST 'letters' has labels in {1..26} (1=A). Remap to {0..25}
        # so downstream views can use 0-indexed class lists naturally.
        # EMNIST also stores its images transposed relative to MNIST —
        # the canonical orientation is image.t() per Cohen 2017 Fig. 2.
        # Apply the transpose once here so flattened bytes mirror MNIST.
        images = ds.data.to(torch.float32).div_(255.0)
        images = images.transpose(1, 2).contiguous().view(-1, IMAGE_DIM)
        labels = ds.targets.to(torch.long).clone() - 1
        return images, labels

    ctor = _DATASET_CTORS[name]
    ds = ctor(root=root, train=train, download=True)
    images = ds.data.to(torch.float32).div_(255.0).view(-1, IMAGE_DIM)
    labels = ds.targets.to(torch.long).clone()
    return images, labels


# ---------------------------------------------------------------------------
# A view over (images, labels, class_subset) with batched sampling
# ---------------------------------------------------------------------------


@dataclass
class TaskDataView:
    """A sampler/eval view over one task's class subset of a dataset.

    `local_to_global` maps each LOCAL class index in the dataset (e.g.
    MNIST digit 3) to the GLOBAL head output index (e.g. 13 in the
    chained-15 bench). When chaining multiple datasets the same local
    class label can mean different global classes.

    All sampling and eval methods return labels in the GLOBAL space.
    """
    name: str
    images: torch.Tensor          # (N, 784) float32 — pre-filtered to local_classes
    labels_global: torch.Tensor   # (N,) int64 — already remapped
    local_classes: List[int]
    global_classes: List[int]

    def sample(
        self,
        batch: int,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Uniform-with-replacement minibatch sample. Cheaper than per-task
        DataLoader+shuffle and matches the patterns in bench_50task."""
        n = self.images.shape[0]
        if n == 0:
            raise RuntimeError(f"task {self.name!r} has 0 examples")
        idx = torch.randint(0, n, (batch,), generator=generator)
        return self.images[idx], self.labels_global[idx]

    def all_examples(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full set — used to construct the fixed eval batches."""
        return self.images, self.labels_global

    def n_examples(self) -> int:
        return int(self.images.shape[0])


# ---------------------------------------------------------------------------
# Dataset bundle — caches train + test tensors per dataset name
# ---------------------------------------------------------------------------


class DatasetBundle:
    """Caches train+test tensors for one or more datasets and slices them
    into per-task views on demand. Construct once per process; passes
    through to TaskDataView for actual sampling.

    Repeated calls to `.task_view(...)` with the same args reuse the
    same filtered tensors (cheap — they're sliced via index_select).
    """

    def __init__(
        self,
        names: Sequence[str],
        root: str = DEFAULT_DATA_ROOT,
    ):
        self.root = root
        self._train: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._test: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for name in names:
            print(f"[datasets] loading {name} (train) ...")
            self._train[name] = _load_split(name, train=True, root=root)
            print(f"[datasets] loading {name} (test)  ...")
            self._test[name] = _load_split(name, train=False, root=root)

    def task_view(
        self,
        dataset_name: str,
        local_classes: Sequence[int],
        global_classes: Sequence[int],
        split: str = "train",
        task_name: Optional[str] = None,
    ) -> TaskDataView:
        """Slice the chosen split to the given local classes and remap
        labels into the global class space."""
        if dataset_name not in self._train:
            raise ValueError(
                f"dataset {dataset_name!r} not loaded; "
                f"loaded={list(self._train)}"
            )
        if len(local_classes) != len(global_classes):
            raise ValueError(
                f"local_classes ({len(local_classes)}) and global_classes "
                f"({len(global_classes)}) must have equal length"
            )
        if split == "train":
            images, labels = self._train[dataset_name]
        elif split == "test":
            images, labels = self._test[dataset_name]
        else:
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")

        local_t = torch.tensor(list(local_classes), dtype=torch.long)
        # Mask: True where label is one of local_classes.
        mask = (labels.unsqueeze(1) == local_t.unsqueeze(0)).any(dim=1)
        keep_idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
        sub_images = images.index_select(0, keep_idx).clone()
        sub_labels_local = labels.index_select(0, keep_idx).clone()

        # Remap local→global. Build a (max_local+1) lookup.
        max_local = int(local_t.max().item())
        remap = torch.full((max_local + 1,), -1, dtype=torch.long)
        for L, G in zip(local_classes, global_classes):
            remap[L] = G
        sub_labels_global = remap[sub_labels_local]
        if int(sub_labels_global.min().item()) < 0:
            raise RuntimeError(
                "label remap produced -1; this means a label slipped "
                "through the class filter — internal bug"
            )

        nm = task_name or f"{dataset_name}_{'_'.join(str(c) for c in local_classes)}"
        return TaskDataView(
            name=nm,
            images=sub_images,
            labels_global=sub_labels_global,
            local_classes=list(local_classes),
            global_classes=list(global_classes),
        )


# ---------------------------------------------------------------------------
# Pre-built curricula (binary tasks, 5-per-dataset)
# ---------------------------------------------------------------------------


@dataclass
class ChainedTaskSpec:
    """A task in the chained curriculum. Resolves to a TaskDataView when
    paired with a DatasetBundle."""
    name: str
    dataset_name: str
    local_classes: List[int]
    global_classes: List[int]


def split_mnist_specs() -> List[ChainedTaskSpec]:
    """Standard split-MNIST: 5 binary tasks; global classes = local classes."""
    out: List[ChainedTaskSpec] = []
    for i in range(5):
        loc = [2 * i, 2 * i + 1]
        out.append(
            ChainedTaskSpec(
                name=f"mnist_{loc[0]}_{loc[1]}",
                dataset_name="mnist",
                local_classes=loc,
                global_classes=list(loc),
            )
        )
    return out


def chained_15_specs() -> List[ChainedTaskSpec]:
    """Chained 15-task curriculum: MNIST → FashionMNIST → EMNIST-letters.

    KMNIST was the originally-planned third block but its only
    torchvision mirror (codh.rois.ac.jp) is unreachable. EMNIST-letters
    (NIST source, reachable) covers the same role: a distinct glyph
    distribution from digits and clothing, 28x28 grayscale, 10 of its
    26 letter classes split into 5 binary tasks (A/B, C/D, E/F, G/H, I/J).

    Global class layout:
      MNIST 0..9                 → global 0..9
      FashionMNIST 0..9          → global 10..19
      EMNIST letters A..J (0..9) → global 20..29
    """
    out: List[ChainedTaskSpec] = []
    blocks = [
        ("mnist", 0),
        ("fashion_mnist", 10),
        ("emnist_letters", 20),
    ]
    for ds, offset in blocks:
        for i in range(5):
            loc = [2 * i, 2 * i + 1]
            glob = [loc[0] + offset, loc[1] + offset]
            out.append(
                ChainedTaskSpec(
                    name=f"{ds}_{loc[0]}_{loc[1]}",
                    dataset_name=ds,
                    local_classes=loc,
                    global_classes=glob,
                )
            )
    return out


def build_task_views(
    bundle: DatasetBundle,
    specs: Sequence[ChainedTaskSpec],
    split: str = "train",
) -> List[TaskDataView]:
    return [
        bundle.task_view(
            dataset_name=s.dataset_name,
            local_classes=s.local_classes,
            global_classes=s.global_classes,
            split=split,
            task_name=s.name,
        )
        for s in specs
    ]


__all__ = [
    "IMAGE_DIM",
    "DEFAULT_DATA_ROOT",
    "TaskDataView",
    "DatasetBundle",
    "ChainedTaskSpec",
    "split_mnist_specs",
    "chained_15_specs",
    "build_task_views",
]
