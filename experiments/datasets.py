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
        DataLoader+shuffle and matches the patterns in bench_50task.

        Use `iter_epoch` instead when you want every sample seen exactly
        once per pass — this is preferred for proper task training; the
        random-with-replacement firehose was producing
        37%-of-samples-never-seen on smoke-budget runs and disrupting
        the model's ability to settle on stable representations.
        """
        n = self.images.shape[0]
        if n == 0:
            raise RuntimeError(f"task {self.name!r} has 0 examples")
        idx = torch.randint(0, n, (batch,), generator=generator)
        return self.images[idx], self.labels_global[idx]

    def iter_epoch(
        self,
        batch: int,
        generator: Optional[torch.Generator] = None,
    ):
        """Yield shuffled minibatches that traverse the task data
        EXACTLY ONCE.

        Each call to iter_epoch produces a fresh shuffle. The last batch
        may be smaller than `batch` — fine for training. Caller can
        loop iter_epoch(...) inside an outer epoch loop to get N
        complete passes through the data.

        Per Gemma's framing: the random-with-replacement firehose is
        the equivalent of bombarding a developing cell with a flickering
        stimulus distribution. Proper epoch iteration lets the network
        commit to representations on stable input shape.
        """
        n = self.images.shape[0]
        if n == 0:
            raise RuntimeError(f"task {self.name!r} has 0 examples")
        perm = torch.randperm(n, generator=generator)
        for start in range(0, n, batch):
            idx = perm[start:start + batch]
            yield self.images[idx], self.labels_global[idx]

    def all_examples(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full set — used to construct the fixed eval batches."""
        return self.images, self.labels_global

    def n_examples(self) -> int:
        return int(self.images.shape[0])


# ---------------------------------------------------------------------------
# Rehearsal memory buffer (Path 2)
# ---------------------------------------------------------------------------


class MemoryBuffer:
    """Bounded per-task memory of past examples for rehearsal during
    continual training.

    For each task, stores a random subset of `samples_per_task` examples
    (image, global_label) drawn from that task's training pool. During
    rehearsal, `sample()` returns a uniform random batch drawn from the
    union of all stored tasks.

    Used in `train_one_task` to mix a rehearsal batch into each training
    step, with the rehearsal CE loss masked to ALL classes seen so far
    (not just the current binary pair) — so the gradient directly fights
    cross-class head drift, the dominant failure mode on full-softmax
    accuracy in the chained-15 bench (head columns from the most-recent
    task dominate argmax across the 30-class output).

    Reservoir-style: each task gets a fixed allotment, sampling is
    uniform across the stored union. Keeps memory bounded; total budget
    is `samples_per_task * n_tasks_seen`.
    """

    def __init__(self, samples_per_task: int = 100):
        self.samples_per_task = int(samples_per_task)
        self._x: List[torch.Tensor] = []
        self._y: List[torch.Tensor] = []

    def add_task(
        self,
        x_pool: torch.Tensor,
        y_global_pool: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> int:
        """Sample `samples_per_task` random examples from the pool and
        append to the buffer. Returns the number of samples actually
        stored (may be less if the pool is small)."""
        n = x_pool.shape[0]
        if n == 0:
            return 0
        k = min(self.samples_per_task, n)
        idx = torch.randperm(n, generator=generator)[:k]
        self._x.append(x_pool[idx].detach().clone())
        self._y.append(y_global_pool[idx].detach().clone())
        return k

    def has_samples(self) -> bool:
        return len(self._x) > 0

    def n_total_samples(self) -> int:
        return sum(int(t.shape[0]) for t in self._x)

    def n_tasks_stored(self) -> int:
        return len(self._x)

    def sample(
        self,
        batch: int,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Uniform-random batch across all stored tasks.

        Returns (None, None) if the buffer is empty. Returned batch may
        be smaller than `batch` if the buffer total is smaller.
        """
        if not self._x:
            return None, None  # type: ignore
        all_x = torch.cat(self._x, dim=0)
        all_y = torch.cat(self._y, dim=0)
        total = all_x.shape[0]
        k = min(batch, total)
        idx = torch.randperm(total, generator=generator)[:k]
        return all_x[idx], all_y[idx]


# ---------------------------------------------------------------------------
# Dataset bundle — caches train + test tensors per dataset name
# ---------------------------------------------------------------------------


class DatasetBundle:
    """Caches train+test tensors for one or more datasets and slices them
    into per-task views on demand. Construct once per process; passes
    through to TaskDataView for actual sampling.

    Repeated calls to `.task_view(...)` with the same args reuse the
    same filtered tensors (cheap — they're sliced via index_select).

    `n_holdout_per_dataset`: optional non-negative int. When >0, the
    FIRST n_holdout samples of each train split are reserved as
    "infancy" data — `task_view` will not draw from them. The held-out
    portion is exposed via `infancy_view(specs)`. Used by the chained-15
    bench to give L0 a brief developmental warmup on disjoint data
    before the continual stream begins. Lickliter (2002) — too much
    sensory input during a developmental window disrupts perceptual
    cascading; default 0 (no holdout) preserves prior bench behavior.
    """

    def __init__(
        self,
        names: Sequence[str],
        root: str = DEFAULT_DATA_ROOT,
        n_holdout_per_dataset: int = 0,
    ):
        if n_holdout_per_dataset < 0:
            raise ValueError(
                f"n_holdout_per_dataset must be >= 0, got {n_holdout_per_dataset}"
            )
        self.root = root
        self.n_holdout_per_dataset = int(n_holdout_per_dataset)
        # Train tensors AFTER holdout is removed — task_view sees these.
        self._train: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._test: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        # Holdout tensors (raw, with original local labels). Keys exist
        # only when n_holdout_per_dataset > 0.
        self._holdout: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for name in names:
            print(f"[datasets] loading {name} (train) ...")
            full_images, full_labels = _load_split(name, train=True, root=root)
            if self.n_holdout_per_dataset > 0:
                hold = self.n_holdout_per_dataset
                if hold > full_images.shape[0]:
                    raise ValueError(
                        f"n_holdout_per_dataset={hold} exceeds {name} "
                        f"train size {full_images.shape[0]}"
                    )
                self._holdout[name] = (
                    full_images[:hold].clone(),
                    full_labels[:hold].clone(),
                )
                self._train[name] = (
                    full_images[hold:].clone(),
                    full_labels[hold:].clone(),
                )
            else:
                self._train[name] = (full_images, full_labels)
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

    def infancy_view(
        self,
        specs: Sequence["ChainedTaskSpec"],
    ) -> TaskDataView:
        """Combine the held-out portions of each dataset into a single
        TaskDataView spanning ALL global classes that appear in `specs`.

        Used for the L0 warmup: brief, disjoint-from-training exposure
        that develops a feature extractor before the continual stream
        starts. Requires the bundle was constructed with
        n_holdout_per_dataset > 0; raises otherwise.

        The returned view's labels_global use the SAME global class
        layout as the chained curriculum, so the warmup classifier
        trains on a head wide enough to cover all 30 classes (10 per
        dataset, 3 datasets).
        """
        if self.n_holdout_per_dataset == 0:
            raise RuntimeError(
                "infancy_view requires the bundle to be built with "
                "n_holdout_per_dataset > 0"
            )

        # Build a per-dataset {local_class -> global_class} map from the
        # chained specs. For datasets that span multiple specs (e.g.
        # MNIST has 5 specs covering classes 0..9), merge the maps.
        per_dataset_remap: Dict[str, Dict[int, int]] = {}
        for s in specs:
            mapping = per_dataset_remap.setdefault(s.dataset_name, {})
            for L, G in zip(s.local_classes, s.global_classes):
                mapping[L] = G

        chunks_x: List[torch.Tensor] = []
        chunks_y: List[torch.Tensor] = []
        for ds_name, remap in per_dataset_remap.items():
            if ds_name not in self._holdout:
                raise RuntimeError(
                    f"dataset {ds_name!r} appears in specs but isn't loaded"
                )
            images, labels_local = self._holdout[ds_name]
            # Mask: keep only samples whose local label appears in remap.
            local_t = torch.tensor(
                sorted(remap.keys()), dtype=torch.long,
            )
            mask = (labels_local.unsqueeze(1) == local_t.unsqueeze(0)).any(dim=1)
            keep = torch.nonzero(mask, as_tuple=False).squeeze(1)
            if keep.numel() == 0:
                continue
            sub_images = images.index_select(0, keep)
            sub_labels_local = labels_local.index_select(0, keep)
            # Remap to global.
            max_local = int(local_t.max().item())
            remap_t = torch.full((max_local + 1,), -1, dtype=torch.long)
            for L, G in remap.items():
                remap_t[L] = G
            sub_labels_global = remap_t[sub_labels_local]
            chunks_x.append(sub_images)
            chunks_y.append(sub_labels_global)

        if not chunks_x:
            raise RuntimeError(
                "infancy_view: no held-out samples matched any spec class"
            )
        all_x = torch.cat(chunks_x, dim=0)
        all_y = torch.cat(chunks_y, dim=0)
        all_globals = sorted({int(g) for m in per_dataset_remap.values()
                              for g in m.values()})
        return TaskDataView(
            name="infancy",
            images=all_x,
            labels_global=all_y,
            local_classes=[-1] * len(all_globals),  # not meaningful here
            global_classes=all_globals,
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
