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


class DifferentialReplayBuffer:
    """Per-class multi-layer differential storage. For each class c at
    consolidation time, store δL_c = L(x_c) - L(0) at every relevant
    layer's output (L0, L1, head). Replay drives the live network to
    maintain the SAME differential signature regardless of how its
    biases / weights drift on subsequent tasks.

    Why differentials, not absolutes (Rocky 2026-05-04): the network
    embodies its memory; forward(0) reveals the network's accumulated
    memory state as bias-and-routing patterns shift across tasks. The
    differential isolates the TASK-SPECIFIC contribution from this
    drifting memory baseline. Storing absolutes (hippocampal) couples
    rehearsal to a specific bias state at consolidation time; storing
    differentials decouples rehearsal from bias drift — a class-c
    differential remains valid even as L1+head biases drift on later
    tasks.

    Storage scales with TOTAL NETWORK SIZE (sum of layer widths), not
    input_dim. Per class: L0_width + L1_width + head_size floats. For
    chained-15 with L0=128, L1≈48, head=30: ~206 floats × 4 = 824 B
    per class. 30 classes total ≈ 25 KB.

    Dimension growth handling: if L1 or head grow after a class's
    differential is stored, the stored differential is zero-padded at
    sample time to match the current dimensionality (new units don't
    contribute to old classes' differentials).
    """

    def __init__(self):
        # class_idx → dict with keys 'dL0', 'dL1', 'dlogit'
        self._deltas: Dict[int, Dict[str, torch.Tensor]] = {}

    def add_class(
        self,
        class_idx: int,
        dL0: torch.Tensor,
        dL1: torch.Tensor,
        dlogit: torch.Tensor,
    ) -> None:
        """Store differentials for one class. Each must be 1-D."""
        for name, t in [("dL0", dL0), ("dL1", dL1), ("dlogit", dlogit)]:
            if t.dim() != 1:
                raise ValueError(
                    f"{name} must be 1-D, got {tuple(t.shape)}"
                )
        self._deltas[int(class_idx)] = {
            "dL0": dL0.detach().clone(),
            "dL1": dL1.detach().clone(),
            "dlogit": dlogit.detach().clone(),
        }

    def has_classes(self) -> bool:
        return len(self._deltas) > 0

    def n_classes_stored(self) -> int:
        return len(self._deltas)

    def stored_classes(self) -> List[int]:
        return sorted(self._deltas.keys())

    def storage_bytes(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for d in self._deltas.values()
            for t in d.values()
        )

    def sample(
        self,
        n_samples: int,
        l0_width: int,
        l1_width: int,
        head_size: int,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (dL0, dL1, dlogit, y) batches. Each delta is zero-
        padded if the layer has grown since storage time. Returns
        (None,)*4 if buffer is empty."""
        if not self._deltas:
            return None, None, None, None  # type: ignore
        classes = list(self._deltas.keys())
        choice_idx = torch.randint(
            0, len(classes), (n_samples,), generator=generator,
        )
        dL0_rows = torch.zeros(n_samples, l0_width)
        dL1_rows = torch.zeros(n_samples, l1_width)
        dlogit_rows = torch.zeros(n_samples, head_size)
        ys: List[int] = []
        for i in range(n_samples):
            c = classes[int(choice_idx[i])]
            d = self._deltas[c]
            d0 = d["dL0"]; d1 = d["dL1"]; dlg = d["dlogit"]
            dL0_rows[i, : min(d0.shape[0], l0_width)] = d0[: min(d0.shape[0], l0_width)]
            dL1_rows[i, : min(d1.shape[0], l1_width)] = d1[: min(d1.shape[0], l1_width)]
            dlogit_rows[i, : min(dlg.shape[0], head_size)] = (
                dlg[: min(dlg.shape[0], head_size)]
            )
            ys.append(c)
        ys_t = torch.tensor(ys, dtype=torch.long)
        return dL0_rows, dL1_rows, dlogit_rows, ys_t


class HippocampalBuffer:
    """Per-class compressed-code storage. K canonical L0 outputs per
    class, stored at consolidation time by forwarding K real samples
    through the (frozen) L0 layer. Replay feeds these codes directly
    into L1 via forward_from_layer(start=1), bypassing L0 entirely.

    Biological mapping: hippocampal place/concept cells store sparse
    codes that index into cortical activation patterns, not raw sensory
    data. The cortex (here L1+head) reconstructs the rich representation
    by integrating the index through its recurrent dynamics. Sharp-wave
    ripples replay these compressed codes during sleep to drive cortical
    consolidation. trioron's frozen L0 plays the role of the cortical
    sensory hierarchy; HippocampalBuffer plays the role of CA3 + place
    cells.

    Why L0 output and not raw input: storage scales with L0_width
    (constant), not input_dim (grows with resolution). MNIST 784 → 128
    is a 6× compression; ImageNet 150528 → 128 is a 1200× compression;
    the buffer for ImageNet-scale problems uses the same RAM as for
    MNIST-scale problems. Real samples retain natural diversity at L0
    output (unlike gradient-ascent engrams which collapse adversarially
    — see experiments/probe_engram_diversity.py for the diagnostic).

    L0 must be frozen for stored codes to remain valid across tasks
    (encoding stable). For arms with trainable L0 (e.g., fixed_ewc) the
    buffer would go stale and either re-encoding-per-task or fallback to
    raw input is needed; for grown_capped_* / grown_uncapped_* arms with
    freeze_l0=True this is the natural design.
    """

    def __init__(self):
        # class_idx → tensor of shape (K, l0_width)
        self._codes: Dict[int, torch.Tensor] = {}

    def add_class(
        self, class_idx: int, codes: torch.Tensor,
    ) -> None:
        """Store K codes for one class. codes must be (K, l0_width).
        Overwrites any prior entry for this class."""
        if codes.dim() != 2:
            raise ValueError(
                f"codes must be 2-D (K, l0_width), got {tuple(codes.shape)}"
            )
        self._codes[int(class_idx)] = codes.detach().clone()

    def has_classes(self) -> bool:
        return len(self._codes) > 0

    def n_classes_stored(self) -> int:
        return len(self._codes)

    def stored_classes(self) -> List[int]:
        return sorted(self._codes.keys())

    def storage_bytes(self) -> int:
        return sum(c.numel() * c.element_size() for c in self._codes.values())

    def sample(
        self,
        n_samples: int,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (z, y) with z shape (n_samples, l0_width) and y
        shape (n_samples,). Uniform-with-replacement across stored
        classes; for each chosen class, uniformly samples one of its
        K stored codes. Returns (None, None) if buffer is empty."""
        if not self._codes:
            return None, None  # type: ignore
        classes = list(self._codes.keys())
        choice_idx = torch.randint(
            0, len(classes), (n_samples,), generator=generator,
        )
        rows = []; ys = []
        for i in range(n_samples):
            c = classes[int(choice_idx[i])]
            bank = self._codes[c]
            ridx = int(torch.randint(
                0, bank.shape[0], (1,), generator=generator,
            ).item())
            rows.append(bank[ridx])
            ys.append(c)
        zs = torch.stack(rows, dim=0)
        ys_t = torch.tensor(ys, dtype=torch.long)
        return zs, ys_t


class EngramBuffer:
    """Per-class synthetic input prototype storage for Engram Replay
    (triparametric pseudo-rehearsal). After each task's consolidation,
    one prototype `x_c` per just-learned class is found by running
    gradient ascent on the input through the *anchored* network to
    maximize logit_c. The resulting `x_c` is what the consolidated
    network considers a canonical class-c input — its "engram."

    During training of subsequent tasks, engrams are sampled in
    minibatches and fed through both the live and the anchored network;
    a KL distillation term keeps the live response on engram inputs
    aligned with the anchored response on those inputs (the trioron-
    native LwF — distilling on synthetic in-distribution past-class
    inputs rather than OOD new-task data).

    Storage cost: ~3 KB per class at 28×28×float32 (784 floats × 4 B).
    For 30 classes ≈ 90 KB total — manageable on edge hardware.

    Why input-space (784-dim) and not L0 output (128-dim): per Rocky's
    framing, the rehearsal signal must traverse all layers, including
    L0. L0-output engrams would constrain the L1-input pattern directly,
    which can lead to "trioron cannibalization" — L1 nodes locked to
    specific engram patterns lose plasticity for repurposing on new
    tasks. Input-space engrams pass through L0's frozen random
    projection first, which decorrelates the gradient signal across
    L1 nodes.
    """

    def __init__(self):
        # class_idx → x_c tensor of shape (input_dim,)
        self._engrams: Dict[int, torch.Tensor] = {}

    def add_class(
        self, class_idx: int, x_c: torch.Tensor,
    ) -> None:
        """Store a single engram for one class. x_c must be 1-D
        (input_dim,). Overwrites any prior entry for this class.
        """
        if x_c.dim() != 1:
            raise ValueError(
                f"x_c must be 1-D (input_dim,), got {tuple(x_c.shape)}"
            )
        self._engrams[int(class_idx)] = x_c.detach().clone()

    def has_classes(self) -> bool:
        return len(self._engrams) > 0

    def n_classes_stored(self) -> int:
        return len(self._engrams)

    def stored_classes(self) -> List[int]:
        return sorted(self._engrams.keys())

    def sample(
        self,
        n_samples: int,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (x, y) with x shape (n_samples, input_dim) and y shape
        (n_samples,). Each row is one stored engram, with class assigned
        uniformly at random across stored classes (with replacement).
        Returns (None, None) if buffer is empty.
        """
        if not self._engrams:
            return None, None  # type: ignore
        classes = list(self._engrams.keys())
        choice_idx = torch.randint(
            0, len(classes), (n_samples,), generator=generator,
        )
        ys = torch.tensor(
            [classes[int(i)] for i in choice_idx], dtype=torch.long,
        )
        rows = [self._engrams[classes[int(i)]] for i in choice_idx]
        xs = torch.stack(rows, dim=0)
        return xs, ys


class BrainstemBuffer:
    """Per-class (μ, σ) Gaussian statistics at a bottleneck layer for
    latent rehearsal (Brainstem-Spark / latent generative replay).

    Stores ONE diagonal Gaussian per global class, computed from the
    L1 output activations on that class's training data at consolidation
    time. At rehearsal time, sample synthetic latents `z ~ N(μ_c, σ_c²)`
    for random classes, feed them directly into the head (bypassing L0
    and L1 entirely). The head learns to classify synthetic in-
    distribution latents, regularizing it against drift on classes
    whose actual training data is no longer available.

    Handles L1 growth: stored stats from earlier tasks are at the L1
    width prevailing at storage time. When `sample()` is called with a
    larger `current_l1_width`, the stored vector is zero-padded — new
    L1 units (initialized fresh) didn't fire on the old class's data,
    so their expected contribution to the old class is ~zero.

    Storage cost is small: 30 classes × (μ, σ) × ~60 dims × 4 bytes ≈
    30 KB total at chained-15 scale.
    """

    def __init__(self):
        # class_idx → (mu, sigma) tensors. Each tensor has shape
        # (l1_width_at_storage,). When the network's L1 has grown past
        # this width, we zero-pad on sample.
        self._stats: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    def add_class(
        self, class_idx: int, mu: torch.Tensor, sigma: torch.Tensor,
    ) -> None:
        """Store stats for one class. mu, sigma must have shape
        (l1_width,). Overwrites any prior entry for this class."""
        if mu.dim() != 1 or sigma.dim() != 1:
            raise ValueError(
                f"mu/sigma must be 1-D, got {mu.shape}, {sigma.shape}"
            )
        if mu.shape != sigma.shape:
            raise ValueError(
                f"mu/sigma shape mismatch: {mu.shape} vs {sigma.shape}"
            )
        self._stats[int(class_idx)] = (
            mu.detach().clone(),
            sigma.detach().clone(),
        )

    def has_classes(self) -> bool:
        return len(self._stats) > 0

    def n_classes_stored(self) -> int:
        return len(self._stats)

    def sample(
        self,
        n_samples: int,
        current_l1_width: int,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (z, y) with z shape (n_samples, current_l1_width) and
        y shape (n_samples,). Each row of z is sampled from a randomly
        chosen stored class's per-feature Gaussian, zero-padded if the
        stored width is smaller than `current_l1_width`."""
        if not self._stats:
            return None, None  # type: ignore
        classes = list(self._stats.keys())
        # Random class assignment per row.
        choice_idx = torch.randint(
            0, len(classes), (n_samples,), generator=generator,
        )
        ys = torch.tensor(
            [classes[int(i)] for i in choice_idx], dtype=torch.long,
        )
        zs = torch.zeros(n_samples, current_l1_width)
        for i in range(n_samples):
            c = classes[int(choice_idx[i])]
            mu, sigma = self._stats[c]
            old_dim = mu.shape[0]
            d = min(old_dim, current_l1_width)
            noise = torch.randn(d, generator=generator)
            zs[i, :d] = mu[:d] + sigma[:d] * noise
            # zs[i, d:] left zero
        return zs, ys


class ManifoldBuffer:
    """Per-class diagonal Gaussian over L0 outputs — trioron-native
    pseudo-rehearsal that samples codes on demand from the consolidated
    L0 distribution rather than storing K real-sample codes.

    Stores ONE diagonal Gaussian per global class, computed from real
    samples of the class forwarded through the (frozen) L0 layer at
    consolidation time. Each replay step samples synthetic z ~ N(μ_c,
    σ_c²) and feeds via forward_from_layer(z, start=1), so L1 AND head
    are supervised — distinct from BrainstemBuffer which stores L1
    stats and bypasses L1 (head-only supervision).

    Compared to HippocampalBuffer (K real codes per class, ~768 KB at
    chained-15 K=50): storage drops to 30 KB total (per-class μ + σ at
    L0 width 128). Buffer/network ratio shrinks as the curriculum grows
    because per-class storage is constant in K.

    Trioron framing: differential's δL0 captures the per-class L0 mean
    (one moment); ManifoldBuffer adds the per-dim variance (second
    moment) and uses both as a sample generator rather than as a KL
    distillation target. The replay path is byte-identical to hippo
    (CE on forward_from_layer output) but the codes are sampled rather
    than stored.

    Frozen-L0 only: stored stats stay valid as long as L0 doesn't drift.
    For trainable-L0 arms the buffer would go stale; the bench gates
    consolidation on arm_l0_frozen.
    """

    def __init__(self):
        self._stats: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    def add_class(
        self, class_idx: int, mu: torch.Tensor, sigma: torch.Tensor,
    ) -> None:
        if mu.dim() != 1 or sigma.dim() != 1:
            raise ValueError(
                f"mu/sigma must be 1-D, got {mu.shape}, {sigma.shape}"
            )
        if mu.shape != sigma.shape:
            raise ValueError(
                f"mu/sigma shape mismatch: {mu.shape} vs {sigma.shape}"
            )
        self._stats[int(class_idx)] = (
            mu.detach().clone(),
            sigma.detach().clone(),
        )

    def has_classes(self) -> bool:
        return len(self._stats) > 0

    def n_classes_stored(self) -> int:
        return len(self._stats)

    def stored_classes(self) -> List[int]:
        return sorted(self._stats.keys())

    def storage_bytes(self) -> int:
        return sum(
            mu.numel() * mu.element_size() + sg.numel() * sg.element_size()
            for (mu, sg) in self._stats.values()
        )

    def sample(
        self,
        n_samples: int,
        noise_scale: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (z, y) with z shape (n_samples, l0_width). Each row
        sampled from a randomly chosen class's per-feature Gaussian.
        L0 is frozen → stored width is constant; no zero-pad logic."""
        if not self._stats:
            return None, None  # type: ignore
        classes = list(self._stats.keys())
        choice_idx = torch.randint(
            0, len(classes), (n_samples,), generator=generator,
        )
        ys = torch.tensor(
            [classes[int(i)] for i in choice_idx], dtype=torch.long,
        )
        d = next(iter(self._stats.values()))[0].shape[0]
        zs = torch.zeros(n_samples, d)
        for i in range(n_samples):
            c = classes[int(choice_idx[i])]
            mu, sigma = self._stats[c]
            noise = torch.randn(d, generator=generator) * noise_scale
            zs[i] = mu + sigma * noise
        return zs, ys


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
