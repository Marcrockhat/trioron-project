"""Pair-mode classifier — when single-look is ambiguous, compare the
input against stored class prototypes via per-sense difference
signatures.

The framing: a single-image fused classifier might confuse two visually
similar classes (Rocky's "two shells smell and feel the same"). The
fix is *experimentation* — bring in a reference and compare. Per-sense
differences cancel common-mode sensor noise, surfacing class-specific
features that absolute readings can't.

Pipeline:
  1. Build per-(sense, class) prototypes once from train data:
     prototypes[sense] = (n_classes, sense_dim) — the class-mean of
     standardized sense readings.
  2. For an ambiguous input + top-K candidate classes, compute per
     (sense, candidate) the L2 distance between the input's
     standardized sense reading and the class prototype.
  3. Per-candidate score = aggregate over senses (uniform-mean by
     default; calibrator gates supported).
  4. Argmin distance → resolved class.

Storage: 100 × Σ sense_dims floats. For greedy-7
(64+12+8+8+6+8+6 = 112 dims) that's 11.2K floats ≈ 45 KB fp32, 22 KB
fp16. Cheap relative to the 1.5 MB substrate.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import torch

from . import apply_sense
from .standardizer import Standardizer
from .organism import SensoryOrganism


# ---------------------------------------------------------------------
# Prototype bank
# ---------------------------------------------------------------------


@dataclass
class PrototypeBank:
    """Per-(sense, class) class-mean of standardized sense readings.

    Fields:
        prototypes: dict mapping sense name → (n_classes, sense_dim).
            Class index = position in `class_order`. If a class wasn't
            seen during fit (no images), its row is the all-zeros
            vector (i.e., the standardized origin).
        class_order: list of class IDs (length n_classes). Index into
            the prototype rows.
        sense_order: list of sense names (length n_senses). Defines a
            canonical ordering for downstream stacking.
    """

    prototypes: Dict[str, torch.Tensor]
    class_order: List[int]
    sense_order: List[str]

    @classmethod
    def fit_from_organism(
        cls,
        org: SensoryOrganism,
        images: torch.Tensor,
        labels: torch.Tensor,
        *,
        batch_size: int = 512,
    ) -> "PrototypeBank":
        """Compute class-mean readings on ``(images, labels)``.

        For each branch ``b`` (whose sense is ``b.sense_name`` and
        whose Standardizer is ``b.standardizer``), accumulates
        Σ standardize(sense(image)) per class label and divides by
        the per-class count. The standardizer is the one baked into
        the donor checkpoint so comparisons stay in the same space the
        donor's L1 was trained on.
        """
        sense_order = [b.sense_name for b in org.branches]
        class_order = list(org.union_classes)
        cls_to_idx = {c: i for i, c in enumerate(class_order)}
        n_classes = len(class_order)

        # Sums and counts, one tensor per sense.
        sums: Dict[str, torch.Tensor] = {}
        counts: Dict[str, torch.Tensor] = {}
        for b in org.branches:
            sense = b.sense_name
            # Sense dim = the input_dim of the donor's L0.
            sense_dim = b.net.layers[0].W.shape[-1]
            sums[sense] = torch.zeros(n_classes, sense_dim)
            counts[sense] = torch.zeros(n_classes)

        N = images.shape[0]
        with torch.no_grad():
            for i in range(0, N, batch_size):
                j = min(i + batch_size, N)
                ys = labels[i:j].long()
                # Map labels to class indices; skip rows whose label
                # isn't in class_order (defensive).
                idx = torch.tensor(
                    [cls_to_idx.get(int(y), -1) for y in ys.tolist()],
                    dtype=torch.long,
                )
                keep = idx >= 0
                if not keep.any():
                    continue
                imgs = images[i:j][keep]
                idx = idx[keep]
                for b in org.branches:
                    sense = b.sense_name
                    raw = apply_sense(sense, imgs)
                    std = b.standardizer
                    z = (raw - std.mean) / std.std
                    sums[sense].index_add_(0, idx, z)
                    counts[sense].index_add_(
                        0, idx, torch.ones(z.shape[0]),
                    )

        prototypes: Dict[str, torch.Tensor] = {}
        for sense, s in sums.items():
            c = counts[sense].clamp_min(1e-6).unsqueeze(-1)
            prototypes[sense] = s / c

        return cls(
            prototypes=prototypes,
            class_order=class_order,
            sense_order=sense_order,
        )

    def storage_bytes(self, dtype: torch.dtype = torch.float32) -> int:
        size = 0
        for t in self.prototypes.values():
            size += t.numel() * torch.tensor([], dtype=dtype).element_size()
        return size

    def to_dict(self) -> Dict:
        return {
            "kind": "sensory_prototype_bank",
            "class_order": list(self.class_order),
            "sense_order": list(self.sense_order),
            "prototypes": {k: v.detach().cpu()
                           for k, v in self.prototypes.items()},
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "PrototypeBank":
        return cls(
            prototypes={k: v for k, v in d["prototypes"].items()},
            class_order=list(d["class_order"]),
            sense_order=list(d["sense_order"]),
        )

    def save(self, path: Union[str, Path]) -> None:
        torch.save(self.to_dict(), str(path))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "PrototypeBank":
        d = torch.load(str(path), map_location="cpu", weights_only=False)
        return cls.from_dict(d)


# ---------------------------------------------------------------------
# Pair classifier
# ---------------------------------------------------------------------


def per_branch_distances(
    org: SensoryOrganism,
    images: torch.Tensor,
    bank: PrototypeBank,
) -> torch.Tensor:
    """Compute squared L2 distance from each image's per-sense
    standardized reading to every class prototype.

    Returns a (B, N_branches, n_classes) tensor of squared distances,
    where the class axis follows ``bank.class_order``.
    """
    if bank.sense_order != [b.sense_name for b in org.branches]:
        raise ValueError(
            "PrototypeBank.sense_order must match organism branch order"
        )
    B = images.shape[0]
    n_branches = len(org.branches)
    n_classes = len(bank.class_order)
    out = images.new_zeros(B, n_branches, n_classes)
    with torch.no_grad():
        for bi, b in enumerate(org.branches):
            raw = apply_sense(b.sense_name, images)
            std = b.standardizer
            z = (raw - std.mean) / std.std                    # (B, sense_dim)
            proto = bank.prototypes[b.sense_name]              # (n_classes, sense_dim)
            # Squared L2 between each row of z and each prototype.
            #   d[b, c] = ||z_b - μ_c||^2
            #          = ||z_b||^2 + ||μ_c||^2 - 2 z_b·μ_c
            zz = (z * z).sum(dim=-1, keepdim=True)             # (B, 1)
            pp = (proto * proto).sum(dim=-1).unsqueeze(0)      # (1, n_classes)
            zp = z @ proto.t()                                  # (B, n_classes)
            out[:, bi] = zz + pp - 2 * zp
    return out


def resolve_pair(
    distances: torch.Tensor,
    candidate_classes: torch.Tensor,
    bank: PrototypeBank,
    *,
    branch_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Given per-branch distances over all classes, pick the candidate
    class with the smallest aggregated distance per image.

    Args:
        distances: (B, N_branches, n_classes) — per-branch squared L2
            distance, output of ``per_branch_distances``.
        candidate_classes: (B, K) — global class IDs of the top-K
            candidates per image.
        bank: PrototypeBank used to compute the distances; provides
            class_order for index lookup.
        branch_weights: optional (B, N_branches) — if supplied, used
            to weight the cross-branch sum. Default = uniform-mean.

    Returns:
        resolved_class: (B,) — picked global class ID per image.
    """
    B, N, _ = distances.shape
    K = candidate_classes.shape[-1]

    # Vectorized class-id → bank-row-index lookup. class_order may be
    # any permutation, so build a 1-D lookup once.
    n_max = max(bank.class_order) + 1
    lookup = torch.full((n_max,), -1, dtype=torch.long)
    for i, c in enumerate(bank.class_order):
        lookup[c] = i
    cand_idx = lookup[candidate_classes.long()]                  # (B, K)
    if (cand_idx < 0).any():
        raise ValueError("candidate_classes contains class IDs not in bank")
    gathered = distances.gather(
        dim=-1,
        index=cand_idx.unsqueeze(1).expand(B, N, K),
    )                                                    # (B, N, K)

    if branch_weights is None:
        agg = gathered.mean(dim=1)                       # (B, K)
    else:
        w = branch_weights.unsqueeze(-1)                 # (B, N, 1)
        agg = (gathered * w).sum(dim=1)                  # (B, K)
    pick = agg.argmin(dim=-1)                            # (B,)
    resolved = candidate_classes.gather(-1, pick.unsqueeze(-1)).squeeze(-1)
    return resolved


__all__ = [
    "PrototypeBank",
    "per_branch_distances",
    "resolve_pair",
]
