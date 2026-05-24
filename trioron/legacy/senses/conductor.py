"""SensoryConductor — late-fusion over multiple (sense, donor) pairs.

Each primitive donor was trained on a single sense's readings and has
its own L0 random projection (different senses → different input
dims → cannot share L0). The conductor wraps an ordered list of
SenseDonor pairs, runs them independently on the same image batch,
and fuses their per-class logits into a single classifier.

This is the multi-modal weak-sensor architecture: each donor is a
"blind man" probing the image through one channel; the conductor is
the integrator that fuses partial reports into a class.

Fusion rules:
  * "mean_logit"        — average of raw logits (default; geometric
                          mean of softmaxes up to a constant)
  * "sum_logit"         — sum of raw logits (equivalent under argmax)
  * "log_prob_mean"     — average of log-softmax
  * "confidence_weighted" — per-image, per-donor weighting by softmax
                            max. Donors that are unsure on a given
                            image contribute less; confident donors
                            dominate. Crucial when senses vary widely
                            in solo strength (otherwise weak donors
                            dilute strong ones at fusion time).

For donors that cover different class subsets, logits are zero-padded
to the union-class layout before fusion. Donors implicitly assert
"abstain" on classes outside their coverage by contributing zero to
that slot.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .standardizer import Standardizer
from . import apply_sense
from ..network import TrioronNetwork


# ---------------------------------------------------------------------
# SenseDonor — one absorbed (sense, standardizer, frozen net) triple
# ---------------------------------------------------------------------


@dataclass
class SenseDonor:
    """One trained primitive donor along with its sense + standardizer."""
    sense_name: str
    standardizer: Standardizer
    net: TrioronNetwork
    classes_covered: List[int] = field(default_factory=list)
    label: str = ""


def load_sense_donor(path: Union[str, Path]) -> SenseDonor:
    """Reconstruct a SenseDonor from a checkpoint produced by
    ``experiments/cifar/train_donor.py``. The checkpoint embeds the
    sense name + Standardizer state alongside the standard donor
    payload."""
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    sense_name = payload.get("sense")
    if sense_name is None:
        raise ValueError(
            f"{path}: missing 'sense' field — not a sense-donor checkpoint"
        )
    std_dict = payload.get("standardizer")
    if std_dict is None:
        raise ValueError(f"{path}: missing 'standardizer' field")
    std = Standardizer.from_dict(std_dict)

    n_nodes = list(payload["n_nodes_per_layer"])
    input_dim = int(payload["input_dim"])
    layer_specs: List = []
    prev = input_dim
    for i, n in enumerate(n_nodes):
        act = "linear" if i == len(n_nodes) - 1 else "relu"
        layer_specs.append((prev, int(n), act))
        prev = int(n)
    net = TrioronNetwork(layer_specs)
    net.load_state_dict(payload["state_dict"])
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return SenseDonor(
        sense_name=sense_name,
        standardizer=std,
        net=net,
        classes_covered=list(payload["classes_covered"]),
        label=payload.get("label", ""),
    )


# ---------------------------------------------------------------------
# SensoryConductor — late-fusion module
# ---------------------------------------------------------------------


class SensoryConductor(nn.Module):
    """Late-fusion classifier over multiple sense donors."""

    def __init__(
        self,
        donors: Sequence[SenseDonor],
        fusion: str = "mean_logit",
    ) -> None:
        super().__init__()
        if not donors:
            raise ValueError("SensoryConductor: donors must be non-empty")
        if fusion not in ("mean_logit", "sum_logit", "log_prob_mean",
                          "confidence_weighted"):
            raise ValueError(f"unknown fusion {fusion!r}")
        self.donors: List[SenseDonor] = list(donors)
        self.fusion = fusion
        # Hold nets so .to(device) propagates.
        self.donor_nets = nn.ModuleList(d.net for d in donors)

        union = sorted({int(c) for d in donors for c in d.classes_covered})
        self.union_classes: List[int] = union
        n_union = len(union)
        cls_to_union = {c: i for i, c in enumerate(union)}

        # Column j of donor.net's head corresponds to the j-th class in
        # `classes_covered` (sorted ascending — matches the bench's
        # incremental head-growth order under sequential-class
        # curricula). Build a per-donor LongTensor mapping each donor
        # head column to a union slot.
        for i, d in enumerate(donors):
            idx_map = torch.tensor(
                [cls_to_union[int(c)] for c in d.classes_covered],
                dtype=torch.long,
            )
            self.register_buffer(f"_idx_map_{i}", idx_map, persistent=False)

        # Buffers for standardizers (one per donor).
        for i, d in enumerate(donors):
            self.register_buffer(f"_std_mean_{i}", d.standardizer.mean.clone(),
                                 persistent=False)
            self.register_buffer(f"_std_std_{i}",  d.standardizer.std.clone(),
                                 persistent=False)

    @property
    def n_classes(self) -> int:
        return len(self.union_classes)

    def per_donor_logits(self, images: torch.Tensor) -> List[torch.Tensor]:
        """Return list of (N, n_union) tensors, one per donor.

        Each tensor is the donor's logits zero-padded out to union-class
        layout (zero on slots the donor doesn't cover).
        """
        N = images.shape[0]
        device = images.device
        outs: List[torch.Tensor] = []
        for i, d in enumerate(self.donors):
            x_raw = apply_sense(d.sense_name, images)
            mean = getattr(self, f"_std_mean_{i}").to(device)
            std = getattr(self, f"_std_std_{i}").to(device)
            x = (x_raw - mean) / std
            logits = d.net(x)               # (N, head_size_d)
            idx_map = getattr(self, f"_idx_map_{i}").to(device)  # (head_size_d,)
            padded = torch.zeros(N, self.n_classes, device=device,
                                 dtype=logits.dtype)
            padded.index_copy_(1, idx_map, logits)
            outs.append(padded)
        return outs

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        per = self.per_donor_logits(images)
        stacked = torch.stack(per, dim=0)  # (D, N, n_union)
        if self.fusion == "sum_logit":
            return stacked.sum(dim=0)
        if self.fusion == "mean_logit":
            return stacked.mean(dim=0)
        if self.fusion == "log_prob_mean":
            return F.log_softmax(stacked, dim=-1).mean(dim=0)
        # confidence_weighted: per-image, per-donor scalar weight =
        # softmax-max (the donor's max class probability for that
        # image). Normalize so weights sum to 1 across donors per
        # image. Donors with peaked posteriors carry more weight than
        # donors that are uniform. Surfaces strong donors on images
        # where they actually know.
        probs = F.softmax(stacked, dim=-1)              # (D, N, C)
        conf = probs.max(dim=-1).values                 # (D, N)
        denom = conf.sum(dim=0, keepdim=True).clamp_min(1e-6)
        weights = (conf / denom).unsqueeze(-1)          # (D, N, 1)
        return (stacked * weights).sum(dim=0)


# ---------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------


def build_conductor(
    donor_paths: Sequence[Union[str, Path]],
    fusion: str = "mean_logit",
) -> SensoryConductor:
    """Load donors from disk and return a SensoryConductor."""
    donors = [load_sense_donor(p) for p in donor_paths]
    return SensoryConductor(donors, fusion=fusion)


# ---------------------------------------------------------------------
# LearnedFusion — small calibration head over per-donor logits
# ---------------------------------------------------------------------


class LearnedFusion(nn.Module):
    """Calibration head over stacked donor logits, parameterized as
    *offsets* from the mean-logit baseline so weight decay pulls
    toward the parameter-free fusion (not toward zero).

    Forms:
      * "scalar"      — one weight per donor (D params). Tiny and
                        robust. fused = sum_d (1/D + δ_d) · logits_d.
      * "scalar_bias" — D weights + C biases. Adds per-class shift.
      * "full"        — D*C weights + C biases. Per-(donor, class)
                        calibration; richest, most overfittable.
    """

    def __init__(
        self,
        n_donors: int,
        n_classes: int,
        form: str = "scalar_bias",
    ) -> None:
        super().__init__()
        if form not in ("scalar", "scalar_bias", "full"):
            raise ValueError(f"unknown form {form!r}")
        self.form = form
        self.n_donors = n_donors
        self.n_classes = n_classes
        if form == "scalar":
            self.delta_w = nn.Parameter(torch.zeros(n_donors))
            self.bias = None
        elif form == "scalar_bias":
            self.delta_w = nn.Parameter(torch.zeros(n_donors))
            self.bias = nn.Parameter(torch.zeros(n_classes))
        else:
            self.delta_w = nn.Parameter(torch.zeros(n_donors, n_classes))
            self.bias = nn.Parameter(torch.zeros(n_classes))

    def forward(self, stacked: torch.Tensor) -> torch.Tensor:
        """stacked: (B, D, C); returns (B, C)."""
        base = 1.0 / max(self.n_donors, 1)
        if self.form == "scalar":
            w = (base + self.delta_w).view(1, self.n_donors, 1)
            return (stacked * w).sum(dim=1)
        if self.form == "scalar_bias":
            w = (base + self.delta_w).view(1, self.n_donors, 1)
            return (stacked * w).sum(dim=1) + self.bias
        # full
        w = (base + self.delta_w).unsqueeze(0)            # (1, D, C)
        return (stacked * w).sum(dim=1) + self.bias


def fuse_with(
    conductor: SensoryConductor,
    images: torch.Tensor,
    fusion_module: LearnedFusion,
) -> torch.Tensor:
    """Run conductor's per-donor logits and apply a learned fusion."""
    per = conductor.per_donor_logits(images)
    stacked_dnc = torch.stack(per, dim=1)  # (B, D, C)
    return fusion_module(stacked_dnc)


__all__ = [
    "SenseDonor",
    "SensoryConductor",
    "LearnedFusion",
    "fuse_with",
    "load_sense_donor",
    "build_conductor",
]
