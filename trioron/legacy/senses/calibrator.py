"""Stage-B calibrator for SensoryOrganism — learns to interpret its
sensors.

Two tiny heads on top of per-branch confidence features:

  * CalibratedRouter — produces per-input gate weights across branches.
    Replaces the naive softmax-over-archive-loglik routing that
    regresses (~10pp full / ~23pp task) because per-branch log-pdf
    magnitudes aren't comparable across senses (eye averages -128,
    taste averages -53; taste wins routing despite eye being the
    stronger predictor — see sensory_organism_stage_a memory).

  * AmbiguityHead — produces a "single look is enough" scalar in
    [0, 1]. Trained against fused-prediction-correctness. Drives
    the Stage-C escalation to pair-mode comparison.

Both heads consume the same per-branch feature bank from
``SensoryOrganism.branch_features`` so they share the forward pass:

    archive_loglik           (B, N)
    archive_top1_logpdf      (B, N)
    archive_margin           (B, N)
    head_softmax_max         (B, N)
    head_logit_margin        (B, N)

Storage: per-branch parameters keep the router structure
interpretable. Each branch learns its own rule for converting its
five raw features into a score; softmax across branches assembles
gates. Total params:
  router:  N · 5 + N            (per-branch linear scoring + bias)
  ambig:   5 · N + 1 (+ bias)   (single linear over flattened features)
For N=7 senses: router 42 params, ambig 38 params. Together ~320 B
in float32 — negligible against the 1.5 MB organism substrate.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


BRANCH_FEATURE_NAMES: List[str] = [
    "archive_loglik",
    "archive_top1_logpdf",
    "archive_margin",
    "head_softmax_max",
    "head_logit_margin",
]


def stack_branch_features(
    feat_dict: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Stack the named per-branch features into a (B, N_branches, F)
    tensor in the canonical order. Drops auxiliary fields like
    ``branch_logits_padded``."""
    cols = [feat_dict[name] for name in BRANCH_FEATURE_NAMES]
    return torch.stack(cols, dim=-1)        # (B, N, F)


# ---------------------------------------------------------------------
# CalibratedRouter — per-branch linear scorer + softmax across branches
# ---------------------------------------------------------------------


class CalibratedRouter(nn.Module):
    """Per-branch linear scorer with branch-specific weights.

    For each branch b: score_b = w_b · feature_b + bias_b.
    Gates = softmax(scores).

    Per-branch parameterization is intentional — each sense has its
    own reliability profile (eye produces wide-σ archives, taste
    narrow), so each one learns its own rule for what its raw signals
    mean. A single shared Linear would force one calibration curve
    for all senses, which is exactly the failure mode that motivated
    this head.
    """

    def __init__(
        self, n_branches: int, n_features: int = len(BRANCH_FEATURE_NAMES),
        temperature: float = 1.0,
    ):
        super().__init__()
        self.n_branches = n_branches
        self.n_features = n_features
        self.temperature = temperature
        # Per-branch weights and biases.
        self.weight = nn.Parameter(torch.zeros(n_branches, n_features))
        self.bias = nn.Parameter(torch.zeros(n_branches))
        # Initialize at near-uniform: zero weights, equal bias →
        # softmax produces 1/N gates, so the calibrator starts at
        # parity with the uniform-routing baseline.
        nn.init.normal_(self.weight, mean=0.0, std=1e-3)

    def forward(self, branch_features: torch.Tensor) -> torch.Tensor:
        """branch_features: (B, N, F). Returns gates (B, N)."""
        # Per-branch dot product: (B, N, F) * (N, F) → (B, N) sum over F.
        scores = (branch_features * self.weight.unsqueeze(0)).sum(dim=-1)
        scores = scores + self.bias.unsqueeze(0)
        return F.softmax(scores / max(self.temperature, 1e-6), dim=-1)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------
# AmbiguityHead — "single look is enough" probability
# ---------------------------------------------------------------------


class AmbiguityHead(nn.Module):
    """Logistic over flattened per-branch features + fused-top
    confidence statistics → P(single-look is sufficient).

    Fused-confidence inputs are computed by the caller from the gated
    fused logits and fed in as ``fused_aux`` (top-softmax, top1-top2
    margin). Keeping that calculation outside the module lets the
    head be reused with arbitrary fusion strategies.
    """

    def __init__(
        self,
        n_branches: int,
        n_features: int = len(BRANCH_FEATURE_NAMES),
        n_fused_aux: int = 2,
    ):
        super().__init__()
        in_dim = n_branches * n_features + n_fused_aux
        self.linear = nn.Linear(in_dim, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.n_branches = n_branches
        self.n_features = n_features
        self.n_fused_aux = n_fused_aux

    def forward(
        self,
        branch_features: torch.Tensor,
        fused_aux: torch.Tensor,
    ) -> torch.Tensor:
        """branch_features: (B, N, F), fused_aux: (B, n_fused_aux).
        Returns probabilities (B,) in [0, 1]."""
        flat = branch_features.flatten(start_dim=1)               # (B, N·F)
        x = torch.cat([flat, fused_aux], dim=-1)                  # (B, N·F + 2)
        return torch.sigmoid(self.linear(x)).squeeze(-1)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------
# Helper — fused logits + fused-confidence aux features
# ---------------------------------------------------------------------


def fuse_with_router(
    branch_logits_padded: torch.Tensor,
    gates: torch.Tensor,
    bias_offset: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """branch_logits_padded: (B, N, n_union); gates: (B, N).
    Returns fused logits (B, n_union)."""
    fused = (branch_logits_padded * gates.unsqueeze(-1)).sum(dim=1)
    if bias_offset is not None:
        fused = fused + bias_offset
    return fused


def fused_confidence_aux(fused_logits: torch.Tensor) -> torch.Tensor:
    """Top-1 softmax probability and top1-top2 logit margin of the
    fused prediction. Shape (B, 2). Used as auxiliary input to the
    ambiguity head."""
    sm = F.softmax(fused_logits, dim=-1)
    top1_p = sm.max(dim=-1).values
    top2 = torch.topk(fused_logits, k=min(2, fused_logits.shape[-1]),
                      dim=-1).values
    margin = top2[:, 0] - top2[:, 1] if top2.shape[-1] > 1 else top2[:, 0]
    return torch.stack([top1_p, margin], dim=-1)


__all__ = [
    "BRANCH_FEATURE_NAMES",
    "stack_branch_features",
    "CalibratedRouter",
    "AmbiguityHead",
    "fuse_with_router",
    "fused_confidence_aux",
]
