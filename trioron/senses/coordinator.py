"""Coordinator — learns to use senses *together*.

Stage B's CalibratedRouter was per-branch scalar (42 params for 7
senses) → effectively "trust eye more, trust pulse less, the same
way for every input/class". Too coarse. Sense reliability varies
*per-class*: eye is great for big-shaped animals (mammal classes),
pulse for fast-moving (vehicles, motion-coded), heat_diffusion for
texture/thermal (food, fabric). A per-(branch, class) weight
captures that.

This module also adds an optional per-image dynamic correction
driven by per-branch confidence features — Stage B's
AmbiguityHead-style signal channeled into per-class deltas instead
of one scalar gate.

Architecture:
  static_W  : (N_branches, n_classes) — init 1/N (uniform parity)
  bias      : (n_classes,)
  dyn       : optional shared Linear(n_features → n_classes)
              applied per-branch to per-branch features → (B, N, C)
              delta. Shared across branches so each branch's delta
              comes from its own features but uses one calibration
              rule (much fewer params; less overfit).

  forward(padded_logits, branch_features):
    delta_W = static_W.unsqueeze(0)              # (1, N, C)
    if dynamic:
      delta_W = delta_W + dyn(branch_features)   # (B, N, C)
    combined = (delta_W * padded_logits).sum(1)  # (B, C)
    return combined + bias

Param count for 12 senses, 100 classes:
  static : 12 × 100 + 100 = 1300
  dynamic: 5 × 100 + 100 = 600
  total  : 1900 (vs Stage B's 42)

Coordinator is initialized at exact uniform-fusion parity so
training only needs to learn deviations.
"""
from __future__ import annotations
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Coordinator(nn.Module):
    """Per-(branch, class) coordination over branch logits.

    Args:
        n_branches: number of senses fused.
        n_classes:  size of the union class layout.
        n_features: number of per-branch confidence features used
            by the dynamic head (default 5, matches
            ``trioron.senses.calibrator.BRANCH_FEATURE_NAMES``).
        dynamic:    enable the per-image correction branch.
        weight_init: 1/N (default) initializes at exact uniform-fusion
            parity. Setting to 0 starts at a "no signal" point but
            then the network has to relearn parity through training.
    """

    def __init__(
        self,
        n_branches: int,
        n_classes: int,
        n_features: int = 5,
        dynamic: bool = True,
        weight_init: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.n_branches = n_branches
        self.n_classes = n_classes
        self.n_features = n_features
        self.dynamic = dynamic
        if weight_init is None:
            weight_init = 1.0 / n_branches
        self.static_W = nn.Parameter(
            torch.full((n_branches, n_classes), weight_init)
        )
        self.bias = nn.Parameter(torch.zeros(n_classes))
        if dynamic:
            self.dyn = nn.Linear(n_features, n_classes)
            nn.init.zeros_(self.dyn.weight)
            nn.init.zeros_(self.dyn.bias)
        else:
            self.dyn = None

    def forward(
        self,
        padded_logits: torch.Tensor,
        branch_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """padded_logits: (B, N, C); branch_features: (B, N, F).

        Returns combined logits (B, C). When ``branch_features`` is
        None and dynamic=True, the dynamic head contribution is
        treated as zero (i.e., static-only forward)."""
        W = self.static_W.unsqueeze(0)                          # (1, N, C)
        if self.dynamic and branch_features is not None:
            W = W + self.dyn(branch_features)                   # (B, N, C)
        combined = (W * padded_logits).sum(dim=1)               # (B, C)
        return combined + self.bias

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


__all__ = ["Coordinator"]
