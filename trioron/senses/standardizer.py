"""Standardizer — per-feature mean/std normalization for sense readings.

Senses return deterministic per-image features at their physical
scale. Some senses (mass_moment) have features with very different
scales (cx in [-1, 1] vs kurtosis in [-2, +30]) that would unbalance
trioron's frozen L0 random projection. A Standardizer fitted on the
training set normalizes them uniformly across train + eval and gets
saved alongside the donor checkpoint so inference applies the same
transform.
"""
from __future__ import annotations
from dataclasses import dataclass

import torch


@dataclass
class Standardizer:
    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def fit(cls, X: torch.Tensor, eps: float = 1e-6) -> "Standardizer":
        """Fit on a (N, D) feature tensor."""
        if X.dim() != 2:
            raise ValueError(f"Standardizer.fit expects 2D, got {tuple(X.shape)}")
        mu = X.mean(dim=0)
        sd = X.std(dim=0, unbiased=False).clamp_min(eps)
        return cls(mean=mu, std=sd)

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        return (X - self.mean.to(X.device)) / self.std.to(X.device)

    def to_dict(self) -> dict:
        return {"mean": self.mean.detach().cpu(), "std": self.std.detach().cpu()}

    @classmethod
    def from_dict(cls, d: dict) -> "Standardizer":
        return cls(mean=d["mean"], std=d["std"])
