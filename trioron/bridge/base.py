"""Encoder protocol + L0 adapter — the structural pieces that let
non-trioron frozen models slot in as input substrates for a trioron
multi-branch organism.

The shared-L0 invariant from the absorption protocol generalizes
cleanly: any frozen encoder common across all donors and the recipient
satisfies the invariant, as long as a fixed projection brings the
encoder's output into the trioron L0 code space. When the encoder's
output dim matches L0_dim exactly, the projection collapses to
identity; otherwise we use a deterministic random projection seeded by
the L0 seed (so the projection itself is part of the shared substrate).
"""
from __future__ import annotations
import math
from typing import Protocol, runtime_checkable

import torch
import torch.nn.functional as F


@runtime_checkable
class Encoder(Protocol):
    """Stateless callable that maps raw input to a fixed-dim feature
    tensor. Implementations should be deterministic (no dropout, no
    sampling) at inference; their parameters should be frozen.

    The protocol is intentionally minimal: ``encode_dim`` reports the
    output dimensionality and ``__call__`` does the work. Bridge
    consumers should not depend on additional methods.
    """
    encode_dim: int

    def __call__(self, batch) -> torch.Tensor:
        ...


class L0Adapter:
    """Projects an encoder's output into the trioron L0 code space.

    Two modes:
      - identity:           encoder_dim == l0_dim, no projection needed.
      - random projection:  fixed seed-derived Kaiming projection, applied
                            once and frozen. The projection is part of
                            the shared substrate — sibling organisms that
                            share an L0 seed AND an encoder choice MUST
                            also share this projection.

    The adapter is itself frozen and stateless after construction; it
    takes part in inference forward only.
    """

    def __init__(
        self,
        encoder_dim: int,
        l0_dim: int,
        l0_seed: int,
        activation: str = "relu",
    ):
        self.encoder_dim = int(encoder_dim)
        self.l0_dim = int(l0_dim)
        self.l0_seed = int(l0_seed)
        self.activation = activation
        if self.encoder_dim == self.l0_dim:
            self.W = None
            self.b = None
        else:
            gen = torch.Generator().manual_seed(self.l0_seed)
            std = math.sqrt(2.0 / self.encoder_dim)
            self.W = torch.empty(
                self.l0_dim, self.encoder_dim,
            ).normal_(0.0, std, generator=gen)
            self.b = torch.zeros(self.l0_dim)

    def is_identity(self) -> bool:
        return self.W is None

    def __call__(self, e: torch.Tensor) -> torch.Tensor:
        """e: (B, encoder_dim). Returns (B, l0_dim) post-activation."""
        if self.W is None:
            z = e
        else:
            if e.dtype != self.W.dtype:
                e = e.to(self.W.dtype)
            z = F.linear(e, self.W, self.b)
        if self.activation == "relu":
            return F.relu(z)
        if self.activation == "linear":
            return z
        raise ValueError(f"Unsupported adapter activation: {self.activation}")


__all__ = ["Encoder", "L0Adapter"]
