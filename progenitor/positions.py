"""Seeded sensor positions — the *spatial S* (shared handshake coordinate basis).

The R·S weight handshake (memory: l0_subspace_factor_trump_card) aligns two
organisms' *weights* by a shared subspace S + a per-organism rotation R. This is
the same idea one layer down, for *positions*: a shared seed defines a shared
coordinate field, so two organisms drawing from it place their sensors at the
SAME points → graft/absorption-aligned by construction.

Positions are drawn from a Gaussian and mapped into the arena's [0,1]³ box via
the normal CDF, which keeps the center-dense ("foveated" / retinal §3.2) shape.
For d-dimensional data the first d axes carry the draw; unused axes sit at 0.5.

NOTE (deferred, per the build plan): this aligns *birth* positions. Once
organisms grow/recycle independently the draw order diverges; keeping alignment
stable under growth is the next handshake problem (the R·S analog of aligning at
the basis level rather than per-index).
"""
from __future__ import annotations

import math

import torch


def sensor_positions(n: int, seed: int, dim: int = 1) -> torch.Tensor:
    """n sensor positions in [0,1]³ from a seeded Gaussian field (the spatial S).

    Same (n, seed) → identical positions across organisms = handshake-aligned.
    """
    g = torch.Generator().manual_seed(seed)
    raw = torch.randn(n, 3, generator=g)
    pos = 0.5 * (1.0 + torch.erf(raw / math.sqrt(2.0)))  # normal CDF → (0,1), center-dense
    if dim < 3:
        pos[:, dim:] = 0.5
    return pos


if __name__ == "__main__":
    a = sensor_positions(5, seed=0xC0FFEE)
    b = sensor_positions(5, seed=0xC0FFEE)
    c = sensor_positions(5, seed=0xBADBAD)
    print("same seed identical (handshake-aligned):", torch.allclose(a, b))
    print("different seed differs:", not torch.allclose(a, c))
    print("positions (x along the 1-D axis, y=z=0.5):")
    for i, p in enumerate(a.tolist()):
        print(f"  sensor {i}: ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})")
