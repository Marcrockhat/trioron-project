"""sense_echolocation — 16-angle radial luminance profile.

For each of 16 angles around the image center, sample luminance along
the ray at increasing radii and integrate. The result is a polar
profile — the "depth signature" you'd get from a 16-direction sonar.
Detects directional asymmetry / silhouette fingerprint independently
of overall mass distribution.
"""
from __future__ import annotations
import math
import torch

ECHOLOCATION_DIM = 16

_LUMA = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32)
_N_ANGLES = 16
_N_RADII = 14   # ~half image extent


def _build_sample_coords(H: int, W: int, device, dtype):
    """Return (n_angles*n_radii, 2) tensor of (y, x) sample positions."""
    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0
    angles = torch.arange(_N_ANGLES, device=device, dtype=dtype) * \
             (2.0 * math.pi / _N_ANGLES)
    radii = torch.arange(1, _N_RADII + 1, device=device, dtype=dtype)
    a, r = torch.meshgrid(angles, radii, indexing="ij")
    y = cy + r * torch.sin(a)
    x = cx + r * torch.cos(a)
    return y.flatten(), x.flatten()


def sense_echolocation(images: torch.Tensor) -> torch.Tensor:
    """Apply the echolocation sense.

    Args:
        images: (N, 3, 32, 32) float32 in [0, 1].

    Returns:
        (N, 16) float32 — per-angle integrated luminance, normalized
        to sum to 1.
    """
    if images.dim() != 4 or images.shape[1] != 3:
        raise ValueError(
            f"sense_echolocation expects (N, 3, H, W); got {tuple(images.shape)}"
        )
    N, _, H, W = images.shape
    device = images.device
    dtype = images.dtype

    luma = (images * _LUMA.to(device).view(1, 3, 1, 1)).sum(dim=1)

    ys, xs = _build_sample_coords(H, W, device, dtype)
    # Bilinear sample: floor + frac.
    y0 = ys.floor().long().clamp(0, H - 1)
    y1 = (y0 + 1).clamp(0, H - 1)
    x0 = xs.floor().long().clamp(0, W - 1)
    x1 = (x0 + 1).clamp(0, W - 1)
    fy = (ys - y0.to(dtype)).clamp(0.0, 1.0)
    fx = (xs - x0.to(dtype)).clamp(0.0, 1.0)

    samples = (
        luma[:, y0, x0] * ((1 - fy) * (1 - fx)) +
        luma[:, y0, x1] * ((1 - fy) * fx) +
        luma[:, y1, x0] * (fy * (1 - fx)) +
        luma[:, y1, x1] * (fy * fx)
    )                                            # (N, n_angles*n_radii)
    samples = samples.reshape(N, _N_ANGLES, _N_RADII)
    profile = samples.sum(dim=2)                  # (N, n_angles)
    total = profile.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return profile / total
