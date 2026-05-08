"""sense_proprioception — directional silhouette extent.

Divides the image into 8 angular sectors around its center and reports
the fraction of total luminance in each sector. Captures shape
asymmetry and directional reach (e.g. a horizontally-elongated object
will dump mass into the E + W sectors). Orthogonal to mass_moment
(which captures inertia globally) and color_smell (color-only).
"""
from __future__ import annotations
import math
import torch

PROPRIOCEPTION_DIM = 8

_LUMA = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32)
_N_SECTORS = 8


def sense_proprioception(images: torch.Tensor) -> torch.Tensor:
    """Apply the proprioception sense.

    Args:
        images: (N, 3, 32, 32) float32 in [0, 1].

    Returns:
        (N, 8) float32 — per-image luminance fraction in each of 8
        angular sectors (sector 0 = east, increasing CCW).
    """
    if images.dim() != 4 or images.shape[1] != 3:
        raise ValueError(
            f"sense_proprioception expects (N, 3, H, W); got {tuple(images.shape)}"
        )
    N, _, H, W = images.shape
    device = images.device
    dtype = images.dtype

    luma = (images * _LUMA.to(device).view(1, 3, 1, 1)).sum(dim=1)
    luma = luma.clamp_min(0.0)

    ys = torch.arange(H, device=device, dtype=dtype).view(1, H, 1)
    xs = torch.arange(W, device=device, dtype=dtype).view(1, 1, W)
    yc = (H - 1) / 2.0
    xc = (W - 1) / 2.0
    dy = ys - yc           # y is row index (down-positive)
    dx = xs - xc

    # Angle in [0, 2*pi). atan2 takes (y, x) — flip y so up is positive.
    angle = torch.atan2(-dy.expand(1, H, W), dx.expand(1, H, W))
    angle = (angle + 2.0 * math.pi) % (2.0 * math.pi)
    sector = (angle / (2.0 * math.pi) * _N_SECTORS).long().clamp_max(_N_SECTORS - 1)
    sector = sector.expand(N, H, W)

    one_hot = torch.zeros(N, H * W, _N_SECTORS, device=device, dtype=dtype)
    one_hot.scatter_(2, sector.reshape(N, H * W, 1), 1.0)
    weighted = one_hot * luma.reshape(N, H * W, 1)
    sector_mass = weighted.sum(dim=1)        # (N, 8)
    total = sector_mass.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return sector_mass / total
