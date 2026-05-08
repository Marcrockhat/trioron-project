"""sense_taste — per-quadrant chromatic vector.

Divides the image into 4 quadrants (NW, NE, SW, SE) and reports the
hue-vector mean (cos(2π·H), sin(2π·H)) per quadrant — a circular-
mean-friendly representation of dominant color direction. 4 quadrants
× 2 components = 8 features. Spatial color layout signal that
color_smell (which sums over the whole image) cannot see.
"""
from __future__ import annotations
import torch

from .color_smell import _rgb_to_hsv

TASTE_DIM = 8


def sense_taste(images: torch.Tensor) -> torch.Tensor:
    """Apply the per-quadrant chromatic-vector sense.

    Args:
        images: (N, 3, 32, 32) float32 in [0, 1].

    Returns:
        (N, 8) float32 — [cos_NW, sin_NW, cos_NE, sin_NE, cos_SW,
        sin_SW, cos_SE, sin_SE], each weighted by quadrant saturation
        so achromatic quadrants contribute small magnitude.
    """
    if images.dim() != 4 or images.shape[1] != 3:
        raise ValueError(
            f"sense_taste expects (N, 3, H, W); got {tuple(images.shape)}"
        )
    N, _, H, W = images.shape
    H2, W2 = H // 2, W // 2

    # (NW, NE, SW, SE) — top-left, top-right, bot-left, bot-right.
    quadrants = [
        images[:, :, :H2,  :W2],
        images[:, :, :H2,  W2:],
        images[:, :, H2:,  :W2],
        images[:, :, H2:,  W2:],
    ]

    feats = []
    for q in quadrants:
        # (N, H/2, W/2, 3) for HSV conversion.
        hsv = _rgb_to_hsv(q.permute(0, 2, 3, 1).reshape(N, -1, 3))
        h = hsv[..., 0]                   # (N, H/2 * W/2)
        s = hsv[..., 1]
        # Saturation-weighted circular mean of hue.
        cos = torch.cos(2.0 * torch.pi * h) * s
        sin = torch.sin(2.0 * torch.pi * h) * s
        denom = s.sum(dim=1).clamp_min(1e-6)
        feats.append(cos.sum(dim=1) / denom)
        feats.append(sin.sum(dim=1) / denom)
    return torch.stack(feats, dim=1)
