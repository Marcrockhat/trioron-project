"""sense_color_smell — chromatic "olfactory" profile.

HSV histogram with 4 hue bins × 3 saturation bins = 12 bin fractions.
Captures dominant color characters (red-saturated, green-muted,
blue-bright, …) without caring about spatial layout. Pairs cleanly
with sense_mass_moment (which captures spatial layout but no color)
and sense_eye (which captures gestalt but in grayscale).
"""
from __future__ import annotations
import torch

COLOR_SMELL_DIM = 12  # 4 hue bins * 3 saturation bins

_N_HUE = 4
_N_SAT = 3


def _rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
    """Vectorized RGB→HSV. Input/output: (..., 3) in [0, 1].

    Hue normalized to [0, 1) (full circle), saturation/value in [0, 1].
    """
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    cmax, argmax = rgb.max(dim=-1)
    cmin = rgb.min(dim=-1).values
    delta = (cmax - cmin).clamp_min(1e-8)

    h = torch.zeros_like(cmax)
    mask_r = (argmax == 0)
    mask_g = (argmax == 1)
    mask_b = (argmax == 2)
    h = torch.where(mask_r, ((g - b) / delta) % 6.0, h)
    h = torch.where(mask_g, ((b - r) / delta) + 2.0, h)
    h = torch.where(mask_b, ((r - g) / delta) + 4.0, h)
    h = h / 6.0
    h = torch.where(cmax == cmin, torch.zeros_like(h), h)

    s = torch.where(cmax > 0, delta / cmax.clamp_min(1e-8), torch.zeros_like(cmax))
    v = cmax
    return torch.stack([h, s, v], dim=-1)


def sense_color_smell(images: torch.Tensor) -> torch.Tensor:
    """Apply the color-smell sense.

    Args:
        images: (N, 3, 32, 32) float32 in [0, 1].

    Returns:
        (N, 12) float32. Each row is a probability distribution over
        the 12 (hue, saturation) bins (sums to 1).
    """
    if images.dim() != 4 or images.shape[1] != 3:
        raise ValueError(
            f"sense_color_smell expects (N, 3, H, W); got {tuple(images.shape)}"
        )
    N, _, H, W = images.shape
    rgb = images.permute(0, 2, 3, 1).reshape(N, H * W, 3)
    hsv = _rgb_to_hsv(rgb)
    h = hsv[..., 0]
    s = hsv[..., 1]

    # Bin assignment.
    h_bin = (h * _N_HUE).long().clamp_max(_N_HUE - 1)
    s_bin = (s * _N_SAT).long().clamp_max(_N_SAT - 1)
    flat_bin = h_bin * _N_SAT + s_bin   # 0..11

    one_hot = torch.zeros(N, H * W, _N_HUE * _N_SAT, device=images.device,
                          dtype=images.dtype)
    one_hot.scatter_(2, flat_bin.unsqueeze(-1), 1.0)
    counts = one_hot.sum(dim=1)
    return counts / counts.sum(dim=1, keepdim=True).clamp_min(1.0)
