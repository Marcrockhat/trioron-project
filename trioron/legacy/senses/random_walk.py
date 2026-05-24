"""sense_random_walk — gradient-field walker statistics.

Treats the luminance gradient field as a drift force on a particle and
reports population-level statistics: net drift direction, diffusion
energy, curl, and direction entropy. A walker dropped on this field
would experience these dynamics; we report the field invariants
directly (deterministic, batch-friendly) rather than simulating walks.
"""
from __future__ import annotations
import math
import torch

RANDOM_WALK_DIM = 5

_LUMA = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32)
_N_DIR_BINS = 4


def sense_random_walk(images: torch.Tensor) -> torch.Tensor:
    """Apply the random-walk (gradient-field) sense.

    Args:
        images: (N, 3, 32, 32) float32 in [0, 1].

    Returns:
        (N, 5) float32: [drift_x, drift_y, diffusion_energy, curl,
        direction_entropy].
    """
    if images.dim() != 4 or images.shape[1] != 3:
        raise ValueError(
            f"sense_random_walk expects (N, 3, H, W); got {tuple(images.shape)}"
        )
    N, _, H, W = images.shape
    device = images.device
    dtype = images.dtype

    luma = (images * _LUMA.to(device).view(1, 3, 1, 1)).sum(dim=1)
    gx = luma[:, :, 2:] - luma[:, :, :-2]
    gy = luma[:, 2:, :] - luma[:, :-2, :]
    # Pad to consistent shape.
    gx = torch.nn.functional.pad(gx, (1, 1, 0, 0))
    gy = torch.nn.functional.pad(gy, (0, 0, 1, 1))

    drift_x = gx.mean(dim=(1, 2))
    drift_y = gy.mean(dim=(1, 2))
    diffusion_energy = (gx.pow(2) + gy.pow(2)).mean(dim=(1, 2))

    # Curl ≈ d(gy)/dx - d(gx)/dy at every pixel; report mean magnitude.
    cgx = gy[:, :, 2:] - gy[:, :, :-2]
    cgy = gx[:, 2:, :] - gx[:, :-2, :]
    cgx = torch.nn.functional.pad(cgx, (1, 1, 0, 0))
    cgy = torch.nn.functional.pad(cgy, (0, 0, 1, 1))
    curl = (cgx - cgy).abs().mean(dim=(1, 2))

    # Direction entropy: 4-bin histogram of gradient angles weighted by
    # magnitude, then Shannon entropy.
    mag = (gx.pow(2) + gy.pow(2)).sqrt().clamp_min(1e-6)
    angle = torch.atan2(gy, gx)            # [-pi, pi]
    bin_idx = ((angle + math.pi) / (2.0 * math.pi) * _N_DIR_BINS).long().clamp_max(
        _N_DIR_BINS - 1
    )
    one_hot = torch.zeros(N, H * W, _N_DIR_BINS, device=device, dtype=dtype)
    one_hot.scatter_(2, bin_idx.reshape(N, H * W, 1), 1.0)
    weighted = one_hot * mag.reshape(N, H * W, 1)
    bin_total = weighted.sum(dim=1)        # (N, 4)
    bin_total = bin_total / bin_total.sum(dim=1, keepdim=True).clamp_min(1e-6)
    entropy = -(bin_total * (bin_total.clamp_min(1e-12)).log()).sum(dim=1)

    return torch.stack([drift_x, drift_y, diffusion_energy, curl, entropy], dim=1)
