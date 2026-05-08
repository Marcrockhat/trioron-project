"""sense_heat_diffusion — image as initial temperature, sampled after
fixed-time diffusion.

Convolves luminance with a Gaussian (closed-form solution to the heat
equation at fixed time). Reports peak value, peak location, total
remaining energy after compared with raw, and 4-quadrant energy
fractions of the diffused field. Captures large-scale spatial
distribution that local-feature senses miss.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

HEAT_DIFFUSION_DIM = 8

_LUMA = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32)


def _gaussian_kernel(sigma: float, ksize: int, device, dtype) -> torch.Tensor:
    half = (ksize - 1) / 2.0
    xs = torch.arange(ksize, device=device, dtype=dtype) - half
    g = torch.exp(-(xs.pow(2)) / (2.0 * sigma * sigma))
    g = g / g.sum()
    return g.view(1, 1, 1, ksize) * g.view(1, 1, ksize, 1)


def sense_heat_diffusion(images: torch.Tensor) -> torch.Tensor:
    """Apply the heat-diffusion sense.

    Args:
        images: (N, 3, 32, 32) float32 in [0, 1].

    Returns:
        (N, 8) float32: [peak_value, peak_x, peak_y, energy_ratio,
        nw_frac, ne_frac, sw_frac, se_frac].
    """
    if images.dim() != 4 or images.shape[1] != 3:
        raise ValueError(
            f"sense_heat_diffusion expects (N, 3, H, W); got {tuple(images.shape)}"
        )
    N, _, H, W = images.shape
    device = images.device
    dtype = images.dtype

    luma = (images * _LUMA.to(device).view(1, 3, 1, 1)).sum(dim=1)
    raw_energy = luma.sum(dim=(1, 2)).clamp_min(1e-6)

    kernel = _gaussian_kernel(sigma=4.0, ksize=15, device=device, dtype=dtype)
    diffused = F.conv2d(luma.unsqueeze(1), kernel, padding=7).squeeze(1)
    diff_energy = diffused.sum(dim=(1, 2)).clamp_min(1e-6)
    energy_ratio = diff_energy / raw_energy

    # Peak.
    flat = diffused.reshape(N, -1)
    peak_val, peak_idx = flat.max(dim=1)
    peak_y = (peak_idx // W).to(dtype) / max(H - 1, 1)
    peak_x = (peak_idx %  W).to(dtype) / max(W - 1, 1)

    # Quadrant energy fractions.
    H2, W2 = H // 2, W // 2
    q_nw = diffused[:, :H2,  :W2].sum(dim=(1, 2))
    q_ne = diffused[:, :H2,  W2:].sum(dim=(1, 2))
    q_sw = diffused[:, H2:,  :W2].sum(dim=(1, 2))
    q_se = diffused[:, H2:,  W2:].sum(dim=(1, 2))
    total = (q_nw + q_ne + q_sw + q_se).clamp_min(1e-6)

    return torch.stack([
        peak_val,
        peak_x, peak_y,
        energy_ratio,
        q_nw / total, q_ne / total, q_sw / total, q_se / total,
    ], dim=1)
