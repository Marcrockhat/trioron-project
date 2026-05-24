"""sense_frequency_print — radial-band power spectrum.

2D FFT magnitude is partitioned into 8 concentric radial annuli; each
sense feature is the total magnitude in one annulus, normalized by
total spectral energy. Probes texture frequency content: smooth-blob
images concentrate energy in low-freq bands; spotted/striped images
push energy into mid/high bands.
"""
from __future__ import annotations
import torch

FREQUENCY_PRINT_DIM = 8

_LUMA = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32)
_N_BANDS = 8


def sense_frequency_print(images: torch.Tensor) -> torch.Tensor:
    """Apply the frequency-print sense.

    Args:
        images: (N, 3, 32, 32) float32 in [0, 1].

    Returns:
        (N, 8) float32 — per-image magnitude fraction in each radial
        annulus (band 0 = lowest freq, band 7 = highest), summing to 1.
    """
    if images.dim() != 4 or images.shape[1] != 3:
        raise ValueError(
            f"sense_frequency_print expects (N, 3, H, W); got {tuple(images.shape)}"
        )
    N, _, H, W = images.shape
    device = images.device
    dtype = images.dtype

    luma = (images * _LUMA.to(device).view(1, 3, 1, 1)).sum(dim=1)
    spec = torch.fft.fftshift(torch.fft.fft2(luma), dim=(-2, -1))
    mag = spec.abs()

    ys = torch.arange(H, device=device, dtype=dtype).view(H, 1)
    xs = torch.arange(W, device=device, dtype=dtype).view(1, W)
    dy = ys - (H - 1) / 2.0
    dx = xs - (W - 1) / 2.0
    radius = (dy.pow(2) + dx.pow(2)).sqrt()
    max_r = float(radius.max().item()) + 1e-6
    band = (radius / max_r * _N_BANDS).long().clamp_max(_N_BANDS - 1)
    band_flat = band.reshape(-1)

    out = torch.zeros(N, _N_BANDS, device=device, dtype=dtype)
    mag_flat = mag.reshape(N, -1)
    for b in range(_N_BANDS):
        mask = (band_flat == b)
        out[:, b] = mag_flat[:, mask].sum(dim=1)
    total = out.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return out / total
