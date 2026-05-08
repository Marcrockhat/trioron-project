"""sense_pulse — 2D autocorrelation summary.

The 2D autocorrelation of an image describes how self-similar it is
under translation. Repetitive textures (stripes, scales, fur) have
strong off-center peaks; smooth/blob images have only the central
peak. Computed via FFT (Wiener-Khinchin: autocorr = IFFT(|FFT|²)).
"""
from __future__ import annotations
import torch

PULSE_DIM = 6

_LUMA = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32)


def sense_pulse(images: torch.Tensor) -> torch.Tensor:
    """Apply the pulse (autocorrelation-summary) sense.

    Args:
        images: (N, 3, 32, 32) float32 in [0, 1].

    Returns:
        (N, 6) float32: [acorr_r1, acorr_r4, acorr_r8, ring_anisotropy,
        offpeak_strength, offpeak_radius].
    """
    if images.dim() != 4 or images.shape[1] != 3:
        raise ValueError(
            f"sense_pulse expects (N, 3, H, W); got {tuple(images.shape)}"
        )
    N, _, H, W = images.shape
    device = images.device
    dtype = images.dtype

    luma = (images * _LUMA.to(device).view(1, 3, 1, 1)).sum(dim=1)
    luma = luma - luma.mean(dim=(1, 2), keepdim=True)

    spec = torch.fft.fft2(luma)
    power = spec.abs().pow(2)
    acorr = torch.fft.ifft2(power).real
    acorr = torch.fft.fftshift(acorr, dim=(-2, -1))

    center = acorr[:, H // 2, W // 2].clamp_min(1e-6)
    acorr_norm = acorr / center.view(N, 1, 1)

    ys = torch.arange(H, device=device, dtype=dtype).view(H, 1) - H / 2
    xs = torch.arange(W, device=device, dtype=dtype).view(1, W) - W / 2
    radius = (ys.pow(2) + xs.pow(2)).sqrt()

    def _ring_mean(r0: float, r1: float) -> torch.Tensor:
        mask = (radius >= r0) & (radius < r1)
        if not mask.any():
            return torch.zeros(N, device=device, dtype=dtype)
        return acorr_norm[:, mask].mean(dim=1)

    acorr_r1 = _ring_mean(0.5, 1.5)
    acorr_r4 = _ring_mean(3.5, 4.5)
    acorr_r8 = _ring_mean(7.5, 8.5)

    # Ring anisotropy: std over ring at r=4 (high = directional texture).
    mask4 = (radius >= 3.5) & (radius < 4.5)
    if mask4.any():
        ring4 = acorr_norm[:, mask4]
        ring_anisotropy = ring4.std(dim=1, unbiased=False)
    else:
        ring_anisotropy = torch.zeros(N, device=device, dtype=dtype)

    # Off-peak: max value outside r<2; its radius.
    mask_off = (radius >= 2.0)
    flat_off = acorr_norm[:, mask_off]
    offpeak_strength = flat_off.max(dim=1).values
    flat_radius = radius[mask_off]
    idx_max = flat_off.argmax(dim=1)
    offpeak_radius = flat_radius[idx_max]

    return torch.stack([
        acorr_r1, acorr_r4, acorr_r8,
        ring_anisotropy, offpeak_strength, offpeak_radius,
    ], dim=1)
