"""sense_sonification — raster-scan-as-sound spectral profile.

Converts the luminance image into a 1D "audio" signal by raster-scan,
then takes its 1D power spectrum and aggregates into 12 logarithmic
frequency bands. Each band = the fraction of acoustic energy in that
band. Surfaces aspects of repetition / scanline texture that the 2D
frequency_print can miss.
"""
from __future__ import annotations
import torch

SONIFICATION_DIM = 12

_LUMA = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32)
_N_BANDS = 12


def sense_sonification(images: torch.Tensor) -> torch.Tensor:
    """Apply the sonification sense.

    Args:
        images: (N, 3, 32, 32) float32 in [0, 1].

    Returns:
        (N, 12) float32 — fraction of 1D spectral energy in each of 12
        log-spaced frequency bands.
    """
    if images.dim() != 4 or images.shape[1] != 3:
        raise ValueError(
            f"sense_sonification expects (N, 3, H, W); got {tuple(images.shape)}"
        )
    N, _, H, W = images.shape
    device = images.device
    dtype = images.dtype

    luma = (images * _LUMA.to(device).view(1, 3, 1, 1)).sum(dim=1)
    audio = luma.reshape(N, H * W)
    audio = audio - audio.mean(dim=1, keepdim=True)

    spec = torch.fft.rfft(audio, dim=1)
    power = spec.abs().pow(2)        # (N, H*W//2 + 1)
    n_bins = power.shape[1]

    # Log-spaced band edges so the 12 bands span low → high freq with
    # roughly equal log-resolution. Bin 0 (DC) is dropped because we
    # already mean-subtracted.
    edges = torch.logspace(
        start=0.0, end=float(torch.log10(torch.tensor(float(n_bins - 1)))),
        steps=_N_BANDS + 1, base=10.0, device=device,
    ).long().clamp_max(n_bins - 1)
    out = torch.zeros(N, _N_BANDS, device=device, dtype=dtype)
    for b in range(_N_BANDS):
        lo, hi = int(edges[b].item()), int(edges[b + 1].item())
        if hi <= lo:
            hi = lo + 1
        out[:, b] = power[:, lo:hi].sum(dim=1)
    total = out.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return out / total
