"""sense_mass_moment — geometric "feel" of the image.

Treats luminance as 2D mass density and reports six inertial properties:
center of mass, principal-axis tilt, major/minor inertia eigenvalues,
luminance kurtosis. Probes shape distribution in a way completely
orthogonal to color or fine texture.
"""
from __future__ import annotations
import torch

MASS_MOMENT_DIM = 6

_LUMA = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32)


def sense_mass_moment(images: torch.Tensor) -> torch.Tensor:
    """Apply the mass-moment sense.

    Args:
        images: (N, 3, 32, 32) float32 in [0, 1].

    Returns:
        (N, 6) float32, deterministic per-image: [cx, cy, theta_n,
        lambda_major, lambda_minor, kurtosis]. cx/cy in [-1, 1]
        (normalized to image extent), theta_n in [-1, 1] (rescaled
        from ±pi/2). Eigenvalues and kurtosis are returned in their
        physical scale; fit a `Standardizer` on training data and
        apply uniformly to train+eval to put them on L0-friendly
        scale.
    """
    if images.dim() != 4 or images.shape[1] != 3:
        raise ValueError(
            f"sense_mass_moment expects (N, 3, H, W); got {tuple(images.shape)}"
        )
    N, _, H, W = images.shape
    device = images.device
    dtype = images.dtype

    luma = (images * _LUMA.to(device).view(1, 3, 1, 1)).sum(dim=1)
    luma = luma.clamp_min(1e-6)
    mass = luma.sum(dim=(1, 2))

    ys = torch.arange(H, device=device, dtype=dtype).view(1, H, 1)
    xs = torch.arange(W, device=device, dtype=dtype).view(1, 1, W)
    ys_n = (ys - (H - 1) / 2.0) / ((H - 1) / 2.0)
    xs_n = (xs - (W - 1) / 2.0) / ((W - 1) / 2.0)

    cx = (luma * xs_n).sum(dim=(1, 2)) / mass
    cy = (luma * ys_n).sum(dim=(1, 2)) / mass

    dx = xs_n - cx.view(N, 1, 1)
    dy = ys_n - cy.view(N, 1, 1)
    Ixx = (luma * dy.pow(2)).sum(dim=(1, 2)) / mass
    Iyy = (luma * dx.pow(2)).sum(dim=(1, 2)) / mass
    Ixy = -(luma * dx * dy).sum(dim=(1, 2)) / mass

    theta = 0.5 * torch.atan2(2.0 * Ixy, Ixx - Iyy)
    theta_n = theta / (torch.pi / 2.0)

    trace = Ixx + Iyy
    det = Ixx * Iyy - Ixy * Ixy
    disc = (trace.pow(2) / 4.0 - det).clamp_min(0.0).sqrt()
    lam_major = trace / 2.0 + disc
    lam_minor = trace / 2.0 - disc

    mu = luma.mean(dim=(1, 2), keepdim=True)
    sigma = luma.std(dim=(1, 2), unbiased=False, keepdim=True).clamp_min(1e-6)
    kurt = (((luma - mu) / sigma).pow(4)).mean(dim=(1, 2)) - 3.0

    return torch.stack([cx, cy, theta_n, lam_major, lam_minor, kurt], dim=1)
