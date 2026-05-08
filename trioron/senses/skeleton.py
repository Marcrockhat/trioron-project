"""sense_skeleton — morphological-shape proxies for object structure.

True skeletonization is iterative and awkward in pure torch; this
sense uses cheaper proxies that capture what skeleton features are
diagnostic of: thresholded-area, perimeter, bounding-box extent, and
left/right and top/bottom symmetries. Together: a rough silhouette
fingerprint.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

SKELETON_DIM = 6

_LUMA = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32)


def sense_skeleton(images: torch.Tensor) -> torch.Tensor:
    """Apply the skeleton-proxy sense.

    Args:
        images: (N, 3, 32, 32) float32 in [0, 1].

    Returns:
        (N, 6) float32: [area_frac, perimeter_frac, aspect_ratio,
        compactness, sym_lr, sym_tb].
    """
    if images.dim() != 4 or images.shape[1] != 3:
        raise ValueError(
            f"sense_skeleton expects (N, 3, H, W); got {tuple(images.shape)}"
        )
    N, _, H, W = images.shape
    device = images.device
    dtype = images.dtype

    luma = (images * _LUMA.to(device).view(1, 3, 1, 1)).sum(dim=1)
    threshold = luma.mean(dim=(1, 2), keepdim=True)
    fg = (luma > threshold).to(dtype)

    area = fg.mean(dim=(1, 2))

    # Perimeter ≈ count of pixels where fg differs from at least one
    # neighbor. Sobel-like 4-neighbor edge detector.
    pad = F.pad(fg.unsqueeze(1), (1, 1, 1, 1), mode="replicate")
    edge = (pad[:, :, 1:-1, 1:-1] != pad[:, :, :-2, 1:-1]).to(dtype) + \
           (pad[:, :, 1:-1, 1:-1] != pad[:, :, 2:,  1:-1]).to(dtype) + \
           (pad[:, :, 1:-1, 1:-1] != pad[:, :, 1:-1, :-2]).to(dtype) + \
           (pad[:, :, 1:-1, 1:-1] != pad[:, :, 1:-1, 2:]).to(dtype)
    edge = (edge > 0).to(dtype).squeeze(1)
    perim = edge.mean(dim=(1, 2))

    # Bounding box of foreground.
    rows_any = fg.sum(dim=2) > 0      # (N, H)
    cols_any = fg.sum(dim=1) > 0      # (N, W)
    row_idx = torch.arange(H, device=device, dtype=dtype).view(1, H)
    col_idx = torch.arange(W, device=device, dtype=dtype).view(1, W)
    rows_present = rows_any.to(dtype)
    cols_present = cols_any.to(dtype)
    h_extent = (rows_present * row_idx).max(dim=1).values - \
               torch.where(rows_any, row_idx, torch.full_like(row_idx, H)).min(dim=1).values
    w_extent = (cols_present * col_idx).max(dim=1).values - \
               torch.where(cols_any, col_idx, torch.full_like(col_idx, W)).min(dim=1).values
    h_extent = h_extent.clamp_min(0.0)
    w_extent = w_extent.clamp_min(0.0)
    aspect = (h_extent + 1.0) / (w_extent + 1.0)

    # Compactness: 4*pi*area / perimeter^2 — disc=1, anything else < 1.
    pi = torch.tensor(3.14159265, device=device, dtype=dtype)
    compactness = (4.0 * pi * area) / (perim.pow(2).clamp_min(1e-6))

    # Symmetries: cosine similarity between left/right halves (and
    # top/bottom) of foreground mask.
    sym_lr = _flip_cos(fg, dim=2)
    sym_tb = _flip_cos(fg, dim=1)

    return torch.stack([area, perim, aspect, compactness, sym_lr, sym_tb], dim=1)


def _flip_cos(fg: torch.Tensor, dim: int) -> torch.Tensor:
    """Cosine sim between fg and its flip along `dim`."""
    a = fg.reshape(fg.shape[0], -1)
    b = fg.flip(dim).reshape(fg.shape[0], -1)
    num = (a * b).sum(dim=1)
    den = (a.pow(2).sum(dim=1).sqrt() * b.pow(2).sum(dim=1).sqrt()).clamp_min(1e-6)
    return num / den
