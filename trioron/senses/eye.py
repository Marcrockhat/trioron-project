"""sense_eye — myopic gestalt vision.

8×8 grayscale luminance thumbnail of a 32×32×3 image. Deliberately
weak: at 8×8 a tiger and a leopard are both a spotted-orange-blob,
an apple and a peach are both a round-red-blob. The other senses
disambiguate; this one anchors gestalt.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

EYE_DIM = 64  # 8 * 8

# Standard ITU-R BT.601 luminance weights.
_LUMA = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32)


def sense_eye(images: torch.Tensor) -> torch.Tensor:
    """Apply the myopic-eye sense.

    Args:
        images: (N, 3, 32, 32) float32 in [0, 1].

    Returns:
        (N, 64) float32. Per-image z-scored so the donor's frozen
        L0 random projection sees a stable distribution.
    """
    if images.dim() != 4 or images.shape[1] != 3:
        raise ValueError(
            f"sense_eye expects (N, 3, H, W); got {tuple(images.shape)}"
        )
    luma = (images * _LUMA.to(images.device).view(1, 3, 1, 1)).sum(dim=1)
    pooled = F.avg_pool2d(luma.unsqueeze(1), kernel_size=4, stride=4).squeeze(1)
    flat = pooled.reshape(pooled.shape[0], -1)
    mean = flat.mean(dim=1, keepdim=True)
    std = flat.std(dim=1, keepdim=True).clamp_min(1e-6)
    return (flat - mean) / std
