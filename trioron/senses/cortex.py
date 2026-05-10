"""sense_cortex — frozen ImageNet-pretrained MobileNetV3-Small features.

The "learned" sense in the bank: complements the closed-form senses
(color_smell, frequency_print, taste, random_walk) with a neural
backbone that captures the invariant gestalt features classical CV
can't. The backbone (~2.5M params) is loaded once, frozen, and used
purely as a feature extractor; trioron's substrate never sees the
weights, only the post-avgpool feature vector.

Two design choices keep cortex commensurate with the classical senses:

  1. ImageNet pretraining (no CIFAR-100 label exposure) — features
     transfer rather than memorize, so the reviewer story stays
     "perception is upstream and frozen."

  2. Random projection 576 → 64 — MobileNetV3-Small's post-avgpool
     features are 576-d, which would dominate a 33-d classical-sense
     concatenation purely by dimensionality. A deterministic-seed
     Gaussian projection brings cortex to 64-d so it enters the
     sensorium at the same scale as the four classical components.
"""
from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

CORTEX_DIM = 64
_RAW_DIM = 576              # MobileNetV3-Small post-avgpool channel count
_PROJ_SEED = 20260509       # frozen, deterministic
_INPUT_SIZE = 96            # upsample 32×32 → 96×96 before MobileNet
_CHUNK = 256                # internal batch size when caller passes a
                            # whole-dataset tensor (CIFAR-100 prep does)

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

_BACKBONE: Optional[nn.Module] = None
_PROJ: Optional[torch.Tensor] = None


def _ensure_loaded(device: torch.device, dtype: torch.dtype) -> None:
    """Lazy-load the frozen backbone and the random projection."""
    global _BACKBONE, _PROJ
    if _BACKBONE is None:
        from torchvision.models import (
            mobilenet_v3_small, MobileNet_V3_Small_Weights,
        )
        m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        m.classifier = nn.Identity()    # forward returns post-avgpool (N, 576)
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        _BACKBONE = m
    if _PROJ is None:
        g = torch.Generator().manual_seed(_PROJ_SEED)
        _PROJ = torch.randn(_RAW_DIM, CORTEX_DIM, generator=g) / (_RAW_DIM ** 0.5)
    if next(_BACKBONE.parameters()).device != device:
        _BACKBONE.to(device)
    if _PROJ.device != device or _PROJ.dtype != dtype:
        _PROJ = _PROJ.to(device=device, dtype=dtype)


def sense_cortex(images: torch.Tensor) -> torch.Tensor:
    """Apply the cortex (learned-feature) sense.

    Args:
        images: (N, 3, H, W) float32 in [0, 1]. CIFAR-style 32×32
        expected; the sense upsamples to 96×96 internally.

    Returns:
        (N, 64) float32 — random-projected MobileNetV3-Small features.
    """
    if images.dim() != 4 or images.shape[1] != 3:
        raise ValueError(
            f"sense_cortex expects (N, 3, H, W); got {tuple(images.shape)}"
        )
    device = images.device
    dtype = images.dtype
    _ensure_loaded(device, dtype)

    mean = _IMAGENET_MEAN.to(device=device, dtype=dtype)
    std = _IMAGENET_STD.to(device=device, dtype=dtype)

    out = torch.empty(images.shape[0], CORTEX_DIM, device=device, dtype=dtype)
    with torch.no_grad():
        for i in range(0, images.shape[0], _CHUNK):
            chunk = images[i:i + _CHUNK]
            if chunk.shape[-2:] != (_INPUT_SIZE, _INPUT_SIZE):
                chunk = F.interpolate(chunk, size=(_INPUT_SIZE, _INPUT_SIZE),
                                      mode="bilinear", align_corners=False)
            chunk = (chunk - mean) / std
            feats = _BACKBONE(chunk)             # (n, 576)
            out[i:i + chunk.shape[0]] = feats @ _PROJ
    return out
