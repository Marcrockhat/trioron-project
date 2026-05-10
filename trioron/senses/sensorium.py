"""sense_sensorium — concatenated multi-modal feature vector.

A single sense that bundles the kept components into one input vector:

    cortex            (64)   learned ImageNet features, random-projected
    color_smell       (12)   global hue × saturation histogram
    frequency_print    (8)   2D FFT radial bands (texture frequency)
    taste              (8)   per-quadrant chromatic vectors (spatial color)
    random_walk        (5)   gradient-field statistics (edge dynamics)
                      ----
    SENSORIUM_DIM     = 97

The bundle replaces the per-sense parallel-donor + conductor pipeline:
trioron now consumes one (N, 97) feature tensor through a single
Standardizer + L0 random projection + substrate, so signal isn't lost
to per-donor dimensionality residuals at fusion time.
"""
from __future__ import annotations
from typing import Tuple

import torch

from .cortex import sense_cortex, CORTEX_DIM
from .color_smell import sense_color_smell, COLOR_SMELL_DIM
from .frequency_print import sense_frequency_print, FREQUENCY_PRINT_DIM
from .taste import sense_taste, TASTE_DIM
from .random_walk import sense_random_walk, RANDOM_WALK_DIM

SENSORIUM_COMPONENTS: Tuple[str, ...] = (
    "cortex", "color_smell", "frequency_print", "taste", "random_walk",
)
SENSORIUM_DIM = (
    CORTEX_DIM + COLOR_SMELL_DIM + FREQUENCY_PRINT_DIM
    + TASTE_DIM + RANDOM_WALK_DIM
)
assert SENSORIUM_DIM == 97, f"sensorium dim drift: {SENSORIUM_DIM}"

CLASSICAL_COMPONENTS: Tuple[str, ...] = (
    "color_smell", "frequency_print", "taste", "random_walk",
)
CLASSICAL_DIM = (
    COLOR_SMELL_DIM + FREQUENCY_PRINT_DIM + TASTE_DIM + RANDOM_WALK_DIM
)
assert CLASSICAL_DIM == 33, f"classical dim drift: {CLASSICAL_DIM}"


def sense_sensorium(images: torch.Tensor) -> torch.Tensor:
    """Apply the bundled sensorium sense.

    Args:
        images: (N, 3, H, W) float32 in [0, 1].

    Returns:
        (N, 97) float32 — concatenation of cortex, color_smell,
        frequency_print, taste, random_walk in that order.
    """
    if images.dim() != 4 or images.shape[1] != 3:
        raise ValueError(
            f"sense_sensorium expects (N, 3, H, W); got {tuple(images.shape)}"
        )
    parts = [
        sense_cortex(images),
        sense_color_smell(images),
        sense_frequency_print(images),
        sense_taste(images),
        sense_random_walk(images),
    ]
    return torch.cat(parts, dim=1)


def sense_classical(images: torch.Tensor) -> torch.Tensor:
    """Apply the four classical (non-cortex) senses concatenated.

    Args:
        images: (N, 3, H, W) float32 in [0, 1].

    Returns:
        (N, 33) float32 — concatenation of color_smell, frequency_print,
        taste, random_walk in that order. The ablation that removes the
        learned cortex sense to isolate classical-CV contribution.
    """
    if images.dim() != 4 or images.shape[1] != 3:
        raise ValueError(
            f"sense_classical expects (N, 3, H, W); got {tuple(images.shape)}"
        )
    parts = [
        sense_color_smell(images),
        sense_frequency_print(images),
        sense_taste(images),
        sense_random_walk(images),
    ]
    return torch.cat(parts, dim=1)
