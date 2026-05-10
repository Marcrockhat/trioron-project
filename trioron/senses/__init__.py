"""trioron.senses — hand-coded weak sensors over images, plus one
learned cortex sense, bundled into the ``sensorium`` concat vector.

Each sense maps a batch of images into a small structured vector.
Trioron's substrate never sees pixels; it only sees sense readings.
For the deployment pipeline the bundled ``sensorium`` sense is the
canonical input — one Standardizer, one L0 projection, one donor.

Each sense module exports:
  * a callable ``sense_<name>(images: (N, 3, H, W) float32 in [0, 1])
    -> (N, dim) float32``
  * an integer ``<NAME>_DIM`` giving the output dimension.

This package's registry maps short names to (callable, dim) tuples so
training scripts can iterate over senses without hardcoding imports.
"""
from __future__ import annotations
from typing import Callable, Dict, Tuple

import torch

from .cortex import sense_cortex, CORTEX_DIM
from .color_smell import sense_color_smell, COLOR_SMELL_DIM
from .frequency_print import sense_frequency_print, FREQUENCY_PRINT_DIM
from .taste import sense_taste, TASTE_DIM
from .random_walk import sense_random_walk, RANDOM_WALK_DIM
from .sensorium import (
    sense_sensorium, SENSORIUM_DIM, SENSORIUM_COMPONENTS,
    sense_classical, CLASSICAL_DIM, CLASSICAL_COMPONENTS,
)
from .standardizer import Standardizer


SenseFn = Callable[[torch.Tensor], torch.Tensor]

SENSES: Dict[str, Tuple[SenseFn, int]] = {
    "cortex":          (sense_cortex,          CORTEX_DIM),
    "color_smell":     (sense_color_smell,     COLOR_SMELL_DIM),
    "frequency_print": (sense_frequency_print, FREQUENCY_PRINT_DIM),
    "taste":           (sense_taste,           TASTE_DIM),
    "random_walk":     (sense_random_walk,     RANDOM_WALK_DIM),
    "sensorium":       (sense_sensorium,       SENSORIUM_DIM),
    "classical":       (sense_classical,       CLASSICAL_DIM),
}


def apply_sense(name: str, images: torch.Tensor) -> torch.Tensor:
    """Run the named sense over a batch of (N, 3, H, W) images."""
    if name not in SENSES:
        raise KeyError(f"unknown sense {name!r}; have: {list(SENSES)}")
    fn, _ = SENSES[name]
    return fn(images)


def sense_dim(name: str) -> int:
    """Return the output dimension of the named sense."""
    if name not in SENSES:
        raise KeyError(f"unknown sense {name!r}; have: {list(SENSES)}")
    return SENSES[name][1]


def _conductor_classes():
    # Lazy to avoid a circular import (conductor imports `apply_sense`
    # from this package). Re-exported for convenience for legacy
    # parallel-donor experiments.
    from .conductor import (
        SenseDonor, SensoryConductor, load_sense_donor, build_conductor,
    )
    return SenseDonor, SensoryConductor, load_sense_donor, build_conductor


__all__ = [
    "SENSES",
    "apply_sense",
    "sense_dim",
    "Standardizer",
    "sense_cortex", "CORTEX_DIM",
    "sense_color_smell", "COLOR_SMELL_DIM",
    "sense_frequency_print", "FREQUENCY_PRINT_DIM",
    "sense_taste", "TASTE_DIM",
    "sense_random_walk", "RANDOM_WALK_DIM",
    "sense_sensorium", "SENSORIUM_DIM", "SENSORIUM_COMPONENTS",
    "sense_classical", "CLASSICAL_DIM", "CLASSICAL_COMPONENTS",
    "SenseDonor", "SensoryConductor",
    "load_sense_donor", "build_conductor",
]


def __getattr__(name):
    if name in {"SenseDonor", "SensoryConductor",
                "load_sense_donor", "build_conductor"}:
        SenseDonor, SensoryConductor, load_sense_donor, build_conductor = (
            _conductor_classes()
        )
        return {"SenseDonor": SenseDonor, "SensoryConductor": SensoryConductor,
                "load_sense_donor": load_sense_donor,
                "build_conductor": build_conductor}[name]
    raise AttributeError(name)
