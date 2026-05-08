"""trioron.senses — hand-coded weak sensors over images.

Each sense is a deliberately information-lossy transform that maps a
batch of images into a small structured vector. trioron's substrate
never sees pixels; it only sees sense readings. The conductor (an
absorbed multi-branch organism) fuses readings from multiple senses,
each contributing partial evidence — the blind-men-and-elephant model.

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

from .eye import sense_eye, EYE_DIM
from .mass_moment import sense_mass_moment, MASS_MOMENT_DIM
from .color_smell import sense_color_smell, COLOR_SMELL_DIM
from .proprioception import sense_proprioception, PROPRIOCEPTION_DIM
from .frequency_print import sense_frequency_print, FREQUENCY_PRINT_DIM
from .sonification import sense_sonification, SONIFICATION_DIM
from .taste import sense_taste, TASTE_DIM
from .heat_diffusion import sense_heat_diffusion, HEAT_DIFFUSION_DIM
from .random_walk import sense_random_walk, RANDOM_WALK_DIM
from .skeleton import sense_skeleton, SKELETON_DIM
from .pulse import sense_pulse, PULSE_DIM
from .echolocation import sense_echolocation, ECHOLOCATION_DIM
from .standardizer import Standardizer


SenseFn = Callable[[torch.Tensor], torch.Tensor]

SENSES: Dict[str, Tuple[SenseFn, int]] = {
    "eye":              (sense_eye,              EYE_DIM),
    "mass_moment":      (sense_mass_moment,      MASS_MOMENT_DIM),
    "color_smell":      (sense_color_smell,      COLOR_SMELL_DIM),
    "proprioception":   (sense_proprioception,   PROPRIOCEPTION_DIM),
    "frequency_print":  (sense_frequency_print,  FREQUENCY_PRINT_DIM),
    "sonification":     (sense_sonification,     SONIFICATION_DIM),
    "taste":            (sense_taste,            TASTE_DIM),
    "heat_diffusion":   (sense_heat_diffusion,   HEAT_DIFFUSION_DIM),
    "random_walk":      (sense_random_walk,      RANDOM_WALK_DIM),
    "skeleton":         (sense_skeleton,         SKELETON_DIM),
    "pulse":            (sense_pulse,            PULSE_DIM),
    "echolocation":     (sense_echolocation,     ECHOLOCATION_DIM),
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
    # from this package). Re-exported for convenience.
    from .conductor import (
        SenseDonor, SensoryConductor, load_sense_donor, build_conductor,
    )
    return SenseDonor, SensoryConductor, load_sense_donor, build_conductor


__all__ = [
    "SENSES",
    "apply_sense",
    "sense_dim",
    "Standardizer",
    "sense_eye", "EYE_DIM",
    "sense_mass_moment", "MASS_MOMENT_DIM",
    "sense_color_smell", "COLOR_SMELL_DIM",
    "sense_proprioception", "PROPRIOCEPTION_DIM",
    "sense_frequency_print", "FREQUENCY_PRINT_DIM",
    "sense_sonification", "SONIFICATION_DIM",
    "sense_taste", "TASTE_DIM",
    "sense_heat_diffusion", "HEAT_DIFFUSION_DIM",
    "sense_random_walk", "RANDOM_WALK_DIM",
    "sense_skeleton", "SKELETON_DIM",
    "sense_pulse", "PULSE_DIM",
    "sense_echolocation", "ECHOLOCATION_DIM",
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
