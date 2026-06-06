"""Animals-by-weight: a one-feature N-class classifier dataset.

A growth probe.  Each animal class is a Gaussian over body weight (kg).  Some
pairs separate cleanly by weight (chicken vs goat vs cow); others DO NOT (chicken
vs cat overlap heavily).  This lets us watch what a growing substrate does when a
class distinction is well-supported by the feature vs when it is information-
theoretically impossible from weight alone -- capacity cannot add information the
input lacks, so growth should not rescue an unseparable pair.

Feature is standardized (zero mean, unit std over the generated sample) so the
substrate sees a well-conditioned input regardless of the raw kg scale.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

# Per-species body-weight priors (kg): (mean, std).  Realistic-ish adult weights.
# chicken & cat overlap (both small); goat & cow are well-separated.
SPECIES = {
    "chicken": (2.5, 0.6),
    "cat":     (4.5, 1.0),
    "goat":    (50.0, 15.0),
    "cow":     (500.0, 80.0),
}

# Default 2-class problem, kept for the binary exercise.
CHICKEN = 0
GOAT = 1


@dataclass
class Animals:
    """A generated animals sample.  x = standardized weight, y = class index,
    kg = raw weight, names = class index -> species name, plus norm stats."""
    x: torch.Tensor          # (N, 1) standardized weight
    y: torch.Tensor          # (N,) long class index
    kg: torch.Tensor         # (N,) raw weight in kg
    names: list[str]         # class index -> species name
    mean_kg: float
    std_kg: float

    @property
    def n_classes(self) -> int:
        return len(self.names)

    def __len__(self) -> int:
        return self.x.shape[0]


def make_animals(
    species: list[str] | None = None,
    n_per_class: int = 256,
    seed: int = 0,
    log: bool = False,
) -> Animals:
    """Generate a balanced N-class animals-by-weight dataset.

    `species` is a list of names into SPECIES; defaults to chicken+goat (the
    original binary problem).  Weights are per-class Gaussians clamped to a
    plausible floor, then standardized across the whole sample.

    `log=True` applies log(kg) BEFORE standardizing.  Weight spans 3 orders of
    magnitude (chicken ~2 kg, cow ~500 kg), so a single global std is dominated
    by cows and crushes the small animals together (chicken/cat gap ~0.01 in std
    space).  Log compresses the heavy tail and re-spreads the light classes, so
    the feature the substrate sees preserves the small-animal separation.  The
    transform is monotonic -> the Bayes ceiling is unchanged; only usability is.
    """
    species = species or ["chicken", "goat"]
    g = torch.Generator().manual_seed(seed)

    kg_parts, y_parts = [], []
    for ci, name in enumerate(species):
        mean, std = SPECIES[name]
        w = (torch.randn(n_per_class, generator=g) * std + mean).clamp_min(0.1)
        kg_parts.append(w)
        y_parts.append(torch.full((n_per_class,), ci, dtype=torch.long))

    kg = torch.cat(kg_parts)
    y = torch.cat(y_parts)

    perm = torch.randperm(kg.numel(), generator=g)
    kg, y = kg[perm], y[perm]

    feat = kg.log() if log else kg
    mean_f = float(feat.mean())
    std_f = float(feat.std())
    x = ((feat - mean_f) / std_f).unsqueeze(1)  # (N, 1)

    return Animals(x=x, y=y, kg=kg, names=list(species), mean_kg=mean_f, std_kg=std_f)


def make_chicken_goat(n_per_class: int = 256, seed: int = 0) -> Animals:
    """The original binary problem (chicken vs goat)."""
    return make_animals(["chicken", "goat"], n_per_class=n_per_class, seed=seed)


def bayes_accuracy(d: Animals) -> float:
    """Best achievable accuracy on this sample: assign each example to the class
    whose TRUE generating Gaussian (in kg space) has highest likelihood, with
    equal priors.  This is the information ceiling given weight as the only
    feature -- no classifier (grown or not) can beat it."""
    log2pi = math.log(2 * math.pi)
    # (N, C) log-likelihood under each species' true Gaussian.
    logp = []
    for name in d.names:
        mean, std = SPECIES[name]
        z = (d.kg - mean) / std
        logp.append(-0.5 * (z * z) - math.log(std) - 0.5 * log2pi)
    logp = torch.stack(logp, dim=1)
    pred = logp.argmax(dim=1)
    return float((pred == d.y).float().mean())


def _describe(d: Animals) -> None:
    print(f"N = {len(d)}  classes = {d.names}")
    for ci, name in enumerate(d.names):
        w = d.kg[d.y == ci]
        print(f"  {name:>8} (class {ci}): mean {w.mean():6.2f} kg  std {w.std():6.2f}  "
              f"range [{w.min():6.2f}, {w.max():7.2f}]")
    print(f"  standardized x: mean {d.x.mean():.3f}  std {d.x.std():.3f}")
    print(f"  Bayes-optimal accuracy (weight-only ceiling): {bayes_accuracy(d):.3f}")


if __name__ == "__main__":
    print("=== binary: chicken vs goat ===")
    _describe(make_animals(["chicken", "goat"], seed=0))
    print("\n=== 4-class: chicken, cat, goat, cow ===")
    _describe(make_animals(["chicken", "cat", "goat", "cow"], seed=0))
