"""The differentiation testbed — 6 animals by body weight (1 feature).

Faithful to the design doc's probe (progenitor_council.md §5): each species is a
weight prior in kg. dog = Uniform(2, 85) is the *disruptor* — it spans
chihuahua→great dane, overlapping cat below and goat above, so the weight-only
Bayes partition is non-linearly separable (dog wins two disjoint intervals). A
linear readout caps ~0.79; the design's claim is that a correct council grows a
DENDRITE for dog and stays linear elsewhere, approaching the 0.852 ceiling.

Self-contained on purpose — the clean room does not import the archived
experiments. Distributions copied verbatim from the original chicken_goat probe.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

_LOG2PI = math.log(2 * math.pi)


@dataclass(frozen=True)
class Normal:
    mean: float
    std: float

    def sample(self, n: int, g: torch.Generator) -> torch.Tensor:
        return torch.randn(n, generator=g) * self.std + self.mean

    def log_prob(self, kg: torch.Tensor) -> torch.Tensor:
        z = (kg - self.mean) / self.std
        return -0.5 * (z * z) - math.log(self.std) - 0.5 * _LOG2PI


@dataclass(frozen=True)
class Uniform:
    lo: float
    hi: float

    def sample(self, n: int, g: torch.Generator) -> torch.Tensor:
        return torch.rand(n, generator=g) * (self.hi - self.lo) + self.lo

    def log_prob(self, kg: torch.Tensor) -> torch.Tensor:
        inside = (kg >= self.lo) & (kg <= self.hi)
        lp = torch.full_like(kg, float("-inf"))
        lp[inside] = -math.log(self.hi - self.lo)
        return lp


# kg priors. chicken/cat overlap (small); goat/cow/elephant cleanly separated;
# dog is the disruptor spanning the cat→goat no-man's-land.
SPECIES: dict[str, object] = {
    "chicken": Normal(2.5, 0.6),
    "cat":     Normal(4.5, 1.0),
    "dog":     Uniform(2.0, 85.0),
    "goat":    Normal(50.0, 15.0),
    "cow":     Normal(500.0, 80.0),
    "elephant": Normal(5000.0, 800.0),
}
SPECIES6: list[str] = ["chicken", "cat", "dog", "goat", "cow", "elephant"]


@dataclass
class Data:
    x: torch.Tensor       # (N, 1) standardized (log) weight — the input feature
    y: torch.Tensor       # (N,) class index
    kg: torch.Tensor      # (N,) raw weight in kg (for the Bayes ceiling)
    names: list[str]
    mean_f: float         # standardization stats (on the log feature)
    std_f: float

    def __len__(self) -> int:
        return self.x.shape[0]


def make_data(species: list[str] | None = None, n_per_class: int = 256,
              seed: int = 0, log: bool = True) -> Data:
    """Balanced N-class weight dataset. log=True applies log(kg) before
    standardizing (weight spans 3 orders of magnitude; log re-spreads the small
    classes). Monotonic → Bayes ceiling unchanged, only usability improves."""
    species = species or SPECIES6
    g = torch.Generator().manual_seed(seed)
    kg_parts, y_parts = [], []
    for ci, name in enumerate(species):
        kg_parts.append(SPECIES[name].sample(n_per_class, g).clamp_min(0.1))
        y_parts.append(torch.full((n_per_class,), ci, dtype=torch.long))
    kg = torch.cat(kg_parts)
    y = torch.cat(y_parts)
    perm = torch.randperm(kg.numel(), generator=g)
    kg, y = kg[perm], y[perm]
    feat = kg.log() if log else kg
    mean_f, std_f = float(feat.mean()), float(feat.std())
    x = ((feat - mean_f) / std_f).unsqueeze(1)
    return Data(x=x, y=y, kg=kg, names=list(species), mean_f=mean_f, std_f=std_f)


def bayes_accuracy(d: Data) -> float:
    """Information ceiling: assign each sample to the species whose TRUE kg
    distribution has highest likelihood (equal priors). No classifier beats it."""
    logp = torch.stack([SPECIES[n].log_prob(d.kg) for n in d.names], dim=1)
    return float((logp.argmax(1) == d.y).float().mean())


def bayes_per_class(d: Data) -> list[float]:
    logp = torch.stack([SPECIES[n].log_prob(d.kg) for n in d.names], dim=1)
    pred = logp.argmax(1)
    return [float((pred[d.y == c] == c).float().mean()) for c in range(len(d.names))]


if __name__ == "__main__":
    d = make_data(seed=0, n_per_class=512)
    print(f"N={len(d)} classes={d.names}")
    bpc = bayes_per_class(d)
    for ci, name in enumerate(d.names):
        w = d.kg[d.y == ci]
        print(f"  {name:>8}: mean {w.mean():8.2f} kg  range [{w.min():6.2f},{w.max():8.2f}]  "
              f"bayes recall {bpc[ci]:.3f}")
    print(f"  weight-only Bayes ceiling: {bayes_accuracy(d):.3f}")
