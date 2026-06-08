"""10-species taxonomy dataset — lean 4 features (2 continuous + 1 discrete + 1 binary).

Extends the weight-only animals probe to a small DISCOVERABLE taxonomy (bird / mammal /
reptile). Lean feature set chosen for the first continual-learning test:

  weight   (continuous, kg)   per-species Normal/Uniform  -- keeps the dog disruptor
  height   (continuous, cm)   per-species Normal/Uniform  -- a correlated-but-distinct axis
  cover    (categorical x4)   feathers/fur/skin/scales    -- clean taxonomic backbone
  eggs     (binary)           0/1                          -- birds+turtle vs mammals

Discrete features are near-deterministic (small label noise) so they pin the GROUP while
weight/height separate WITHIN a group. All features are generated independently per class,
so the naive-Bayes ceiling (sum of per-feature log-likelihoods) is EXACT and cheap.

NN feature vector (7-d): [log-weight_z, log-height_z, onehot(cover)[4], eggs].
Bayes ceiling is computed in the RAW kg/cm space (where the generators are defined).

Run: python3 -m experiments.growth_exercise.taxonomy
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from experiments.growth_exercise.chicken_goat import Normal, Uniform

COVERS = ["feathers", "fur", "skin", "scales"]
P_COVER = 0.95   # prob the observed cover matches the species' true cover (else uniform other)
P_EGGS = 0.98    # prob the observed eggs bit matches truth


@dataclass
class Species:
    weight: object   # Normal | Uniform  (kg)
    height: object   # Normal | Uniform  (cm)
    cover: int       # index into COVERS
    eggs: int        # 0 / 1
    group: str       # taxonomic group (for reporting only)


SPECIES: dict[str, Species] = {
    # Birds
    "chicken":  Species(Normal(2.5, 0.6),    Normal(40, 5),     0, 1, "bird"),
    "duck":     Species(Normal(2.0, 0.4),    Normal(35, 5),     0, 1, "bird"),
    "canary":   Species(Normal(0.02, 0.005), Normal(12, 2),     0, 1, "bird"),
    # Mammals
    "cat":      Species(Normal(4.5, 1.0),    Normal(25, 4),     1, 0, "mammal"),
    "dog":      Species(Uniform(2, 85),      Uniform(15, 90),   1, 0, "mammal"),
    "goat":     Species(Normal(50, 15),      Normal(70, 10),    1, 0, "mammal"),
    "cow":      Species(Normal(500, 80),     Normal(150, 15),   1, 0, "mammal"),
    "pig":      Species(Normal(120, 30),     Normal(70, 12),    2, 0, "mammal"),
    "elephant": Species(Normal(5000, 800),   Normal(300, 30),   2, 0, "mammal"),
    # Reptile
    "turtle":   Species(Normal(4, 1.5),      Normal(15, 4),     3, 1, "reptile"),
}

ALL = list(SPECIES.keys())


@dataclass
class Taxon:
    X: torch.Tensor          # (N, 7) standardized NN features
    y: torch.Tensor          # (N,) class index into `names`
    kg: torch.Tensor         # (N,) raw weight
    cm: torch.Tensor         # (N,) raw height
    cover: torch.Tensor      # (N,) observed cover index
    eggs: torch.Tensor       # (N,) observed eggs bit
    names: list[str]
    norm: tuple              # (wmean, wstd, hmean, hstd) in log space

    def __len__(self) -> int:
        return self.X.shape[0]


def _cat_log_prob(obs: torch.Tensor, true_val: int, n_cat: int, p: float) -> torch.Tensor:
    """log P(observed | species) for a near-deterministic categorical: p on the true
    value, (1-p) spread uniformly over the other n_cat-1 values."""
    other = (1.0 - p) / (n_cat - 1)
    return torch.where(obs == true_val,
                       torch.tensor(math.log(p)),
                       torch.tensor(math.log(other)))


def make_taxonomy(species: list[str] | None = None, n_per_class: int = 256,
                  seed: int = 0) -> Taxon:
    species = species or ALL
    g = torch.Generator().manual_seed(seed)

    kg_p, cm_p, cov_p, egg_p, y_p = [], [], [], [], []
    for ci, name in enumerate(species):
        s = SPECIES[name]
        kg_p.append(s.weight.sample(n_per_class, g).clamp_min(0.005))
        cm_p.append(s.height.sample(n_per_class, g).clamp_min(1.0))
        # categorical cover with label noise
        u = torch.rand(n_per_class, generator=g)
        cov = torch.full((n_per_class,), s.cover, dtype=torch.long)
        flip = u > P_COVER
        if flip.any():
            alts = torch.randint(0, len(COVERS), (int(flip.sum()),), generator=g)
            cov[flip] = alts
        cov_p.append(cov)
        # binary eggs with noise
        e = torch.full((n_per_class,), s.eggs, dtype=torch.long)
        ef = torch.rand(n_per_class, generator=g) > P_EGGS
        e[ef] = 1 - e[ef]
        egg_p.append(e)
        y_p.append(torch.full((n_per_class,), ci, dtype=torch.long))

    kg = torch.cat(kg_p); cm = torch.cat(cm_p)
    cover = torch.cat(cov_p); eggs = torch.cat(egg_p); y = torch.cat(y_p)

    perm = torch.randperm(kg.numel(), generator=g)
    kg, cm, cover, eggs, y = kg[perm], cm[perm], cover[perm], eggs[perm], y[perm]

    lw, lh = kg.log(), cm.log()
    wmean, wstd = float(lw.mean()), float(lw.std())
    hmean, hstd = float(lh.mean()), float(lh.std())
    onehot = torch.nn.functional.one_hot(cover, len(COVERS)).float()
    X = torch.cat([((lw - wmean) / wstd).unsqueeze(1),
                   ((lh - hmean) / hstd).unsqueeze(1),
                   onehot,
                   eggs.float().unsqueeze(1)], dim=1)
    return Taxon(X, y, kg, cm, cover, eggs, list(species), (wmean, wstd, hmean, hstd))


def bayes_accuracy(d: Taxon, use: tuple[str, ...] = ("weight", "height", "cover", "eggs")) -> float:
    """Exact naive-Bayes ceiling on this sample using the chosen feature subset
    (independent features, equal priors -> joint factorizes)."""
    logp = torch.zeros(len(d), len(d.names))
    for ci, name in enumerate(d.names):
        s = SPECIES[name]
        lp = torch.zeros(len(d))
        if "weight" in use: lp = lp + s.weight.log_prob(d.kg)
        if "height" in use: lp = lp + s.height.log_prob(d.cm)
        if "cover" in use:  lp = lp + _cat_log_prob(d.cover, s.cover, len(COVERS), P_COVER)
        if "eggs" in use:   lp = lp + _cat_log_prob(d.eggs, s.eggs, 2, P_EGGS)
        logp[:, ci] = lp
    return float((logp.argmax(1) == d.y).float().mean())


def main() -> None:
    d = make_taxonomy(n_per_class=512, seed=1)
    print(f"{len(d.names)} species, {len(d)} samples, feature dim = {d.X.shape[1]}")
    print(f"groups: " + ", ".join(f"{n}({SPECIES[n].group[0]})" for n in d.names))
    print()
    print("Bayes ceiling as features accumulate (shows each feature's lift):")
    for use in [("weight",), ("weight", "height"), ("weight", "height", "cover"),
                ("weight", "height", "cover", "eggs")]:
        acc = bayes_accuracy(d, use)
        print(f"  {'+'.join(use):28} {acc:.3f}")


if __name__ == "__main__":
    main()
