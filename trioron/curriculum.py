"""Parameterized contrastive curriculum.

Generalizes incubator.ContrastiveCurriculum (the hardcoded 5-pair §5.3
curriculum) to arbitrary state dimension and arbitrary pair structure.
Used by the harder-curriculum benchmark that revisits §13 past the
5-task floor — does the architecture keep working at 20+ tasks?

Each pair is described by a `ContrastivePairSpec`:

  - `a_settings` and `b_settings` are lists of (dim_idx, value) pairs
    that pin specific state dimensions to specific values on each side
    of the pair.
  - All other dimensions are sampled iid uniform per call so the network
    cannot shortcut on incidental correlations.

`build_progressive_pairs` produces a curriculum of single-dim pairs
followed by compound (two-dim XOR) pairs. The single → compound boundary
is the structural distribution shift the harder benchmark uses to stress
capacity: compound pairs cannot be solved by any single-dim projection
and force the network to compose multiple latent directions.
"""

from __future__ import annotations
from dataclasses import dataclass
import random
from typing import List, Optional, Tuple

import torch


@dataclass
class ContrastivePairSpec:
    """Spec for one contrastive task. `a_settings` and `b_settings` are
    [(dim_idx, value)] lists pinning the named dims on each side; all
    other dims sample iid uniform."""

    name: str
    a_settings: List[Tuple[int, float]]
    b_settings: List[Tuple[int, float]]


class ParameterizedContrastiveCurriculum:
    """Sample contrastive batches over an arbitrary state dimension and
    pair structure. Mirrors incubator.ContrastiveCurriculum's API
    (`sample_pair(name, batch) -> (a, b)`) so existing experiment code
    can swap one curriculum for the other without changes elsewhere."""

    def __init__(
        self,
        state_dim: int,
        pair_specs: List[ContrastivePairSpec],
        seed: Optional[int] = None,
    ):
        if state_dim < 1:
            raise ValueError("state_dim must be >= 1")
        self.state_dim = int(state_dim)
        if len(pair_specs) == 0:
            raise ValueError("Need at least one pair spec.")

        self.pair_specs = {p.name: p for p in pair_specs}
        if len(self.pair_specs) != len(pair_specs):
            raise ValueError("Duplicate pair names in pair_specs")
        for spec in pair_specs:
            for d, _ in spec.a_settings + spec.b_settings:
                if not (0 <= d < state_dim):
                    raise ValueError(
                        f"Pair {spec.name!r}: dim {d} outside [0, {state_dim})"
                    )

        self.pair_names: List[str] = [p.name for p in pair_specs]
        self._rng = torch.Generator()
        if seed is not None:
            self._rng.manual_seed(seed)

    def sample_pair(self, name: str, batch: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if name not in self.pair_specs:
            raise ValueError(
                f"Unknown pair {name!r}; valid: {list(self.pair_specs.keys())[:5]}…"
            )
        spec = self.pair_specs[name]
        a = torch.rand(batch, self.state_dim, generator=self._rng)
        b = torch.rand(batch, self.state_dim, generator=self._rng)
        for d, val in spec.a_settings:
            a[:, d] = val
        for d, val in spec.b_settings:
            b[:, d] = val
        return a, b


def build_progressive_pairs(
    state_dim: int = 16,
    n_single: int = 12,
    n_compound: int = 8,
    seed: int = 0,
    low: float = 0.1,
    high: float = 0.9,
) -> List[ContrastivePairSpec]:
    """Build n_single single-dim pairs (dims 0..n_single-1) followed by
    n_compound two-dim XOR pairs over randomly chosen dim pairs.

    Pair name format:  single_DD  /  compound_DD  (zero-padded index).
    Curriculum order = single first, compound second — that boundary is
    the structural distribution shift the bench tracks."""
    if n_single > state_dim:
        raise ValueError(
            f"n_single={n_single} > state_dim={state_dim}; need >= 1 dim per single pair"
        )
    if n_single < 0 or n_compound < 0:
        raise ValueError("n_single and n_compound must be non-negative")
    if not (0.0 <= low < high <= 1.0):
        raise ValueError("require 0 <= low < high <= 1")

    rng = random.Random(seed)
    specs: List[ContrastivePairSpec] = []

    for d in range(n_single):
        specs.append(
            ContrastivePairSpec(
                name=f"single_{d:02d}",
                a_settings=[(d, low)],
                b_settings=[(d, high)],
            )
        )

    for i in range(n_compound):
        d1 = rng.randrange(state_dim)
        d2 = rng.randrange(state_dim)
        while d2 == d1:
            d2 = rng.randrange(state_dim)
        specs.append(
            ContrastivePairSpec(
                name=f"compound_{i:02d}",
                a_settings=[(d1, low), (d2, high)],
                b_settings=[(d1, high), (d2, low)],
            )
        )

    return specs
