"""The manifold adapter for PCLL — μ/σ sketches over pocket space. [D15]

See docs/design/mixed_stream_growth.md §5 (approved, Rocky s032). The
fifth feature of the rebuild: reuse `trioron.learning.manifold`'s
`ManifoldAstrocyte` (spec §4.5, the headline-CL sketch) over POCKET
space (code_dim = the controller's read width, receptor + composed
dims). Each learned class's sketch RIDES ITS OWN astrocyte arena row
(one row per class, D5/D11 — existence, weight and lineage stay
arena-visible; no second bookkeeping row).

Update: at each boundary the meeting feeds the period's per-class
member pockets (canonical frame) to that class's sketch — streaming
Welford μ/σ; the signature already holds first moments, the sketch
adds spread, over the FULL deposit history (a rolling buffer holds
only the last BUF members; the sketch remembers everything since the
class was born — that asymmetry is what the consumers buy).

Consumers (design §5, priority order):

  1. ANNEALING — after divisions self-arrest, `samples()` synthesizes
     deposits that re-anchor the rolling buffers (and hence the
     buffer-mean templates) against membership-pollution drift — the
     slow late decay of the s031 freeze run.
  2. σ-READOUT — `log_likelihood()` replaces the raw equal-weight
     matched filter: per-dim evidence weighted by the class's own
     spread (a tight dim testifies loudly, a wide one self-discounts)
     — the measured raw→Bayes-ceiling half of the s031 gap.
  3. REPLAY GUARD — `replay()` exposes per-class pseudo-deposits for
     future gradient training over grown tissue (spec §4.5 semantics);
     `state_dict()` round-trips through ship/wake.

Width changes (composer spawn/prune reshape the read space): sketches
REBUILD from the class's rolling buffer at the new width — history
beyond the buffer is traded for consistency; recorded, not hidden.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch

from trioron.core.receptor import N_QUANTA
from trioron.learning.manifold import ManifoldAstrocyte

SIGMA_FLOOR = 2.0    # pockets — quantization grain; a near-constant dim
                     # must not collapse the likelihood (1e-6 σ explodes)
ANNEAL_N = 40        # synthetic deposits injected per quiescent boundary
MIN_SKETCH = 60      # don't consume a sketch below this many members
                     # (= division's MIN_MEMBERS: judgable evidence)


class PCLLManifold:
    """Per-class pocket-space sketches keyed by class NAME, each backed
    by the class's existing astrocyte arena row."""

    def __init__(self, arena) -> None:
        self.arena = arena
        self.sketches: Dict[str, ManifoldAstrocyte] = {}
        self._births = 0

    # ── lifecycle ─────────────────────────────────────────────────

    def adopt(self, name: str, cell_id: int, width: int) -> ManifoldAstrocyte:
        """A class exists (birth or division child) — give it a sketch
        on its OWN astrocyte row."""
        self._births += 1
        astro = ManifoldAstrocyte(self.arena, cell_id, self._births, width)
        self.sketches[name] = astro
        return astro

    def retire(self, name: str) -> None:
        self.sketches.pop(name, None)

    def update(self, name: str, q: torch.Tensor) -> None:
        """Feed a class's REAL window members (canonical pockets [n, W])
        — synthetic deposits never update the sketch."""
        astro = self.sketches.get(name)
        if astro is not None and len(q):
            astro.update(q)

    def rebuild(self, name: str, cell_id: int, q: torch.Tensor) -> None:
        """Re-derive a sketch from a buffer's pockets (division children;
        width changes). The buffer is the best available history."""
        astro = self.adopt(name, cell_id, q.shape[1])
        if len(q):
            astro.update(q)

    # ── consumer 1: annealing ─────────────────────────────────────

    def anneal_phasors(self, name: str, n: int = ANNEAL_N
                       ) -> Optional[torch.Tensor]:
        """Synthesize n pseudo-deposits as PHASORS for the class's
        rolling buffer — the re-anchor against membership-pollution
        drift. None below the evidence floor."""
        astro = self.sketches.get(name)
        if astro is None or astro._n < MIN_SKETCH:
            return None
        q = astro.sample(n).clamp(1, N_QUANTA - 1)
        return torch.exp(1j * 2 * math.pi * q / N_QUANTA)

    # ── consumer 2: σ-weighted readout ────────────────────────────

    def log_likelihood(self, q: torch.Tensor, names: List[str]
                       ) -> torch.Tensor:
        """[B, C] diagonal-Gaussian log p(pockets | class) with the
        pocket-grain σ floor; −inf where no consumable sketch exists
        (the caller falls back to the raw filter for that class)."""
        out = torch.full((q.shape[0], len(names)), -float("inf"))
        for k, name in enumerate(names):
            astro = self.sketches.get(name)
            if astro is None or astro._n < MIN_SKETCH:
                continue
            sigma = astro.sigma.clamp(min=SIGMA_FLOOR)
            diff = q[:, :astro.code_dim] - astro.mu
            out[:, k] = -0.5 * ((diff / sigma) ** 2
                                + 2 * sigma.log()).sum(dim=-1)
        return out

    # ── consumer 3: replay guard + persistence ────────────────────

    def replay(self, n_per_class: int = 32) -> List[tuple]:
        """(name, pocket samples) per consumable class — spec-§4.5
        pseudo-rehearsal for gradient training over grown tissue."""
        return [(name, astro.sample(n_per_class).clamp(1, N_QUANTA - 1))
                for name, astro in self.sketches.items()
                if astro._n >= MIN_SKETCH]

    def state_dict(self) -> dict:
        return {name: {
            "cell_id": astro.cell_id,
            "n": astro._n,
            "mean": astro._mean.clone(),
            "m2": astro._m2.clone(),
        } for name, astro in self.sketches.items()}

    def load_state_dict(self, state: dict) -> None:
        self.sketches = {}
        for name, s in state.items():
            astro = self.adopt(name, s["cell_id"], s["mean"].numel())
            astro._n = s["n"]
            astro._mean = s["mean"].clone()
            astro._m2 = s["m2"].clone()
