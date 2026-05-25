"""Manifold replay — storage-free pseudo-rehearsal via diagonal-Gaussian sketches.

See spec §4.5.  Archive entries are manifold astrocytes in the arena.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from trioron.core.arena import Arena
from trioron.core.epigenome import (
    LINEAR, PERCEPTION, OUTPUT, set_gene, clear_gene, has_gene,
)
from trioron.core.roles import CellRole
from trioron.core.state import CellState


@dataclass
class ManifoldConfig:
    replay_steps_per_class: int = 20
    code_dim: int = 128


# ── Manifold astrocyte lifecycle ─────────────────────────────────


def get_code_boundary(arena: Arena) -> torch.Tensor:
    """Return cell ids that form the stable code-space boundary.

    For a minimal base this is the perception cells; for a frozen base
    it would be the frozen base's output cells.
    """
    mask = arena.alive & has_gene(arena.epigenome, PERCEPTION).bool()
    return mask.nonzero(as_tuple=False).squeeze(-1).to(torch.int32)


class ManifoldAstrocyte:
    """Wraps a single manifold archive entry stored as an arena cell."""

    def __init__(self, arena: Arena, cell_id: int, class_id: int, code_dim: int) -> None:
        self.arena = arena
        self.cell_id = cell_id
        self.class_id = class_id
        self.code_dim = code_dim
        self._n: int = 0
        self._mean = torch.zeros(code_dim, device=arena.device)
        self._m2 = torch.zeros(code_dim, device=arena.device)

    def update(self, code_batch: torch.Tensor) -> None:
        """Online Welford update from a batch of code-space activations ``[B, code_dim]``."""
        for x in code_batch:
            self._n += 1
            delta = x - self._mean
            self._mean += delta / self._n
            delta2 = x - self._mean
            self._m2 += delta * delta2

    def finalize(self) -> None:
        """Write (mu, sigma) into the arena cell's bias/edge_weight as compact storage."""
        if self._n < 2:
            sigma = torch.ones(self.code_dim, device=self.arena.device) * 0.01
        else:
            sigma = (self._m2 / (self._n - 1)).sqrt().clamp(min=1e-6)
        a = self.arena
        cid = self.cell_id
        with torch.no_grad():
            a.bias[cid] = 0.0
        a.engagement[cid] = float(self._n)

    @property
    def mu(self) -> torch.Tensor:
        return self._mean.clone()

    @property
    def sigma(self) -> torch.Tensor:
        if self._n < 2:
            return torch.ones(self.code_dim, device=self.arena.device) * 0.01
        return (self._m2 / (self._n - 1)).sqrt().clamp(min=1e-6)

    def sample(self, batch_size: int) -> torch.Tensor:
        """Draw ``batch_size`` pseudo-samples from the diagonal Gaussian."""
        eps = torch.randn(batch_size, self.code_dim, device=self.arena.device)
        return self.mu + self.sigma * eps

    def log_likelihood(self, code_batch: torch.Tensor) -> torch.Tensor:
        """Diagonal-Gaussian log p(code | class) for each sample. Returns [B]."""
        sigma = self.sigma
        diff = code_batch[:, :self.code_dim] - self.mu
        return -0.5 * ((diff / sigma) ** 2 + sigma.log() * 2).sum(dim=-1)


class ManifoldArchive:
    """Manages all manifold astrocytes across the curriculum."""

    def __init__(self, arena: Arena, cfg: ManifoldConfig | None = None) -> None:
        self.arena = arena
        self.cfg = cfg or ManifoldConfig()
        self._astrocytes: dict[int, ManifoldAstrocyte] = {}

    def has_class(self, class_id: int) -> bool:
        return class_id in self._astrocytes

    @property
    def n_classes(self) -> int:
        return len(self._astrocytes)

    @property
    def class_ids(self) -> list[int]:
        return list(self._astrocytes.keys())

    def spawn(self, class_id: int, code_dim: int | None = None) -> ManifoldAstrocyte:
        """Create a manifold astrocyte for a new class."""
        dim = code_dim or self.cfg.code_dim
        a = self.arena
        cell_ids = a.alloc(1)
        cid = int(cell_ids[0].item())

        a.forward_inclusion[cid] = False
        epi = int(a.epigenome[cid].item())
        epi = clear_gene(epi, LINEAR)
        a.epigenome[cid] = epi
        a.lineage_root[cid] = cid
        a.parent[cid] = -1
        a.position[cid] = torch.tensor([1.0, class_id * 0.01, 0.9], device=a.device)

        astro = ManifoldAstrocyte(a, cid, class_id, dim)
        self._astrocytes[class_id] = astro
        return astro

    def get(self, class_id: int) -> ManifoldAstrocyte | None:
        return self._astrocytes.get(class_id)

    def update_class(self, class_id: int, code_batch: torch.Tensor) -> None:
        """Accumulate code-space activations for a class seen this batch."""
        astro = self._astrocytes.get(class_id)
        if astro is None:
            astro = self.spawn(class_id, code_batch.shape[-1])
        if self.arena.state[astro.cell_id] == CellState.ACTIVE:
            astro.update(code_batch)

    def finalize_all(self) -> None:
        """Freeze all active manifold astrocytes (called at dream-cycle consolidation)."""
        for astro in self._astrocytes.values():
            if self.arena.state[astro.cell_id] == CellState.ACTIVE:
                astro.finalize()
                self.arena.state[astro.cell_id] = CellState.DORMANT

    def replay_batches(
        self,
        batch_size: int,
        exclude_class: int | None = None,
        exclude_classes: set[int] | None = None,
    ) -> list[tuple[torch.Tensor, int]]:
        """Generate replay pseudo-samples for all archived classes.

        Returns list of ``(samples, class_id)`` tuples, one per past class,
        each with ``cfg.replay_steps_per_class`` batches collapsed into one.
        """
        skip: set[int] = set()
        if exclude_class is not None:
            skip.add(exclude_class)
        if exclude_classes is not None:
            skip.update(exclude_classes)

        result = []
        for cid, astro in self._astrocytes.items():
            if cid in skip:
                continue
            if self.arena.state[astro.cell_id] != CellState.DORMANT:
                continue
            samples = astro.sample(batch_size * self.cfg.replay_steps_per_class)
            result.append((samples, cid))
        return result

    def storage_bytes(self) -> int:
        """Total archive storage in bytes (mu + sigma per class)."""
        per_class = 0
        for astro in self._astrocytes.values():
            per_class += astro.code_dim * 4 * 2
        return per_class
