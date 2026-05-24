"""Cell — the indivisible unit of the substrate.  See spec §2.1.

The Cell is a thin read/write facade over arena slot fields.  It is
NOT used in the scheduler's hot path (commitment 1); it exists for
the user-facing API and debugging.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .state import CellState
from .roles import CellRole

if TYPE_CHECKING:
    from .arena import Arena


class Cell:
    __slots__ = ("_arena", "_id")

    def __init__(self, arena: Arena, cell_id: int) -> None:
        self._arena = arena
        self._id = cell_id

    # ── Identity ──────────────────────────────────────────────────

    @property
    def id(self) -> int:
        return self._id

    @property
    def alive(self) -> bool:
        return bool(self._arena.alive[self._id].item())

    # ── Weights / bias ────────────────────────────────────────────

    @property
    def bias(self) -> float:
        return float(self._arena.bias[self._id].item())

    @bias.setter
    def bias(self, value: float) -> None:
        self._arena.bias[self._id] = value

    def inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(source_ids, weights) for all incoming edges."""
        return self._arena.inputs_of(self._id)

    @property
    def fan_in(self) -> int:
        return self._arena.fan_in(self._id)

    @property
    def output_dim(self) -> int:
        return int(self._arena.output_dim[self._id].item())

    # ── Statistics ────────────────────────────────────────────────

    @property
    def engagement(self) -> float:
        return float(self._arena.engagement[self._id].item())

    @property
    def utility(self) -> float:
        return float(self._arena.utility[self._id].item())

    @property
    def age(self) -> int:
        return int(self._arena.age[self._id].item())

    # ── Spatial ───────────────────────────────────────────────────

    @property
    def position(self) -> torch.Tensor:
        """Float32[3] view into the arena position tensor."""
        return self._arena.position[self._id]

    # ── Genetics ──────────────────────────────────────────────────

    @property
    def epigenome(self) -> int:
        return int(self._arena.epigenome[self._id].item())

    @property
    def phenotype(self) -> int:
        return int(self._arena.phenotype_cache[self._id].item())

    # ── Lineage ───────────────────────────────────────────────────

    @property
    def parent_id(self) -> int:
        return int(self._arena.parent[self._id].item())

    @property
    def lineage_root_id(self) -> int:
        return int(self._arena.lineage_root[self._id].item())

    # ── Lifecycle ─────────────────────────────────────────────────

    @property
    def state(self) -> CellState:
        return CellState(int(self._arena.state[self._id].item()))

    @property
    def is_active(self) -> bool:
        return self.state == CellState.ACTIVE

    @property
    def is_dormant(self) -> bool:
        return self.state == CellState.DORMANT

    @property
    def rank(self) -> int:
        return int(self._arena.rank[self._id].item())

    @property
    def forward_inclusion(self) -> bool:
        return bool(self._arena.forward_inclusion[self._id].item())

    @property
    def role(self) -> CellRole:
        return CellRole.NEURON if self.forward_inclusion else CellRole.ASTROCYTE

    # ── Display ───────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"Cell(id={self._id}, state={self.state.name}, "
            f"rank={self.rank}, role={self.role.name})"
        )
