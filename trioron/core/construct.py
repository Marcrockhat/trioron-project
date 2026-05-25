"""Construction — build a substrate from a base recipe.  See spec §2.9."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch

from .arena import Arena
from .cell import Cell
from .envelope import Envelope
from .graph import CellGraph
from .lineage import Lineage
from .scheduler import ForwardFn, Scheduler
from .state import CellState


# ── Base protocol ─────────────────────────────────────────────────

@runtime_checkable
class Base(Protocol):
    """A base produces an initial cell population inside an arena.

    Bases are callables that receive a :class:`Substrate` (with an
    empty arena) and populate it with founding cells and edges.
    """

    def __call__(self, substrate: Substrate) -> None: ...


# ── Substrate ─────────────────────────────────────────────────────

class Substrate:
    """Top-level aggregate: arena + graph + lineage + scheduler.

    Created by :func:`construct`.  Provides the user-facing API for
    forward passes, cell inspection, and training-loop hooks.
    """

    def __init__(
        self,
        arena: Arena,
        dispatch_table: dict[int, ForwardFn] | None = None,
    ) -> None:
        self.arena = arena
        self.envelope = arena.envelope
        self.graph = CellGraph(arena)
        self.lineage = Lineage(arena)
        self.scheduler = Scheduler(arena, dispatch_table)
        self.morphogen = None  # set by developmental base if used

    # ── Forward ───────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scheduler.forward(x)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    # ── Cell access ───────────────────────────────────────────────

    def cell(self, cell_id: int) -> Cell:
        return Cell(self.arena, cell_id)

    # ── Task boundary ─────────────────────────────────────────────

    def compile(self) -> None:
        """Recompute ranks and rebuild the dispatch plan."""
        self.graph.recompute_ranks()
        self.scheduler.compile()

    def end_task(self) -> None:
        """Call after each task to update envelope stagnation and
        mark ranks for recomputation."""
        self.envelope.end_task(self.arena.n_cells)
        self.arena.rank_dirty = True

    # ── Training helpers ──────────────────────────────────────────

    def trainable_tensors(self) -> list[torch.Tensor]:
        """Return tensors that should be passed to the optimizer."""
        tensors = [self.arena.bias, self.arena.edge_weight]
        if self.morphogen is not None:
            tensors.extend(self.morphogen.trainable_tensors())
        return tensors

    def prepare_training(self) -> None:
        """Enable gradients on weight tensors and compile."""
        self.arena.bias.requires_grad_(True)
        self.arena.edge_weight.requires_grad_(True)
        if self.morphogen is not None:
            self.morphogen.prepare_training()
        self.compile()

    def zero_dormant_grads(self) -> None:
        """Gradient mask for dormant cells (commitment 5)."""
        self.scheduler.zero_dormant_grads()

    # ── Info ──────────────────────────────────────────────────────

    @property
    def last_activations(self) -> torch.Tensor | None:
        """Per-cell activations [batch, capacity] from the most recent forward pass."""
        return self.scheduler._last_activations

    @property
    def n_cells(self) -> int:
        return self.arena.n_cells

    @property
    def n_edges(self) -> int:
        return self.arena.n_edges

    @property
    def pressure(self) -> float:
        return self.arena.pressure()

    def __repr__(self) -> str:
        return (
            f"Substrate(cells={self.n_cells}, edges={self.n_edges}, "
            f"pressure={self.pressure:.2f})"
        )


# ── Constructor ───────────────────────────────────────────────────

def construct(
    base: Base,
    envelope: Envelope | None = None,
    *,
    dispatch_table: dict[int, ForwardFn] | None = None,
    device: str | torch.device = "cpu",
    capacity: int | None = None,
) -> Substrate:
    """Build a substrate from a base recipe and an envelope.

    >>> substrate = construct(
    ...     base=my_base,
    ...     envelope=Envelope(max_cells=1000),
    ... )
    """
    if envelope is None:
        envelope = Envelope()
    arena = Arena(envelope, device=device, capacity=capacity)
    substrate = Substrate(arena, dispatch_table)
    base(substrate)
    substrate.compile()
    return substrate
