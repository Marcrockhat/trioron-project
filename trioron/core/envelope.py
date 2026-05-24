"""Envelope — resource budget (cells, edges, bytes).  See spec §2.5."""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Envelope:
    """Bounded growth medium.  Caps are optional; None = uncapped."""

    max_cells: int | None = None
    max_edges: int | None = None
    max_parameter_bytes: int | None = None

    _stagnation: int = field(default=0, init=False, repr=False)
    _prev_cell_count: int = field(default=0, init=False, repr=False)

    # ── Derived signals ───────────────────────────────────────────

    def pressure(self, n_cells: int, n_edges: int, param_bytes: int) -> float:
        """Max utilisation across all set caps; 0.0 when fully uncapped."""
        ratios: list[float] = []
        if self.max_cells is not None and self.max_cells > 0:
            ratios.append(n_cells / self.max_cells)
        if self.max_edges is not None and self.max_edges > 0:
            ratios.append(n_edges / self.max_edges)
        if self.max_parameter_bytes is not None and self.max_parameter_bytes > 0:
            ratios.append(param_bytes / self.max_parameter_bytes)
        return max(ratios) if ratios else 0.0

    def niche_capacity(self, n_cells: int) -> float:
        """Additional cells allowed before hitting the cell cap."""
        if self.max_cells is not None:
            return max(0, self.max_cells - n_cells)
        return math.inf

    @property
    def stagnation(self) -> int:
        """Consecutive tasks where cell count did not increase."""
        return self._stagnation

    # ── Enforcement ───────────────────────────────────────────────

    def allows_growth(
        self,
        n_cells: int,
        n_edges: int,
        param_bytes: int,
        *,
        add_cells: int = 0,
        add_edges: int = 0,
        add_bytes: int = 0,
    ) -> bool:
        if self.max_cells is not None and n_cells + add_cells > self.max_cells:
            return False
        if self.max_edges is not None and n_edges + add_edges > self.max_edges:
            return False
        if (
            self.max_parameter_bytes is not None
            and param_bytes + add_bytes > self.max_parameter_bytes
        ):
            return False
        return True

    # ── Task boundary bookkeeping ─────────────────────────────────

    def end_task(self, n_cells: int) -> None:
        """Call at every task boundary to update stagnation."""
        if n_cells <= self._prev_cell_count:
            self._stagnation += 1
        else:
            self._stagnation = 0
        self._prev_cell_count = n_cells
