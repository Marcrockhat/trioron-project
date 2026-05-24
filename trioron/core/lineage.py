"""Lineage tree — parent/child division history.  See spec §2.6."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .arena import Arena


class Lineage:
    """Read-only lineage operations backed by the arena's parent/root tensors."""

    __slots__ = ("_arena",)

    def __init__(self, arena: Arena) -> None:
        self._arena = arena

    def parent_of(self, cell_id: int) -> int:
        """Direct parent, or -1 for founding cells.  O(1)."""
        return int(self._arena.parent[cell_id].item())

    def lineage_root_of(self, cell_id: int) -> int:
        """Founding ancestor.  O(1) (cached on the cell)."""
        return int(self._arena.lineage_root[cell_id].item())

    def siblings_of(self, cell_id: int) -> torch.Tensor:
        """Int32 tensor of cells sharing the same direct parent."""
        pid = self._arena.parent[cell_id].item()
        if pid == -1:
            return torch.tensor([], dtype=torch.int32, device=self._arena.device)
        alive = self._arena.alive_ids()
        parents = self._arena.parent[alive]
        mask = (parents == pid) & (alive != cell_id)
        return alive[mask]

    def descendants_of(self, cell_id: int) -> torch.Tensor:
        """All alive descendants (breadth-first).  O(alive cells)."""
        alive = self._arena.alive_ids()
        found: list[int] = []
        frontier = {cell_id}
        while frontier:
            parents_set = frontier
            frontier = set()
            for cid in alive.tolist():
                pid = int(self._arena.parent[cid].item())
                if pid in parents_set and cid != cell_id:
                    found.append(cid)
                    frontier.add(cid)
        return torch.tensor(found, dtype=torch.int32, device=self._arena.device)

    def lineage_tree_of(self, root_id: int) -> torch.Tensor:
        """All alive cells whose lineage_root is *root_id*."""
        alive = self._arena.alive_ids()
        roots = self._arena.lineage_root[alive]
        return alive[roots == root_id]
