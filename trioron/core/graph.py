"""CellGraph — directed cell-to-cell connectivity.  See spec §2.3."""
from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import torch

from .epigenome import RECURRENT, has_gene

if TYPE_CHECKING:
    from .arena import Arena


class CellGraph:
    """Structural API over the arena's edge storage.

    Enforces the three invariants (§2.3):
      1. No dangling edges
      2. No phenotype/cycle violation
      3. No envelope overflow
    """

    __slots__ = ("_arena",)

    def __init__(self, arena: Arena) -> None:
        self._arena = arena

    # ── Edge mutation ─────────────────────────────────────────────

    def add_edge(
        self,
        src: int,
        dst: int,
        weight: float | None = None,
    ) -> None:
        """Add one directed edge src→dst with invariant checks."""
        a = self._arena

        # Invariant 1 — no dangling edges
        if not a.alive[src]:
            raise ValueError(f"Source cell {src} is not alive")
        if not a.alive[dst]:
            raise ValueError(f"Destination cell {dst} is not alive")

        # Invariant 3 — envelope capacity
        byte_cost = 4  # one float32 edge weight
        if not a.allows_growth(add_edges=1, add_bytes=byte_cost):
            raise RuntimeError("Envelope does not allow another edge")

        # Invariant 2 — cycle check (skip for self-edges on recurrent cells)
        if src != dst and self._creates_forbidden_cycle(src, dst):
            raise ValueError(
                f"Edge {src}→{dst} would create a cycle through "
                f"non-recurrent cells"
            )

        w = torch.tensor([weight if weight is not None else 0.0])
        a.add_edges(
            torch.tensor([src], dtype=torch.int32),
            torch.tensor([dst], dtype=torch.int32),
            w,
        )

    def add_edges_unchecked(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> None:
        """Batch-add edges without per-edge invariant checks.

        Use only during construction or when the caller guarantees
        invariants hold (e.g. bases that build acyclic topologies).
        """
        self._arena.add_edges(src, dst, weights)

    # ── Rank computation ──────────────────────────────────────────

    def recompute_ranks(self) -> None:
        """BFS topological rank (Kahn's algorithm).  O(cells + edges).

        Back-edges into recurrent cells and self-edges are excluded
        from the DAG used for ranking.
        """
        a = self._arena
        alive = set(a.alive_ids().tolist())
        if not alive:
            a.rank_dirty = False
            return

        # Build forward adjacency and in-degree
        successors: dict[int, list[int]] = {c: [] for c in alive}
        in_deg: dict[int, int] = {c: 0 for c in alive}

        src = a.edge_src[: a.edge_cursor]
        dst = a.edge_dst[: a.edge_cursor]

        for i in range(a.edge_cursor):
            s = int(src[i].item())
            d = int(dst[i].item())
            if s == d or s not in alive or d not in alive:
                continue
            successors[s].append(d)
            in_deg[d] += 1

        # Kahn's BFS
        a.rank.fill_(0)
        queue: deque[int] = deque()
        for c in alive:
            if in_deg[c] == 0:
                queue.append(c)

        visited = 0
        while queue:
            u = queue.popleft()
            visited += 1
            u_rank = int(a.rank[u].item())
            for v in successors[u]:
                new_rank = u_rank + 1
                if new_rank > int(a.rank[v].item()):
                    a.rank[v] = new_rank
                in_deg[v] -= 1
                if in_deg[v] == 0:
                    queue.append(v)

        if visited < len(alive):
            # Cells still in a cycle — assign max_rank + 1 provisionally
            max_rank = int(a.rank.max().item())
            for c in alive:
                if in_deg[c] > 0:
                    a.rank[c] = max_rank + 1

        a.rank_dirty = False

    # ── Internal helpers ──────────────────────────────────────────

    def _creates_forbidden_cycle(self, src: int, dst: int) -> bool:
        """Return True if adding src→dst closes a cycle through
        any cell that does not have the RECURRENT gene."""
        a = self._arena

        # DFS from dst looking for src
        visited: set[int] = set()
        stack = [dst]
        while stack:
            u = stack.pop()
            if u == src:
                # Found cycle — check every cell on it has RECURRENT
                return not self._all_recurrent_on_path(dst, src)
            if u in visited:
                continue
            visited.add(u)
            # Follow outgoing edges from u
            edge_src = a.edge_src[: a.edge_cursor]
            edge_dst = a.edge_dst[: a.edge_cursor]
            for i in range(a.edge_cursor):
                if int(edge_src[i].item()) == u:
                    v = int(edge_dst[i].item())
                    if v not in visited and a.alive[v]:
                        stack.append(v)
        return False

    def _all_recurrent_on_path(self, start: int, end: int) -> bool:
        """Check that every cell on any path start→end has RECURRENT."""
        a = self._arena
        if not has_gene(int(a.epigenome[start].item()), RECURRENT):
            return False
        if not has_gene(int(a.epigenome[end].item()), RECURRENT):
            return False

        visited: set[int] = set()
        stack = [start]
        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            if not has_gene(int(a.epigenome[u].item()), RECURRENT):
                return False
            if u == end:
                continue
            edge_src = a.edge_src[: a.edge_cursor]
            edge_dst = a.edge_dst[: a.edge_cursor]
            for i in range(a.edge_cursor):
                if int(edge_src[i].item()) == u:
                    v = int(edge_dst[i].item())
                    if v not in visited and a.alive[v]:
                        stack.append(v)
        return True
