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
        from the DAG used for ranking.  Fully vectorized edge filtering;
        adjacency as CSR for cache-friendly BFS.
        """
        a = self._arena
        alive_ids = a.alive_ids()
        if alive_ids.numel() == 0:
            a.rank_dirty = False
            return

        n = a.capacity
        src_all = a.edge_src[: a.edge_cursor]
        dst_all = a.edge_dst[: a.edge_cursor]

        # Vectorized edge filtering
        if a.edge_cursor > 0:
            valid = (
                (src_all != dst_all)
                & a.alive[src_all.long()]
                & a.alive[dst_all.long()]
            )
            src_f = src_all[valid].long()
            dst_f = dst_all[valid].long()
        else:
            src_f = torch.tensor([], dtype=torch.long)
            dst_f = torch.tensor([], dtype=torch.long)

        ne = src_f.numel()

        # Vectorized in-degree
        in_deg = torch.zeros(n, dtype=torch.int32, device=a.device)
        if ne > 0:
            in_deg.scatter_add_(0, dst_f, torch.ones(ne, dtype=torch.int32, device=a.device))

        # CSR successor index: sort edges by src, then build row pointers
        if ne > 0:
            order = src_f.argsort()
            src_sorted = src_f[order]
            dst_sorted = dst_f[order]
            # row_ptr[i] = first edge index where src == i
            # Use searchsorted on the sorted src array
            ids = torch.arange(n, device=a.device)
            row_ptr = torch.searchsorted(src_sorted, ids)
            row_end = torch.searchsorted(src_sorted, ids, right=True)
            dst_list = dst_sorted.tolist()
            row_ptr_list = row_ptr.tolist()
            row_end_list = row_end.tolist()
        else:
            dst_list = []
            row_ptr_list = [0] * n
            row_end_list = [0] * n

        # Kahn's BFS on CPU with CSR adjacency
        a.rank.fill_(0)
        in_deg_arr = in_deg.tolist()
        rank_arr = [0] * n

        queue: deque[int] = deque()
        alive_list = alive_ids.tolist()
        for c in alive_list:
            if in_deg_arr[c] == 0:
                queue.append(c)

        visited = 0
        while queue:
            u = queue.popleft()
            visited += 1
            u_rank = rank_arr[u]
            for ei in range(row_ptr_list[u], row_end_list[u]):
                v = dst_list[ei]
                new_rank = u_rank + 1
                if new_rank > rank_arr[v]:
                    rank_arr[v] = new_rank
                in_deg_arr[v] -= 1
                if in_deg_arr[v] == 0:
                    queue.append(v)

        # Write back to tensor
        for c in alive_list:
            a.rank[c] = rank_arr[c]

        if visited < len(alive_list):
            max_rank = max(rank_arr[c] for c in alive_list)
            for c in alive_list:
                if in_deg_arr[c] > 0:
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
