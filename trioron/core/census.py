"""Structural census — the arena's countable anatomy.  See design
docs/design/mixed_stream_growth.md §1 [D11].

The structural contract: every growth event has a declared arena delta,
and the proof the organism grows is that these numbers move — computing
cells, edges, depth. The census is standing instrumentation (emitted in
every PCLL MeetingReport), not a probe.

Kinds are derived from arena fields alone:
  computing    — forward_inclusion=True (tissue in the forward path)
  germline     — forward_inclusion=False, epigenome != 0 (the progenitor
                 and her council seats: bookkeeping with genes)
  astrocyte    — forward_inclusion=False, epigenome == 0 (class rows and
                 other markers: bookkeeping without genes)
Depth is measured over computing cells only — bookkeeping rows all sit
at rank 0 and must not dilute it.
"""
from __future__ import annotations

from dataclasses import dataclass

from .arena import Arena
from .epigenome import EXPRESSION_GENES, GENE_NAMES, has_gene
from .state import CellState


@dataclass(frozen=True)
class Census:
    cells: int                 # allocated rows (arena.cursor)
    computing: int             # forward_inclusion=True
    dormant: int               # computing rows currently in CellState.DORMANT
    germline: int              # bookkeeping rows with genes (progenitor+council)
    astrocytes: int            # bookkeeping rows without genes (class rows)
    edges: int                 # arena.edge_cursor
    depth: int                 # max rank among computing cells (0 = flat)
    ranks: tuple[int, ...]     # distinct ranks among computing cells
    genes: tuple[tuple[str, int], ...]  # expressed-gene counts, computing only

    def __str__(self) -> str:
        g = ", ".join(f"{n}:{c}" for n, c in self.genes if c)
        return (f"census[cells {self.cells} = {self.computing} computing"
                f" ({self.dormant} dormant) + {self.germline} germline"
                f" + {self.astrocytes} astrocyte | edges {self.edges}"
                f" | depth {self.depth} ranks {list(self.ranks)} | {g}]")

    def delta(self, before: "Census") -> str:
        """One-line structural delta vs an earlier census (gate output)."""
        parts = []
        for f in ("computing", "astrocytes", "edges", "depth"):
            d = getattr(self, f) - getattr(before, f)
            if d:
                parts.append(f"{f} {'+' if d > 0 else ''}{d}")
        return "Δ[" + (", ".join(parts) if parts else "no structural change") + "]"


def census(arena: Arena) -> Census:
    n = arena.cursor
    fi = arena.forward_inclusion[:n]
    epi = arena.epigenome[:n]
    state = arena.state[:n]
    comp_ranks = arena.rank[:n][fi]
    ranks = tuple(sorted(set(comp_ranks.tolist()))) if int(fi.sum()) else ()
    genes = tuple(
        (GENE_NAMES[g], int((has_gene(epi, g).bool() & fi).sum()))
        for g in EXPRESSION_GENES
    )
    return Census(
        cells=n,
        computing=int(fi.sum()),
        dormant=int((fi & (state == CellState.DORMANT)).sum()),
        germline=int(((~fi) & (epi != 0)).sum()),
        astrocytes=int(((~fi) & (epi == 0)).sum()),
        edges=arena.edge_cursor,
        depth=max(ranks) if ranks else 0,
        ranks=ranks,
        genes=genes,
    )
