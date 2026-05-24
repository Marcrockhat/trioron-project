"""Seeded base — perception + interior + output cells.  See spec §2.10."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from trioron.core.epigenome import (
    LINEAR, PERCEPTION, OUTPUT, CREDIT_ELIGIBLE, set_gene,
)

if TYPE_CHECKING:
    from trioron.core.construct import Substrate


class Seeded:
    """Substrate with ``interior_cells`` active neurons between perception and output."""

    def __init__(self, input_dim: int, initial_classes: int, interior_cells: int = 32) -> None:
        self.input_dim = input_dim
        self.initial_classes = initial_classes
        self.interior_cells = interior_cells

    def __call__(self, substrate: Substrate) -> None:
        a = substrate.arena

        perc_ids = a.alloc(self.input_dim)
        int_ids = a.alloc(self.interior_cells)
        out_ids = a.alloc(self.initial_classes)

        for pid in perc_ids.tolist():
            epi = int(a.epigenome[pid].item())
            epi = set_gene(epi, PERCEPTION)
            a.epigenome[pid] = epi
            a.position[pid] = torch.tensor([0.0, pid / max(self.input_dim - 1, 1), 0.5])
            a.rank[pid] = 0

        for idx, iid in enumerate(int_ids.tolist()):
            epi = int(a.epigenome[iid].item())
            epi = set_gene(epi, CREDIT_ELIGIBLE)
            a.epigenome[iid] = epi
            a.position[iid] = torch.tensor([
                0.5,
                idx / max(self.interior_cells - 1, 1),
                0.5,
            ])
            a.rank[iid] = 1

        for i, oid in enumerate(out_ids.tolist()):
            epi = int(a.epigenome[oid].item())
            epi = set_gene(epi, OUTPUT)
            epi = set_gene(epi, CREDIT_ELIGIBLE)
            a.epigenome[oid] = epi
            a.position[oid] = torch.tensor([1.0, i / max(self.initial_classes - 1, 1), 0.5])
            a.rank[oid] = 2

        a.refresh_all_phenotypes()

        # perception → interior (fully connected)
        src_pi = perc_ids.repeat(self.interior_cells)
        dst_pi = int_ids.repeat_interleave(self.input_dim)
        a.add_edges(src_pi, dst_pi)

        # interior → output (fully connected)
        src_io = int_ids.repeat(self.initial_classes)
        dst_io = out_ids.repeat_interleave(self.interior_cells)
        a.add_edges(src_io, dst_io)


def seeded(input_dim: int, initial_classes: int, interior_cells: int = 32) -> Seeded:
    return Seeded(input_dim, initial_classes, interior_cells)
