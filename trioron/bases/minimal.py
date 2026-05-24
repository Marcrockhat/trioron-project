"""Minimal base — perception strip + output strip, no interior.  See spec §2.10."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from trioron.core.epigenome import (
    LINEAR, PERCEPTION, OUTPUT, CREDIT_ELIGIBLE, set_gene,
)

if TYPE_CHECKING:
    from trioron.core.construct import Substrate


class Minimal:
    """Smallest viable substrate: ``input_dim`` perception cells connected
    directly to ``initial_classes`` output cells."""

    def __init__(self, input_dim: int, initial_classes: int) -> None:
        self.input_dim = input_dim
        self.initial_classes = initial_classes

    def __call__(self, substrate: Substrate) -> None:
        a = substrate.arena

        perc_ids = a.alloc(self.input_dim)
        out_ids = a.alloc(self.initial_classes)

        for pid in perc_ids.tolist():
            epi = int(a.epigenome[pid].item())
            epi = set_gene(epi, PERCEPTION)
            a.epigenome[pid] = epi
            a.position[pid] = torch.tensor([0.0, pid / max(self.input_dim - 1, 1), 0.5])
            a.rank[pid] = 0

        for i, oid in enumerate(out_ids.tolist()):
            epi = int(a.epigenome[oid].item())
            epi = set_gene(epi, OUTPUT)
            epi = set_gene(epi, CREDIT_ELIGIBLE)
            a.epigenome[oid] = epi
            a.position[oid] = torch.tensor([1.0, i / max(self.initial_classes - 1, 1), 0.5])

        a.refresh_all_phenotypes()

        src = perc_ids.repeat(self.initial_classes)
        dst = out_ids.repeat_interleave(self.input_dim)
        a.add_edges(src, dst)


def minimal(input_dim: int, initial_classes: int) -> Minimal:
    """Convenience factory matching spec §2.10 API."""
    return Minimal(input_dim, initial_classes)
