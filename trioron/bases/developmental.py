"""Developmental base — stem cells + axon guidance.  See spec §5.7.6."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from trioron.core.epigenome import (
    LINEAR, PERCEPTION, OUTPUT, CREDIT_ELIGIBLE, set_gene,
)
from trioron.lifecycle.developmental import (
    spawn_stem, axon_guidance, differentiate_all_stems,
    MorphogenField, GuidanceConfig,
)

if TYPE_CHECKING:
    from trioron.core.construct import Substrate


class Developmental:
    """Cell-first base: perception + output + undifferentiated stem pool.

    Unlike ``seeded``, no interior cells are pre-assigned a phenotype
    and no edges are pre-wired between layers. Stem cells differentiate
    on first gradient; edges form via proximity-based axon guidance.

    Args:
        input_dim: number of perception cells.
        initial_classes: number of output cells.
        stem_pool: number of undifferentiated stem cells.
        guidance: axon guidance configuration.
    """

    def __init__(
        self,
        input_dim: int,
        initial_classes: int,
        stem_pool: int = 64,
        guidance: GuidanceConfig | None = None,
    ) -> None:
        self.input_dim = input_dim
        self.initial_classes = initial_classes
        self.stem_pool = stem_pool
        self.guidance = guidance or GuidanceConfig()

    def __call__(self, substrate: Substrate) -> None:
        a = substrate.arena

        # 1. Perception cells at z=0, x=0.5, y spread
        perc_ids = a.alloc(self.input_dim)
        for i, pid in enumerate(perc_ids.tolist()):
            epi = int(a.epigenome[pid].item())
            epi = set_gene(epi, PERCEPTION)
            a.epigenome[pid] = epi
            a.position[pid] = torch.tensor([
                0.5,
                i / max(self.input_dim - 1, 1),
                0.0,
            ], device=a.device)
            a.rank[pid] = 0

        # 2. Output cells at z=1, x=0.5, y spread
        out_ids = a.alloc(self.initial_classes)
        for i, oid in enumerate(out_ids.tolist()):
            epi = int(a.epigenome[oid].item())
            epi = set_gene(epi, OUTPUT)
            epi = set_gene(epi, CREDIT_ELIGIBLE)
            a.epigenome[oid] = epi
            a.position[oid] = torch.tensor([
                0.5,
                i / max(self.initial_classes - 1, 1),
                1.0,
            ], device=a.device)
            a.rank[oid] = 2

        a.refresh_all_phenotypes()

        # 3. Stem cells in the interior (column from z=0.05 to z=0.95)
        stem_ids = spawn_stem(a, self.stem_pool)

        # 4. Initialize morphogen field on the substrate
        morphogen = MorphogenField(device=a.device)
        substrate.morphogen = morphogen

        # 5. Axon guidance — proximity-based wiring
        axon_guidance(a, cfg=self.guidance)

        # 6. Differentiate stems immediately (first-gradient trigger
        #    is conceptual; for practical training we need committed
        #    phenotypes before the first forward pass, otherwise the
        #    scheduler can't dispatch them properly)
        differentiate_all_stems(a, morphogen)


def developmental(
    input_dim: int,
    initial_classes: int,
    stem_pool: int = 64,
    guidance: GuidanceConfig | None = None,
) -> Developmental:
    """Convenience factory for developmental base."""
    return Developmental(input_dim, initial_classes, stem_pool, guidance)
