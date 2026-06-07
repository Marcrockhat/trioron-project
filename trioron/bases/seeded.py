"""Seeded base — perception + interior + output cells.  See spec §2.10."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from trioron.core.epigenome import (
    LINEAR, DENDRITE, PERCEPTION, OUTPUT, CREDIT_ELIGIBLE, set_gene,
)

if TYPE_CHECKING:
    from trioron.core.construct import Substrate


class Seeded:
    """Substrate with one or more ``interior_cells``-wide layers between perception and output.

    ``interior_layers > 1`` stacks bipartite hidden layers in series (DAG-safe,
    no cycles): perception → interior_1 → interior_2 → ... → output. Each
    interior layer is fully connected to the next. h_interior_ids picks up
    every interior cell across all layers, so the H-vector is the concatenation
    over layers.
    """

    def __init__(self, input_dim: int, initial_classes: int,
                 interior_cells: int = 32, interior_layers: int = 1,
                 nonlinear: bool = False) -> None:
        self.input_dim = input_dim
        self.initial_classes = initial_classes
        self.interior_cells = interior_cells
        self.interior_layers = max(1, interior_layers)
        # nonlinear=True gives interior cells the quad (dendrite) phenotype
        # σ(z)=z+z² instead of pure-linear — the substrate's only nonlinearity.
        # Output cells stay linear (logits). Default False preserves the v1
        # linear behaviour / CI numbers; flip per experiment, train with grad
        # clipping (see trioron/phenotype/dendrite.py).
        self.nonlinear = nonlinear

    def __call__(self, substrate: Substrate) -> None:
        a = substrate.arena

        perc_ids = a.alloc(self.input_dim)
        interior_layer_ids: list[torch.Tensor] = [a.alloc(self.interior_cells)
                                                  for _ in range(self.interior_layers)]
        out_ids = a.alloc(self.initial_classes)

        for pid in perc_ids.tolist():
            epi = int(a.epigenome[pid].item())
            epi = set_gene(epi, PERCEPTION)
            a.epigenome[pid] = epi
            a.position[pid] = torch.tensor([0.0, pid / max(self.input_dim - 1, 1), 0.5])
            a.rank[pid] = 0

        for layer_idx, int_ids in enumerate(interior_layer_ids):
            rank = layer_idx + 1
            for idx, iid in enumerate(int_ids.tolist()):
                epi = int(a.epigenome[iid].item())
                epi = set_gene(epi, CREDIT_ELIGIBLE)
                # nonlinear: dendrite gene set here; the quad only engages once
                # the fan-in is split into ≥2 branches, done after wiring below
                # (spec §3.6: a K=1 dendrite is byte-identical to linear).
                if self.nonlinear:
                    epi = (epi & ~(1 << LINEAR)) | (1 << DENDRITE)
                a.epigenome[iid] = epi
                a.position[iid] = torch.tensor([
                    (layer_idx + 1) / (self.interior_layers + 1),
                    idx / max(self.interior_cells - 1, 1),
                    0.5,
                ])
                a.rank[iid] = rank

        output_rank = self.interior_layers + 1
        for i, oid in enumerate(out_ids.tolist()):
            epi = int(a.epigenome[oid].item())
            epi = set_gene(epi, OUTPUT)
            epi = set_gene(epi, CREDIT_ELIGIBLE)
            a.epigenome[oid] = epi
            a.position[oid] = torch.tensor([1.0, i / max(self.initial_classes - 1, 1), 0.5])
            a.rank[oid] = output_rank

        a.refresh_all_phenotypes()

        # perception → interior_1 (fully connected)
        first = interior_layer_ids[0]
        src_pi = perc_ids.repeat(self.interior_cells)
        dst_pi = first.repeat_interleave(self.input_dim)
        a.add_edges(src_pi, dst_pi)

        # interior_i → interior_{i+1} (fully connected, layer-to-layer)
        for layer_idx in range(self.interior_layers - 1):
            prev = interior_layer_ids[layer_idx]
            nxt = interior_layer_ids[layer_idx + 1]
            src = prev.repeat(self.interior_cells)
            dst = nxt.repeat_interleave(self.interior_cells)
            a.add_edges(src, dst)

        # interior_last → output (fully connected)
        last = interior_layer_ids[-1]
        src_io = last.repeat(self.initial_classes)
        dst_io = out_ids.repeat_interleave(self.interior_cells)
        a.add_edges(src_io, dst_io)

        # nonlinear: engage the quad by splitting each interior cell's fan-in
        # into two branches (spec §3.6 — quad needs K≥2). Second half → branch 1.
        if self.nonlinear:
            for int_ids in interior_layer_ids:
                for iid in int_ids.tolist():
                    srcs, _ = a.inputs_of(iid)
                    if srcs.numel() >= 2:
                        half = srcs.numel() // 2
                        a.grow_branch(iid, srcs[half:])
            a.refresh_all_phenotypes()


def seeded(input_dim: int, initial_classes: int,
           interior_cells: int = 32, interior_layers: int = 1,
           nonlinear: bool = False) -> Seeded:
    return Seeded(input_dim, initial_classes, interior_cells, interior_layers,
                  nonlinear=nonlinear)
