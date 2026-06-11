"""Epigenome — per-cell mutable marks.  See spec §2.2."""
from __future__ import annotations

import torch

# ── Gene bit positions ────────────────────────────────────────────
LINEAR: int = 0
ATTENTION: int = 1
CONV: int = 2
RECURRENT: int = 3
DENDRITE: int = 4
PERCEPTION: int = 5
OUTPUT: int = 6
CREDIT_ELIGIBLE: int = 7
RECYCLABLE: int = 8
WEIGHT_TIED_LINEAGE: int = 9
MIRROR: int = 10  # marker gene: cell fires on BOTH own-action and observed-action
                  # (apprenticing substrate). NOT an expression gene — mirror cells
                  # dispatch as LINEAR; the gene is for identity/credit-gating/lesion.
RECEPTOR: int = 11  # PCLL phase injection (spec §10.2): co-expressed with PERCEPTION;
                    # the scheduler injects the receptor phase θ = 2π·q/1000 instead of
                    # the raw input value. Marker gene, not an expression gene —
                    # receptor cells skip dispatch like all perception cells.
TANH: int = 12      # bounded-saturation expression gene (spec §3.10, s030):
                    # y = tanh(b + Σ w·a). Composes safely in depth (|y| ≤ 1) —
                    # the bounded counterpart to the unbounded dendrite quad.

EXPRESSION_GENES: tuple[int, ...] = (LINEAR, ATTENTION, CONV, RECURRENT, DENDRITE, TANH)

GENE_NAMES: dict[int, str] = {
    LINEAR: "linear", ATTENTION: "attention", CONV: "conv",
    RECURRENT: "recurrent", DENDRITE: "dendrite", PERCEPTION: "perception",
    OUTPUT: "output", CREDIT_ELIGIBLE: "credit_eligible",
    RECYCLABLE: "recyclable", WEIGHT_TIED_LINEAGE: "weight_tied_lineage",
    MIRROR: "mirror", RECEPTOR: "receptor", TANH: "tanh",
}


# ── Bit helpers (work on scalars and tensors) ─────────────────────

def has_gene(epigenome: int | torch.Tensor, gene: int) -> int | torch.Tensor:
    return (epigenome >> gene) & 1


def set_gene(epigenome: int | torch.Tensor, gene: int) -> int | torch.Tensor:
    return epigenome | (1 << gene)


def clear_gene(epigenome: int | torch.Tensor, gene: int) -> int | torch.Tensor:
    return epigenome & ~(1 << gene)


def primary_phenotype(epigenome: int) -> int:
    """Lowest set expression gene, defaulting to LINEAR."""
    for gene in EXPRESSION_GENES:
        if (epigenome >> gene) & 1:
            return gene
    return LINEAR
