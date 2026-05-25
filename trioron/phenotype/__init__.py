"""Phenotype dispatch table — maps gene bits to batched forward ops.

See spec §3.1.  New phenotypes register here via :func:`register`.
"""
from __future__ import annotations

from trioron.core.epigenome import LINEAR, ATTENTION, CONV, RECURRENT, DENDRITE
from trioron.core.scheduler import ForwardFn
from . import linear

_REGISTRY: dict[int, ForwardFn] = {}


def register(gene_bit: int, fn: ForwardFn) -> None:
    """Register a batched forward function for *gene_bit*."""
    _REGISTRY[gene_bit] = fn


def default_dispatch_table() -> dict[int, ForwardFn]:
    """Return a dispatch table with all shipped phenotypes."""
    return dict(_REGISTRY)


register(LINEAR, linear.forward_batch)
register(CONV, linear.forward_batch)
register(ATTENTION, linear.forward_batch)
register(RECURRENT, linear.forward_batch)
register(DENDRITE, linear.forward_batch)
