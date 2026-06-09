"""Phenotype dispatch table — maps gene bits to batched forward ops.

See spec §3.1.  New phenotypes register here via :func:`register`.
"""
from __future__ import annotations

from trioron.core.epigenome import LINEAR, ATTENTION, CONV, RECURRENT, DENDRITE
from trioron.core.scheduler import ForwardFn
from . import linear
from . import dendrite
from . import recurrent
from . import attention
from . import conv

_REGISTRY: dict[int, ForwardFn] = {}


def register(gene_bit: int, fn: ForwardFn) -> None:
    """Register a batched forward function for *gene_bit*."""
    _REGISTRY[gene_bit] = fn


def default_dispatch_table() -> dict[int, ForwardFn]:
    """Return a dispatch table with all shipped phenotypes."""
    return dict(_REGISTRY)


register(LINEAR, linear.forward_batch)
register(CONV, conv.forward_batch)           # real lineage weight-tying; reduces to linear at own-root
register(ATTENTION, attention.forward_batch) # real SDPA over fan-in; reduces to linear at 1 token
register(RECURRENT, recurrent.forward_batch) # real self/lateral unroll; reduces to linear w/o back-edge
register(DENDRITE, dendrite.forward_batch)   # real quad σ(z)=z+z², not a stub
