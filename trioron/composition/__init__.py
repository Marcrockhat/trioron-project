"""Composition primitives for cross-seed donor absorption.

The L0 handshake translator (paper L0_HANDSHAKE_BRIEF.md) replaces the
shared-seed invariant with a closed-form translator built from each donor's
publicly-available L0 random projection matrix.
"""
from .translator import (
    L0Translator,
    compose_with_translator,
    transform_archive_to_canonical,
)
from .subspace import (
    PROTOCOL_SEED,
    build_protocol_subspace,
    build_donor_rotation,
    build_factored_l0_weight,
    factor_l0_in_place,
)

__all__ = [
    "L0Translator",
    "compose_with_translator",
    "transform_archive_to_canonical",
    "PROTOCOL_SEED",
    "build_protocol_subspace",
    "build_donor_rotation",
    "build_factored_l0_weight",
    "factor_l0_in_place",
]
