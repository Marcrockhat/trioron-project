"""Extend — add capacity to a woken substrate.  See spec §5.4."""
from __future__ import annotations

from trioron.core.construct import Substrate
from trioron.core.envelope import Envelope


def extend_envelope(
    substrate: Substrate,
    *,
    max_cells: int | None = ...,
    max_edges: int | None = ...,
    max_parameter_bytes: int | None = ...,
) -> None:
    """Lift one or more envelope caps to allow further growth.

    Pass ``None`` to uncap; omit (sentinel ``...``) to keep the current value.
    The new cap takes effect at the next growth attempt.

    Example::

        extend_envelope(substrate, max_parameter_bytes=8_000_000)
    """
    env = substrate.envelope
    if max_cells is not ...:
        env.max_cells = max_cells
    if max_edges is not ...:
        env.max_edges = max_edges
    if max_parameter_bytes is not ...:
        env.max_parameter_bytes = max_parameter_bytes
