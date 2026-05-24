"""Compose — combine multiple bases sequentially.  See spec §2.10."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trioron.core.construct import Base, Substrate


class Compose:
    """Run multiple bases in sequence, each populating the same arena.

    Example::

        substrate = construct(
            base=compose(
                seeded(784, 10, interior_cells=32),
                frozen("donor.pt"),
            ),
            envelope=Envelope(),
        )
    """

    def __init__(self, *bases: Base) -> None:
        self.bases = bases

    def __call__(self, substrate: Substrate) -> None:
        for base in self.bases:
            base(substrate)


def compose(*bases: Base) -> Compose:
    """Convenience factory for sequential base composition."""
    return Compose(*bases)
