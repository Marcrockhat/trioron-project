"""Rejuvenation — dormant-to-active recovery.  See spec §4.4."""
from __future__ import annotations

from trioron.core.arena import Arena
from trioron.core.state import CellState


DEFAULT_COOLDOWN_TASKS: int = 2


def rejuvenate(arena: Arena, cell_id: int) -> bool:
    """Transition a dormant cell back to active.

    Returns True if the cell was dormant and is now active, False otherwise.
    """
    if arena.state[cell_id] != CellState.DORMANT:
        return False
    arena.state[cell_id] = CellState.ACTIVE
    return True


def find_rejuvenation_candidates(
    arena: Arena,
    engagement_threshold: float = 0.15,
) -> list[int]:
    """Find dormant cells whose engagement suggests they should regain plasticity.

    Returns cell ids where engagement exceeds ``engagement_threshold``
    (half the default credit lock threshold).
    """
    dormant_mask = (
        arena.alive
        & (arena.state == CellState.DORMANT)
        & arena.forward_inclusion  # skip astrocytes (storage-only cells)
    )
    dormant_ids = dormant_mask.nonzero(as_tuple=False).squeeze(-1)
    if dormant_ids.numel() == 0:
        return []
    engaged = arena.engagement[dormant_ids] >= engagement_threshold
    return dormant_ids[engaged].tolist()
