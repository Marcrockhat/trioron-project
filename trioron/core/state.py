"""Cell state transitions — active / dormant lifecycle.  See spec §2.8."""
from __future__ import annotations

import enum


class CellState(enum.IntEnum):
    ACTIVE = 0
    DORMANT = 1
