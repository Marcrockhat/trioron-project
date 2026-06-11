"""Two-driver stress → growth routing + habituation (design §6–8).

The resolution status of a period is exclusive (empty XOR frustrated XOR
resolved), so the stress→driver mapping is the §7 table, applied directly:

  empty      → grow SENSATION       (new receptor — reach toward the world)
  frustrated → grow DISCRIMINATION  (composers that sharpen the margin)
  resolved   → no growth

The conserved council vote economy (mirrors step4_grow.py: per-cell votes,
Σ constant, ±VOTE_PER_EVENT per outcome spread across the other group, per-cell
floor) tracks each driver group's earned credibility. Because the period status
already picks the GROUP, the votes' role here is within-group prioritisation in
the full organism (which phenotype/site to spawn) — recorded and asserted
conserved, not yet arbitrating between drivers.

Habituation tames the void (§8): empty-stress drive decays ×HABITUATION_DECAY on
each FRUITLESS sensory growth and resets to 1 the moment a growth finds signal.
Below DRIVE_FLOOR the organism settles into ACCEPTED-EMPTY — empty stays a valid
terminal answer; growth stops. The skull (parameter envelope) remains the hard
backstop, uncapped for the trial.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from resolve import EMPTY, FRUSTRATED, RESOLVED

SENSATION = "sensation"
DISCRIMINATION = "discrimination"
GROUPS = (SENSATION, DISCRIMINATION)

CELLS_PER_GROUP = 10
VOTE_PER_EVENT = 1.0     # one whole vote moves per outcome (conserved)
GROUP_VOTE_FLOOR = 1.0   # a group's council can fall to at most this many votes
HABITUATION_DECAY = 0.5  # empty-drive multiplier per fruitless sensory growth
DRIVE_FLOOR = 0.2        # below this, empty is ACCEPTED (terminal, no growth)


def _vote_transfer(votes: Dict[int, float], src: List[int], dst: List[int],
                   amount: float, floor: float) -> float:
    """Move up to *amount* total votes from src cells to dst cells, equally per
    src cell, never below the per-cell floor (step4_grow.py semantics)."""
    share = amount / len(src)
    moved = 0.0
    for c in src:
        take = min(share, max(0.0, votes[c] - floor))
        votes[c] -= take
        moved += take
    add = moved / len(dst)
    for c in dst:
        votes[c] += add
    return moved


class StressRouter:
    def __init__(self) -> None:
        self.cells = {g: [i + GROUPS.index(g) * CELLS_PER_GROUP
                          for i in range(CELLS_PER_GROUP)] for g in GROUPS}
        self.votes: Dict[int, float] = {c: 1.0 for g in GROUPS for c in self.cells[g]}
        self.floor = GROUP_VOTE_FLOOR / CELLS_PER_GROUP
        self.empty_drive = 1.0

    def group_votes(self, group: str) -> float:
        return sum(self.votes[c] for c in self.cells[group])

    def total_votes(self) -> float:
        return sum(self.votes.values())

    def decide(self, status: str) -> Optional[str]:
        """Which driver group grows this period boundary; None = no growth
        (resolved, or accepted-empty after habituation)."""
        if status == RESOLVED:
            return None
        if status == EMPTY:
            return SENSATION if self.empty_drive >= DRIVE_FLOOR else None
        return DISCRIMINATION

    def report(self, group: str, success: bool) -> None:
        """Outcome of the grown period: success = the stress it answered is gone
        (empty → something clears the floor; ambiguity → resolved)."""
        other = [c for g in GROUPS if g != group for c in self.cells[g]]
        if success:
            _vote_transfer(self.votes, other, self.cells[group],
                           VOTE_PER_EVENT, self.floor)
            if group == SENSATION:
                self.empty_drive = 1.0          # stress was justified — re-sensitise
        else:
            _vote_transfer(self.votes, self.cells[group], other,
                           VOTE_PER_EVENT, self.floor)
            if group == SENSATION:
                self.empty_drive *= HABITUATION_DECAY
