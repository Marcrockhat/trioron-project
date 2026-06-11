"""Stress drivers → the council vote economy.  See spec §10.6.

The resolution status of a period is exclusive (empty XOR frustrated XOR
resolved), so the stress→driver mapping is the design-§7 table, applied
directly — votes never arbitrate BETWEEN drivers:

  empty      → grow SENSATION       (the progenitor reaches toward the
                                     world: attach/re-equip a receptor)
  frustrated → grow DISCRIMINATION  (recruit a sensed-but-unread feature;
                                     in the gradient path, spawn the
                                     winning composer phenotype)
  resolved   → no growth

The economy is the ONE council book (step4_grow `_vote_transfer`
semantics: equal share per src seat, per-seat floor, Σ conserved). Sides:

  SENSATION       — 4 bookkeeping seats held by the progenitor (the
                    perception group's ledger key, decision D4: the live
                    receptor population is what is voted ON, it does not
                    vote). 4 seats × floor ¼ ⇒ the side can pay exactly 3
                    whole votes before its floor — aligning with the
                    habituation walk 1→0.5→0.25→0.125 < DRIVE_FLOOR after
                    exactly 3 fruitless growths.
  DISCRIMINATION  — the 24 council phenotype cells (6 genes × 4). Stress
                    transfers spread equally across the side, so the
                    BETWEEN-phenotype ranking (step4_grow's economy) is
                    preserved — one book, two readouts.

Habituation tames the void (design §8): the empty drive decays
×HABITUATION_DECAY per FRUITLESS sensory growth and resets to 1 the
moment a growth finds signal. Below DRIVE_FLOOR the organism settles
into ACCEPTED-EMPTY — empty stays a valid terminal answer; growth stops.

Settlement: a growth decision is settled at the NEXT boundary — success
= the stress it answered is gone (empty → something clears the floor;
ambiguity → resolved). The router holds the pending driver; the meeting
calls settle() then decide(). A decision implies the progenitor acts
before the next boundary.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .resolve import EMPTY, FRUSTRATED, RESOLVED

SENSATION = "sensation"
DISCRIMINATION = "discrimination"

PERCEPTION_SEATS = 4     # the progenitor's book — same multiplicity as a gene group
VOTE_PER_EVENT = 1.0     # one whole vote moves per outcome (conserved)
GROUP_VOTE_FLOOR = 1.0   # a side's seats can fall to at most this many votes total
HABITUATION_DECAY = 0.5  # empty-drive multiplier per fruitless sensory growth
DRIVE_FLOOR = 0.2        # below this, empty is ACCEPTED (terminal, no growth)


def _vote_transfer(votes: Dict, src: List, dst: List,
                   amount: float, floor: float) -> float:
    """Move up to *amount* total votes from src seats to dst seats, equally
    per src seat, never below the per-seat floor (step4_grow semantics)."""
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
    """The two-driver routing over a Germline's conserved vote book."""

    def __init__(self, germline) -> None:
        self.germline = germline
        self.sensation_seats: List = list(germline.perception_seats)
        self.discrimination_seats: List = [
            cid for ids in germline.council_ids.values() for cid in ids
        ]
        self.empty_drive = 1.0
        self.pending: Optional[str] = None

    # ── readouts ──────────────────────────────────────────────────

    def _seats(self, group: str) -> List:
        return (self.sensation_seats if group == SENSATION
                else self.discrimination_seats)

    def group_votes(self, group: str) -> float:
        return sum(self.germline.votes[c] for c in self._seats(group))

    def total_votes(self) -> float:
        return sum(self.germline.votes.values())

    def winner_phenotype(self) -> int:
        """Within-discrimination prioritisation: the composer gene the
        gradient growth path would spawn (step4_grow ranking)."""
        g = self.germline
        return max(g.council_ids,
                   key=lambda p: sum(g.votes[c] for c in g.council_ids[p]))

    # ── the meeting hooks ─────────────────────────────────────────

    def settle(self, status: str) -> Optional[bool]:
        """Settle the pending growth against this period's status. Returns
        the outcome (None if nothing was pending)."""
        if self.pending is None:
            return None
        group, self.pending = self.pending, None
        success = (status != EMPTY if group == SENSATION
                   else status == RESOLVED)
        self._report(group, success)
        return success

    def decide(self, status: str) -> Optional[str]:
        """Which driver grows this boundary; None = no growth (resolved, or
        accepted-empty after habituation). Sets the pending settlement."""
        if status == RESOLVED:
            return None
        if status == EMPTY:
            group = SENSATION if self.empty_drive >= DRIVE_FLOOR else None
        else:
            group = DISCRIMINATION
        if group is not None:
            self.pending = group
        return group

    def _report(self, group: str, success: bool) -> None:
        votes = self.germline.votes
        mine, other = self._seats(group), self._seats(
            DISCRIMINATION if group == SENSATION else SENSATION)
        if success:
            _vote_transfer(votes, other, mine, VOTE_PER_EVENT,
                           GROUP_VOTE_FLOOR / len(other))
            if group == SENSATION:
                self.empty_drive = 1.0      # stress was justified — re-sensitise
        else:
            _vote_transfer(votes, mine, other, VOTE_PER_EVENT,
                           GROUP_VOTE_FLOOR / len(mine))
            if group == SENSATION:
                self.empty_drive *= HABITUATION_DECAY
