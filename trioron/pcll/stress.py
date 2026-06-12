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

Settlement [D13]: a growth decision carries the CLASS it answered and
is settled against THAT class's stress, never the period's global
status. The s031 false-credit confound: "is the stress gone at the next
boundary?" pays a growth whenever the WORLD changes underneath it (a
new class arrives → RESOLVED → credit), which drained sensation to its
floor and inflated every phenotype group to a flat 4.500 while no
growth ever executed. Each pending entry is (driver, subject):

  subject None     — global-status predicate (sensation growths keep
                     the s030 semantics: EMPTY is the period's stress,
                     not a class's — something clears the floor =
                     success).
  subject present  — settles ONLY on the subject's own testimony, via
                     the caller's testify(subject) verdict: True/False
                     settles, None defers to a later boundary (the
                     world talking about something else is not
                     evidence about this growth; silence pays nobody).

Vote transfers keep step4 semantics unchanged (equal share per seat,
per-seat floors, Σ=28 conserved) — only the success predicate moved.
The meeting calls settle() then decide(); a consumer that executes
growths binds them to the decision via attach_subjects().
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

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
        # Pending settlements [D13]: (driver, subject). Subject is opaque
        # to the router — the consumer that grew supplies the testimony.
        self.pending: List[Tuple[str, Any]] = []
        # Audit log: (driver, subject, success) per settled decision —
        # the book saturates at its floors; this keeps the full signal.
        self.settlements: List[Tuple[str, Any, bool]] = []

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

    def settle(self, status: str,
               testify: Optional[Callable[[Any], Optional[bool]]] = None
               ) -> List[bool]:
        """Settle pending growths [D13]. Subject-less entries settle against
        the period's global status (the s030 sensation semantics).
        Subject-bearing entries settle ONLY on the subject's own testimony:
        testify(subject) → True/False settles, None defers the entry to a
        later boundary. Returns the outcomes settled this call."""
        held: List[Tuple[str, Any]] = []
        out: List[bool] = []
        for group, subject in self.pending:
            if subject is None:
                success = (status != EMPTY if group == SENSATION
                           else status == RESOLVED)
            else:
                verdict = testify(subject) if testify is not None else None
                if verdict is None:
                    held.append((group, subject))
                    continue
                success = bool(verdict)
            gene = getattr(subject, "gene", None)
            if gene is not None:
                self._report_gene(gene, success)   # composer spawn [D14]
            else:
                self._report(group, success)
            self.settlements.append((group, subject, success))
            out.append(success)
        self.pending = held
        return out

    def decide(self, status: str, subject: Any = None) -> Optional[str]:
        """Which driver grows this boundary; None = no growth (resolved, or
        accepted-empty after habituation). Queues the pending settlement,
        carrying the class the growth answers when the caller knows it."""
        if status == RESOLVED:
            return None
        if status == EMPTY:
            group = SENSATION if self.empty_drive >= DRIVE_FLOOR else None
        else:
            group = DISCRIMINATION
        if group is not None:
            self.pending.append((group, subject))
        return group

    def attach_subjects(self, subjects: List[Any]) -> None:
        """Bind executed growths to the boundary's decision [D13]: the
        (driver, None) entry decide() just queued fans out to one pending
        per subject answered — a division answers ONE class; a boundary may
        commit several."""
        if not subjects or not self.pending or self.pending[-1][1] is not None:
            return
        group, _ = self.pending[-1]
        self.pending[-1:] = [(group, s) for s in subjects]

    def _report_gene(self, gene: int, success: bool) -> None:
        """Composer-spawn settlement [D14]: the winning gene's council
        group earns (or repays) within the discrimination side — the
        step4_grow economy, at last with its consumer. Success pulls one
        vote equally from the OTHER gene groups; failure bleeds the
        gene's group back. Per-seat floors keep every group ≥
        GROUP_VOTE_FLOOR; Σ=28 conserved."""
        votes = self.germline.votes
        mine = list(self.germline.council_ids[gene])
        others = [c for g, ids in self.germline.council_ids.items()
                  if g != gene for c in ids]
        src, dst = (others, mine) if success else (mine, others)
        _vote_transfer(votes, src, dst, VOTE_PER_EVENT,
                       GROUP_VOTE_FLOOR / len(mine))

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
