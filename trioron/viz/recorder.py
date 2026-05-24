"""Snapshot recorder — captures substrate state at significant events.  See spec §7.1."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .snapshot import Event, Snapshot, capture_full, capture_delta

if TYPE_CHECKING:
    from trioron.core.arena import Arena


class Recorder:
    """Records substrate snapshots to a directory.

    Attach to a substrate via ``substrate.attach_recorder(recorder)``.
    Call the trigger methods at appropriate points in the training loop.
    """

    def __init__(
        self,
        snapshot_dir: str | Path,
        *,
        on_task_boundary: bool = True,
        on_dream_consolidate: bool = True,
        on_growth_event: bool | str = "full",
        on_recycle_event: bool = False,
        sample_growth_every_n: int = 1,
    ) -> None:
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        self.on_task_boundary = on_task_boundary
        self.on_dream_consolidate = on_dream_consolidate
        self.on_growth_event = on_growth_event
        self.on_recycle_event = on_recycle_event
        self.sample_growth_every_n = max(1, sample_growth_every_n)

        self._snapshot_counter: int = 0
        self._growth_counter: int = 0
        self._pending_events: list[Event] = []
        self._snapshots: list[Snapshot] = []

    def record_event(self, event: Event) -> None:
        """Buffer an event for the next snapshot."""
        self._pending_events.append(event)

    # ── Trigger methods ──────────────────────────────────────────

    def on_task_start(self, arena: Arena, task_index: int) -> None:
        if self.on_task_boundary:
            self._write_full(arena, task_index, tag="task_start")

    def on_task_end(self, arena: Arena, task_index: int) -> None:
        if self.on_task_boundary:
            self._write_full(arena, task_index, tag="task_end")

    def on_dream(self, arena: Arena, task_index: int) -> None:
        if self.on_dream_consolidate:
            self._write_full(arena, task_index, tag="dream")

    def on_growth(self, arena: Arena, task_index: int, child_id: int) -> None:
        if not self.on_growth_event:
            return
        self._growth_counter += 1
        if self._growth_counter % self.sample_growth_every_n != 0:
            return
        if self.on_growth_event == "full":
            self._write_full(arena, task_index, tag="growth")
        else:
            self._write_delta(arena, [child_id], task_index, tag="growth")

    def on_recycle(self, arena: Arena, task_index: int, cell_id: int) -> None:
        if self.on_recycle_event:
            self._write_delta(arena, [cell_id], task_index, tag="recycle")

    # ── Internal ─────────────────────────────────────────────────

    def _write_full(self, arena: Arena, task_index: int, tag: str) -> None:
        snap = capture_full(arena, task_index)
        snap.events_since_last = list(self._pending_events)
        self._pending_events.clear()
        self._write(snap, tag)

    def _write_delta(self, arena: Arena, changed_ids: list[int], task_index: int, tag: str) -> None:
        snap = capture_delta(arena, changed_ids, task_index)
        snap.events_since_last = list(self._pending_events)
        self._pending_events.clear()
        self._write(snap, tag)

    def _write(self, snap: Snapshot, tag: str) -> None:
        idx = self._snapshot_counter
        self._snapshot_counter += 1
        filename = f"t{snap.task_index:03d}_{idx:04d}_{tag}.json"
        snap.write(self.snapshot_dir / filename)
        self._snapshots.append(snap)

    @property
    def snapshot_count(self) -> int:
        return self._snapshot_counter

    @property
    def snapshots(self) -> list[Snapshot]:
        return list(self._snapshots)
