"""Tests for visualization pipeline — snapshot, recorder, detect, export."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

from trioron.core import Envelope, construct
from trioron.core.state import CellState
from trioron.bases import minimal
from trioron.phenotype import default_dispatch_table
from trioron.viz import Recorder, capture_full, detect_all, detect_from_directory, export_html


def _make_substrate(input_dim=4, n_classes=3):
    return construct(
        base=minimal(input_dim, n_classes),
        envelope=Envelope(),
        dispatch_table=default_dispatch_table(),
        capacity=128,
    )


class TestSnapshot:

    def test_capture_full_cell_count(self):
        sub = _make_substrate()
        snap = capture_full(sub.arena, task_index=0)
        assert len(snap.cells) == 7
        assert len(snap.edges) == 12

    def test_snapshot_json_round_trip(self):
        sub = _make_substrate()
        snap = capture_full(sub.arena, task_index=1)
        j = snap.to_json()
        data = json.loads(j)
        assert data["schema_version"] == "trioron-v2.0"
        assert data["task_index"] == 1
        assert len(data["cells"]) == 7

    def test_snapshot_write_to_disk(self):
        sub = _make_substrate()
        snap = capture_full(sub.arena)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "snap.json"
            snap.write(path)
            assert path.exists()
            data = json.loads(path.read_text())
            assert len(data["cells"]) == 7

    def test_capture_delta(self):
        from trioron.viz.snapshot import capture_delta
        sub = _make_substrate()
        snap = capture_delta(sub.arena, [0, 1], task_index=2)
        assert snap.kind == "delta"
        assert len(snap.cells) == 2


class TestRecorder:

    def test_task_boundary_snapshots(self):
        sub = _make_substrate()
        with tempfile.TemporaryDirectory() as d:
            rec = Recorder(d)
            rec.on_task_start(sub.arena, 0)
            rec.on_task_end(sub.arena, 0)
            assert rec.snapshot_count == 2
            files = list(Path(d).glob("*.json"))
            assert len(files) == 2

    def test_growth_event_recording(self):
        sub = _make_substrate()
        with tempfile.TemporaryDirectory() as d:
            rec = Recorder(d, on_task_boundary=False, on_growth_event="full")
            rec.on_growth(sub.arena, 0, child_id=5)
            assert rec.snapshot_count == 1

    def test_growth_sampling(self):
        sub = _make_substrate()
        with tempfile.TemporaryDirectory() as d:
            rec = Recorder(d, on_task_boundary=False, sample_growth_every_n=3)
            for i in range(6):
                rec.on_growth(sub.arena, 0, child_id=i)
            assert rec.snapshot_count == 2

    def test_event_buffering(self):
        sub = _make_substrate()
        with tempfile.TemporaryDirectory() as d:
            rec = Recorder(d)
            from trioron.viz.snapshot import Event
            rec.record_event(Event(type="lock", cell_id=4, task=0))
            rec.on_task_end(sub.arena, 0)
            snap = rec.snapshots[0]
            assert len(snap.events_since_last) == 1
            assert snap.events_since_last[0].type == "lock"


class TestDetect:

    def test_detect_dormant_region(self):
        sub = _make_substrate()
        out_ids = [4, 5, 6]
        for cid in out_ids:
            sub.arena.state[cid] = CellState.DORMANT
        snap = capture_full(sub.arena)
        structures = detect_all(snap)
        dormant_types = [s for s in structures if s.type == "dormant_region"]
        assert len(dormant_types) == 1
        assert len(dormant_types[0].cell_ids) == 3

    def test_detect_from_directory(self):
        sub = _make_substrate()
        with tempfile.TemporaryDirectory() as d:
            snap = capture_full(sub.arena, task_index=0)
            snap.write(Path(d) / "t000_0000_test.json")
            result = detect_from_directory(d)
            assert len(result["structures_by_snapshot"]) == 1


class TestExport:

    def test_export_html(self):
        sub = _make_substrate()
        with tempfile.TemporaryDirectory() as d:
            snap_dir = Path(d) / "snaps"
            snap_dir.mkdir()
            snap = capture_full(sub.arena)
            snap.write(snap_dir / "t000_0000_test.json")

            out = Path(d) / "viewer.html"
            export_html(snap_dir, out)
            assert out.exists()
            content = out.read_text()
            assert "Trioron" in content
            assert "SNAPSHOTS" in content
