"""Snapshot data structures — the JSON schema for substrate state.  See spec §7.1."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import torch

from trioron.core.arena import Arena
from trioron.core.epigenome import EXPRESSION_GENES, has_gene
from trioron.core.state import CellState

_PHENOTYPE_NAMES = {0: "linear", 1: "attention", 2: "conv", 3: "recurrent", 4: "dendrite"}
_STATE_NAMES = {int(CellState.ACTIVE): "active", int(CellState.DORMANT): "dormant"}


@dataclass
class Event:
    type: str
    cell_id: int
    task: int
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class CellRecord:
    id: int
    state: str
    position: list[float]
    epigenome: int
    phenotype: str
    output_dim: int
    lineage_root: int
    parent: int
    rank: int
    engagement: float
    utility: float
    age: int
    forward_inclusion: bool


@dataclass
class EdgeRecord:
    src: int
    dst: int
    weight: float


@dataclass
class Snapshot:
    schema_version: str = "trioron-v2.0"
    task_index: int = 0
    timestamp_iso: str = ""
    kind: str = "full"
    cells: list[CellRecord] = field(default_factory=list)
    edges: list[EdgeRecord] = field(default_factory=list)
    events_since_last: list[Event] = field(default_factory=list)
    envelope: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "task_index": self.task_index,
            "timestamp_iso": self.timestamp_iso,
            "kind": self.kind,
            "envelope": self.envelope,
            "cells": [asdict(c) for c in self.cells],
            "edges": [asdict(e) for e in self.edges],
            "events_since_last_snapshot": [asdict(ev) for ev in self.events_since_last],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())


def capture_full(arena: Arena, task_index: int = 0) -> Snapshot:
    """Capture a full snapshot of the arena's current state."""
    snap = Snapshot(
        task_index=task_index,
        timestamp_iso=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        kind="full",
        envelope={
            "max_cells": arena.envelope.max_cells,
            "max_edges": arena.envelope.max_edges,
            "max_parameter_bytes": arena.envelope.max_parameter_bytes,
        },
    )

    alive_ids = arena.alive_ids()
    for cid in alive_ids.tolist():
        snap.cells.append(CellRecord(
            id=cid,
            state=_STATE_NAMES.get(int(arena.state[cid].item()), "active"),
            position=arena.position[cid].tolist(),
            epigenome=int(arena.epigenome[cid].item()),
            phenotype=_PHENOTYPE_NAMES.get(int(arena.phenotype_cache[cid].item()), "linear"),
            output_dim=int(arena.output_dim[cid].item()),
            lineage_root=int(arena.lineage_root[cid].item()),
            parent=int(arena.parent[cid].item()),
            rank=int(arena.rank[cid].item()),
            engagement=round(float(arena.engagement[cid].item()), 6),
            utility=round(float(arena.utility[cid].item()), 6),
            age=int(arena.age[cid].item()),
            forward_inclusion=bool(arena.forward_inclusion[cid].item()),
        ))

    for i in range(arena.edge_cursor):
        snap.edges.append(EdgeRecord(
            src=int(arena.edge_src[i].item()),
            dst=int(arena.edge_dst[i].item()),
            weight=round(float(arena.edge_weight[i].item()), 6),
        ))

    return snap


def capture_delta(arena: Arena, changed_ids: list[int], task_index: int = 0) -> Snapshot:
    """Capture a delta snapshot with only the specified cells."""
    snap = Snapshot(
        task_index=task_index,
        timestamp_iso=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        kind="delta",
    )

    for cid in changed_ids:
        if not arena.alive[cid]:
            continue
        snap.cells.append(CellRecord(
            id=cid,
            state=_STATE_NAMES.get(int(arena.state[cid].item()), "active"),
            position=arena.position[cid].tolist(),
            epigenome=int(arena.epigenome[cid].item()),
            phenotype=_PHENOTYPE_NAMES.get(int(arena.phenotype_cache[cid].item()), "linear"),
            output_dim=int(arena.output_dim[cid].item()),
            lineage_root=int(arena.lineage_root[cid].item()),
            parent=int(arena.parent[cid].item()),
            rank=int(arena.rank[cid].item()),
            engagement=round(float(arena.engagement[cid].item()), 6),
            utility=round(float(arena.utility[cid].item()), 6),
            age=int(arena.age[cid].item()),
            forward_inclusion=bool(arena.forward_inclusion[cid].item()),
        ))

    return snap
