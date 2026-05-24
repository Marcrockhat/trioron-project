"""Structure detector — identifies named patterns in snapshots.  See spec §7.2."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from .snapshot import Snapshot


@dataclass
class Structure:
    id: str
    type: str
    task_first_seen: int
    cell_ids: list[int]
    label: str
    confidence: float = 1.0
    extra: dict[str, Any] = field(default_factory=dict)


def detect_attention_heads(snap: Snapshot) -> list[Structure]:
    """Find 4-cell lineage groups with Q/K/V/OUT pattern."""
    # Phase 1: attention phenotype not yet implemented, return empty
    return []


def detect_conv_banks(snap: Snapshot) -> list[Structure]:
    """Find astrocyte + receptor lineage groups with conv gene."""
    return []


def detect_dormant_regions(snap: Snapshot) -> list[Structure]:
    """Find clusters of dormant cells co-positioned in space."""
    structures = []
    dormant = [c for c in snap.cells if c.state == "dormant"]
    if len(dormant) >= 3:
        structures.append(Structure(
            id=f"dormant-t{snap.task_index}",
            type="dormant_region",
            task_first_seen=snap.task_index,
            cell_ids=[c.id for c in dormant],
            label=f"dormant region ({len(dormant)} cells)",
            confidence=0.9,
        ))
    return structures


def detect_lineage_clusters(snap: Snapshot) -> list[Structure]:
    """Group cells by lineage root and flag large clusters."""
    from collections import Counter
    roots = Counter(c.lineage_root for c in snap.cells if c.lineage_root >= 0)
    structures = []
    for root, count in roots.items():
        if count >= 4:
            members = [c.id for c in snap.cells if c.lineage_root == root]
            structures.append(Structure(
                id=f"lineage-{root}-t{snap.task_index}",
                type="lineage_cluster",
                task_first_seen=snap.task_index,
                cell_ids=members,
                label=f"lineage {root} ({count} cells)",
                confidence=0.8,
            ))
    return structures


def detect_all(snap: Snapshot) -> list[Structure]:
    """Run all detection templates on a snapshot."""
    structures = []
    structures.extend(detect_attention_heads(snap))
    structures.extend(detect_conv_banks(snap))
    structures.extend(detect_dormant_regions(snap))
    structures.extend(detect_lineage_clusters(snap))
    return structures


def detect_from_directory(snapshot_dir: str | Path, output: str | Path | None = None) -> dict:
    """Run detection on all snapshots in a directory."""
    snap_dir = Path(snapshot_dir)
    result = {"structures_by_snapshot": []}

    for json_file in sorted(snap_dir.glob("*.json")):
        import json as _json
        data = _json.loads(json_file.read_text())
        snap = _snapshot_from_dict(data)
        structures = detect_all(snap)
        result["structures_by_snapshot"].append({
            "file": json_file.name,
            "task_index": snap.task_index,
            "structures": [asdict(s) for s in structures],
        })

    if output:
        out_path = Path(output)
        out_path.write_text(json.dumps(result, indent=2))

    return result


def _snapshot_from_dict(data: dict) -> Snapshot:
    """Reconstruct a minimal Snapshot from a JSON dict for detection."""
    from .snapshot import CellRecord
    snap = Snapshot(
        task_index=data.get("task_index", 0),
        kind=data.get("kind", "full"),
    )
    for c in data.get("cells", []):
        snap.cells.append(CellRecord(**c))
    return snap
