"""Ship — serialize a substrate to a checkpoint.  See spec §5.4."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from trioron.core.arena import Arena
from trioron.core.construct import Substrate
from trioron.core.state import CellState


_SCHEMA_VERSION: int = 1


def ship(
    substrate: Substrate,
    path: str | Path,
    *,
    consolidate: bool = False,
    quantize_dormant: bool = False,
    task_index: int | None = None,
    optimizer_state: dict | None = None,
) -> Path:
    """Serialize the substrate to ``path``.

    Args:
        substrate: the substrate to ship.
        path: output file path (typically ``substrate_<task>.pt``).
        consolidate: if True, run a final consolidation-dream pass before saving.
        quantize_dormant: if True, quantize dormant cells' weights to int8.
        task_index: current task index (stored in metadata).
        optimizer_state: optional optimizer state dict to include.

    Returns:
        The path written.
    """
    path = Path(path)
    a = substrate.arena

    state: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "task_index": task_index,
    }

    state["arena"] = _serialize_arena(a)

    if quantize_dormant:
        state["arena"]["quantized_dormant"] = _quantize_dormant_weights(a)

    state["envelope"] = {
        "max_cells": a.envelope.max_cells,
        "max_edges": a.envelope.max_edges,
        "max_parameter_bytes": a.envelope.max_parameter_bytes,
    }

    if optimizer_state is not None:
        state["optimizer_state"] = optimizer_state

    torch.save(state, path)
    return path


def _serialize_arena(arena: Arena) -> dict[str, Any]:
    """Snapshot all arena tensors."""
    a = arena
    return {
        "capacity": a.capacity,
        "cursor": a.cursor,
        "edge_cursor": a.edge_cursor,
        "bias": a.bias[:a.cursor].clone(),
        "engagement": a.engagement[:a.cursor].clone(),
        "utility": a.utility[:a.cursor].clone(),
        "position": a.position[:a.cursor].clone(),
        "epigenome": a.epigenome[:a.cursor].clone(),
        "phenotype_cache": a.phenotype_cache[:a.cursor].clone(),
        "lineage_root": a.lineage_root[:a.cursor].clone(),
        "parent": a.parent[:a.cursor].clone(),
        "rank": a.rank[:a.cursor].clone(),
        "state": a.state[:a.cursor].clone(),
        "age": a.age[:a.cursor].clone(),
        "output_dim": a.output_dim[:a.cursor].clone(),
        "forward_inclusion": a.forward_inclusion[:a.cursor].clone(),
        "alive": a.alive[:a.cursor].clone(),
        "edge_src": a.edge_src[:a.edge_cursor].clone(),
        "edge_dst": a.edge_dst[:a.edge_cursor].clone(),
        "edge_weight": a.edge_weight[:a.edge_cursor].clone(),
    }


def _quantize_dormant_weights(arena: Arena) -> dict[str, torch.Tensor]:
    """Int8 quantization of dormant cells' bias values (one scale per cell)."""
    a = arena
    dormant = a.alive[:a.cursor] & (a.state[:a.cursor] == CellState.DORMANT)
    d_ids = dormant.nonzero(as_tuple=False).squeeze(-1)

    if d_ids.numel() == 0:
        return {"cell_ids": d_ids, "scales": torch.tensor([]), "quantized_bias": torch.tensor([], dtype=torch.int8)}

    biases = a.bias[d_ids]
    scales = biases.abs().clamp(min=1e-8)
    quantized = (biases / scales * 127).round().clamp(-128, 127).to(torch.int8)

    return {
        "cell_ids": d_ids.clone(),
        "scales": scales.clone(),
        "quantized_bias": quantized,
    }
