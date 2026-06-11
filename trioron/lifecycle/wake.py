"""Wake — deserialize a shipped substrate.  See spec §5.4."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from trioron.core.arena import Arena
from trioron.core.construct import Substrate
from trioron.core.envelope import Envelope
from trioron.core.state import CellState


def wake(
    path: str | Path,
    *,
    dispatch_table: dict | None = None,
    device: str | torch.device = "cpu",
    capacity: int | None = None,
) -> Substrate:
    """Deserialize a substrate from a checkpoint.

    The returned substrate is immediately runnable for inference.
    Call :func:`wake_for_training` to prepare for further training.

    Args:
        path: checkpoint file path.
        dispatch_table: phenotype dispatch table (required for forward passes).
        device: target device.
        capacity: arena capacity override (must be >= checkpoint cursor).

    Returns:
        A ready-to-use Substrate.
    """
    state = torch.load(Path(path), map_location=device, weights_only=False)
    arena_state = state["arena"]
    env_state = state.get("envelope", {})

    envelope = Envelope(
        max_cells=env_state.get("max_cells"),
        max_edges=env_state.get("max_edges"),
        max_parameter_bytes=env_state.get("max_parameter_bytes"),
    )

    saved_cursor = arena_state["cursor"]
    cap = capacity or max(arena_state.get("capacity", saved_cursor * 2), saved_cursor * 2)

    arena = Arena(envelope, device=device, capacity=cap)
    _restore_arena(arena, arena_state)

    substrate = Substrate(arena, dispatch_table)
    substrate.compile()

    substrate._checkpoint_meta = {
        "schema_version": state.get("schema_version", 0),
        "task_index": state.get("task_index"),
        "has_optimizer_state": "optimizer_state" in state,
    }

    if "optimizer_state" in state:
        substrate._optimizer_state = state["optimizer_state"]

    if "pcll" in state:
        # restore via: PCLLController(substrate).load_state_dict(substrate._pcll_state)
        substrate._pcll_state = state["pcll"]

    return substrate


def wake_for_training(substrate: Substrate) -> None:
    """Prepare a woken substrate for further training.

    Resets frustration counters and engagement EMAs for active cells,
    then enables gradients and recompiles.
    """
    a = substrate.arena
    active = a.alive & (a.state == CellState.ACTIVE)
    a.engagement[active] = 0.0
    a.utility[active] = 0.0

    substrate.prepare_training()


def get_optimizer_state(substrate: Substrate) -> dict | None:
    """Retrieve stored optimizer state from a woken substrate, if any."""
    return getattr(substrate, "_optimizer_state", None)


def _restore_arena(arena: Arena, saved: dict) -> None:
    """Copy saved tensors back into a fresh arena."""
    a = arena
    n = saved["cursor"]
    a.cursor = n

    a.bias[:n] = saved["bias"].to(a.device)
    a.engagement[:n] = saved["engagement"].to(a.device)
    a.utility[:n] = saved["utility"].to(a.device)
    a.position[:n] = saved["position"].to(a.device)
    a.epigenome[:n] = saved["epigenome"].to(a.device)
    a.phenotype_cache[:n] = saved["phenotype_cache"].to(a.device)
    a.lineage_root[:n] = saved["lineage_root"].to(a.device)
    a.parent[:n] = saved["parent"].to(a.device)
    a.rank[:n] = saved["rank"].to(a.device)
    a.state[:n] = saved["state"].to(a.device)
    a.age[:n] = saved["age"].to(a.device)
    a.output_dim[:n] = saved["output_dim"].to(a.device)
    a.forward_inclusion[:n] = saved["forward_inclusion"].to(a.device)
    if "division_mode" in saved:
        a.division_mode[:n] = saved["division_mode"].to(a.device)
    a.alive[:n] = saved["alive"].to(a.device)
    # Dendritic compartmentalization (spec §3.6) — restore branch structure +
    # learned α so a grown dendrite stays a dendrite across ship/wake.
    if "n_branches" in saved:
        a.n_branches[:n] = saved["n_branches"].to(a.device)
        with torch.no_grad():
            cols = min(a.branch_alpha.shape[1], saved["branch_alpha"].shape[1])
            a.branch_alpha[:n, :cols] = saved["branch_alpha"][:, :cols].to(a.device)

    if "receptor_levels" in saved:  # PCLL (spec §10.3)
        a.receptor_levels[:n] = saved["receptor_levels"].to(a.device)
        a.lockin_re[:n] = saved["lockin_re"].to(a.device)
        a.lockin_im[:n] = saved["lockin_im"].to(a.device)
        a.lockin_n[:n] = saved["lockin_n"].to(a.device)

    ec = saved["edge_cursor"]
    a.edge_cursor = ec
    a.edge_src[:ec] = saved["edge_src"].to(a.device)
    a.edge_dst[:ec] = saved["edge_dst"].to(a.device)
    with torch.no_grad():
        a.edge_weight[:ec] = saved["edge_weight"].to(a.device)
    if "edge_branch" in saved:
        a.edge_branch[:ec] = saved["edge_branch"].to(a.device)

    a.rank_dirty = True
