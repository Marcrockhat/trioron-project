"""Tests for cell_position buffer — the locus coordinate primitive.

Phase A: pure plumbing. No forward-path consumer yet. These tests
verify the buffer exists, extends/contracts correctly with grow_node /
prune_node, round-trips through state_dict, and v1 donors get zero-
defaults.

Phase B+ (orchestrator, position-aware connectivity) tests come later
once develop_positions and grow_input_local land.
"""

from __future__ import annotations

import pytest
import torch

from trioron.node import TrioronLayer


# ---------- default state ----------

def test_cell_position_buffer_exists():
    layer = TrioronLayer(fan_in=4, n_nodes=3)
    assert hasattr(layer, "cell_position")
    assert layer.cell_position.shape == (3, 3)
    assert layer.cell_position.dtype == torch.float32


def test_cell_position_default_zero():
    layer = TrioronLayer(fan_in=4, n_nodes=5)
    assert (layer.cell_position == 0.0).all()


def test_position_dim_attribute():
    layer = TrioronLayer(fan_in=4, n_nodes=3)
    assert layer.position_dim == 3


# ---------- grow_node / prune_node ----------

def test_grow_node_appends_zero_position_row():
    layer = TrioronLayer(fan_in=4, n_nodes=2)
    # Drift existing positions so we can see they're preserved.
    with torch.no_grad():
        layer.cell_position[0] = torch.tensor([1.0, 2.0, 3.0])
        layer.cell_position[1] = torch.tensor([4.0, 5.0, 6.0])
    new_idx = layer.grow_node()
    assert new_idx == 2
    assert layer.cell_position.shape == (3, 3)
    # Existing rows preserved.
    assert torch.allclose(
        layer.cell_position[0], torch.tensor([1.0, 2.0, 3.0])
    )
    assert torch.allclose(
        layer.cell_position[1], torch.tensor([4.0, 5.0, 6.0])
    )
    # New row defaulted to zero — caller (or developmental orchestrator)
    # is responsible for assigning a meaningful position.
    assert (layer.cell_position[2] == 0.0).all()


def test_prune_node_drops_correct_position_row():
    layer = TrioronLayer(fan_in=4, n_nodes=3)
    with torch.no_grad():
        layer.cell_position[0] = torch.tensor([1.0, 0.0, 0.0])
        layer.cell_position[1] = torch.tensor([0.0, 1.0, 0.0])
        layer.cell_position[2] = torch.tensor([0.0, 0.0, 1.0])
    layer.prune_node(1)
    assert layer.cell_position.shape == (2, 3)
    assert torch.allclose(
        layer.cell_position[0], torch.tensor([1.0, 0.0, 0.0])
    )
    # Index 1 after prune is what used to be index 2.
    assert torch.allclose(
        layer.cell_position[1], torch.tensor([0.0, 0.0, 1.0])
    )


def test_grow_then_prune_node_preserves_positions():
    """Stress-test: grow → assign → grow → prune middle. Surviving
    positions should be exactly correct."""
    layer = TrioronLayer(fan_in=4, n_nodes=2)
    with torch.no_grad():
        layer.cell_position[0] = torch.tensor([0.1, 0.1, 0.1])
        layer.cell_position[1] = torch.tensor([0.2, 0.2, 0.2])
    layer.grow_node()
    with torch.no_grad():
        layer.cell_position[2] = torch.tensor([0.3, 0.3, 0.3])
    layer.grow_node()
    with torch.no_grad():
        layer.cell_position[3] = torch.tensor([0.4, 0.4, 0.4])
    layer.prune_node(1)
    # Surviving cells were originally 0, 2, 3 → now at indices 0, 1, 2.
    assert layer.cell_position.shape == (3, 3)
    assert torch.allclose(layer.cell_position[0], torch.tensor([0.1] * 3))
    assert torch.allclose(layer.cell_position[1], torch.tensor([0.3] * 3))
    assert torch.allclose(layer.cell_position[2], torch.tensor([0.4] * 3))


# ---------- grow_input / prune_input ----------

def test_grow_input_does_not_change_cell_position():
    """grow_input changes fan_in (per-column), not n_nodes. cell_position
    is per-cell so it should be untouched."""
    layer = TrioronLayer(fan_in=4, n_nodes=3)
    with torch.no_grad():
        layer.cell_position[0] = torch.tensor([1.0, 2.0, 3.0])
        layer.cell_position[1] = torch.tensor([4.0, 5.0, 6.0])
        layer.cell_position[2] = torch.tensor([7.0, 8.0, 9.0])
    before = layer.cell_position.clone()
    layer.grow_input()
    assert layer.cell_position.shape == (3, 3)
    assert torch.equal(layer.cell_position, before)


def test_prune_input_does_not_change_cell_position():
    layer = TrioronLayer(fan_in=4, n_nodes=2)
    with torch.no_grad():
        layer.cell_position[0] = torch.tensor([1.0, 1.0, 1.0])
        layer.cell_position[1] = torch.tensor([2.0, 2.0, 2.0])
    before = layer.cell_position.clone()
    layer.prune_input(2)
    assert layer.cell_position.shape == (2, 3)
    assert torch.equal(layer.cell_position, before)


# ---------- state_dict round-trip ----------

def test_state_dict_round_trip_preserves_positions():
    src = TrioronLayer(fan_in=4, n_nodes=3)
    with torch.no_grad():
        src.cell_position[0] = torch.tensor([0.5, 0.5, 0.5])
        src.cell_position[1] = torch.tensor([-0.3, 0.2, 0.7])
    sd = src.state_dict()
    dst = TrioronLayer(fan_in=4, n_nodes=3)
    dst.load_state_dict(sd)
    assert torch.equal(dst.cell_position, src.cell_position)


def test_v1_state_dict_load_defaults_cell_position_to_zero():
    """A v1 donor without cell_position (and without other 2.0 keys)
    should load cleanly, with cell_position injected as zero."""
    src = TrioronLayer(fan_in=4, n_nodes=3)
    sd = src.state_dict()
    # Strip the Axis 5 keys + cell_position to simulate v1.
    for k in (
        "input_sources", "input_archived",
        "axonal_gain", "axonal_gain_anchor",
        "branch_id", "branch_weight", "branch_weight_anchor",
        "fisher_branch_weight", "B_per_node",
        "internal_stress", "branch_utility", "dendrite_orphan",
        "cell_position",
    ):
        sd.pop(k, None)
    dst = TrioronLayer(fan_in=4, n_nodes=3)
    dst.load_state_dict(sd)
    assert dst.cell_position.shape == (3, 3)
    assert (dst.cell_position == 0.0).all()


# ---------- no forward consumer yet ----------

def test_setting_cell_position_does_not_affect_forward():
    """Phase A is purely additive — no forward-path code reads
    cell_position yet. The byte-identity property must hold whether
    positions are zero or non-zero."""
    torch.manual_seed(31)
    layer = TrioronLayer(fan_in=4, n_nodes=3, activation="relu")
    x = torch.randn(5, 4)
    y_before = layer(x).detach()
    with torch.no_grad():
        layer.cell_position[:] = torch.randn(3, 3) * 10.0
    y_after = layer(x).detach()
    assert torch.equal(y_before, y_after)
