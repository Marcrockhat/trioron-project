"""Tests for lifecycle (growth) and dream cycle."""
from __future__ import annotations

import torch

from trioron.core import Envelope, construct
from trioron.core.state import CellState
from trioron.bases import minimal
from trioron.phenotype import default_dispatch_table
from trioron.learning import (
    CreditTracker, ManifoldArchive, dream_cycle, DreamConfig,
)
from trioron.learning.rejuvenate import rejuvenate
from trioron.lifecycle import divide, check_growth_trigger, GrowthConfig


def _make_substrate(input_dim=4, n_classes=3, capacity=256):
    return construct(
        base=minimal(input_dim, n_classes),
        envelope=Envelope(),
        dispatch_table=default_dispatch_table(),
        capacity=capacity,
    )


# ── Growth ───────────────────────────────────────────────────────


class TestGrowth:

    def test_divide_creates_child(self):
        sub = _make_substrate()
        before = sub.n_cells
        out_ids = sub.scheduler._plan.output_ids
        parent = int(out_ids[0].item())
        event = divide(sub.arena, parent)
        assert event is not None
        assert sub.n_cells == before + 1
        assert sub.arena.parent[event.child_id] == parent

    def test_child_inherits_epigenome(self):
        sub = _make_substrate()
        out_ids = sub.scheduler._plan.output_ids
        parent = int(out_ids[0].item())
        parent_epi = int(sub.arena.epigenome[parent].item())
        event = divide(sub.arena, parent)
        child_epi = int(sub.arena.epigenome[event.child_id].item())
        assert child_epi == parent_epi

    def test_child_has_edges(self):
        sub = _make_substrate()
        out_ids = sub.scheduler._plan.output_ids
        parent = int(out_ids[0].item())
        event = divide(sub.arena, parent)
        assert event.n_inherited_edges > 0

    def test_envelope_blocks_growth(self):
        sub = construct(
            base=minimal(4, 3),
            envelope=Envelope(max_cells=7),
            dispatch_table=default_dispatch_table(),
            capacity=8,
        )
        out_ids = sub.scheduler._plan.output_ids
        parent = int(out_ids[0].item())
        event = divide(sub.arena, parent)
        assert event is None

    def test_growth_trigger_logic(self):
        assert not check_growth_trigger(1.5, 50)
        assert check_growth_trigger(2.5, 60)
        assert not check_growth_trigger(2.5, 10)

    def test_plasticity_gate_blocks_mature_cell(self):
        # design §3.5: a matured (high-λ) cell loses divisibility. Gate OFF (default)
        # divides any cell; gate ON refuses a parent whose λ exceeds the threshold while
        # still dividing a plastic (low-λ) one.
        sub = _make_substrate()
        out_ids = sub.scheduler._plan.output_ids
        parent = int(out_ids[0].item())
        cfg = GrowthConfig(divide_lambda_max=1e-2)

        # plastic parent (λ at default ~0) divides under the gate
        sub.arena.node_lambda[parent] = 0.0
        assert divide(sub.arena, parent, cfg) is not None

        # matured parent (λ above threshold) is refused
        other = int(out_ids[1].item())
        sub.arena.node_lambda[other] = 1.0
        assert divide(sub.arena, other, cfg) is None
        # ...but the same mature cell divides with the gate OFF (legacy behavior preserved)
        assert divide(sub.arena, other, GrowthConfig()) is not None

    def test_divide_and_forward(self):
        sub = _make_substrate()
        sub.prepare_training()
        out_ids = sub.scheduler._plan.output_ids
        parent = int(out_ids[0].item())
        divide(sub.arena, parent)
        sub.compile()

        x = torch.randn(2, 4)
        logits = sub(x)
        # child inherits OUTPUT gene, so 4 output cells now
        assert logits.shape == (2, 4)


# ── Rejuvenate ───────────────────────────────────────────────────


class TestRejuvenate:

    def test_rejuvenate_dormant(self):
        sub = _make_substrate()
        out_ids = sub.scheduler._plan.output_ids
        cid = int(out_ids[0].item())
        sub.arena.state[cid] = CellState.DORMANT
        assert rejuvenate(sub.arena, cid)
        assert sub.arena.state[cid] == CellState.ACTIVE

    def test_rejuvenate_active_noop(self):
        sub = _make_substrate()
        out_ids = sub.scheduler._plan.output_ids
        cid = int(out_ids[0].item())
        assert not rejuvenate(sub.arena, cid)


# ── Dream Cycle ──────────────────────────────────────────────────


class TestDreamCycle:

    def test_dream_runs_without_archive(self):
        sub = _make_substrate()
        sub.prepare_training()
        credit = CreditTracker(sub.arena)
        archive = ManifoldArchive(sub.arena)
        result = dream_cycle(sub, credit, archive)
        assert result.n_locked == 0
        assert result.n_rejuvenated == 0

    def test_dream_locks_eligible_cells(self):
        sub = _make_substrate()
        sub.prepare_training()
        credit = CreditTracker(sub.arena)
        archive = ManifoldArchive(sub.arena)

        out_ids = sub.scheduler._plan.output_ids.long()
        sub.arena.engagement[out_ids] = 0.5
        sub.arena.utility[out_ids] = 1e-5
        for _ in range(2):
            credit.consolidate()

        result = dream_cycle(sub, credit, archive)
        assert result.n_locked > 0

    def test_dream_with_replay(self):
        sub = _make_substrate()
        sub.prepare_training()
        credit = CreditTracker(sub.arena)
        archive = ManifoldArchive(sub.arena)

        for c in range(3):
            archive.update_class(c, torch.randn(50, 4))
        archive.finalize_all()

        cfg = DreamConfig(replay_batch_size=8, replay_lr=0.001)
        result = dream_cycle(sub, credit, archive, current_class=2, cfg=cfg)
        assert result.replay_loss > 0

    def test_dream_rejuvenates_candidates(self):
        sub = _make_substrate()
        sub.prepare_training()
        credit = CreditTracker(sub.arena)
        archive = ManifoldArchive(sub.arena)

        out_ids = sub.scheduler._plan.output_ids
        cid = int(out_ids[0].item())
        sub.arena.state[cid] = CellState.DORMANT
        sub.arena.engagement[cid] = 0.5

        result = dream_cycle(sub, credit, archive)
        assert cid in result.rejuvenated_ids
