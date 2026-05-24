"""Tests for learning subsystem — credit, frustration, manifold."""
from __future__ import annotations

import torch

from trioron.core import Envelope, construct
from trioron.core.state import CellState
from trioron.core.epigenome import CREDIT_ELIGIBLE, has_gene
from trioron.bases import minimal
from trioron.phenotype import default_dispatch_table
from trioron.learning import CreditTracker, FrustrationDetector, ManifoldArchive


def _make_substrate(input_dim=4, n_classes=3):
    return construct(
        base=minimal(input_dim, n_classes),
        envelope=Envelope(),
        dispatch_table=default_dispatch_table(),
        capacity=256,
    )


# ── Credit ───────────────────────────────────────────────────────


class TestCredit:

    def test_engagement_update(self):
        sub = _make_substrate()
        sub.prepare_training()
        tracker = CreditTracker(sub.arena)
        act = torch.ones(8, sub.arena.capacity) * 0.5
        tracker.update_engagement(act)
        out_ids = sub.scheduler._plan.output_ids.long()
        assert (sub.arena.engagement[out_ids] > 0).all()

    def test_no_lock_before_threshold(self):
        sub = _make_substrate()
        sub.prepare_training()
        tracker = CreditTracker(sub.arena)
        locked = tracker.consolidate()
        assert len(locked) == 0

    def test_lock_after_sustained_engagement(self):
        sub = _make_substrate()
        sub.prepare_training()
        tracker = CreditTracker(sub.arena)

        out_ids = sub.scheduler._plan.output_ids.long()
        sub.arena.engagement[out_ids] = 0.5
        sub.arena.utility[out_ids] = 1e-5

        for _ in range(3):
            tracker.consolidate()

        dormant = sub.arena.state[out_ids] == CellState.DORMANT
        assert dormant.any(), "At least one output cell should lock"

    def test_cooldown_prevents_immediate_relock(self):
        sub = _make_substrate()
        sub.prepare_training()
        tracker = CreditTracker(sub.arena)

        out_ids = sub.scheduler._plan.output_ids
        cid = int(out_ids[0].item())
        sub.arena.engagement[cid] = 0.5
        sub.arena.utility[cid] = 1e-5

        for _ in range(3):
            tracker.consolidate()
        assert sub.arena.state[cid] == CellState.DORMANT

        sub.arena.state[cid] = CellState.ACTIVE
        tracker.set_cooldown(cid, 2)
        sub.arena.engagement[cid] = 0.5
        sub.arena.utility[cid] = 1e-5

        locked = tracker.consolidate()
        assert cid not in locked, "Cell under cooldown should not relock"


# ── Frustration ──────────────────────────────────────────────────


class TestFrustration:

    def test_starts_at_one(self):
        det = FrustrationDetector()
        assert det.multiplier == 1.0

    def test_stagnant_loss_increases_multiplier(self):
        det = FrustrationDetector()
        for _ in range(300):
            det.step(1.0)
        assert det.multiplier > 1.0

    def test_reset_clears(self):
        det = FrustrationDetector()
        for _ in range(300):
            det.step(1.0)
        assert det.is_frustrated
        det.reset()
        assert det.multiplier == 1.0

    def test_improvement_resets_stuck(self):
        det = FrustrationDetector()
        for _ in range(100):
            det.step(1.0)
        for i in range(100):
            det.step(1.0 - 0.005 * i)
        assert not det.is_frustrated


# ── Manifold ─────────────────────────────────────────────────────


class TestManifold:

    def test_spawn_and_update(self):
        sub = _make_substrate()
        archive = ManifoldArchive(sub.arena)
        code = torch.randn(32, 128)
        archive.update_class(0, code)
        assert archive.has_class(0)
        assert archive.n_classes == 1

    def test_sample_shape(self):
        sub = _make_substrate()
        archive = ManifoldArchive(sub.arena)
        code = torch.randn(100, 128)
        archive.update_class(0, code)
        astro = archive.get(0)
        samples = astro.sample(16)
        assert samples.shape == (16, 128)

    def test_finalize_locks_astrocyte(self):
        sub = _make_substrate()
        archive = ManifoldArchive(sub.arena)
        code = torch.randn(50, 128)
        archive.update_class(0, code)
        archive.finalize_all()
        astro = archive.get(0)
        assert sub.arena.state[astro.cell_id] == CellState.DORMANT

    def test_replay_excludes_current(self):
        sub = _make_substrate()
        archive = ManifoldArchive(sub.arena)
        for c in range(3):
            archive.update_class(c, torch.randn(50, 128))
        archive.finalize_all()
        batches = archive.replay_batches(8, exclude_class=1)
        class_ids = [cid for _, cid in batches]
        assert 1 not in class_ids
        assert len(class_ids) == 2

    def test_storage_bytes(self):
        sub = _make_substrate()
        archive = ManifoldArchive(sub.arena)
        archive.update_class(0, torch.randn(10, 128))
        assert archive.storage_bytes() == 128 * 4 * 2
