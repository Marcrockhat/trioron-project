"""Tests for streaming context-shift detection (learning/shift.py)."""
from __future__ import annotations

import torch

from trioron.learning import ShiftDetector, SurpriseBaseline


def _stream(mu, n, dim=8, noise=1.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    return mu + noise * torch.randn(n, dim, generator=g)


class TestSurpriseBaseline:

    def test_tracks_stationary_mean(self):
        base = SurpriseBaseline(dim=8)
        xs = _stream(torch.full((8,), 3.0), 400)
        for x in xs:
            base.update(x)
        assert (base.mu - 3.0).abs().max() < 0.5

    def test_alpha_rises_with_drift_snr(self):
        # High-drift stream should derive a larger alpha than a static one.
        torch.manual_seed(0)
        static = SurpriseBaseline(dim=4)
        drifting = SurpriseBaseline(dim=4)
        level = torch.zeros(4)
        for _ in range(500):
            static.update(torch.randn(4) * 0.5)
            level = level + torch.randn(4) * 2.0     # strong random-walk drift
            drifting.update(level + torch.randn(4) * 0.5)
        assert drifting.alpha.mean() > static.alpha.mean()

    def test_surprise_spikes_on_context_switch(self):
        base = SurpriseBaseline(dim=8)
        for x in _stream(torch.zeros(8), 300):
            s_before = base.update(x)
        s_after = base.update(torch.full((8,), 8.0))
        assert float(s_after.mean()) > 4 * max(float(s_before.mean()), 0.1)


class TestShiftDetector:

    def test_no_false_fires_on_stationary(self):
        det = ShiftDetector(dim=8)
        for x in _stream(torch.zeros(8), 1000):
            det.update(x)
        assert len(det.events) == 0

    def test_detects_abrupt_switch_promptly(self):
        det = ShiftDetector(dim=8)
        for x in _stream(torch.zeros(8), 500):
            det.update(x)
        for x in _stream(torch.full((8,), 5.0), 50, seed=1):
            det.update(x)
        assert len(det.events) == 1
        assert det.events[0].step - 500 <= 10, "shift declared too slowly"

    def test_recovers_after_reanchor(self):
        det = ShiftDetector(dim=8)
        for x in _stream(torch.zeros(8), 500):
            det.update(x)
        for x in _stream(torch.full((8,), 5.0), 400, seed=1):
            det.update(x)
        # One shift, then the new context becomes the quiet baseline.
        assert len(det.events) == 1
        assert det.z < det.z_threshold

    def test_two_switches_two_events(self):
        det = ShiftDetector(dim=8)
        for x in _stream(torch.zeros(8), 400):
            det.update(x)
        for x in _stream(torch.full((8,), 5.0), 400, seed=1):
            det.update(x)
        for x in _stream(torch.full((8,), -5.0), 400, seed=2):
            det.update(x)
        assert len(det.events) == 2

    def test_slow_drift_absorbed_without_firing(self):
        # Random-walk drift is what the Kalman alpha exists to track.
        torch.manual_seed(3)
        det = ShiftDetector(dim=8)
        level = torch.zeros(8)
        for _ in range(1500):
            level = level + torch.randn(8) * 0.05
            det.update(level + torch.randn(8))
        assert len(det.events) == 0
