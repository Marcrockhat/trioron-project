"""PCLL substrate machinery (spec §10) — fast CI guards.

The full I1 validation is experiments/progenitor/run_i1_substrate.py
(exact parity with the standalone schedule-learning PASS); these tests
pin the substrate plumbing: receptor injection, mask rule, deposits,
boundary meeting, frustration adapter, ship/wake round-trip.
"""
from __future__ import annotations

import math
import random

import torch

from trioron.core import construct
from trioron.core.epigenome import PERCEPTION, RECEPTOR, set_gene
from trioron.lifecycle.ship import ship
from trioron.lifecycle.wake import wake
from trioron.pcll import (
    EMPTY,
    N_QUANTA,
    PCLLController,
    RESOLVED,
    evidence_mask,
)

F = 4


def _receptor_base(n: int):
    def base(sub):
        a = sub.arena
        ids = a.alloc(n)
        for cid in ids.tolist():
            epi = int(a.epigenome[cid].item())
            a.epigenome[cid] = set_gene(set_gene(epi, PERCEPTION), RECEPTOR)
    return base


def _organism(n: int = F + 1):
    sub = construct(_receptor_base(n), capacity=16)
    return sub, PCLLController(sub)


def _period(ranges, rng, n_obs=300):
    """[n_obs, F+1] pocket-valued samples + the gain-reference sentinel."""
    qs = [[rng.randint(lo, hi) for lo, hi in ranges] for _ in range(n_obs)]
    x = torch.tensor(qs, dtype=torch.float32)
    return torch.cat([x, torch.full((n_obs, 1), float(N_QUANTA))], dim=1)


CLASS_A = [(100, 200), (400, 500), (700, 800), (250, 350)]
CLASS_B = [(300, 400), (600, 700), (100, 200), (850, 950)]
NOISE = [(0, N_QUANTA)] * F


class TestReceptorInjection:
    def test_activation_is_phase(self):
        sub, controller = _organism()
        rng = random.Random(0)
        x = _period(CLASS_A, rng, n_obs=8)
        controller.observe(x)
        q = sub.scheduler._last_receptor_q
        ids = sub.scheduler._plan.receptor_ids.long()
        assert torch.allclose(
            sub.last_activations[:, ids], 2 * math.pi * q / N_QUANTA
        )

    def test_sentinel_makes_quantizer_identity(self):
        sub, controller = _organism()
        rng = random.Random(0)
        x = _period(CLASS_A, rng, n_obs=8)
        controller.observe(x)
        q = sub.scheduler._last_receptor_q
        assert torch.equal(q, x)  # pockets round-trip exactly

    def test_mask_rule_reference_not_evidence(self):
        sub, controller = _organism()
        a = sub.arena
        # flat input: every feature equals the sample max → all saturated
        controller.observe(torch.full((10, F + 1), 7.0))
        ids = controller.receptor_ids.long()
        assert a.lockin_n[ids].sum().item() == 0.0  # deposits nothing
        q = torch.tensor([[0.0, 500.0, 1000.0]])
        assert evidence_mask(q).tolist() == [[False, True, False]]


class TestBoundaryMeeting:
    def test_meeting_runs_at_end_task_and_resets(self):
        sub, controller = _organism()
        a = sub.arena
        rng = random.Random(1)
        controller.observe(_period(CLASS_A, rng))
        ids = controller.receptor_ids.long()
        assert a.lockin_n[ids].sum().item() > 0
        report = sub.end_task()
        assert report is not None and report.event == "birth"
        assert a.lockin_n[ids].sum().item() == 0.0  # accumulators reset

    def test_discovery_birth_match_and_resolution(self):
        sub, controller = _organism()
        rng = random.Random(2)
        for ranges in (CLASS_A, CLASS_B):
            controller.observe(_period(ranges, rng))
            assert sub.end_task().event == "birth"
        controller.observe(_period(CLASS_A, rng))
        report = sub.end_task()
        assert report.event == "match"
        assert report.resolution.status == RESOLVED
        assert not controller.frustration.is_frustrated

    def test_noise_reads_empty_and_stresses_adapter(self):
        sub, controller = _organism()
        rng = random.Random(3)
        controller.observe(_period(NOISE, rng))
        report = sub.end_task()
        assert report.event == "empty"
        assert controller.frustration.is_frustrated
        assert controller.frustration.multiplier > 5.0  # the ceiling

    def test_end_task_without_controller_returns_none(self):
        sub = construct(_receptor_base(F + 1), capacity=16)
        assert sub.end_task() is None


class TestShipWake:
    def test_lockin_state_round_trips(self, tmp_path):
        sub, controller = _organism()
        a = sub.arena
        rng = random.Random(4)
        controller.observe(_period(CLASS_A, rng))  # mid-period state
        a.receptor_levels[2] = 3
        path = ship(sub, tmp_path / "pcll.pt")
        woken = wake(path)
        n = a.cursor
        for name in ("lockin_re", "lockin_im", "lockin_n", "receptor_levels"):
            assert torch.equal(
                getattr(woken.arena, name)[:n], getattr(a, name)[:n]
            ), f"{name} did not round-trip"
        # the woken plan re-derives receptor identity from the epigenome
        assert torch.equal(
            woken.scheduler._plan.receptor_ids,
            sub.scheduler._plan.receptor_ids,
        )
