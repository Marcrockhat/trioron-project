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


class TestPerceptionGenesis:
    """Period-1 developmental program (spec §10.6, integration phase I2)."""

    @staticmethod
    def _period(rng, n=300, col3="noise"):
        import random as _r
        rows = []
        for _ in range(n):
            rows.append([
                0.0,                                   # col0: dead → starve
                1.0 if rng.random() < 0.9 else 0.0,    # col1: binary → k=2
                4.0 + rng.random(),                    # col2: narrow → coherent
                rng.random() * 10.0,                   # col3: broad → incoherent
            ])
        return torch.tensor(rows)

    def _grow(self):
        import random
        from trioron.pcll import PerceptionGenesis, germline_base
        rng = random.Random(0)
        sub = construct(germline_base, capacity=64)
        pg = PerceptionGenesis(sub)
        pg.feed(self._period(rng))
        sitting = sub.end_task()
        return sub, pg, sitting, rng

    def test_first_sitting_verdicts(self):
        sub, pg, sitting, _ = self._grow()
        assert sitting.starved == [0]
        assert sitting.discrete == {1: 2}
        kinds = {c: v.kind for c, v in sitting.verdicts.items()}
        assert kinds[2] == "continuous" and kinds[3] == "continuous"
        # natal replay: the period-1 class is LEARNED at the sitting (s030)
        assert sitting.event == "birth" and sitting.class_name is not None
        assert len(pg.controller.learner.classes) == 1
        # starved column: receptor withdrawn, cell dormant, mapping intact
        from trioron.core.epigenome import RECEPTOR, has_gene
        a = sub.arena
        cid = int(pg.perception_ids[0].item())
        assert not has_gene(int(a.epigenome[cid].item()), RECEPTOR)
        assert int(a.state[cid].item()) == 1  # CellState.DORMANT
        assert sub.scheduler._plan.perception_ids.numel() == 4  # mapping holds
        assert sub.scheduler._plan.receptor_ids.numel() == 3

    def test_handover_and_codec_labeled_lines(self):
        sub, pg, _, rng = self._grow()
        assert sub._pcll is pg.controller  # controller attached in pg's place
        pg.controller.observe(self._period(rng, n=8))
        q = sub.scheduler._last_receptor_q
        cols = sub.scheduler._plan.receptor_cols.tolist()
        b = cols.index(1)  # the binary column's receptor index
        assert set(q[:, b].tolist()) <= {250.0, 750.0}  # k=2 labeled lines

    def test_habituation_retires_noise_keeps_real(self):
        from trioron.pcll import RETIRE_PATIENCE
        sub, pg, _, rng = self._grow()
        ctrl = pg.controller
        cols = sub.scheduler._plan.receptor_cols.tolist()
        retired_at = {}
        for p in range(1, RETIRE_PATIENCE + 2):
            ctrl.observe(self._period(rng))
            report = sub.end_task()
            for r in report.retired:
                retired_at[cols[r]] = p
        # the natal replay (s030) is the first testimony, so the noise column
        # retires after RETIRE_PATIENCE-1 further periods — 3 testimonies total
        assert retired_at == {3: RETIRE_PATIENCE - 1}
        kept = {cols[i] for i in range(len(cols)) if ctrl.read_mask[i]}
        assert kept == {1, 2}
        # shadow semantics: retired receptor keeps its gene and deposits
        from trioron.core.epigenome import RECEPTOR, has_gene
        a = sub.arena
        noise_cell = int(sub.scheduler._plan.receptor_ids[cols.index(3)].item())
        assert has_gene(int(a.epigenome[noise_cell].item()), RECEPTOR)
        ctrl.observe(self._period(rng, n=50))
        assert a.lockin_n[noise_cell].item() > 0

    def test_germline_never_forward_and_votes_conserved(self):
        from trioron.core.epigenome import EXPRESSION_GENES
        sub, pg, _, _ = self._grow()
        g = pg.germline
        a = sub.arena
        germ = [g.progenitor_id] + [c for ids in g.council_ids.values() for c in ids]
        assert len(germ) == 1 + 4 * len(EXPRESSION_GENES)  # 25 with tanh (s030)
        assert not a.forward_inclusion[torch.tensor(germ)].any()
        # one book: 4 council seats per gene + 4 progenitor-held perception seats
        assert abs(sum(g.votes.values()) - (4.0 * len(EXPRESSION_GENES) + 4.0)) < 1e-9
        assert len(g.perception_seats) == 4
        for cid in pg.perception_ids.tolist():
            assert int(a.parent[cid].item()) == g.progenitor_id


class TestStress:
    """Stress drivers + vote economy on the germline book (spec §10.6)."""

    def _organism(self, **kw):
        import random
        from trioron.pcll import Germline, PCLLController, StressRouter, germline_base
        sub = construct(germline_base, capacity=64)
        germ = Germline(sub)
        ids = germ.spawn_perception(F + 1)
        germ.equip_receptors(ids)
        sub.compile()
        router = StressRouter(germ)
        ctrl = PCLLController(sub, stress=router, **kw)
        return sub, germ, router, ctrl, random.Random(0)

    def test_decide_routing_and_habituation_floor(self):
        from trioron.pcll import (DISCRIMINATION, DRIVE_FLOOR, EMPTY, FRUSTRATED,
                                  RESOLVED, SENSATION)
        _, _, router, _, _ = self._organism()
        assert router.decide(RESOLVED) is None
        assert router.decide(FRUSTRATED) == DISCRIMINATION
        router.pending = []
        assert router.decide(EMPTY) == SENSATION
        router.empty_drive = DRIVE_FLOOR / 2
        router.pending = []
        assert router.decide(EMPTY) is None    # accepted-empty

    def test_settle_moves_votes_conserved_and_drive(self):
        from trioron.pcll import EMPTY, RESOLVED, SENSATION
        _, germ, router, _, _ = self._organism()
        total = sum(germ.votes.values())
        router.pending = [(SENSATION, None)]
        router.settle(EMPTY)                   # fruitless → sensation pays
        assert abs(sum(germ.votes.values()) - total) < 1e-9
        assert router.group_votes(SENSATION) == 3.0
        assert router.empty_drive == 0.5
        router.pending = [(SENSATION, None)]
        router.settle(RESOLVED)                # justified → re-sensitised, repaid
        assert router.empty_drive == 1.0
        assert abs(sum(germ.votes.values()) - total) < 1e-9

    def test_per_class_settlement_defers_without_testimony(self):
        # [D13] the false-credit confound: a period about ANOTHER class
        # settles nothing — the world changing is not evidence
        from trioron.pcll import DISCRIMINATION, FRUSTRATED, RESOLVED
        _, germ, router, _, _ = self._organism()
        before = dict(germ.votes)
        assert router.decide(FRUSTRATED, subject="u7") == DISCRIMINATION
        out = router.settle(RESOLVED, testify=lambda s: None)
        assert out == [] and router.pending == [(DISCRIMINATION, "u7")]
        assert germ.votes == before            # no votes moved
        assert router.settlements == []

    def test_per_class_settlement_pays_on_subject_testimony(self):
        from trioron.pcll import (DISCRIMINATION, FRUSTRATED, RESOLVED,
                                  SENSATION)
        _, germ, router, _, _ = self._organism()
        total = sum(germ.votes.values())
        router.decide(FRUSTRATED, subject="u7")
        out = router.settle(RESOLVED, testify=lambda s: s == "u7")
        assert out == [True] and router.pending == []
        assert router.settlements == [(DISCRIMINATION, "u7", True)]
        assert router.group_votes(SENSATION) == 3.0   # discrimination earned
        assert abs(sum(germ.votes.values()) - total) < 1e-9
        router.decide(FRUSTRATED, subject="u9")
        out = router.settle(FRUSTRATED, testify=lambda s: False)
        assert out == [False]                  # failure repays
        assert abs(router.group_votes(SENSATION) - 4.0) < 1e-9
        assert abs(sum(germ.votes.values()) - total) < 1e-9

    def test_attach_subjects_fans_out(self):
        from trioron.pcll import DISCRIMINATION, FRUSTRATED
        _, _, router, _, _ = self._organism()
        router.decide(FRUSTRATED)              # subject unknown at decide time
        router.attach_subjects(["a", "b", "c"])
        assert router.pending == [(DISCRIMINATION, "a"),
                                  (DISCRIMINATION, "b"),
                                  (DISCRIMINATION, "c")]
        # a subject-bearing tail is never clobbered
        router.attach_subjects(["d"])
        assert len(router.pending) == 3

    def test_meeting_emits_grow_decision(self):
        from trioron.pcll import SENSATION
        sub, _, router, ctrl, rng = self._organism()
        noise = torch.tensor(
            [[rng.randint(0, N_QUANTA) for _ in range(F)] + [float(N_QUANTA)]
             for _ in range(300)], dtype=torch.float32)
        ctrl.observe(noise)
        report = sub.end_task()
        assert report.status == "empty" and report.grow == SENSATION
        assert router.pending == [(SENSATION, None)]

    def test_refresh_receptors_remaps_learner_by_cell_id(self):
        sub, germ, _, ctrl, rng = self._organism()
        # learn one class, then attach a new receptor and verify remap
        ctrl.observe(_period(CLASS_A, rng))
        sub.end_task()
        cls = ctrl.learner.classes[0]
        active_by_cell = {int(ctrl.receptor_ids[i]): bool(cls.active[i])
                          for i in range(len(ctrl.receptor_ids))}
        new = germ.spawn_perception(1)
        germ.equip_receptors(new)
        ctrl.refresh_receptors()
        assert ctrl.learner.F == F + 2
        for i, cid in enumerate(ctrl.receptor_ids.tolist()):
            want = active_by_cell.get(cid, False)  # new receptor: inactive
            assert bool(cls.active[i]) == want
        assert ctrl.read_mask.all() and int(ctrl._streak.sum()) == 0

    def test_winner_phenotype_is_expression_gene(self):
        from trioron.core.epigenome import EXPRESSION_GENES
        _, germ, router, _, _ = self._organism()
        p = router.winner_phenotype()
        assert p in EXPRESSION_GENES
        germ.votes[germ.council_ids[EXPRESSION_GENES[2]][0]] += 1.0
        assert router.winner_phenotype() == EXPRESSION_GENES[2]


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
