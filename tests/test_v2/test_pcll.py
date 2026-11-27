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


class TestD20Membership:
    """[D20] the margin gate, consolidation, and the reject readout."""

    def _mixed(self, **kw):
        import math
        from trioron.pcll import (Germline, MixedStreamController,
                                  StressRouter, germline_base)
        sub = construct(germline_base, capacity=64)
        germ = Germline(sub)
        ids = germ.spawn_perception(4)
        germ.equip_receptors(ids)
        sub.compile()
        m = MixedStreamController(sub, stress=StressRouter(germ), **kw)
        # two well-separated classes, by hand
        g = torch.Generator().manual_seed(0)
        qa = 250 + 12 * torch.randn(120, 4, generator=g)
        qb = 700 + 12 * torch.randn(120, 4, generator=g)
        z = lambda q: torch.exp(1j * 2 * math.pi * q / 1000)
        m.classes = [type(m)._dummy_class("u1"), type(m)._dummy_class("u2")] \
            if hasattr(type(m), "_dummy_class") else None
        from trioron.pcll.signature import LearnedClass
        mk = lambda n: LearnedClass(n, torch.zeros(4, dtype=torch.complex64),
                                    torch.zeros(4),
                                    torch.ones(4, dtype=torch.bool))
        m.classes = [mk("u1"), mk("u2")]
        m.bufs = [z(qa), z(qb)]
        m._new_node("u1", None)
        m._new_node("u2", None)
        return m, z

    def test_gate_refuses_low_margin(self):
        # gate_k scaled to the 4-dim fixture: a coherent sample's margin
        # is ≈ Σ|T|/σ ≈ 4/√2 ≈ 2.8 (the 12-dim world sits at 4-6).
        # The gate engages only once SETTLED (quiescence deferral).
        m, z = self._mixed(gate_k=2.0)
        m._settled = m.SETTLE_STREAK
        g = torch.Generator().manual_seed(1)
        sure = z(250 + 5 * torch.randn(10, 4, generator=g))
        junk = z(torch.rand(10, 4, generator=g) * 1000)
        member = m._assign(torch.cat([sure, junk]))
        assert (member[:10] == 0).all()
        assert (member[10:] == -1).float().mean() > 0.5

    def test_consolidate_moves_misplaced_members(self):
        import math
        m, z = self._mixed(consolidate=True, member_margin=True)
        g = torch.Generator().manual_seed(2)
        stray = z(250 + 5 * torch.randn(30, 4, generator=g))
        m.bufs[1] = torch.cat([m.bufs[1], stray])   # pollute class 2
        moved = m._consolidate()
        assert moved >= 30                          # strays went home

    def test_margins_in_chunks_equal_the_whole_table(self, monkeypatch):
        """[s070] the best class and its margin are taken inside each
        row chunk; the members x classes table is never held whole.
        Bit-identical to the whole table, raw-E and margin ranking."""
        import trioron.pcll.mixed as mx
        m, z = self._mixed()
        g = torch.Generator().manual_seed(4)
        m.bufs.append(z(1000 * torch.rand(80, 4, generator=g)))  # a weak template
        Z = z(1000 * torch.rand(501, 4, generator=g))
        T = torch.stack([b.mean(0) for b in m.bufs])
        E = (Z.unsqueeze(1) * T.conj().unsqueeze(0)).real.sum(-1)
        sig = ((T.abs() ** 2).sum(-1) / 2).sqrt().clamp_min(1e-9)
        monkeypatch.setattr(mx, "EVIDENCE_ELEMS", 7 * 3 * 4)     # 7 rows a chunk
        assert torch.equal(m._evidence(Z, T), E)
        for settled, ref in ((0, E), (m.SETTLE_STREAK, E / sig)):
            m._settled = settled
            best, marg = m._margins(Z)
            want = ref.argmax(1)
            assert torch.equal(best, want)
            assert torch.equal(
                marg, (E / sig).gather(1, want.unsqueeze(1)).squeeze(1))
        best, marg = m._margins(Z[:0])                           # no rows
        assert best.shape == (0,) and marg.shape == (0,)

    def test_classify_reject_option(self):
        m, z = self._mixed()
        g = torch.Generator().manual_seed(3)
        x_known = 0.25 + 0.005 * torch.randn(10, 4, generator=g)
        x_junk = torch.rand(10, 4, generator=g)
        # classify drives the real forward path: feed raw x
        pred_k, marg_k = m.classify(x_known, k_reject=None)
        assert marg_k.shape == (10,)
        pred_j, marg_j = m.classify(x_junk, k_reject=float(marg_k.min()))
        assert (pred_j == -1).any() or marg_j.min() >= marg_k.min()


def test_member_names_reads_current_members_not_ledger():
    """s066: PhasecyteLeaf.member_names() = majority tag of each class's
    current buffer members; names() stays the lifetime-ledger readout."""
    from trioron.pcll import PhasecyteLeaf
    g = torch.Generator().manual_seed(0)
    centers = torch.randn(4, 16, generator=g) * 3
    y = torch.arange(4).repeat_interleave(60)
    X = centers[y] + 0.3 * torch.randn(len(y), 16, generator=g)
    p = torch.randperm(len(y), generator=g)
    X, y = X[p], y[p]
    leaf = PhasecyteLeaf(lambda x: x, X[:60], 0, 0, window=60, class_cap=None, verbose=False)
    assert (leaf.member_names() == -1).all()            # nothing told yet
    for i in range(0, len(X), 60):
        leaf.observe(X[i:i + 60], [f"g{int(f)}" for f in y[i:i + 60]])
    names = leaf.member_names()
    assert names.shape == (len(leaf.mixed.classes),) and names.dtype == torch.long
    assert leaf.names().shape == names.shape
    best, _ = leaf.mixed.classify(X)
    ok = best >= 0
    assert (names[best[ok]] == y[ok]).float().mean() > 0.8


def test_class_rows_outgrow_the_arena():
    """s069: with no cell cap a class always gets its arena row — the arena
    enlarges at the boundary instead of leaving the class rowless (-2)."""
    from trioron.pcll import PhasecyteLeaf
    from trioron.pcll import mixed as mx
    g = torch.Generator().manual_seed(0)
    centres = torch.randn(12, 6, generator=g) * 3
    X = (centres[torch.randint(12, (3000,), generator=g)] + 0.05 * torch.randn(3000, 6, generator=g))
    mk = lambda cap: PhasecyteLeaf(lambda x: x, X[:200], 0, 0, window=200, class_cap=None,
                                   composer=False, capacity=cap, verbose=False)
    start = mk(8192).sub.arena.cursor + mx.ROW_HEADROOM      # an arena 'full' from the first class on
    leaf = mk(start)
    a = leaf.sub.arena
    for i in range(200, 3000, 200):
        leaf.mixed.observe(X[i:i + 200]); leaf.sub.end_task()
    assert len(leaf.mixed.classes) > 1
    assert a.capacity > start
    assert all(c.cell_id >= 0 for c in leaf.mixed.classes)


def test_sitting_cap_bounds_the_judging():
    """s069: under the sitting cap at most cap classes are judged a boundary,
    a class with nothing new is not judged again, and classes are still
    discovered; off (the default) nothing is recorded."""
    from trioron.pcll import PhasecyteLeaf
    from trioron.organs.sitting import SittingCap
    g = torch.Generator().manual_seed(0)
    centres = torch.randn(12, 6, generator=g) * 3
    X = (centres[torch.randint(12, (6000,), generator=g)] + 0.05 * torch.randn(6000, 6, generator=g))
    mk = lambda: PhasecyteLeaf(lambda x: x, X[:200], 0, 0, window=200, class_cap=None, composer=False, verbose=False)
    plain, capped = mk(), mk()
    assert plain.mixed.sitting is None
    capped.mixed.sitting = SittingCap()
    for i in range(200, 6000, 200):
        for leaf in (plain, capped):
            leaf.mixed.observe(X[i:i + 200]); leaf.sub.end_task()
    work = capped.mixed.sitting.work
    assert work and all(w["judged"] <= w["cap"] and w["judged"] <= w["due"] for w in work)
    assert sum(w["divisions"] for w in work) > 0 and len(capped.mixed.classes) > 1
    assert all(w["settled"] >= 1 for w in work[1:])      # the round ran at every sitting, in instalments
    s = SittingCap()
    assert s.choose(["a", "b"], 1) == [0, 1]            # never judged: both due
    assert s.choose(["a", "b"], 2) == []                # nothing new: nobody judged
    s.receive(["a", "b"], torch.tensor([1, 1, 0, -1]))
    assert s.choose(["a", "b"], 3) == [1, 0]            # busiest first


def test_stale_classes_fold_into_kin():
    """s069: with fold on, a class the world stops feeding fades and is folded
    into kin — the class count falls, its sketch history is kept in the survivor."""
    from trioron.pcll import PhasecyteLeaf
    from trioron.organs.sitting import SittingCap, FADE
    assert abs(25 * FADE ** 60 - 1.0) < 1e-6
    g = torch.Generator().manual_seed(0)
    centres = torch.randn(12, 6, generator=g) * 3
    draw = lambda n, hi: centres[torch.randint(hi, (n,), generator=g)] + 0.05 * torch.randn(n, 6, generator=g)
    X = draw(4000, 12)
    leaf = PhasecyteLeaf(lambda x: x, X[:200], 0, 0, window=200, class_cap=None, composer=False, verbose=False)
    leaf.mixed.sitting = cap = SittingCap(fold=True)
    for i in range(200, 4000, 200):
        leaf.mixed.observe(X[i:i + 200]); leaf.sub.end_task()
    before = len(leaf.mixed.classes)
    total = sum(s._n for s in leaf.mixed.manifold.sketches.values())
    assert before > 3 and not cap.folds
    for _ in range(150):                               # the world narrows to two of its states
        leaf.mixed.observe(draw(200, 2)); leaf.sub.end_task()
    assert cap.folds and len(leaf.mixed.classes) < before
    assert len(leaf.mixed.classes) == len(leaf.mixed.bufs) == len(leaf.mixed.tags)
    assert sum(s._n for s in leaf.mixed.manifold.sketches.values()) >= total   # nothing deleted
    best, _ = leaf.mixed.classify(draw(200, 2))
    assert (best >= 0).all()


def test_numa_ledger_settles_or_takes_back():
    """s071/s072: a provisional pair settles (Numa) after two coherent
    batches of world members on its split dim, and is handed back (Mima)
    when a batch is incoherent or a child goes unfed by the world for the
    trial length; a pair the world feeds slowly stays on trial; a pair that
    lost a child is void."""
    from trioron.organs.numa import NumaLedger
    g = torch.Generator().manual_seed(0)
    ph = lambda ang: torch.exp(1j * ang)
    tight = lambda n, c: ph(c + 0.05 * torch.randn(n, 1, generator=g))
    loose = lambda n: ph(6.28 * torch.rand(n, 1, generator=g))
    names = ["a", "b"]
    member = torch.cat([torch.zeros(25), torch.ones(25)]).long()
    led = NumaLedger()
    led.born(0, "a", "b", 0, 0.72)
    assert led.on_trial("a") and led.on_trial("b")
    led.receive(names, member, torch.cat([tight(25, 0.0), tight(25, 2.0)]))
    assert led.verdicts(1, names) == [] and led.on_trial("a")       # first batch passed: not yet settled
    led.receive(names, member, torch.cat([tight(25, 0.0), tight(25, 2.0)]))
    assert led.verdicts(2, names) == [] and not led.on_trial("a")   # the re-test passed: Numa
    assert led.counts == {"born": 1, "numa": 1, "mima": 0, "gone": 0}
    led.born(2, "a", "b", 0, 0.72)                                  # a split the world does not hold
    led.receive(names, member, torch.cat([tight(25, 0.0), loose(25)]))
    assert led.verdicts(3, names) == [("b", "a")] and led.counts["mima"] == 1
    led.born(3, "a", "b", 0, 0.72)                                  # the world never feeds it
    assert led.verdicts(62, names) == [] and led.verdicts(63, names) == [("b", "a")]
    led.born(63, "a", "b", 0, 0.72)                                 # fed slowly: on trial past the old limit, then Numa
    one = torch.tensor([0, 1])
    for t in range(64, 64 + 50):                                    # one member each, every third boundary
        if t % 3 == 0:
            led.receive(names, one, torch.cat([tight(1, 0.0), tight(1, 2.0)]))
        assert led.verdicts(t, names) == [] and led.on_trial("a")
    for t in range(114, 300):
        led.receive(names, one, torch.cat([tight(1, 0.0), tight(1, 2.0)]))
        assert led.verdicts(t, names) == []
        if not led.on_trial("a"):
            break
    assert t > 63 + led.trial and led.counts["numa"] == 2 and led.taken_back[-1]["why"] == "starved"
    led.born(t, "a", "b", 0, 0.72)                                  # one child left by another road
    assert led.verdicts(t + 1, ["a"]) == [] and led.counts["gone"] == 1 and led.net == 2 - 2
    assert len(led.taken_back) == 2 and not led.pairs and not led.child


def test_numa_rule_on_a_leaf():
    """s071: with the rule on, classes are still discovered and real modes
    settle; a provisional class is neither divided nor replayed; a pair the
    world never feeds is joined again with nothing deleted. Off (the
    default) the leaf is exactly as before."""
    from trioron.pcll import PhasecyteLeaf
    from trioron.organs.numa import NumaLedger
    g = torch.Generator().manual_seed(0)
    centres = torch.randn(12, 6, generator=g) * 3
    draw = lambda n, hi: centres[torch.randint(hi, (n,), generator=g)] + 0.05 * torch.randn(n, 6, generator=g)
    X = draw(6000, 12)
    mk = lambda: PhasecyteLeaf(lambda x: x, X[:200], 0, 0, window=200, class_cap=None, composer=False, verbose=False)
    plain, ruled = mk(), mk()
    assert plain.mixed.numa is None
    ruled.mixed.numa = led = NumaLedger()
    for i in range(200, 6000, 200):
        for leaf in (plain, ruled):
            leaf.mixed.observe(X[i:i + 200]); leaf.sub.end_task()
    assert led.counts["born"] > 0 and led.counts["numa"] > 0
    assert 1 < len(ruled.mixed.classes) <= len(plain.mixed.classes)
    assert led.counts["born"] == led.counts["numa"] + led.counts["mima"] + led.counts["gone"] + len(led.pairs)
    best, _ = ruled.mixed.classify(draw(400, 12))
    assert (best >= 0).float().mean() > 0.9
    # a pair on credit that the world stops feeding: taken back, nothing deleted
    m = ruled.mixed
    for key in list(led.pairs):
        led._close(key, m.periods, "gone")
    a, b = m.classes[-2].name, m.classes[-1].name
    led.born(m.periods, a, b, 0, 0.72)
    before = len(m.classes)
    total = sum(s._n for s in m.manifold.sketches.values())
    held = len(m.bufs[-1])
    quiet = centres[:1] + 0.05 * torch.randn(200, 6, generator=g)   # the world narrows to one state
    for _ in range(led.trial + 1):
        m.observe(quiet); ruled.sub.end_task()
        if led.on_trial(b):
            assert len(m.bufs[[c.name for c in m.classes].index(b)]) <= held   # no replays on credit
    names = [c.name for c in m.classes]
    assert not led.on_trial(a) and led.taken_back and (a in names) != (b in names)
    assert len(m.classes) == len(m.bufs) == len(m.tags) < before + 2 * led.counts["born"]
    assert sum(s._n for s in m.manifold.sketches.values()) >= total


def test_core_names_no_organ_rule():
    """s072 (Rocky: "clearly separate the organs and the triorons core"): nothing under
    trioron/core or trioron/pcll imports trioron/organs; the organs import the core."""
    import pathlib, re
    root = pathlib.Path(__file__).resolve().parents[2] / "trioron"
    for sub in ("core", "pcll", "learning", "lifecycle"):
        for f in (root / sub).rglob("*.py"):
            assert not re.search(r"^\s*(from|import)\s+(trioron\.organs|\.\.organs|\.\.\.organs)", f.read_text(), re.M), f
    import trioron.organs as organs
    assert organs.NumaLedger.__module__ == "trioron.organs.numa"
    import trioron.pcll.numa as old                      # bodies saved before the move still load
    assert old.NumaLedger is organs.NumaLedger


def test_fourth_organ_tries_divisions_for_a_leaf():
    """s072: with the fourth organ as an organ's court, a candidate division is held there and
    fed by the world only; the organ divides when the pair settles; classes are still
    discovered, more slowly; the fine reading is at least as fine as the organ's own; a pair
    one of whose children the world never feeds falls asleep and nothing is divided."""
    from trioron.pcll import PhasecyteLeaf
    from trioron.organs import FourthOrgan
    g = torch.Generator().manual_seed(0)
    centres = torch.randn(12, 6, generator=g) * 3
    draw = lambda n, hi: centres[torch.randint(hi, (n,), generator=g)] + 0.05 * torch.randn(n, 6, generator=g)
    X = draw(12000, 12)
    mk = lambda: PhasecyteLeaf(lambda x: x, X[:200], 0, 0, window=200, class_cap=None, composer=False, verbose=False)
    plain, ruled = mk(), mk()
    assert plain.mixed.court is None
    ruled.mixed.court = org = FourthOrgan()
    seen = []
    for i in range(200, 12000, 200):
        for leaf in (plain, ruled):
            leaf.mixed.observe(X[i:i + 200]); leaf.sub.end_task()
        seen.append(len(ruled.mixed.classes))
    c = org.counts
    assert c["held"] > 0 and c["numa"] > 0 and c["carried"] == c["numa"]
    assert 1 < len(ruled.mixed.classes) <= len(plain.mixed.classes)
    assert seen[0] == 1                                   # nothing is divided at once: the trial comes first
    m = ruled.mixed
    T = draw(600, 12)
    best, _ = m.classify(T)
    assert (best >= 0).float().mean() > 0.9
    Z = torch.exp(1j * 2 * math.pi * m.pockets_of(T) / N_QUANTA)
    fine = org.refine([k.name for k in m.classes], best, Z)
    assert len(set(fine)) >= len(set(best.tolist()))
    # a pair the world feeds on one side only: asleep, the parent whole, not heard again at once
    k = int(torch.bincount(best[best >= 0]).argmax())
    name, b = m.classes[k].name, m.bufs[k]
    side = torch.zeros(len(b), dtype=torch.bool); side[: len(b) // 2] = True
    far = b.clone(); far[side] = far[side] * torch.exp(torch.tensor(1j * 2.5))     # a child nowhere in the world
    org.held.pop(name, None); org.rest.clear()
    before, slept = len(m.classes), c["starved"]
    out = org.hear([x.name for x in m.classes], [far if j == k else x for j, x in enumerate(m.bufs)], [(k, (side, 0))], m.periods)
    assert out == [] and 0 in org.held[name]
    quiet = centres[:1] + 0.05 * torch.randn(200, 6, generator=g)
    for _ in range(org.trial + 2):
        m.observe(draw(200, 12)); ruled.sub.end_task()
    assert c["starved"] + c["failed"] > slept + 0 or name not in [x.name for x in m.classes]
    assert org.size()["members"] >= 0 and len(m.classes) == len(m.bufs) == len(m.tags)
