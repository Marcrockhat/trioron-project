"""Retinal compression (design §3.2/§3.6) — fast CI guards.

Pins the phase-1 wide-progenitor mechanics: pooled-receptor injection
in the scheduler, the redundancy merge at the first sitting (adjacent
duplicate columns -> one region sensor with imposed position), the
no-geometry no-op (status-quo regression), husk treatment of merged
members, and ship/wake round-trip of the pooling apertures.
"""
from __future__ import annotations

import math

import torch

from trioron.core import construct
from trioron.core.epigenome import PERCEPTION, RECEPTOR, has_gene, set_gene
from trioron.core.receptor import N_QUANTA, quantize_frame
from trioron.core.state import CellState
from trioron.lifecycle.ship import ship
from trioron.lifecycle.wake import wake
from trioron.pcll import PerceptionGenesis, germline_base
from trioron.pcll.retina import (REDUNDANT_R, adjacent_pairs, grid_positions,
                                 pair_redundancy)


def _world(n: int = 240, seed: int = 0) -> torch.Tensor:
    """A 3x3 world: col 4 (centre) independent signal, col 0 == col 1
    (duplicate pair -> must merge), col 8 high-amplitude carrier (keeps
    the duplicates off the per-sample ceiling), col 2 constant (starve),
    rest independent."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 9, generator=g)
    x[:, 1] = x[:, 0]              # adjacent duplicates (cols 0,1 on row 0)
    x[:, 2] = 0.7                  # constant -> census starve
    x[:, 8] = 2.0 + torch.rand(n, generator=g)   # per-sample max carrier
    return x


def _genesis(x: torch.Tensor, shape=(3, 3)):
    sub = construct(germline_base, capacity=64)
    pg = PerceptionGenesis(sub, input_shape=shape)
    pg.feed(x)
    report = sub.end_task()
    return sub, pg, report


class TestGridGeometry:
    def test_positions_imposed_on_perception(self):
        x = _world()
        sub, pg, _ = _genesis(x)
        pos = sub.arena.position[pg.perception_ids.long()]
        # col 0 -> (0,0); col 8 -> (1,1); col 5 (row 1, col 2) -> (1, .5)
        assert torch.allclose(pos[0, :2], torch.tensor([0.0, 0.0]))
        assert torch.allclose(pos[8, :2], torch.tensor([1.0, 1.0]))
        assert torch.allclose(pos[5, :2], torch.tensor([1.0, 0.5]))
        assert torch.allclose(pos[:, 2], torch.full((9,), 1.0 / 3.0))

    def test_adjacent_pairs_grid(self):
        pairs = adjacent_pairs(list(range(9)), (3, 3))
        assert (0, 1) in pairs and (0, 3) in pairs
        assert (2, 3) not in pairs          # row wrap is not adjacency
        assert len(pairs) == 12             # 2*W*H - W - H


class TestRedundancyStatistic:
    def test_duplicate_pair_saturates(self):
        x = _world()
        eligible = [0, 1, 3, 4]
        r, cnt = pair_redundancy([x], eligible, [(0, 1), (0, 3)])
        assert float(r[0]) > REDUNDANT_R    # duplicates: R_diff -> 1
        assert float(r[1]) < REDUNDANT_R    # independents stay below
        assert float(cnt[0]) >= 60

    def test_phase_offset_invariance(self):
        # col j = col i shifted by a constant value offset: still redundant
        g = torch.Generator().manual_seed(1)
        x = torch.rand(240, 4, generator=g)
        x[:, 1] = (x[:, 0] + 0.2)
        x[:, 3] = 3.0 + torch.rand(240, generator=g)  # carrier
        r, _ = pair_redundancy([x], [0, 1, 2, 3], [(0, 1), (0, 2)])
        assert float(r[0]) > REDUNDANT_R
        assert float(r[1]) < REDUNDANT_R


class TestFirstSittingMerge:
    def test_region_sensor_spawned(self):
        x = _world()
        sub, pg, report = _genesis(x)
        assert len(report.regions) == 1
        region = report.regions[0]
        assert region.members == [0, 1]
        assert report.merged == {0: region.cell_id, 1: region.cell_id}
        # imposed position = centroid of (0,0) and (0.5,0), scale sqrt(2)/3
        cx, cy, scale = region.position
        assert abs(cx - 0.25) < 1e-6 and abs(cy) < 1e-6
        assert abs(scale - math.sqrt(2) / 3) < 1e-6

    def test_members_become_husks(self):
        x = _world()
        sub, pg, report = _genesis(x)
        a = sub.arena
        for c in (0, 1):
            cid = int(pg.perception_ids[c].item())
            assert not bool(has_gene(a.epigenome[cid], RECEPTOR))
            assert int(a.state[cid]) == CellState.DORMANT
        assert 0 not in report.kept and 1 not in report.kept

    def test_pooled_injection_value(self):
        x = _world()
        sub, pg, report = _genesis(x)
        plan = sub.scheduler._plan
        assert plan.pooled_ids.numel() == 1
        sub.forward(x[:8])
        q = sub.scheduler._last_receptor_q
        # live continuous receptor cols (post-starve, post-merge frame)
        live = [c for c in range(9) if c in report.kept]
        pooled_val = x[:8, [0, 1]].mean(dim=1, keepdim=True)
        frame_vals = torch.cat([x[:8, live], pooled_val], dim=1)
        q_expect, _, _ = quantize_frame(frame_vals)
        assert torch.allclose(q, q_expect)

    def test_starved_column_not_merged(self):
        x = _world()
        _, _, report = _genesis(x)
        assert 2 in report.starved
        assert 2 not in report.merged


class TestStatusQuoRegression:
    def test_no_geometry_no_merge(self):
        x = _world()
        sub = construct(germline_base, capacity=64)
        pg = PerceptionGenesis(sub)          # no input_shape
        pg.feed(x)
        report = sub.end_task()
        assert report.regions == [] and report.merged == {}
        assert sub.arena.pool_dst.numel() == 0
        plan = sub.scheduler._plan
        assert plan.pooled_ids.numel() == 0
        assert torch.equal(plan.column_ids, plan.perception_ids)

    def test_plan_receptor_order_columns_then_pooled(self):
        x = _world()
        sub, pg, report = _genesis(x)
        plan = sub.scheduler._plan
        n_cols = plan.receptor_cols.numel()
        assert plan.receptor_ids.numel() == n_cols + 1
        assert int(plan.receptor_ids[-1]) == report.regions[0].cell_id


class TestShipWakeRoundTrip:
    def test_pool_survives_ship_wake(self, tmp_path):
        x = _world()
        sub, pg, report = _genesis(x)
        path = tmp_path / "organism.pt"
        ship(sub, str(path))
        sub2 = wake(str(path))
        a2 = sub2.arena
        assert torch.equal(a2.pool_src, sub.arena.pool_src)
        assert torch.equal(a2.pool_dst, sub.arena.pool_dst)
        assert torch.allclose(a2.pool_w, sub.arena.pool_w)
        sub2.compile()
        plan = sub2.scheduler._plan
        assert plan.pooled_ids.numel() == 1
