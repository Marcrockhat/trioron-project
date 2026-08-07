"""Tests for the H-space manifold router (learning/router.py)."""
from __future__ import annotations

import torch

from trioron.core import Envelope, construct
from trioron.bases import minimal, seeded
from trioron.phenotype import default_dispatch_table
from trioron.learning import (
    ManifoldArchive,
    ManifoldRouter,
    build_h_archive_from_data,
    build_h_archive_from_manifold,
)
from trioron.learning.router import NEG_LL


def _make_substrate(input_dim=4, n_classes=4):
    return construct(
        base=minimal(input_dim, n_classes),
        envelope=Envelope(),
        dispatch_table=default_dispatch_table(),
        capacity=256,
    )


def _make_seeded_substrate(input_dim=4, n_classes=4):
    """Base with interior cells — H-space routing needs a non-empty H."""
    return construct(
        base=seeded(input_dim, n_classes, interior_cells=16, nonlinear=True),
        envelope=Envelope(),
        dispatch_table=default_dispatch_table(),
        capacity=256,
    )


def _separable_codes(dim=6, n=64, spread=8.0, seed=0):
    """Three well-separated Gaussian clusters in code space."""
    g = torch.Generator().manual_seed(seed)
    codes, labels = {}, [0, 1, 2]
    for cid in labels:
        mu = torch.zeros(dim)
        mu[cid] = spread
        codes[cid] = mu + torch.randn(n, dim, generator=g)
    return codes


def _fit_archive(arena, codes, full_cov=False):
    archive = ManifoldArchive(arena, full_cov=full_cov)
    for cid, batch in codes.items():
        archive.update_class(cid, batch)
    archive.finalize_all()
    return archive


class TestRouteClass:

    def test_recovers_separable_classes(self):
        sub = _make_substrate()
        codes = _separable_codes()
        router = ManifoldRouter(_fit_archive(sub.arena, codes))
        for cid, batch in codes.items():
            pred = router.route_class(batch)
            assert (pred == cid).float().mean() > 0.95

    def test_full_cov_recovers_separable_classes(self):
        sub = _make_substrate()
        codes = _separable_codes()
        router = ManifoldRouter(_fit_archive(sub.arena, codes, full_cov=True))
        for cid, batch in codes.items():
            pred = router.route_class(batch)
            assert (pred == cid).float().mean() > 0.95

    def test_missing_class_never_wins(self):
        sub = _make_substrate()
        codes = _separable_codes()
        archive = _fit_archive(sub.arena, codes)
        router = ManifoldRouter(archive)
        ll = router.class_log_likelihood(codes[0], n_classes=5)
        assert (ll[:, 3] == NEG_LL).all() and (ll[:, 4] == NEG_LL).all()
        assert (router.route_class(codes[0], n_classes=5) < 3).all()


class TestRouteGroup:

    def test_group_pick_and_prediction(self):
        sub = _make_substrate()
        codes = _separable_codes()
        router = ManifoldRouter(_fit_archive(sub.arena, codes))
        groups = [[0, 1], [2]]

        assert (router.route_group(codes[0], groups) == 0).float().mean() > 0.95
        assert (router.route_group(codes[2], groups) == 1).float().mean() > 0.95

        # Head logits favor class 1 everywhere; group routing must confine
        # group-1 queries to class 2 regardless.
        logits = torch.zeros(codes[2].shape[0], 3)
        logits[:, 1] = 10.0
        pred = router.route_prediction(codes[2], logits, groups)
        assert ((pred == 2) | (pred == 1)).all()
        assert (pred[router.route_group(codes[2], groups) == 1] == 2).all()


class TestHArchiveBuilders:

    def test_build_from_data_routes_substrate_codes(self):
        torch.manual_seed(0)
        sub = _make_seeded_substrate()
        sub.prepare_training()
        x = torch.randn(120, 4)
        y = (x[:, 0] > 0).long()  # two input-separable classes
        x[y == 1] += 3.0

        h_archive = build_h_archive_from_data(sub, [(x, y)])
        assert h_archive.n_classes == 2

        router = ManifoldRouter(h_archive)
        _ = sub(x)
        from trioron.learning.manifold import get_interior_ids
        h_codes = sub.last_activations[:, get_interior_ids(sub.arena).long()]
        acc = (router.route_class(h_codes) == y).float().mean()
        assert acc > 0.7, f"H-space routing at {acc:.2f}, expected > chance"

    def test_build_from_manifold_storage_free(self):
        torch.manual_seed(0)
        sub = _make_seeded_substrate()
        sub.prepare_training()
        x = torch.randn(120, 4)
        y = (x[:, 0] > 0).long()
        x[y == 1] += 3.0

        perc = ManifoldArchive(sub.arena)
        for cid in (0, 1):
            perc.update_class(cid, x[y == cid])
        perc.finalize_all()

        h_archive = build_h_archive_from_manifold(
            sub, perc, n_perc=4, samples_per_class=100,
        )
        assert h_archive.n_classes == 2

        router = ManifoldRouter(h_archive)
        _ = sub(x)
        from trioron.learning.manifold import get_interior_ids
        h_codes = sub.last_activations[:, get_interior_ids(sub.arena).long()]
        acc = (router.route_class(h_codes) == y).float().mean()
        assert acc > 0.7, f"storage-free routing at {acc:.2f}, expected > chance"
