"""v1 donor API smoke — build_donor -> absorb -> load_organism through the
public ``trioron.api`` surface (MANUAL §7 / §12 flow). Regression for the
``_extra_state`` dict in TrioronLayer.state_dict() that broke build_donor
(``'dict' object has no attribute 'detach'``) — s051."""
from __future__ import annotations

import torch

from trioron.api import (TaskData, TrioronConfig, build_donor, absorb,
                         load_organism)


def _task(name, classes, seed):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(60, 16, generator=g)
    y = torch.tensor(classes).repeat(30)
    return TaskData(name=name, X_train=X, y_train=y, X_test=X[:20],
                    y_test=y[:20], classes=classes)


def test_build_donor_absorb_load(tmp_path):
    a = tmp_path / "A.pt"
    b = tmp_path / "B.pt"
    build_donor(label="A", tasks=[_task("a", [0, 1], 0)], seed=42,
                epochs_per_task=1, out_path=a,
                config=TrioronConfig(cap_bytes=32_000))
    build_donor(label="B", tasks=[_task("b", [2, 3], 1)], seed=42,
                epochs_per_task=1, out_path=b,
                config=TrioronConfig(cap_bytes=32_000))
    payload = torch.load(a, weights_only=False)
    assert payload["kind"] == "trioron_donor"
    out = absorb(donor_paths=[a, b], out_path=tmp_path / "org.pt")
    org = load_organism(out)
    assert sorted(org.union_classes) == [0, 1, 2, 3]
