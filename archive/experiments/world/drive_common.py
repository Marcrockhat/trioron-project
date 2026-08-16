"""Shared loaders for the s050 zero-master (drive-only) vocabulary arms.

Leaves/router were saved by world_drive_vocab.py as the ordered list of
trainable_tensors() (s049 ckpt rule); rebuild the same topology with the
same seed and copy tensors back."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch

from trioron.bases import seeded
from trioron.core import Envelope, construct
from trioron.phenotype import default_dispatch_table

from experiments.world.router_trioron import PERCEPT_DIM
from experiments.world.world_dream_newleaf import build_leaf_sub
from experiments.world.world_drive_vocab import DRIVES

RUNS = Path(__file__).resolve().parents[3] / "runs" / "drive_vocab"


def _load_into(sub, path):
    ts = torch.load(path)
    live = sub.trainable_tensors()
    assert len(ts) == len(live), (len(ts), len(live))
    with torch.no_grad():
        for a, b in zip(live, ts):
            a.copy_(b)
    return sub


def load_drive_leaves(seed):
    """Same seed line as world_drive_vocab: leaf k built from seed+100k(+900)."""
    leaves, names = [], []
    for k, drive in enumerate(DRIVES):
        sub = build_leaf_sub(seed + 100 * k + 900)
        _load_into(sub, RUNS / f"{drive}_seed{seed}.pt")
        leaves.append(sub); names.append(f"DRIVE_{drive.upper()}")
    return leaves, names


def build_router_sub(seed, n):
    torch.manual_seed(seed)
    sub = construct(
        base=seeded(PERCEPT_DIM, n, interior_cells=32, nonlinear=True),
        envelope=Envelope(max_parameter_bytes=400_000),
        dispatch_table=default_dispatch_table(), capacity=2048,
        sparsity_k=0)
    sub.compile()
    sub.prepare_training()
    return sub


def load_drive_router(seed, n=len(DRIVES)):
    return _load_into(build_router_sub(seed, n), RUNS / f"router_seed{seed}.pt")
