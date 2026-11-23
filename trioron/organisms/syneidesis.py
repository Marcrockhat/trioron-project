"""Syneidesis — the organism that grows on a reflex brain (s064, stage 1).

Working name (Greek *syneidesis*, "knowing-with": conscience / awareness).
Stage 1 of docs/design/incubator_matron.md: read the reflex brain's hidden
state for a query and predict whether the reflex's answer is WRONG.  That
prediction is the organism's awareness signal and, later, its escalation
trigger.

The package side holds only the sense and the leaf.  The reflex brain, the
questions and the grading are world-side (``experiments/incubator/``) and
inject native tensors: hidden states ``[N, d]`` and labels ``[N]`` (1 = wrong).
"""
from __future__ import annotations

import torch

from trioron.senses.hidden_state import HiddenStateSense
from .base import Organism, chunked, leaf, train_leaf


class Syneidesis(Organism):
    name = "syneidesis"

    def __init__(self, seed: int, *, k: int = 64, interior: int = 48, dup: bool = True) -> None:
        super().__init__(seed)
        self.k, self.interior, self.dup = k, interior, dup
        self.sense: HiddenStateSense | None = None
        self.doubt_leaf = None

    # ---- acquisition schedule (each returns self) ---------------------------
    def wake(self, H: torch.Tensor) -> "Syneidesis":
        """Fit the sense on an unlabeled stream of reflex hidden states."""
        self.sense = HiddenStateSense(H, self.k, dup=self.dup)
        return self

    def fit(self, H: torch.Tensor, wrong: torch.Tensor, *, epochs: int = 30,
            lr: float = 1e-3) -> "Syneidesis":
        """Learn right/wrong from graded reflex answers (1 = the reflex was wrong)."""
        if self.sense is None:
            self.wake(H)
        if self.doubt_leaf is None:
            self.doubt_leaf = leaf(self.sense.out_dim, 2, self.seed, interior=self.interior)
        train_leaf(self.doubt_leaf, self.sense(H), wrong.long(), self.seed, epochs, lr=lr)
        return self

    # ---- inference ----------------------------------------------------------
    @torch.no_grad()
    def doubt(self, H: torch.Tensor) -> torch.Tensor:
        """``[N]`` probability that the reflex's answer is wrong."""
        return chunked(self.doubt_leaf, self.sense(H)).softmax(1)[:, 1]

    def substrates(self) -> list:
        return [] if self.doubt_leaf is None else [self.doubt_leaf]
