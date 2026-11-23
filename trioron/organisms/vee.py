"""Vee — the organism that grows on a reflex brain (s064 stage 1, s065 stage 2).

Named by Rocky (from Eve and Veda).
Stage 1 of docs/design/incubator_matron.md: read the reflex brain's hidden
state for a query and predict whether the reflex's answer is WRONG.  That
prediction is the organism's awareness signal and, later, its escalation
trigger.

Stage 2 (fact memory): a second leaf, the KEY leaf, maps the reflex's state
while it reads a question to a short key, trained so the same question in
different words lands on the same key and a different question about the same
subject does not.  ``remember`` stores key -> text (one row per fact; the
memory grows with what the organism knows), ``recall`` returns the nearest
stored fact and how close it is.  Keys are stored as embedded (``e`` numbers
per fact), so the key leaf must not drift after facts are stored.

The package side holds only the sense and the leaf.  The reflex brain, the
questions and the grading are world-side (``experiments/incubator/``) and
inject native tensors: hidden states ``[N, d]`` and labels ``[N]`` (1 = wrong).
"""
from __future__ import annotations

import torch

from trioron.senses.hidden_state import HiddenStateSense
from .base import Organism, chunked, leaf, train_contrastive, train_leaf


class Vee(Organism):
    name = "vee"

    def __init__(self, seed: int, *, k: int = 64, interior: int = 48, dup: bool = True) -> None:
        super().__init__(seed)
        self.k, self.interior, self.dup = k, interior, dup
        self.sense: HiddenStateSense | None = None
        self.doubt_leaf = None
        # stage 2: fact memory
        self.key_sense: HiddenStateSense | None = None
        self.key_leaf = None
        self.keys = torch.empty(0, 0)            # [n_facts, e], unit rows
        self.facts: list[str] = []

    # ---- acquisition schedule (each returns self) ---------------------------
    def wake(self, H: torch.Tensor) -> "Vee":
        """Fit the sense on an unlabeled stream of reflex hidden states."""
        self.sense = HiddenStateSense(H, self.k, dup=self.dup)
        return self

    def fit(self, H: torch.Tensor, wrong: torch.Tensor, *, epochs: int = 30,
            lr: float = 1e-3) -> "Vee":
        """Learn right/wrong from graded reflex answers (1 = the reflex was wrong)."""
        if self.sense is None:
            self.wake(H)
        if self.doubt_leaf is None:
            self.doubt_leaf = leaf(self.sense.out_dim, 2, self.seed, interior=self.interior)
        train_leaf(self.doubt_leaf, self.sense(H), wrong.long(), self.seed, epochs, lr=lr)
        return self

    def learn_keys(self, H: torch.Tensor, group: torch.Tensor, block: torch.Tensor, *,
                   k: int = 256, e: int = 32, epochs: int = 30, lr: float = 1e-3) -> "Vee":
        """Learn the key space from questions seen in several wordings.
        ``group``: same id = the same question reworded; ``block``: same id =
        same subject (other questions of a block are the hard negatives)."""
        if self.key_sense is None:
            self.key_sense = HiddenStateSense(H, k, dup=self.dup)
            self.key_leaf = leaf(self.key_sense.out_dim, e, self.seed, interior=self.interior)
        train_contrastive(self.key_leaf, self.key_leaf.trainable_tensors(), self.key_sense(H),
                          group, block, self.seed, epochs, lr=lr)
        with torch.no_grad(): self.key_leaf(self.key_sense(H[:1]))   # drop grad-bearing activations
        return self

    def remember(self, H: torch.Tensor, texts: list[str]) -> "Vee":
        """Store one fact per row: key of the question's state -> its text."""
        K = self.key(H)
        self.keys = K if not self.facts else torch.cat([self.keys, K])
        self.facts += list(texts)
        return self

    # ---- inference ----------------------------------------------------------
    @torch.no_grad()
    def doubt(self, H: torch.Tensor) -> torch.Tensor:
        """``[N]`` probability that the reflex's answer is wrong."""
        return chunked(self.doubt_leaf, self.sense(H)).softmax(1)[:, 1]

    @torch.no_grad()
    def key(self, H: torch.Tensor) -> torch.Tensor:
        """``[N, e]`` unit-length memory keys."""
        return torch.nn.functional.normalize(chunked(self.key_leaf, self.key_sense(H)), dim=1)

    @torch.no_grad()
    def recall(self, H: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Nearest stored fact per row: ``(index [N], cosine [N])``.  The caller
        decides the override threshold; ``self.facts[i]`` is the stored text."""
        score, idx = (self.key(H) @ self.keys.T).max(1)
        return idx, score

    def memory_bytes(self) -> int:
        """Bytes the stored facts take (keys as float16 + text)."""
        return self.keys.numel() * 2 + sum(len(t.encode()) for t in self.facts)

    def substrates(self) -> list:
        return [s for s in (self.doubt_leaf, self.key_leaf) if s is not None]
