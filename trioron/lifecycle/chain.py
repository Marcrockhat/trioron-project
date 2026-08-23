"""Link chains — depth by growth on a stall trigger.  s059/s060 promotion.

A *link* is one substrate ``Seeded(H + E -> H, nonlinear quad)``. A chain
composes links in series on a shared evidence vector ``e``:
``h_i = link_i([h_{i-1}, e])``, ``h_0 = 0``; a substrate *head link*
``Seeded(H -> C)`` reads the last state. Two growth modes:

* ``tied``  — ONE link re-applied R times (BPTT through R applications);
  growth = R += 1.  Validated: hop-2 .92 vs shallow .49 (s059); language
  rung 0 .905/.878 comp at 1/4 the params of fresh links (s060).
* ``grown`` — spawn a fresh link on stall, ALL links stay trainable
  (joint). Frozen-greedy (credit-lock BEFORE the new link settles) was
  chance on every task (s059) — hence no lock at spawn here.

Growth trigger = ``StallTrigger`` (the Numa patience rule of s059: train
loss fails to drop ``rel_tol`` for ``patience`` epochs while test acc <
target). PATIENCE must outlast the loss plateau (3 fails, 8 passes).
The detector is epoch-grained; ``learning.frustration.FrustrationDetector``
is the batch-grained sibling used inside a single substrate.

Everything trainable here is substrate (``trainable_tensors``); the head
is a substrate link, not a torch Linear.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F

from trioron.bases.seeded import Seeded
from trioron.core import construct


def new_link(in_dim: int, out_dim: int, seed: int, *, interior: int = 48,
             layers: int = 1, fan_in_init: bool = False):
    """``fan_in_init``: rescale the arena's std-.01 edge init to 1/sqrt(fan_in)
    per target cell. Used for the HEAD link — two std-.01 substrates in series
    emit ~1e-5 logits and the chain sits at ln 2 (s060+ smoke); a torch head
    had been supplying that gain implicitly."""
    torch.manual_seed(seed)
    sub = construct(Seeded(in_dim, out_dim, interior_cells=interior,
                           interior_layers=layers, nonlinear=True),
                    capacity=in_dim + interior * layers + out_dim + 8)
    if fan_in_init:
        a = sub.arena
        with torch.no_grad():
            n = a.edge_cursor
            dst = a.edge_dst[:n]
            fan = torch.bincount(dst, minlength=a.edge_dst.numel()).float().clamp(min=1)
            a.edge_weight[:n] = torch.randn(n) / fan[dst].sqrt()
    sub.prepare_training()
    return sub


class LinkChain:
    """Substrate-only chain. ``mode`` = 'tied' | 'grown'."""

    def __init__(self, E: int, C: int, seed: int, *, H: int = 32,
                 interior: int = 48, head_interior: int = 16,
                 mode: str = "tied", max_links: int = 4, ctx: int = 0,
                 norm: str = "none", link_fan_in: bool = False) -> None:
        assert mode in ("tied", "grown", "seq")
        # seq: ONE link eats one token per step (Axis-7 style): h_t = link([h_{t-1}, e_t]),
        # e viewed as [N, L, E]. Re-application = token position, so an operator token
        # (NOT) is the SAME weights at every depth — the precondition for learning it as
        # an involution (s061 not_k: flattened input makes each slot a separate feature,
        # zero-shot parity of unseen slots = exactly 0). Growth = R extra passes over the
        # final state (R-1 "thinking" steps with a zero token).
        # ctx : (seq) leading ``ctx`` input dims are a per-step context (the scene), fed
        #       with every token; the rest is the token stream.
        # norm: (seq) state squash between steps: 'none' | 'rms' | 'tanh' | 'clip'. s062 probe on the
        #       NOT-only language world: rms HURTS (ln 2 stall persists; it amplifies the
        #       std-.01 init noise), link_fan_in=True breaks the stall (.61 vs .50 in 6 ep).
        # link_fan_in: fan-in-scaled edge init on the links too (the head always has it).
        #       Needed in seq mode: 10 std-.01 applications in series vanish the signal.
        self.E, self.C, self.H, self.seed = E, C, H, seed
        self.interior, self.mode, self.max_links = interior, mode, max_links
        self.ctx, self.norm, self.link_fan_in = ctx, norm, link_fan_in
        self.links = [new_link(H + ctx + E, H, seed * 100, interior=interior,
                               fan_in_init=link_fan_in)]
        self.head = new_link(H, C, seed * 100 + 99, interior=head_interior,
                             fan_in_init=True)
        self.R = 1                       # applications (tied) / links (grown)

    @property
    def depth(self) -> int:
        return self.R if self.mode in ("tied", "seq") else len(self.links)

    def can_grow(self) -> bool:
        return self.depth < self.max_links

    def grow(self) -> None:
        if self.mode in ("tied", "seq"):
            self.R += 1
        else:
            self.links.append(new_link(self.H + self.ctx + self.E, self.H,
                                       self.seed * 100 + len(self.links),
                                       interior=self.interior, fan_in_init=self.link_fan_in))

    def forward(self, e: torch.Tensor, **_) -> torch.Tensor:
        h = torch.zeros(len(e), self.H)
        if self.mode == "seq":
            c, toks = e[:, :self.ctx], e[:, self.ctx:].reshape(len(e), -1, self.E)
            link = self.links[0]
            steps = [toks[:, t] for t in range(toks.shape[1])]
            steps += [torch.zeros(len(e), self.E)] * (self.R - 1)
            for x in steps:
                h = self._normed(link(torch.cat([h, c, x], 1)))
            return self.head(h)
        seq = [self.links[0]] * self.R if self.mode == "tied" else self.links
        for link in seq:
            h = link(torch.cat([h, e], 1))
        return self.head(h)

    def _normed(self, h: torch.Tensor) -> torch.Tensor:
        if self.norm == "rms":
            return h * torch.rsqrt(h.pow(2).mean(1, keepdim=True) + 1e-6)
        if self.norm == "tanh":          # s063: bounded state between seq steps (crisp-state
            return torch.tanh(h)         # hypothesis; also tames the quad blow-up over L steps)
        if self.norm == "clip":
            return h.clamp(-3.0, 3.0)
        return h

    def params(self) -> List[torch.Tensor]:
        return (sum((l.trainable_tensors() for l in self.links), [])
                + self.head.trainable_tensors())

    def n_params(self) -> int:
        return sum(int((p != 0).sum()) for p in self.params())


@dataclass
class StallTrigger:
    patience: int = 8
    rel_tol: float = 0.01
    target: float = 0.99

    def __post_init__(self) -> None:
        self.best, self.bad = float("inf"), 0

    def step(self, train_loss: float, test_acc: float) -> str:
        """'settled' (target reached) | 'stalled' | 'run'."""
        if test_acc >= self.target:
            return "settled"
        if train_loss < self.best * (1 - self.rel_tol):
            self.best, self.bad = train_loss, 0
        else:
            self.bad += 1
        return "stalled" if self.bad >= self.patience else "run"


@torch.no_grad()
def accuracy(model, e: torch.Tensor, y: torch.Tensor) -> float:
    return (model.forward(e).argmax(1) == y).float().mean().item()


def fit_stage(model, e, y, et, yt, epochs: int, seed: int, *, lr: float = 3e-3,
              bs: int = 128, trigger: StallTrigger | None = None, penalty=None):
    """One growth stage. Returns (stalled, test_acc, train_loss).
    ``penalty``: optional callable -> scalar tensor added to the loss (e.g. the
    epigenetic-lock EWC pull on a transferred link, s063 Mode E)."""
    trig = trigger or StallTrigger()
    opt = torch.optim.Adam(model.params(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    acc, tot = 0.0, float("nan")
    for _ in range(epochs):
        perm = torch.randperm(len(e), generator=g)
        tot = 0.0
        for i in range(0, len(e), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = F.cross_entropy(model.forward(e[idx]), y[idx])
            if penalty is not None: loss = loss + penalty()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.params(), 1.0)
            opt.step()
            tot += loss.item() * len(idx)
        tot /= len(e)
        acc = accuracy(model, et, yt)
        state = trig.step(tot, acc)
        if state == "settled":
            return False, acc, tot
        if state == "stalled":
            return True, acc, tot
    return True, acc, tot


def fit_grow(model, e, y, et, yt, *, epochs: int, seed: int, patience: int = 8,
             target: float = 0.99, log=print, penalty=None):
    """stall -> grow -> joint train, until settled or ``max_links``.
    Credit is NOT locked at spawn (lock-after-settlement rule, s059)."""
    stage = 1
    while True:
        stalled, acc, loss = fit_stage(
            model, e, y, et, yt, epochs, seed * 7 + stage,
            trigger=StallTrigger(patience=patience, target=target), penalty=penalty)
        log(f"    stage {stage} depth {model.depth}: acc {acc:.3f} "
            f"loss {loss:.3f} stalled={stalled}")
        if not stalled or not model.can_grow():
            return model.depth
        model.grow(); stage += 1
