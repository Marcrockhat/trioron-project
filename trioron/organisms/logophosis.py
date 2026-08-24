"""Logophosis — the pure-trioron language organism (logo- word/reason + -phosis a
becoming; named by Rocky, s061; stable API s062).

Everything that learns is substrate:

* **Link 0** — ``ChordLink0``: a ``PhasecyteLeaf`` over CHORD-encoded tokens
  (``trioron.senses.chord``) discovers the word inventory WITHOUT gradients and
  emits matched-filter evidence per token (centered branch -> softmax = a crisp
  symbol; genuinely ambiguous tokens stay soft = doubt carried). ``link0="oracle"``
  keeps the one-hot control.
* **Chain** — ``trioron.lifecycle.chain.LinkChain``: ``seq`` (one token per step,
  scene as per-step context, fan-in link init; the NOT operator is the SAME
  weights at every nesting depth), ``tied`` (flattened sentence, one link
  re-applied), ``grown`` (fresh links on stall). Substrate head link.
* **Growth** — ``StallTrigger`` -> grow -> joint train (``fit_grow``),
  lock-after-settle.

Native input: ``scene [N, scene_dim]`` float + ``ids [N, L]`` long token ids
(0 = PAD, fixed ``L``). Output: ``comprehend -> [N, n_out]`` probabilities.

    org = Logophosis(scene_dim, class_cap=32, seed=0, mode="seq")
    org.wake(ids_train)                       # Link-0 inventory discovery
    org.fit(S, ids, y, S_te, ids_te, y_te)    # stall -> grow -> joint, returns depth
    p = org.comprehend(S_new, ids_new)        # [N, 2]
    org.save("logophosis.pt"); Logophosis.load("logophosis.pt")
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from trioron.lifecycle.chain import LinkChain, fit_grow
from trioron.organisms.base import Organism
from trioron.pcll.nest import PhasecyteLeaf
from trioron.senses.chord import chord_table, chords_of


def _identity(x: torch.Tensor) -> torch.Tensor:   # module-level: pickle-safe sense
    return x


class ChordLink0:
    """Phasecyte symbol layer on the token chord stream. No gradients.

    ``ev_mode``: 'plain' (raw matched filter; dense, unusable downstream),
    'centered' (s039 common-mode fix), 'softmax' (centered / tau -> crisp symbol).
    """

    def __init__(self, ids_train: torch.Tensor, class_cap: int, seed: int, *,
                 wake: int = 20000, window: int = 1000, ev_mode: str = "softmax",
                 ev_tau: float = 1.0, bins: int = 32, k: int = 3, jitter: float = 0.05,
                 capacity: int = 4096, verbose: bool = True) -> None:
        self.class_cap, self.ev_mode, self.ev_tau, self.jitter, self.window = class_cap, ev_mode, ev_tau, jitter, window
        self.table = chord_table(class_cap, bins=bins, k=k, seed=seed)
        self.gen = torch.Generator().manual_seed(1000 + seed)
        ids = ids_train.flatten(); ids = ids[ids != 0]
        g = torch.Generator().manual_seed(seed)
        ids = ids[torch.randperm(len(ids), generator=g)[:wake]]
        X = chords_of(ids, self.table, self.gen, self.jitter)
        self.leaf = PhasecyteLeaf(_identity, X, seed, 0, window=window, class_cap=class_cap,
                                  input_shape=None, capacity=capacity, verbose=verbose)
        labels = [f"c{int(i)}" for i in ids]          # naming taps only (not a training signal)
        for i in range(0, len(X), window):
            self.leaf.observe(X[i:i + window], labels[i:i + window])
        names = self.leaf.names()
        self.n_classes = len(names)
        self.n_named = len(set(int(v) for v in names if v >= 0))
        if verbose:
            print(f"  [link0] internal classes {self.n_classes}, named {self.n_named} words", flush=True)

    def observe_more(self, ids: torch.Tensor, n: int = 20000) -> None:
        """Continued inventory discovery on a new token stream (no gradients; new
        words get new classes, old classes keep their indices). s063 Mode E."""
        ids = ids.flatten(); ids = ids[ids != 0]
        ids = ids[torch.randperm(len(ids), generator=self.gen)[:n]]
        X = chords_of(ids, self.table, self.gen, self.jitter)
        labels = [f"c{int(i)}" for i in ids]
        w = self.window
        for i in range(0, len(X), w):
            self.leaf.observe(X[i:i + w], labels[i:i + w])
        names = self.leaf.names()
        self.n_classes, self.n_named = len(names), len(set(int(v) for v in names if v >= 0))

    @torch.no_grad()
    def evidence(self, ids: torch.Tensor, chunk: int = 4000) -> torch.Tensor:
        """``[N, L, class_cap]`` per-token symbol evidence; PAD rows zero. Chunked over
        sentences: the whole-set chord tensor was the ~2.9 GB Link-0 peak (OOM, s062)."""
        N, L = ids.shape
        out = torch.zeros(N, L, self.class_cap)
        c = min(self.n_classes, self.class_cap)
        for i in range(0, N, chunk):
            sub = ids[i:i + chunk]
            X = chords_of(sub, self.table, self.gen, self.jitter).flatten(0, 1)
            ev_p, ev_c = self.leaf.evidence(_identity, X)
            ev = (ev_c if self.ev_mode != "plain" else ev_p).float()
            if self.ev_mode == "softmax":
                ev = torch.softmax(ev / self.ev_tau, 1)
            ev = ev * (sub.flatten() != 0).unsqueeze(1)
            out[i:i + chunk, :, :c] = ev[:, :c].view(len(sub), L, c)
        return out


class OracleLink0:
    """One-hot control for Link 0."""

    def __init__(self, class_cap: int) -> None:
        self.class_cap = class_cap
        self.n_classes = self.n_named = class_cap

    def evidence(self, ids: torch.Tensor) -> torch.Tensor:
        e = F.one_hot(ids, self.class_cap).float()
        e[ids == 0] = 0
        return e


class Logophosis(Organism):
    name = "logophosis"

    def __init__(self, scene_dim: int, class_cap: int, seed: int, *, n_out: int = 2,
                 mode: str = "seq", link0: str = "phasecyte", max_links: int = 4,
                 H: int = 32, interior: int = 48, seq_norm: str = "none",
                 link_fan_in: Optional[bool] = None, seq_dup: bool = False,
                 link0_kw: Optional[dict] = None) -> None:
        super().__init__(seed)
        assert mode in ("seq", "tied", "grown") and link0 in ("phasecyte", "oracle")
        self.scene_dim, self.class_cap, self.n_out, self.mode = scene_dim, class_cap, n_out, mode
        self.link0_kind, self.link0_kw = link0, dict(link0_kw or {})
        self.max_links, self.H, self.interior, self.seq_norm = max_links, H, interior, seq_norm
        self.seq_dup = seq_dup
        self.link_fan_in = (mode == "seq") if link_fan_in is None else link_fan_in   # s062 probe
        self.link0 = OracleLink0(class_cap) if link0 == "oracle" else None
        self.chain: Optional[LinkChain] = None
        self.L: Optional[int] = None

    # ── Link 0 ──
    def wake(self, ids_train: torch.Tensor):
        """Discover the symbol inventory from the token stream (no-op for oracle)."""
        if self.link0_kind == "phasecyte":
            self.link0 = ChordLink0(ids_train, self.class_cap, self.seed, **self.link0_kw)
        return self

    def rewake(self, ids_new: torch.Tensor):
        """Mode E: extend the Link-0 inventory on a new world's tokens (no-op for oracle)."""
        if self.link0_kind == "phasecyte":
            self.link0.observe_more(ids_new)
            print(f"  [link0] after rewake: classes {self.link0.n_classes}, named {self.link0.n_named}", flush=True)
        return self

    def lock_links(self, strength: float, lam: float = 1.0) -> None:
        """Epigenetic lock on the chain link(s): anchor now, λ=``lam`` on every cell; the
        EWC pull ``strength``·penalty is added to the training loss in ``fit``. The head
        stays free. ``strength=0`` clears the lock. (s063 Mode E transfer.)"""
        from trioron.learning.epigenetic_lock import anchor, set_lambda
        self._lock = strength
        if strength > 0:
            for l in self.chain.links:
                a = l.arena
                anchor(a); set_lambda(a, torch.full_like(a.node_lambda, lam), "absolute")

    def _penalty(self):
        from trioron.learning.epigenetic_lock import ewc_penalty
        s = getattr(self, "_lock", 0.0)
        if not s: return None
        return lambda: s * sum(ewc_penalty(l.arena) for l in self.chain.links)

    def encode(self, scene: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        """Chain input: ``[scene | evidence flattened]`` (seq: scene is the context head)."""
        if self.link0 is None:
            raise RuntimeError("call wake(ids_train) before encode/fit")
        if self.L is None: self.L = ids.shape[1]
        assert ids.shape[1] == self.L, f"token length {ids.shape[1]} != organism L {self.L}"
        return torch.cat([scene.float(), self.link0.evidence(ids).flatten(1)], 1)

    # ── chain ──
    def _build_chain(self) -> None:
        if self.mode == "seq":
            self.chain = LinkChain(self.class_cap, self.n_out, self.seed, H=self.H,
                                   interior=self.interior, mode="seq", max_links=self.max_links,
                                   ctx=self.scene_dim, norm=self.seq_norm,
                                   link_fan_in=self.link_fan_in, dup=self.seq_dup)
        else:
            self.chain = LinkChain(self.scene_dim + self.L * self.class_cap, self.n_out, self.seed,
                                   H=self.H, interior=self.interior, mode=self.mode,
                                   max_links=self.max_links)

    def fit(self, scene, ids, y, scene_te, ids_te, y_te, *, epochs: int = 20,
            patience: int = 8, target: float = 0.99, log=print) -> int:
        """stall -> grow -> joint train until settled or ``max_links``; returns depth."""
        X, Xt = self.encode(scene, ids), self.encode(scene_te, ids_te)
        if self.chain is None: self._build_chain()
        return fit_grow(self.chain, X, y, Xt, y_te, epochs=epochs, seed=self.seed,
                        patience=patience, target=target, log=log, penalty=self._penalty())

    @torch.no_grad()
    def comprehend(self, scene: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        """``[N, n_out]`` probabilities."""
        return torch.softmax(self.chain.forward(self.encode(scene, ids)), 1)

    @torch.no_grad()
    def accuracy(self, scene, ids, y) -> float:
        return (self.comprehend(scene, ids).argmax(1) == y).float().mean().item()

    @property
    def depth(self) -> int:
        return self.chain.depth if self.chain else 0

    def substrates(self) -> list:
        return ([] if self.chain is None else self.chain.links + [self.chain.head])

    def n_params(self) -> int:
        return self.chain.n_params() if self.chain else 0
