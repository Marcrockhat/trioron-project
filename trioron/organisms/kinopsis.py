"""Kinopsis — the motion-sight organism (kine- motion + opsis sight; s057, stable API s062).

One saveable object: a frozen shape organ (static-shape leaves ``shape`` / ``whole`` /
``fill``, colour-grouped), a motion-silhouette leaf ``msil`` (shape-from-motion), a
velocity leaf ``vel``, an evidence ``Router`` (~60 torch params — the only non-substrate
learner) and every calibration statistic. Built by a continual schedule; each stage
method returns ``self``:

    org = Kinopsis.from_organ(leaves, std, empty_mshape, seed)   # stage 1, OLD world (frozen)
    org.settle_heads(S_tr, Y_tr, old_tr, y_old_tr)   # stage 2a (optional): replay-settle organ heads
    org.acquire_msil(S_tr, y_shape)                   # stage 2b: motion-grouped shape leaf, frozen
    org.fit_router(S_tr, E_tr, E_frozen, y_shape)     # stage 2c: router alone, voters frozen
    org.acquire_vel(Zm_tr, y_vel)                     # stage 3: velocity leaf (paste-and-go)
    org.shape(S, E) -> [N, 5] routed log-prob;  org.velocity(Zm) -> [N, 17] logits

Inputs are the organism's native feature streams (dicts of ``[N, d]`` tensors keyed
``shape``/``whole``/``fill``/``mshape``), the label-free 11-d evidence vector ``E``
(last column = raw motion-mask area = the hard gate), and the motion feature ``Zm``.
The world side (feature extraction, grouping, splits) lives in ``experiments/``.

Intended use beyond the bench: a frozen perceptual AUGMENTATION organ for any image /
frame-packet classifier (shape log-prob, velocity logits, voters' H-codes as features).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from trioron.learning.manifold import get_interior_ids
from trioron.organisms.base import Organism, Std, chunked, leaf, train_leaf

SHAPE_CLASSES, VEL_CLASSES = 5, 17
VOTERS = ("shape", "whole", "msil")


def conf(L: torch.Tensor) -> torch.Tensor:
    """Per voter: max prob, entropy."""
    p = F.softmax(L, 1)
    return torch.stack([p.max(1).values, -(p * torch.log(p + 1e-8)).sum(1)], 1)


def vote(Ls: List[torch.Tensor], w: torch.Tensor) -> torch.Tensor:
    """Weighted log-softmax vote; ``w`` is ``[N, K]`` or per-class ``[N, K, C]``."""
    return sum((w[:, k:k + 1] if w.dim() == 2 else w[:, k]) * F.log_softmax(L, 1)
               for k, L in enumerate(Ls))


class Router(torch.nn.Module):
    """softmax over voters; the MOTION voter is hard-masked when its motion mask is
    empty (fires-only-when-moving): a leaf that has never seen a class cannot be
    allowed to vote against it from an empty input."""

    def __init__(self, d: int, K: int, gate_col=None, motion_idx=None) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(d, K)
        torch.nn.init.zeros_(self.lin.weight); torch.nn.init.zeros_(self.lin.bias)
        self.gc, self.mi = gate_col, motion_idx

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        z = self.lin(e)
        if self.gc is not None:
            off = e[:, self.gc] <= 0; z = z.clone(); z[off, self.mi] = -1e4
        return F.softmax(z, 1)


class QDAVoter:
    """H-space ManifoldRouter in miniature: per-class full-cov Gaussian over the
    concatenated voter H-codes; log-likelihood as logits."""

    def __init__(self, H: torch.Tensor, y: torch.Tensor, ridge: float = 0.1) -> None:
        self.mu, self.P, self.ld = {}, {}, {}
        for c in y.unique().tolist():
            Hc = H[y == c]; cov = torch.cov(Hc.T) + ridge * torch.eye(H.shape[1])
            self.mu[c] = Hc.mean(0); self.P[c] = torch.linalg.inv(cov); self.ld[c] = torch.logdet(cov)

    def __call__(self, H: torch.Tensor) -> torch.Tensor:
        out = []
        for c in sorted(self.mu):
            d = H - self.mu[c]
            out.append(-0.5 * ((d @ self.P[c]) * d).sum(1) - 0.5 * self.ld[c])
        return torch.stack(out, 1)


class Kinopsis(Organism):
    name = "kinopsis"

    def __init__(self, seed: int, leaves: Dict[str, object], std: Dict[str, tuple],
                 empty_mshape: torch.Tensor) -> None:
        """``leaves``: frozen organ substrates keyed shape/whole/fill; ``std``: per-stream
        (mu, sd) shape-world calibration; ``empty_mshape``: the msil stream descriptor of
        an empty motion mask (what a static image feeds the motion voter)."""
        super().__init__(seed)
        self.leaves, self.std, self.empty_mshape = leaves, std, empty_mshape
        self.h_ids = {k: get_interior_ids(sub.arena).long() for k, sub in leaves.items()}
        with torch.no_grad():   # drop grad-bearing last_activations (deepcopy/pickle)
            for k, sub in leaves.items(): sub(torch.zeros(1, std[k][0].numel()))
        self.msil = self.vel = self.router = self.stdm = self.qda = None
        self.ev_mu = self.ev_sd = self.h_mu = self.h_sd = None
        self.settled, self.use_h, self.per_class, self.K = False, False, False, 0

    @classmethod
    def from_organ(cls, organ, empty_mshape: torch.Tensor, seed: int) -> "Kinopsis":
        """Stage 1: adopt a trained shape organ (``.leaves``, ``.std``)."""
        return cls(seed, organ.leaves, organ.std, empty_mshape)

    def norm(self, k: str, Z: torch.Tensor) -> torch.Tensor:
        mu, sd = self.std[k]; return ((Z - mu) / sd).float()

    # ── stage 2a: replay settle of the organ heads (s054 recipe) ──
    def settle_heads(self, S_tr, Y_tr, old_tr, y_old_tr, n: int = 2000, steps: int = 200,
                     rank: int = 32):
        """Class-balanced full-cov archive samples of every OLD class mixed into each
        batch; only the head rows (bias + edges into output cells) move."""
        g = torch.Generator().manual_seed(100 + self.seed)
        idx = torch.randperm(len(Y_tr["y_shape"]), generator=g)[:n]
        for k, sub in self.leaves.items():
            ykey = "y_shape" if k != "fill" else "y_fill"
            yk_old = y_old_tr[ykey]; Zo = self.norm(k, old_tr[k]); arch = {}
            for c in yk_old.unique().tolist():
                Zc = Zo[yk_old == c]; mu = Zc.mean(0)
                cov = torch.cov(Zc.T) + 0.1 * torch.eye(Zc.shape[1])
                ev, V = torch.linalg.eigh(cov); ev = ev.clamp(min=1e-8); r = min(rank, len(ev))
                resid = float(ev[:-r].mean()) ** 0.5 if r < len(ev) else 0.0   # full-rank: no residual
                arch[c] = (mu, V[:, -r:] * ev[-r:].sqrt(), resid)
            for p in sub.trainable_tensors(): p.requires_grad_(True)
            a = sub.arena; out_ids = sub.scheduler._plan.output_ids.long()
            hc = torch.zeros(a.capacity, dtype=torch.bool); hc[out_ids] = True
            he = torch.isin(a.edge_dst.long(), out_ids)
            opt = torch.optim.Adam([a.bias, a.edge_weight], lr=1e-3)
            yk = Y_tr[ykey]; Zk = self.norm(k, S_tr[k]); gg = torch.Generator().manual_seed(self.seed)
            per = max(2, 128 // len(arch))
            for _ in range(steps):
                bi = idx[torch.randint(len(idx), (128,), generator=gg)]; xs, ys = [Zk[bi]], [yk[bi]]
                for c, (mu, A, rs) in arch.items():
                    xs.append(mu + torch.randn(per, A.shape[1], generator=gg) @ A.t()
                              + rs * torch.randn(per, mu.numel(), generator=gg))
                    ys.append(torch.full((per,), c, dtype=torch.long))
                opt.zero_grad(); F.cross_entropy(sub(torch.cat(xs)), torch.cat(ys)).backward()
                a.bias.grad[~hc] = 0.0; a.edge_weight.grad[~he] = 0.0; sub.zero_dormant_grads(); opt.step()
            for p in sub.trainable_tensors(): p.requires_grad_(False)
        self.settled = True; return self

    # ── stage 2b/c: msil leaf + router ──
    def acquire_msil(self, S_tr, y_tr, epochs: int = 8):
        Z = self.norm("shape", S_tr["mshape"])
        self.msil = train_leaf(leaf(Z.shape[1], SHAPE_CLASSES, self.seed), Z, y_tr, self.seed, epochs)
        self.h_ids["msil"] = get_interior_ids(self.msil.arena).long(); return self

    def voters(self, S, with_h: bool = False):
        """List of voter logits [shape, whole, msil(, qda)]; ``with_h`` also returns the
        concatenated H-codes ``[N, 3*48]``."""
        m = S["mshape"] if "mshape" in S else self.empty_mshape.expand(len(S["shape"]), -1)
        outs = [chunked(self.leaves[k], self.norm(k, S[k]), h_ids=self.h_ids[k]) for k in ("shape", "whole")]
        outs.append(chunked(self.msil, self.norm("shape", m), h_ids=self.h_ids["msil"]) if self.msil is not None
                    else (torch.zeros_like(outs[0][0]), torch.zeros_like(outs[0][1])))
        Ls = [o[0] for o in outs]; H = torch.cat([o[1] for o in outs], 1)
        if self.qda is not None: Ls.append(self.qda(H))
        return (Ls, H) if with_h else Ls

    def _feat(self, Ls, E, H=None):
        cols = [(E - self.ev_mu) / self.ev_sd, *[conf(L) for L in Ls]]
        if self.use_h: cols.append((H - self.h_mu) / self.h_sd)
        return torch.cat(cols + [E[:, 9:10]], 1)   # last col = raw motion-mask area (hard gate)

    def _w(self, X):
        """Router weights ``[N, K]`` or, per-class, ``[N, K, C]``."""
        z = self.router.lin(X); z = z.view(len(X), self.K, SHAPE_CLASSES) if self.per_class else z
        off = X[:, -1] <= 0; z = z.clone(); z[off, 2] = -1e4   # motion-fed voters silent when the mask is empty
        if self.qda is not None: z[off, 3] = -1e4
        return F.softmax(z, 1)

    def fit_router(self, S_tr, E_tr, E_frz, y_tr, steps: int = 300, use_h: bool = False,
                   per_class: bool = False, qda: bool = False, wd: float = 0.0):
        """Router alone (voters frozen) on the moving train packets + their FROZEN replay
        (``E_frz``: evidence of the mid frame repeated = nothing moves). ``use_h``: voter
        H-codes join the evidence; ``per_class``: weights per voter AND class; ``qda``:
        H-space full-cov Gaussian voter, gated with msil."""
        self.ev_mu, self.ev_sd = E_tr.mean(0), E_tr.std(0) + 1e-6; self.use_h, self.per_class = use_h, per_class
        S_frz = {k: S_tr[k] for k in ("shape", "whole")}
        if qda: _, Htr = self.voters(S_tr, with_h=True); self.qda = QDAVoter(Htr, y_tr)
        (Vm, Hm), (Vf, Hf) = self.voters(S_tr, with_h=True), self.voters(S_frz, with_h=True)
        self.h_mu, self.h_sd = Hm.mean(0), Hm.std(0) + 1e-6; self.K = len(Vm)
        V = [torch.cat([a, b]) for a, b in zip(Vm, Vf)]
        X = torch.cat([self._feat(Vm, E_tr, Hm), self._feat(Vf, E_frz, Hf)]); y = torch.cat([y_tr, y_tr])
        self.router = Router(X.shape[1], self.K * (SHAPE_CLASSES if per_class else 1))
        opt = torch.optim.Adam(self.router.parameters(), lr=1e-2, weight_decay=wd)
        g = torch.Generator().manual_seed(self.seed)
        for _ in range(steps):
            bi = torch.randint(len(y), (256,), generator=g); opt.zero_grad()
            F.cross_entropy(vote([L[bi] for L in V], self._w(X[bi])), y[bi]).backward(); opt.step()
        for p in self.router.parameters(): p.requires_grad_(False)
        return self

    # ── stage 3: velocity ──
    def acquire_vel(self, Zm_tr, y_vel, epochs: int = 8):
        self.stdm = Std(Zm_tr)
        self.vel = train_leaf(leaf(Zm_tr.shape[1], VEL_CLASSES, self.seed), self.stdm(Zm_tr), y_vel, self.seed, epochs)
        return self

    # ── inference ──
    @torch.no_grad()
    def shape(self, S, E: Optional[torch.Tensor] = None, mode: str = "router") -> torch.Tensor:
        """``[N, 5]`` routed log-probability (``mode='uniform'`` = equal vote)."""
        Ls, H = self.voters(S, with_h=True)
        if mode == "router" and self.router is not None and E is not None:
            return vote(Ls, self._w(self._feat(Ls, E, H)))
        w = torch.ones(1, len(Ls)) if self.msil is not None else torch.tensor([[1., 1., 0.]])
        if "mshape" not in S: w = torch.tensor([[1., 1.] + [0.] * (len(Ls) - 2)])
        return vote(Ls, w.expand(len(Ls[0]), -1))

    @torch.no_grad()
    def velocity(self, Zm: torch.Tensor) -> torch.Tensor:
        """``[N, 17]`` velocity-class logits."""
        return chunked(self.vel, self.stdm(Zm))

    def substrates(self) -> list:
        return list(self.leaves.values()) + [s for s in (self.msil, self.vel) if s is not None]

    def torch_modules(self) -> list:
        return [self.router] if self.router is not None else []


MotionOrganism = Kinopsis   # s057 first name
