"""Phasor cell vs quad cell (s065, Rocky's resonance question).

A phasor cell has complex weights and fires on the squared magnitude of its
summed input: h = |sum_j w_j x_j|^2.  Two inputs in phase add (resonance), out
of phase cancel.  With REAL inputs this is exactly two quad cells summed, so
the cell can only differ when inputs carry phase; and a quad cell given
(cos, sin) of each phase can represent the same function.  What the phasor
cell has by construction is that it sees only RELATIVE phase.  The question is
whether that built-in bias helps learning and generalisation.

Standalone torch (not the arena): one hidden layer of M cells + linear head.
Arms, hidden-layer parameter counts in the table:
  quad one-hot   z = Wx + b, h = z + z^2, symbols one-hot (current practice)
  quad cos/sin   same cell, each symbol as (cos phi, sin phi)
  phasor         h = |Wx|^2, W complex, each symbol as exp(i phi)
Tasks (N symbols, phi_s = 2 pi s / N):
  same/diff      two symbols; is a == b?  Train on pairs of 12 symbols, test on
                 pairs of the 4 HELD-OUT symbols (balanced)
  offset         (b - a) mod N, N classes; train on half the (a, b) pairs, test on the rest
  binding        3 objects, each with a colour and a shape; every object tags its
                 features with a random phase (fresh per sample).  Is colour 0 on
                 the same object as shape 0?  (no one-hot arm: without tags the
                 input is the same for both answers)

Run from repo root:  python3 experiments/progenitor/phasor_vs_quad.py
Env: SEEDS=5 M=8 STEPS=3000 N=16
"""
from __future__ import annotations
import math, os
import torch
import torch.nn.functional as F

N = int(os.environ.get("N", 16))
M = int(os.environ.get("M", 8))
STEPS = int(os.environ.get("STEPS", 3000))
PHI = 2 * math.pi * torch.arange(N) / N


class Quad(torch.nn.Module):
    def __init__(self, d, n_out):
        super().__init__()
        self.w, self.head = torch.nn.Linear(d, M), torch.nn.Linear(M, n_out)

    def forward(self, x):
        z = self.w(x)
        return self.head(z + z * z)


class Phasor(torch.nn.Module):
    def __init__(self, d, n_out):
        super().__init__()
        self.re = torch.nn.Parameter(torch.randn(M, d) / d ** 0.5)
        self.im = torch.nn.Parameter(torch.randn(M, d) / d ** 0.5)
        self.head = torch.nn.Linear(M, n_out)

    def forward(self, x):                                   # x complex [B, d]
        z = x @ torch.complex(self.re, self.im).T
        return self.head(z.real ** 2 + z.imag ** 2)


def onehot(s):  return F.one_hot(s, N).float().flatten(1)
def cossin(s):  return torch.cat([torch.cos(PHI[s]), torch.sin(PHI[s])], 1)
def phase(s):   return torch.polar(torch.ones(s.shape), PHI[s])


def fit(model, xtr, ytr, seed):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(STEPS):
        bi = torch.randint(len(ytr), (min(256, len(ytr)),))
        opt.zero_grad()
        F.cross_entropy(model(xtr[bi]), ytr[bi]).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model


def acc(model, x, y):
    with torch.no_grad():
        return (model(x).argmax(1) == y).float().mean().item()


def task_same(g):
    perm = torch.randperm(N, generator=g)
    seen, held = perm[:12], perm[12:]

    def pairs(sym):                                         # balanced same / different
        a = sym[torch.randint(len(sym), (4000,), generator=g)]
        b = sym[torch.randint(len(sym), (4000,), generator=g)]
        b = torch.where(torch.rand(4000, generator=g) < 0.5, a, b)
        return torch.stack([a, b], 1), (a == b).long()
    return pairs(seen), pairs(held), 2


def task_offset(g):
    ab = torch.cartesian_prod(torch.arange(N), torch.arange(N))
    ab = ab[torch.randperm(len(ab), generator=g)]
    y = (ab[:, 1] - ab[:, 0]) % N
    h = len(ab) // 2
    return (ab[:h], y[:h]), (ab[h:], y[h:]), N


def binding_data(g, n, C=4):
    """3 objects, colour and shape drawn without replacement; x = [colour slots | shape slots],
    each slot holds the phase tag of the object that owns it (0 if absent)."""
    x = torch.zeros(n, 2 * C, dtype=torch.cfloat)
    y = torch.zeros(n, dtype=torch.long)
    for i in range(n):
        col = torch.randperm(C, generator=g)[:3]; shp = torch.randperm(C, generator=g)[:3]
        if torch.rand(1, generator=g) < 0.5:                 # force half the samples positive
            col[0], shp[0] = 0, 0
            col[1:] = torch.arange(1, C)[torch.randperm(C - 1, generator=g)[:2]]
            shp[1:] = torch.arange(1, C)[torch.randperm(C - 1, generator=g)[:2]]
        tag = torch.polar(torch.ones(3), 2 * math.pi * torch.rand(3, generator=g))
        x[i, col] = tag; x[i, C + shp] = tag
        y[i] = int(any(col[k] == 0 and shp[k] == 0 for k in range(3)))
    return x, y


if __name__ == "__main__":
    seeds = range(int(os.environ.get("SEEDS", 5)))
    print(f"# phasor vs quad  N={N} M={M} steps={STEPS} seeds={list(seeds)}")
    res: dict[str, list] = {}
    npar: dict[str, int] = {}
    for seed in seeds:
        g = torch.Generator().manual_seed(seed)
        for tname, mk in (("same/diff (held-out symbols)", task_same), ("offset (held-out pairs)", task_offset)):
            (str_, ytr), (ste, yte), n_out = mk(g)
            for arm, enc, cls in (("quad one-hot", onehot, Quad), ("quad cos/sin", cossin, Quad),
                                  ("phasor", phase, Phasor)):
                xtr, xte = enc(str_), enc(ste)
                torch.manual_seed(seed)
                mdl = fit(cls(xtr.shape[1], n_out), xtr, ytr, seed)
                res.setdefault(f"{tname:30} {arm}", []).append((acc(mdl, xtr, ytr), acc(mdl, xte, yte)))
                npar[f"{tname:30} {arm}"] = sum(p.numel() for p in mdl.parameters())
        xtr, ytr = binding_data(g, 4000); xte, yte = binding_data(g, 2000)
        for arm, enc, cls in (("quad re/im", lambda x: torch.cat([x.real, x.imag], 1), Quad),
                              ("phasor", lambda x: x, Phasor)):
            for ntr in (100, 4000):
                torch.manual_seed(seed)
                mdl = fit(cls(enc(xtr).shape[1], 2), enc(xtr[:ntr]), ytr[:ntr], seed)
                key = f"{'binding, ' + str(ntr) + ' train samples':30} {arm}"
                res.setdefault(key, []).append((acc(mdl, enc(xtr[:ntr]), ytr[:ntr]), acc(mdl, enc(xte), yte)))
                npar[key] = sum(p.numel() for p in mdl.parameters())
        print(f"  seed {seed} done", flush=True)
    print(f"{'task':30} {'arm':14} {'params':>6} {'train':>12} {'test':>12}")
    for k in sorted(res):
        t = torch.tensor(res[k])
        sd = t.std(0) if len(t) > 1 else torch.zeros(2)
        print(f"{k:45} {npar[k]:>6} {t[:, 0].mean():.3f}±{sd[0]:.3f}  {t[:, 1].mean():.3f}±{sd[1]:.3f}")
