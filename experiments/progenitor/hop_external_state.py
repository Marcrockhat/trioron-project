"""s064 — external-state test (docs/design/incubator_matron.md §2.1, row 3).

Question: can trioron do DEEP pointer chasing by iterating a SHALLOW settled
step over a WRITTEN symbol, instead of carrying an analog state through depth?

World (same as logic_chain.make_hop): 8 slots, each a pointer in 0..7 (random
map f).  Target of the end-to-end task: f^k(0).

Step organ: input [start one-hot (8) | evidence(f) (64)], duplicated into both
dendritic halves (branch-partition fix, spec §3.6), ONE quad link, linear head
-> 8 classes.  Trained on RANDOM start positions to output f^STEP(start).
  step1: f(s)   = sum_i [s=i][f(i)=v]            degree 2 -> 1 quad layer
  step2: f(f(s))                                 degree 3 -> 2 quad layers

Iteration (no further training): apply the organ k/STEP times; between
applications the state is
  snap : argmax -> one-hot      (a written symbol; the external-state arm)
  soft : softmax probabilities  (analog hand-over; the control)

PASS: snap accuracy at depth k tracks p^(k/STEP) for single-step accuracy p,
far above the end-to-end tied_dup numbers (hop-3 .706, hop-4 .608, s064).

Run from repo root:  python3 experiments/progenitor/hop_external_state.py
Env: SEEDS=3 STEPS=1,2 EPOCHS=40 NTRAIN=8000 KMAX=8 THREADS=4
"""
from __future__ import annotations
import os, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(__file__))
from logic_chain import new_link, evidence, H

L = 8
EPOCHS = int(os.environ.get("EPOCHS", 40))
NTRAIN = int(os.environ.get("NTRAIN", 8000))
KMAX = int(os.environ.get("KMAX", 8))
STOP = 0.998


def make_maps(n, g):
    return torch.randint(0, L, (n, L), generator=g)


def chase(f, start, k):
    pos = start.clone()
    ar = torch.arange(len(f))
    for _ in range(k):
        pos = f[ar, pos]
    return pos


class StepOrgan:
    def __init__(self, step, seed):
        self.step = step
        self.link = new_link(2 * (L + L * 8), seed * 100, layers=step)
        torch.manual_seed(seed)
        self.head = nn.Linear(H, L)

    def forward(self, s, e):
        """s: [N, 8] state (one-hot or soft); e: [N, 64] evidence."""
        return self.head(self.link(torch.cat([s, e, s, e], 1)))

    def params(self):
        return self.link.trainable_tensors() + list(self.head.parameters())

    def n_params(self):
        return sum(p.numel() for p in self.params())


def fit(organ, f, ft, seed, lr=3e-3, bs=128):
    g = torch.Generator().manual_seed(seed)
    e, et = evidence(f), evidence(ft)
    st = torch.randint(0, L, (len(ft),), generator=g)
    yt = chase(ft, st, organ.step)
    opt = torch.optim.Adam(organ.params(), lr=lr)
    acc = 0.0
    for ep in range(EPOCHS):
        s = torch.randint(0, L, (len(f),), generator=g)      # fresh starts each epoch
        y = chase(f, s, organ.step)
        perm = torch.randperm(len(f), generator=g)
        for i in range(0, len(f), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = F.cross_entropy(organ.forward(F.one_hot(s[idx], L).float(), e[idx]), y[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(organ.params(), 1.0)
            opt.step()
        with torch.no_grad():
            acc = (organ.forward(F.one_hot(st, L).float(), et).argmax(1) == yt).float().mean().item()
        if acc >= STOP:
            break
    return acc, ep + 1


@torch.no_grad()
def iterate(organ, f, k, mode):
    """f^k(0) by k/step applications; returns accuracy."""
    e = evidence(f)
    s = F.one_hot(torch.zeros(len(f), dtype=torch.long), L).float()
    for _ in range(k // organ.step):
        logits = organ.forward(s, e)
        s = F.one_hot(logits.argmax(1), L).float() if mode == "snap" else logits.softmax(1)
    return (s.argmax(1) == chase(f, torch.zeros(len(f), dtype=torch.long), k)).float().mean().item()


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("THREADS", 4)))
    seeds = range(int(os.environ.get("SEEDS", 3)))
    steps = [int(x) for x in os.environ.get("STEPS", "1,2").split(",")]
    print(f"# external-state hop  seeds={list(seeds)} epochs<={EPOCHS} ntrain={NTRAIN}", flush=True)
    for step in steps:
        ks = [k for k in range(step, KMAX + 1, step)]
        rows = {("snap", k): [] for k in ks} | {("soft", k): [] for k in ks}
        ps, t0 = [], time.time()
        for seed in seeds:
            g = torch.Generator().manual_seed(1000 * seed + 7)
            f, ft, fe = make_maps(NTRAIN, g), make_maps(4000, g), make_maps(4000, g)
            organ = StepOrgan(step, seed)
            p, ep = fit(organ, f, ft, seed)
            ps.append(p)
            print(f"  step{step} s{seed}: single-step acc {p:.4f} after {ep} ep, "
                  f"params {organ.n_params()}", flush=True)
            for mode in ("snap", "soft"):
                for k in ks:
                    rows[(mode, k)].append(iterate(organ, fe, k, mode))
        p = torch.tensor(ps)
        print(f"step{step}: single-step p = {p.mean():.4f}±{p.std():.4f}   ({time.time() - t0:.0f}s)")
        print(f"  {'k':>3} {'snap':>13} {'soft':>13} {'p^(k/step)':>11}")
        for k in ks:
            a, b = torch.tensor(rows[("snap", k)]), torch.tensor(rows[("soft", k)])
            print(f"  {k:>3} {a.mean():.3f}±{a.std():.3f}   {b.mean():.3f}±{b.std():.3f}   "
                  f"{p.mean().item() ** (k // step):>9.3f}", flush=True)
