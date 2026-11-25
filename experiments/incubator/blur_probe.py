"""Organ blur: does an organ that starts blurred and sharpens slowly, by need, do as
well with fewer classes?  (Rocky s067: "baby eyes are blur when they were born";
"it need to be sharpen at very slow rate probably relative to her size".)

Toy, on a fresh leaf (not Vee).  Readings: c is 0.2 or 0.6 (a FINE difference), plus
EXTRA on/off readings (-1 / +1); every reading varies by NOISE from moment to moment.
The matron names every moment "low" or "high" (c).  Telling them apart needs a fine
view of c only; a coarse view is enough for everything else.

  sharp       the organ sees full resolution from the start (as Vee does)
  developing  the organ sees each reading in LEVELS steps over its range, starting at 2.
              It may sharpen by ONE step only when
                settled      no division for PATIENCE sittings, where
                             PATIENCE = BASE + classes // PER  (slower the bigger she is)
                unresolved   the matron still corrects her (label error > NEED)
              and the step is ON TRIAL: if the error has not fallen after PATIENCE more
              sittings, it is taken back.
Senses form at genesis on full-resolution readings (a receptor that saw only two
levels would become a two-level receptor for life).

Reported: steps taken / taken back, final levels, classes, high/low on unseen moments
(class named from its members' labels), sittings.

Run from repo root:  python3 experiments/incubator/blur_probe.py
Env: SEEDS=2 N=5760 WINDOW=48 EXTRA=4 NOISES=0.05,0.15 BASE=3 PER=8 NEED=0.1
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.getcwd())
from trioron.pcll import PhasecyteLeaf

E = lambda k, d: type(d)(os.environ.get(k, d))
SEEDS, N, W, EXTRA = E("SEEDS", 2), E("N", 5760), E("WINDOW", 48), E("EXTRA", 4)
NOISES = [float(x) for x in E("NOISES", "0.05,0.15").split(",")]
BASE, PER, NEED = E("BASE", 3), E("PER", 8), E("NEED", 0.1)
LO, HI = -1.5, 1.5


def data(seed, noise, n):
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, 2, (n,), generator=g)
    c = torch.where(y == 1, 0.6, 0.2)
    ex = torch.randint(0, 2, (n, EXTRA), generator=g).float() * 2 - 1
    return torch.cat([c[:, None], ex], 1) + noise * torch.randn(n, 1 + EXTRA, generator=g), y


def see(X, levels):
    """What a blurred organ sees: each reading in ``levels`` steps over its range (None = sharp)."""
    if levels is None: return X
    k = ((X.clamp(LO, HI - 1e-6) - LO) / (HI - LO) * levels).floor()
    return LO + (k + 0.5) * (HI - LO) / levels


def run(seed, noise, developing):
    X, y = data(seed, noise, N + 960)
    leaf = PhasecyteLeaf(lambda x: x, X[:480], seed, 0, window=W, class_cap=None, manifold=True, composer=True, verbose=False)
    m = leaf.mixed
    levels, quiet, trial, steps, back, errs, last_n = (2 if developing else None), 0, None, 0, 0, [], 1
    for i in range(0, N, W):
        xb, yb = see(X[i:i + W], levels), y[i:i + W]
        if len(m.classes) and m.label_taps is not None:               # what she would have said, before the matron speaks
            names = leaf.member_names()
            c = m.classify(xb)[0]
            errs.append((torch.where(c >= 0, names[c.clamp(min=0)], -1) != yb).float().mean().item())
        m.observe(xb, labels=[f"g{int(v)}" for v in yb]); leaf.sub.end_task()
        quiet = quiet + 1 if len(m.classes) == last_n else 0
        last_n = len(m.classes)
        if not developing or len(errs) < 3: continue
        patience, err = BASE + len(m.classes) // PER, float(np.mean(errs[-3:]))
        if trial is not None:                                           # a step on trial
            if trial[1] <= 0:
                if err >= trial[0] - 0.02: levels -= 1; back += 1       # it did not pay: taken back
                trial, quiet = None, 0
            else: trial = (trial[0], trial[1] - 1)
        elif quiet >= patience and err > NEED:                          # settled, and still corrected
            trial, levels, steps = (err, patience), levels + 1, steps + 1
    cls = m.classify(see(X, levels))[0]
    names = leaf.member_names()
    pred = torch.where(cls[N:] >= 0, names[cls[N:].clamp(min=0)], -1)
    return steps, back, levels or 0, len(m.classes), (pred == y[N:]).float().mean().item(), float(np.mean(errs[-10:]))


if __name__ == "__main__":
    print(f"# blur probe: fine reading 0.2 / 0.6 + {EXTRA} on/off readings; {N} moments, {W} a sitting; patience {BASE} + classes // {PER}; seeds={SEEDS}")
    print(f"{'noise':6} {'organ':11} | steps | taken back | final levels | classes | high/low on unseen | corrected in her last 10 sittings")
    for noise in NOISES:
        for dev in (False, True):
            r = np.array([run(s, noise, dev) for s in range(SEEDS)], float).mean(0)
            print(f"{noise:<6} {'developing' if dev else 'sharp':11} | {r[0]:>5.1f} | {r[1]:>10.1f} | {r[2]:>12.1f} | {r[3]:>7.1f} | {r[4]:>18.2f} | {r[5]:.2f}", flush=True)
