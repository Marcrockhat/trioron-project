"""Can the matron's directed labels make the composer grow? (Rocky s067)

Vee never held a composer trial in 120 days: a trial needs a class whose members
disagree on the raw readings (mean coherence < composer.TRIGGER_R), and division
slices every class tight first.  A LABEL pulls moments that look different into one
class — so a relation taught by the matron should produce such a class.

Toy, on a fresh leaf (not Vee): two readings a, b in {-1, +1}; the matron says
"same" when a == b, "different" otherwise — a relation no single reading expresses.
EXTRA further readings are present: KIND=tight (steady readings with small noise) or
KIND=switch (independent on/off readings, as in Vee's world).

Reported per arm: composer trials ever held, composer cells grown, classes, and
same/different read from the class on unseen moments (class named from its members;
the unlabelled arm is named afterwards by majority, measurement only).

Run from repo root:  python3 experiments/incubator/composer_probe.py
BLURS = how much every reading varies from moment to moment (Rocky s067: "when you say
fish, it is fish all the time. In the human world fish can be fiiish, fishhh, fsh";
"baby eyes are blur when they were born").  0.05 = Vee's world as scripted.

Env: SEEDS=3 N=4800 WINDOW=48 EXTRAS=0,6,30 BLURS=0.05,0.3,0.6
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.getcwd())
from trioron.pcll import PhasecyteLeaf

E = lambda k, d: type(d)(os.environ.get(k, d))
SEEDS, N, W = E("SEEDS", 3), E("N", 4800), E("WINDOW", 48)
EXTRAS = [int(x) for x in E("EXTRAS", "0,6,30").split(",")]
BLURS = [float(x) for x in E("BLURS", "0.05,0.3,0.6").split(",")]


def data(seed, extra, kind, n, blur=0.05):
    g = torch.Generator().manual_seed(seed)
    ab = torch.randint(0, 2, (n, 2), generator=g).float() * 2 - 1
    y = (ab[:, 0] == ab[:, 1]).long()
    ex = torch.full((n, extra), 0.3) if kind == "tight" else torch.randint(0, 2, (n, extra), generator=g).float() * 2 - 1
    X = torch.cat([ab, ex], 1) + blur * torch.randn(n, 2 + extra, generator=g)
    return X, y


def run(seed, extra, kind, told, blur):
    X, y = data(seed, extra, kind, N + 960, blur)
    leaf = PhasecyteLeaf(lambda x: x, X[:480], seed, 0, window=W, class_cap=None, manifold=True, composer=True, verbose=False)
    for i in range(0, N, W):
        leaf.mixed.observe(X[i:i + W], labels=[("same", "diff")[1 - int(v)] for v in y[i:i + W]] if told else None)
        leaf.sub.end_task()
    m = leaf.mixed
    cls = m.classify(X)[0]
    names = torch.full((len(m.classes),), -1)
    for c in cls[:N].unique():
        if c >= 0: names[c] = y[:N][cls[:N] == c].mode().values
    pred = torch.where(cls[N:] >= 0, names[cls[N:].clamp(min=0)], -1)
    return m._trial_seed, len(m.specs), len(m.classes), (pred == y[N:]).float().mean().item()


if __name__ == "__main__":
    print(f"# composer probe: same/different over two readings; {N} moments, {W} a sitting, seeds={SEEDS}")
    print(f"{'blur':5} {'extra readings':16} {'matron':8} | trials held | composer cells | classes | same/different on unseen (chance .50)")
    for blur, kind, extra, told in ((b, k, e, t) for b in BLURS for k in ("tight", "switch") for e in EXTRAS for t in (False, True)):
            if True:
                r = np.array([run(s, extra, kind, told, blur) for s in range(SEEDS)], float).mean(0)
                print(f"{blur:<5} {extra:3} {kind:12} {'labels' if told else 'silent':8} | {r[0]:>11.1f} | {r[1]:>14.1f} | {r[2]:>7.1f} | {r[3]:.2f}", flush=True)
