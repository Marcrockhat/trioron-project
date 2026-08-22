"""s060+ — language comprehension, PURE trioron ("run 1+", Rocky).

Everything that learns is substrate:
  Link 0  : PhasecyteLeaf over CHORD-encoded tokens (math-generated spectra, language_world
            .chord_table) — discovers the word inventory without gradients; emits matched-
            filter evidence [L_MAX, n_classes] (padded to CLASS_CAP). LINK0=oracle keeps the
            one-hot control.
  Chain   : trioron.lifecycle.chain.LinkChain (tied | grown), substrate head link.
  Growth  : StallTrigger -> grow -> joint train (fit_grow), lock-after-settle.
Env: SEED_LIST/SEEDS, ARMS=tied,grown, LINK0=phasecyte|oracle, N, EPOCHS, PATIENCE, MAX_LINKS,
     DTRAIN, L_MAX, MAX_OBJ, THREADS, WAKE (tokens observed by Link 0), WINDOW.
Run from repo root.
"""
from __future__ import annotations
import os, sys, time
import torch
sys.path.insert(0, os.getcwd())
from experiments.progenitor import language_world as LW
from trioron.lifecycle.chain import LinkChain, fit_grow, accuracy
from trioron.pcll.nest import PhasecyteLeaf

N = int(os.environ.get("N", 20000))
DTRAIN = int(os.environ.get("DTRAIN", 1))
STAGE_EP = int(os.environ.get("EPOCHS", 20))
PATIENCE = int(os.environ.get("PATIENCE", 8))
MAX_LINKS = int(os.environ.get("MAX_LINKS", 4))
LINK0 = os.environ.get("LINK0", "phasecyte")
WAKE = int(os.environ.get("WAKE", 20000))
WINDOW = int(os.environ.get("WINDOW", 1000))
EV_MODE = os.environ.get("EV_MODE", "softmax")   # plain | centered | softmax
EV_TAU = float(os.environ.get("EV_TAU", 1.0))
SD = LW.MAX_OBJ * LW.OBJ_DIM


class ChordLink0:
    """Phasecyte symbol layer on the token chord stream. No gradients."""

    def __init__(self, seed, E_train):
        self.table = LW.chord_table(seed)
        self.gen = torch.Generator().manual_seed(1000 + seed)
        ids = LW.ids_of(E_train).flatten()
        ids = ids[ids != 0]
        g = torch.Generator().manual_seed(seed)
        ids = ids[torch.randperm(len(ids), generator=g)[:WAKE]]
        X = LW.chords_of(ids, self.table, self.gen)
        self.sense = lambda x: x
        self.leaf = PhasecyteLeaf(self.sense, X, seed, 0, window=WINDOW,
                                  class_cap=LW.CLASS_CAP, input_shape=None,
                                  capacity=4096, verbose=True)
        labels = [f"c{int(i)}" for i in ids]
        for i in range(0, len(X), WINDOW):
            self.leaf.observe(X[i:i + WINDOW], labels[i:i + WINDOW])
        names = self.leaf.names()
        self.n_cls = len(names)
        found = len(set(int(v) for v in names if v >= 0))
        print(f"  [link0] internal classes {self.n_cls}, named {found}/{len(LW.VOCAB) - 1} words",
              flush=True)

    @torch.no_grad()
    def evidence(self, E):
        ids = LW.ids_of(E)                       # [N, L]
        X = LW.chords_of(ids, self.table, self.gen).flatten(0, 1)
        ev_p, ev_c = self.leaf.evidence(self.sense, X)   # [N*L, n_cls]
        ev = (ev_c if EV_MODE != "plain" else ev_p).float()
        if EV_MODE == "softmax":                          # snap to a crisp symbol
            ev = torch.softmax(ev / EV_TAU, 1)
        ev = ev * (ids.flatten() != 0).unsqueeze(1)       # PAD = no evidence
        out = torch.zeros(len(ev), LW.CLASS_CAP)
        out[:, :min(self.n_cls, LW.CLASS_CAP)] = ev[:, :LW.CLASS_CAP]
        return out.view(len(E), -1)


def sets(stage, depth, seed, n, hold):
    S, E, Y, _ = LW.build(n, stage, depth, seed, hold=hold)
    return S, E, Y


def run(arm, seed):
    Str, Etr, ytr = sets(2, DTRAIN, 100 + seed, N, LW.HOLD_OUT)
    Ste, Ete, yte = sets(2, DTRAIN, 200 + seed, 4000, LW.HOLD_OUT)
    allpairs = [(a, n) for a in LW.COLORS + LW.SIZES for n in LW.SHAPES + ["thing"]]
    comp_hold = [p for p in allpairs if p not in LW.HOLD_OUT]
    Sc, Ec, yc = sets(2, DTRAIN, 300 + seed, 4000, comp_hold)
    Sd, Ed, yd = sets(2, DTRAIN + 1, 400 + seed, 4000, LW.HOLD_OUT)
    t0 = time.time()
    if LINK0 == "phasecyte":
        l0 = ChordLink0(seed, Etr)
        enc = lambda S, E: torch.cat([S, l0.evidence(E)], 1)
    else:
        enc = lambda S, E: torch.cat([S, E.flatten(1)], 1)
    Xtr, Xte, Xc, Xd = enc(Str, Etr), enc(Ste, Ete), enc(Sc, Ec), enc(Sd, Ed)
    print(f"  [link0 {LINK0}] evidence built {time.time() - t0:.0f}s", flush=True)
    m = LinkChain(Xtr.shape[1], 2, seed, mode=arm, max_links=MAX_LINKS)
    d = fit_grow(m, Xtr, ytr, Xte, yte, epochs=STAGE_EP, seed=seed, patience=PATIENCE,
                 target=0.99, log=lambda s: print(f"    [{arm} seed{seed}]{s}", flush=True))
    return dict(indist=accuracy(m, Xte, yte), comp=accuracy(m, Xc, yc),
                depth=accuracy(m, Xd, yd), links=d, params=m.n_params(),
                secs=time.time() - t0)


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("THREADS", 4)))
    seeds = ([int(x) for x in os.environ["SEED_LIST"].split(",")] if os.environ.get("SEED_LIST")
             else range(int(os.environ.get("SEEDS", 3))))
    arms = os.environ.get("ARMS", "tied,grown").split(",")
    print(f"# language PURE link0={LINK0} N={N} dtrain={DTRAIN} ep={STAGE_EP} "
          f"L_MAX={LW.L_MAX} MAX_OBJ={LW.MAX_OBJ} seeds={list(seeds)}", flush=True)
    for arm in arms:
        rs = [run(arm, s) for s in seeds]
        f = lambda k: torch.tensor([r[k] for r in rs])
        print(f"{arm:12s} indist {f('indist').mean():.3f}±{f('indist').std():.3f}  "
              f"comp {f('comp').mean():.3f}±{f('comp').std():.3f}  "
              f"depth+1 {f('depth').mean():.3f}±{f('depth').std():.3f}  "
              f"links {[r['links'] for r in rs]}  params {[r['params'] for r in rs]}  "
              f"{f('secs').mean():.0f}s", flush=True)
