"""Logophosis Mode E transfer — rung-1 FULL world (not/and/or) in seq mode (s063).

Arms (chain mode seq, same budget):
  scratch  fresh organism on the full world (NEXT-2 of s062: does seq keep the tied 40K
           .93/.90 and add `not not`?)
  warm     organism transferred from the NOT-only world (INIT), Link-0 rewoken on the
           new tokens, all links free
  lock     as warm, but the NOT link is epigenetically locked (anchor + lambda, EWC pull
           LOCK*penalty); head free
Scores: full-world in-dist / comp / depth+1 (`not not` inside and/or); RETAINED = the
NOT-only world's in-dist (0-2 NOTs) and not^3 zero-shot, i.e. does the primitive survive.
Env: SEEDS/SEED_LIST, ARMS, INIT (path with '{seed}'), LOCK, N, EPOCHS, PATIENCE, MAX_LINKS,
     DTRAIN, SEQ_NORM, THREADS, WAKE, WINDOW (as logophosis.py). Run from repo root.
"""
from __future__ import annotations
import os, sys, time
import torch
sys.path.insert(0, os.getcwd())
from experiments.progenitor import language_world as LW
from experiments.progenitor.logophosis import sets, LINK0_KW, SD
from trioron.organisms import Logophosis

N = int(os.environ.get("N", 40000))
DTRAIN = int(os.environ.get("DTRAIN", 1))
STAGE_EP = int(os.environ.get("EPOCHS", 10))
PATIENCE = int(os.environ.get("PATIENCE", 6))
MAX_LINKS = int(os.environ.get("MAX_LINKS", 2))
SEQ_NORM = os.environ.get("SEQ_NORM", "tanh")
INIT = os.environ.get("INIT", "outputs/organisms/logophosis_notonly_seq_tanh_s{seed}.pt")
LOCK = float(os.environ.get("LOCK", 1.0))


def not_only_sets(seed):
    """The transferred organism's home world (NOT_ONLY, depths 0-2 mixed) + not^3."""
    old = LW.NOT_ONLY; LW.NOT_ONLY = True
    try:
        home = sets(2, 2, 200 + seed, 4000, LW.HOLD_OUT, mix=True)
        not3 = sets(2, 3, 401 + seed, 4000, LW.HOLD_OUT)
    finally:
        LW.NOT_ONLY = old
    return home, not3


def run(arm, seed):
    assert not LW.NOT_ONLY, "run with NOT_ONLY unset (full world)"
    Str, Itr, ytr = sets(2, DTRAIN, 100 + seed, N, LW.HOLD_OUT)
    Ste, Ite, yte = sets(2, DTRAIN, 200 + seed, 4000, LW.HOLD_OUT)
    allpairs = [(a, n) for a in LW.COLORS + LW.SIZES for n in LW.SHAPES + ["thing"]]
    comp_hold = [p for p in allpairs if p not in LW.HOLD_OUT]
    Sc, Ic, yc = sets(2, DTRAIN, 300 + seed, 4000, comp_hold)
    Sd, Id, yd = sets(2, DTRAIN + 1, 401 + seed, 4000, LW.HOLD_OUT)
    home, not3 = not_only_sets(seed)
    t0 = time.time()
    log = lambda s: print(f"    [{arm} seed{seed}]{s}", flush=True)
    if arm == "scratch":
        org = Logophosis(SD, LW.CLASS_CAP, seed, mode="seq", max_links=MAX_LINKS,
                         seq_norm=SEQ_NORM, link0_kw=LINK0_KW).wake(Itr)
        r0 = {}
    else:
        org = Logophosis.load(INIT.format(seed=seed))
        r0 = dict(home0=org.accuracy(*home), not3_0=org.accuracy(*not3))
        log(f" loaded: home {r0['home0']:.3f} not3 {r0['not3_0']:.3f} "
            f"full-world zero-shot {org.accuracy(Ste, Ite, yte):.3f}")
        org.rewake(Itr)
        log(f" after rewake: home {org.accuracy(*home):.3f} not3 {org.accuracy(*not3):.3f}")
        org.lock_links(LOCK if arm == "lock" else 0.0)
    d = org.fit(Str, Itr, ytr, Ste, Ite, yte, epochs=STAGE_EP, patience=PATIENCE, target=0.99, log=log)
    r = dict(indist=org.accuracy(Ste, Ite, yte), comp=org.accuracy(Sc, Ic, yc),
             notnot=org.accuracy(Sd, Id, yd), home=org.accuracy(*home), not3=org.accuracy(*not3),
             links=d, params=org.n_params(), secs=time.time() - t0, **r0)
    log(f" indist {r['indist']:.3f} comp {r['comp']:.3f} notnot {r['notnot']:.3f} | retained home "
        f"{r['home']:.3f} not3 {r['not3']:.3f}")
    return r


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("THREADS", 4)))
    seeds = ([int(x) for x in os.environ["SEED_LIST"].split(",")] if os.environ.get("SEED_LIST")
             else range(int(os.environ.get("SEEDS", 3))))
    arms = os.environ.get("ARMS", "scratch,warm,lock").split(",")
    print(f"# Logophosis transfer FULL world N={N} dtrain={DTRAIN} ep={STAGE_EP} L_MAX={LW.L_MAX} "
          f"MAX_OBJ={LW.MAX_OBJ} seeds={list(seeds)} seq_norm={SEQ_NORM} lock={LOCK} init={INIT}", flush=True)
    for arm in arms:
        rs = [run(arm, s) for s in seeds]
        f = lambda k: torch.tensor([r[k] for r in rs])
        m = lambda k: f"{f(k).mean():.3f}±{f(k).std():.3f}" if len(rs) > 1 else f"{f(k).mean():.3f}"
        print(f"{arm:8s} indist {m('indist')}  comp {m('comp')}  notnot {m('notnot')}  | retained "
              f"home {m('home')}  not3 {m('not3')}  links {[r['links'] for r in rs]}  "
              f"params {[r['params'] for r in rs]}  {f('secs').mean():.0f}s", flush=True)
