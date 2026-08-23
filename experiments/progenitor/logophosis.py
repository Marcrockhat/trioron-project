"""Logophosis bench runner — rung-1 (negation) comprehension on the grounded language world.

The organism itself is the package class ``trioron.organisms.Logophosis`` (Phasecyte Link 0
on chord words + LinkChain + substrate head; see its docstring). This file is the WORLD side:
builds the splits, wakes/fits the organism, scores in-dist / compositional / deeper-nesting.

Env: SEED_LIST/SEEDS, ARMS=tied,grown,seq (chain mode), LINK0=phasecyte|oracle, N, EPOCHS,
     PATIENCE, MAX_LINKS, DTRAIN, L_MAX, MAX_OBJ, THREADS, WAKE, WINDOW, EV_MODE, EV_TAU,
     NOT_ONLY=1 (depth = exact NOT count), DEPTH_MIX=1 (train on depths 0..DTRAIN),
     TEST_DEPTHS=1,2 (nesting offsets beyond DTRAIN), SEQ_NORM=none|rms|tanh|clip, SAVE=path (use '{seed}' for per-seed files).
Run from repo root.
"""
from __future__ import annotations
import os, sys, time
import torch
sys.path.insert(0, os.getcwd())
from experiments.progenitor import language_world as LW
from trioron.organisms import Logophosis

N = int(os.environ.get("N", 20000))
DTRAIN = int(os.environ.get("DTRAIN", 1))
STAGE_EP = int(os.environ.get("EPOCHS", 20))
PATIENCE = int(os.environ.get("PATIENCE", 8))
MAX_LINKS = int(os.environ.get("MAX_LINKS", 4))
LINK0 = os.environ.get("LINK0", "phasecyte")
LINK0_KW = dict(wake=int(os.environ.get("WAKE", 20000)), window=int(os.environ.get("WINDOW", 1000)),
                ev_mode=os.environ.get("EV_MODE", "softmax"), ev_tau=float(os.environ.get("EV_TAU", 1.0)),
                bins=LW.CHORD_BINS, k=LW.CHORD_K, jitter=LW.CHORD_JITTER)
DEPTH_MIX = os.environ.get("DEPTH_MIX", "0") == "1"
SEQ_NORM = os.environ.get("SEQ_NORM", "none")
TEST_DEPTHS = [int(x) for x in os.environ.get("TEST_DEPTHS", "1").split(",")]
SD = LW.MAX_OBJ * LW.OBJ_DIM


def sets(stage, depth, seed, n, hold, mix=False):
    """(scene, ids, y); ``mix`` = uniform over depths 0..depth."""
    if mix and depth > 0:
        parts = [LW.build(n // (depth + 1), stage, d, seed + 17 * d, hold=hold)[:3]
                 for d in range(depth + 1)]
        S, E, Y = (torch.cat([p[i] for p in parts]) for i in range(3))
    else:
        S, E, Y, _ = LW.build(n, stage, depth, seed, hold=hold)
    return S, LW.ids_of(E), Y


def run(arm, seed):
    Str, Itr, ytr = sets(2, DTRAIN, 100 + seed, N, LW.HOLD_OUT, mix=DEPTH_MIX)
    Ste, Ite, yte = sets(2, DTRAIN, 200 + seed, 4000, LW.HOLD_OUT, mix=DEPTH_MIX)
    allpairs = [(a, n) for a in LW.COLORS + LW.SIZES for n in LW.SHAPES + ["thing"]]
    comp_hold = [p for p in allpairs if p not in LW.HOLD_OUT]
    Sc, Ic, yc = sets(2, DTRAIN, 300 + seed, 4000, comp_hold)
    deeper = {k: sets(2, DTRAIN + k, 400 + seed + k, 4000, LW.HOLD_OUT) for k in TEST_DEPTHS}
    t0 = time.time()
    org = Logophosis(SD, LW.CLASS_CAP, seed, mode=arm, link0=LINK0, max_links=MAX_LINKS,
                     seq_norm=SEQ_NORM, link_fan_in=os.environ.get("LINK_FAN_IN", "1") == "1", link0_kw=LINK0_KW).wake(Itr)
    print(f"  [link0 {LINK0}] woke in {time.time() - t0:.0f}s", flush=True)
    d = org.fit(Str, Itr, ytr, Ste, Ite, yte, epochs=STAGE_EP, patience=PATIENCE, target=0.99,
                log=lambda s: print(f"    [{arm} seed{seed}]{s}", flush=True))
    r = dict(indist=org.accuracy(Ste, Ite, yte), comp=org.accuracy(Sc, Ic, yc),
             links=d, params=org.n_params(), secs=time.time() - t0)
    for k, (S, I, y) in deeper.items():
        r[f"depth+{k}"] = org.accuracy(S, I, y)
        print(f"    [{arm} seed{seed}] depth+{k} ({DTRAIN + k} nest): {r[f'depth+{k}']:.3f}", flush=True)
    if os.environ.get("SAVE"):                       # '{seed}' in the path -> one file per seed
        path = os.environ["SAVE"].format(seed=seed)
        print(f"    saved {path} {org.save(path) / 1024:.0f} KiB", flush=True)
    return r


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("THREADS", 4)))
    seeds = ([int(x) for x in os.environ["SEED_LIST"].split(",")] if os.environ.get("SEED_LIST")
             else range(int(os.environ.get("SEEDS", 3))))
    arms = os.environ.get("ARMS", "tied,grown").split(",")
    print(f"# Logophosis PURE link0={LINK0} N={N} dtrain={DTRAIN} mix={DEPTH_MIX} not_only={LW.NOT_ONLY} "
          f"ep={STAGE_EP} L_MAX={LW.L_MAX} MAX_OBJ={LW.MAX_OBJ} seeds={list(seeds)} "
          f"test_depths={TEST_DEPTHS} seq_norm={SEQ_NORM}", flush=True)
    for arm in arms:
        rs = [run(arm, s) for s in seeds]
        f = lambda k: torch.tensor([r[k] for r in rs])
        sd = lambda t: f"{t.std():.3f}" if len(t) > 1 else "n/a"
        print(f"{arm:12s} indist {f('indist').mean():.3f}±{sd(f('indist'))}  "
              f"comp {f('comp').mean():.3f}±{sd(f('comp'))}  "
              + "".join(f"depth+{k} {f(f'depth+{k}').mean():.3f}±{sd(f(f'depth+{k}'))}  "
                        for k in TEST_DEPTHS)
              + f"links {[r['links'] for r in rs]}  params {[r['params'] for r in rs]}  "
              f"{f('secs').mean():.0f}s", flush=True)
