"""s060 — language comprehension on the grounded mini-world (Stage 1/2 of the arc).

Input  : concat[scene (44-d), sentence evidence (L_MAX x CLASS_CAP one-hot, flattened 768-d)]
Label  : truth of the sentence in the scene (balanced 2-way).
Arms   : tied (one trioron link re-applied, R grown on stall)  |  grown_joint chain  |
         mlp3 (ReLU 3x48)  |  bow (linear on bag-of-words + scene: LEAK CHECK)  |
         gru (token-by-token GRU with scene concatenated at every step: reference)
Modes  :
  JOINT      : train on all 3 vocab stages at once, depth<=DTRAIN. Test: in-dist, COMPOSITIONAL
               (held-out adj-noun pairs), DEPTH (depth DTRAIN+1, incl. "not not").
  CONTINUAL  : vocab stage 0 -> 1 -> 2 sequentially (same model), test every stage's set after
               each stage -> forgetting matrix.
Run from repo root:  python3 experiments/progenitor/language_chain.py
Env: SEEDS, ARMS, MODE=joint|continual, N=20000, EPOCHS, PATIENCE, DTRAIN=1
"""
from __future__ import annotations
import os, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.getcwd())
import experiments.progenitor.logic_chain as LC
from experiments.progenitor import language_world as LW

N = int(os.environ.get("N", 20000))
DTRAIN = int(os.environ.get("DTRAIN", 1))
STAGE_EP = int(os.environ.get("EPOCHS", 20))
LC.PATIENCE = int(os.environ.get("PATIENCE", 8))
LC.MAX_LINKS = int(os.environ.get("MAX_LINKS", 4))
SD = LW.MAX_OBJ * LW.OBJ_DIM


def flat(S, E):
    return torch.cat([S, E.flatten(1)], 1)


class BoW:
    def __init__(self, Ein, C, seed):
        torch.manual_seed(seed)
        self.lin = nn.Linear(SD + LW.CLASS_CAP, C)
    def grow(self): pass
    def forward(self, x, **_):
        S, E = x[:, :SD], x[:, SD:].view(len(x), LW.L_MAX, LW.CLASS_CAP)
        return self.lin(torch.cat([S, E.sum(1)], 1))
    def params(self): return list(self.lin.parameters())


class GRUTok:
    def __init__(self, Ein, C, seed):
        torch.manual_seed(seed)
        self.cell = nn.GRUCell(SD + LW.CLASS_CAP, LC.H); self.head = nn.Linear(LC.H, C)
    def grow(self): pass
    def forward(self, x, **_):
        S, E = x[:, :SD], x[:, SD:].view(len(x), LW.L_MAX, LW.CLASS_CAP)
        h = torch.zeros(len(x), LC.H)
        for t in range(LW.L_MAX):
            h = self.cell(torch.cat([S, E[:, t]], 1), h)
        return self.head(h)
    def params(self): return list(self.cell.parameters()) + list(self.head.parameters())


def make(arm, Ein, seed):
    if arm == "grown_joint": return LC.Chain(Ein, 2, seed, joint=True)
    return {"tied": LC.Tied, "mlp3": LC.MLP3, "bow": BoW, "gru": GRUTok}[arm](Ein, 2, seed)


GROWS = {"tied": LC.MAX_LINKS - 1, "grown_joint": LC.MAX_LINKS - 1}


def fit(model, arm, X, y, Xt, yt, seed):
    grows = GROWS.get(arm, 0)
    epochs = STAGE_EP if grows else STAGE_EP * LC.MAX_LINKS
    depth = 1
    while True:
        stalled, acc, _ = LC.train_stage(model, X, y, Xt, yt, epochs, seed * 7 + depth)
        print(f"    [{arm} seed{seed}] stage {depth}: acc {acc:.3f} stalled={stalled}", flush=True)
        if not stalled or grows == 0: break
        model.grow(); grows -= 1; depth += 1
    return depth


def sets(stage, depth, seed, n, hold):
    S, E, Y, raw = LW.build(n, stage, depth, seed, hold=hold)
    return flat(S, E), Y


def run_joint(arm, seed):
    Xtr, ytr = sets(2, DTRAIN, 100 + seed, N, LW.HOLD_OUT)
    Xte, yte = sets(2, DTRAIN, 200 + seed, 4000, LW.HOLD_OUT)
    # compositional: ONLY held-out pairs -> build with hold = all pairs except HOLD_OUT
    allpairs = [(a, n) for a in LW.COLORS + LW.SIZES for n in LW.SHAPES + ["thing"]]
    comp_hold = [p for p in allpairs if p not in LW.HOLD_OUT]
    Xc, yc = sets(2, DTRAIN, 300 + seed, 4000, comp_hold)
    Xd, yd = sets(2, DTRAIN + 1, 400 + seed, 4000, LW.HOLD_OUT)
    m = make(arm, Xtr.shape[1], seed)
    t0 = time.time()
    d = fit(m, arm, Xtr, ytr, Xte, yte, seed)
    return dict(indist=LC.accuracy(m, Xte, yte), comp=LC.accuracy(m, Xc, yc),
                depth=LC.accuracy(m, Xd, yd), links=d, secs=time.time() - t0)


def run_continual(arm, seed):
    tests = [sets(st, DTRAIN, 200 + 10 * st + seed, 3000, LW.HOLD_OUT) for st in range(3)]
    m = None
    mat = []
    for st in range(3):
        Xtr, ytr = sets(st, DTRAIN, 100 + 10 * st + seed, N // 2, LW.HOLD_OUT)
        if m is None: m = make(arm, Xtr.shape[1], seed)
        fit(m, arm, Xtr, ytr, *tests[st], seed + st)
        mat.append([LC.accuracy(m, *t) for t in tests])
    return mat


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("THREADS", 4)))
    seeds = ([int(x) for x in os.environ["SEED_LIST"].split(",")] if os.environ.get("SEED_LIST")
             else range(int(os.environ.get("SEEDS", 3))))
    arms = os.environ.get("ARMS", "bow,mlp3,gru,tied,grown_joint").split(",")
    mode = os.environ.get("MODE", "joint")
    print(f"# language {mode}  N={N} dtrain={DTRAIN} ep={STAGE_EP} seeds={list(seeds)}")
    for arm in arms:
        if mode == "joint":
            rs = [run_joint(arm, s) for s in seeds]
            f = lambda k: torch.tensor([r[k] for r in rs])
            print(f"{arm:12} indist {f('indist').mean():.3f}±{f('indist').std():.3f}  "
                  f"comp {f('comp').mean():.3f}±{f('comp').std():.3f}  "
                  f"depth+1 {f('depth').mean():.3f}±{f('depth').std():.3f}  "
                  f"links {[r['links'] for r in rs]}  {sum(r['secs'] for r in rs):.0f}s", flush=True)
        else:
            ms = torch.tensor([run_continual(arm, s) for s in seeds]).mean(0)
            print(f"{arm:12} after-stage x test-stage acc matrix (rows=after stage 0,1,2):")
            for i, row in enumerate(ms.tolist()):
                print(f"   {i}: " + "  ".join(f"{v:.3f}" for v in row), flush=True)
