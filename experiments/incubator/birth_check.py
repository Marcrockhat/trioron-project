"""Germline check 1b (docs/design/vee_development.md §6.6, last bullet): with the
language block turned down so that she can read her world, can she still tell
sentences apart after birth?

ONE organism per (seed, language scale), with her 148-number input for life:
  genesis   broad sample of her world + spontaneous activity (germline_checks.genesis_pool)
  embryo    DAYS days of the scripted world, text present in P_TEXT of the moments, no labels
  birth     the 40 pilot facts in four waves of 10 (helper rewordings, told, ~50 rows a
            boundary, old waves never shown again); every wording is heard at a moment of
            the continuing world, presence = the matron
Read:
  world     busy / memory full / night / window from her classes on unseen silent days,
            after the embryo and again after the 40 facts (k-means = control)
  facts     held-out wordings, heard at moments of ANOTHER world run: top-1 (class named
            from its current members) and hit at the threshold that lets 5 % of untaught
            questions through; wave 0 reported separately (earliest memories)
  controls  running class mean of the same 148 numbers, and of the language block alone
Scale "organs" = TWO leaves in one body, same 148-number input, language x1: an ear that
reads the language block and a body sense that reads everything else; both live the whole
life (the ear hears silence as zeros); the world is read from the body sense, facts from the ear.

Run from repo root:  python3 experiments/incubator/birth_check.py
Env: SEEDS=3 DAYS=20 TEST_DAYS=5 SCALES=1,0.5,0.25  SENSE=q|both (both = refit on Wikipedia
     sentences + questions, §6.2)  AMBIENT=q|wiki  HOLD=10
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(__file__))
import environment
import germline_checks as G
import vee_memory as vm
from world import HIDDEN, PER_DAY, World
from trioron.pcll import PhasecyteLeaf
from trioron.senses.hidden_state import HiddenStateSense

OUT = "outputs/incubator"
SCALES = [s if s == "organs" else float(s) for s in os.environ.get("SCALES", "1,0.5,0.25,organs").split(",")]
SENSE, AMBIENT, HOLD = os.environ.get("SENSE", "q"), os.environ.get("AMBIENT", "q"), int(os.environ.get("HOLD", 10))
LANG, RES, W = G.LANG, G.RES, G.WINDOW
MATRON, NOBODY = torch.tensor([-1.0, 1.0, -1.0]), torch.tensor([1.0, -1.0, -1.0])


def wiki_states() -> torch.Tensor:
    z = np.load(f"{OUT}/wiki_sent.npz")
    return torch.cat([torch.tensor(z[p].astype(np.float32)) for p in ("h_last", "h_mean")], 1)


def heard(S: torch.Tensor, seed: int, presence=MATRON) -> torch.Tensor:
    """Each language row heard at one moment of a running world: [N, 148]."""
    w = World(seed)
    env = torch.tensor([environment.vector(w.step()[0]) for _ in range(len(S))]).float()
    return torch.cat([S, env, presence.expand(len(S), 3), torch.zeros(len(S), RES)], 1)


def scaled(X: torch.Tensor, s: float) -> torch.Tensor:
    X = X.clone(); X[:, :LANG] *= s
    return X


class Organs:
    """Two phasecyte leaves over disjoint parts of the same input."""
    def __init__(self, pool, seed):
        mk = lambda P, k: PhasecyteLeaf(lambda x: x, P, seed, k, window=W, class_cap=None, manifold=True, composer=True, verbose=False)
        self.ear, self.body = mk(pool[:, :LANG], 0), mk(pool[:, LANG:], 1)
        self.mixed = self.body.mixed                      # world_reading reads the body sense

    def observe(self, X, labels=None):
        self.ear.mixed.observe(X[:, :LANG], labels=labels); self.ear.sub.end_task()
        self.body.mixed.observe(X[:, LANG:], labels=None); self.body.sub.end_task()


def world_reading(leaf, X, Hd, n, seed):
    """{state: (her classes, k-means)} on the unseen days, classes voted on the lived days."""
    if isinstance(leaf, Organs): X = X[:, LANG:]            # control on the same numbers she reads
    cls = leaf.mixed.classify(X)[0]
    km = torch.cdist(X, G.kmeans(X[:n], len(leaf.mixed.classes), seed)).argmin(1)
    return {h: (G.readable(cls, Hd[:, j], n)[0], G.readable(km, Hd[:, j], n)[0]) for j, h in enumerate(HIDDEN)}


if __name__ == "__main__":
    H2, group2, block2, variant2, texts2 = vm.stage2()
    Hp, qid, wave, orig, qa = vm.pilot()
    pq = {q for q, _ in qa.values()}
    other = ~torch.isin(block2, block2[torch.tensor([t in pq for t in texts2])].unique())
    fit = H2[other] if SENSE == "q" else torch.cat([wiki_states(), H2[other]])
    sense = HiddenStateSense(fit, LANG, dup=False)
    amb = sense(H2[other]) if AMBIENT == "q" else sense(wiki_states())
    topic = (G.kmeans(amb, 6, 0)[None] - amb[:, None]).norm(dim=2).argmin(1)
    Sp = sense(Hp)
    ids = {q: i for i, q in enumerate(dict.fromkeys(q for q, v in sorted(zip(qid, wave.tolist()), key=lambda t: t[1] if t[1] >= 0 else 99) if v >= 0))}
    fact = torch.tensor([ids.get(q, -1) if v >= 0 else -1 for q, v in zip(qid, wave.tolist())])
    n = G.DAYS * PER_DAY
    print(f"# birth check  seeds={G.SEEDS}  embryo {G.DAYS} days, text in {G.P_TEXT} of moments; then 40 facts in 4 waves; "
          f"sense={SENSE} ambient={AMBIENT} ({len(amb)} texts)", flush=True)
    R: dict[float, list] = {}
    for seed in range(G.SEEDS):
        g = torch.Generator().manual_seed(seed)
        held = torch.zeros(len(Hp), dtype=torch.bool)
        for f in range(40):
            idx = ((fact == f) & ~orig).nonzero().squeeze(1)
            held[idx[torch.randperm(len(idx), generator=g)[:HOLD]]] = True
        order = []
        for w in range(4):
            idx = ((wave == w) & ~held).nonzero().squeeze(1)
            order.append(idx[torch.randperm(len(idx), generator=g)])
        X0, Hd, _, _ = G.stream(seed, amb, topic)
        pool0 = G.genesis_pool(seed, amb, X0.shape[1])
        lesson0 = {w: heard(Sp[o], 700 + 10 * seed + w) for w, o in enumerate(order)}
        test0 = heard(Sp, 900 + seed)                                  # every pilot row at a moment of another world run
        for s in SCALES:
            org = s == "organs"
            sc = 1.0 if org else s
            X, pool = scaled(X0, sc), scaled(pool0, sc)
            if org:
                leaf = Organs(pool, seed); feed = leaf.observe
                fl, cut = leaf.ear, (lambda A: A[:, :LANG])
            else:
                leaf = PhasecyteLeaf(lambda x: x, pool, seed, 0, window=W, class_cap=None, manifold=True, composer=True, verbose=False)
                def feed(xb, labels=None, leaf=leaf):
                    leaf.mixed.observe(xb, labels=labels); leaf.sub.end_task()
                fl, cut = leaf, (lambda A: A)
            for i in range(0, n, W):
                feed(X[i:i + W])
            c_embryo = len(fl.mixed.classes)
            before = world_reading(leaf, X, Hd, n, seed)
            T = scaled(test0, sc)
            mu = torch.zeros(40, T.shape[1])
            per_wave = []
            for w, o in enumerate(order):
                L = scaled(lesson0[w], sc)
                nw = max(1, round(len(L) / 50))
                for xb, fb in zip(L.tensor_split(nw), fact[o].tensor_split(nw)):
                    feed(xb, labels=[f"g{int(f)}" for f in fb])
                mu.index_add_(0, fact[o], L)
            names = fl.member_names()
            best, marg = fl.mixed.classify(cut(T))
            pred = torch.where(best >= 0, names[best.clamp(min=0)], -1)
            tau = torch.quantile(marg[fact < 0], 0.95)
            right = pred == fact
            hit = right & (marg > tau)
            false = ((pred >= 0) & (marg > tau))[fact < 0].float().mean().item()
            wv = lambda m, v: m[held & (wave == v)].float().mean().item()
            cm = lambda A, B: ((F.normalize(A, dim=1) @ F.normalize(B, dim=1).T).argmax(1) == fact)[held].float().mean().item()
            after = world_reading(leaf, X, Hd, n, seed)
            row = [c_embryo, len(fl.mixed.classes), wv(right, 0), right[held].float().mean().item(), wv(hit, 0), hit[held].float().mean().item(),
                   false, cm(T, mu), cm(T[:, :LANG], mu[:, :LANG])]
            row += [before[h][k] for h in ("busy", "full", "night", "window") for k in (0, 1)]
            row += [after[h][0] for h in ("busy", "full", "night", "window")]
            R.setdefault(s, []).append(row)
            print(f"  seed {seed} language x{s}: classes {row[0]} -> {row[1]}  facts top1 w0 {row[2]:.2f} all {row[3]:.2f}  hit w0 {row[4]:.2f} all {row[5]:.2f}  "
                  f"class mean {row[7]:.2f} (language only {row[8]:.2f})  busy {row[9]:.2f}->{row[17]:.2f}  full {row[11]:.2f}->{row[18]:.2f}", flush=True)
    print(f"\n{'language':9} | classes embryo -> after facts | facts top-1: w0  all | hit: w0  all | untaught answered | class mean 148 / language | "
          f"world after embryo, hers/k-means: busy  full  night  window | world after facts, hers: busy full night window")
    for s, rows in R.items():
        r = np.array(rows, float).mean(0)
        print(f"{('x' + str(s)) if s != 'organs' else s:<9} | {r[0]:>14.0f} -> {r[1]:<11.0f} | {r[2]:>15.2f} {r[3]:.2f} | {r[4]:>7.2f} {r[5]:.2f} | {r[6]:>17.2f} | {r[7]:>14.2f} / {r[8]:.2f}     | "
              + "  ".join(f"{r[9 + 2 * k]:.2f}/{r[10 + 2 * k]:.2f}" for k in range(4)) + " | " + "  ".join(f"{r[17 + k]:.2f}" for k in range(4)))
