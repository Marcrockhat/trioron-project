"""Germline checks (docs/design/vee_development.md §6.4) — small experiments
before anything is raised; no organism is kept.

Her input per moment: language 64 | environment 33 | presence 3 | reserved 48.
Stream: the scripted world (world.py); in half the moments a text is present
(here: a random stage-2 question reading — the Wikipedia pool is not collected
yet), otherwise the language block is zero.  No labels.

  1 silent channels are harmless   same stream with and without the 48 reserved zeros
  2 a silent channel can wake      8 reserved channels start carrying a new hidden
                                   state ("kick") half-way; control = live from the start
  3 no block drowns another        how well each hidden world state, and the kind of
                                   text present, can be read from her classes

"Read from her classes" = every class is mapped to the majority value among
the moments it claimed while growing (measurement only), then scored on later
days she has not lived.  base = always guessing the commoner value.
Control = k-means with the same number of clusters on the same vectors.

Run from repo root:  python3 experiments/incubator/germline_checks.py
Check 1 is scored with the adjusted Rand index (1 = the same grouping of moments, 0 =
chance): class numbers of two organisms are not comparable, groupings are.

Env: SEEDS=3 DAYS=20 TEST_DAYS=5  GENESIS=first|rich (first = her first 48 moments; rich = see genesis_pool)
     P_TEXT=0.5  LANG_SCALE=1.0 (the language block is multiplied by this, §6.6)  AMBIENT=q|wiki
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(__file__))
import environment
from world import HIDDEN, PER_DAY, World
from trioron.pcll import PhasecyteLeaf
from trioron.senses.hidden_state import HiddenStateSense

E = lambda k, d: int(os.environ.get(k, d))
SEEDS, DAYS, TEST = E("SEEDS", 3), E("DAYS", 20), E("TEST_DAYS", 5)
LANG, RES, WINDOW = 64, 48, 48
P_TEXT = float(os.environ.get("P_TEXT", 0.5))                  # share of moments with a text present
LANG_SCALE = float(os.environ.get("LANG_SCALE", 1.0))          # loudness of the language block (plan §6.6)


def stream(seed, S2, topic, kick_from=None):
    """[T, 148] moments, hidden states [T, 5], text kind [T] (-1 none), kick [T]."""
    w, g, gk = World(seed), np.random.default_rng(1000 + seed), np.random.default_rng(2000 + seed)
    X, Hd, K, kick, k = [], [], [], [], False
    for t in range((DAYS + TEST) * PER_DAY):
        raw, hid = w.step()
        lang, kind = torch.zeros(LANG), -1
        if g.random() < P_TEXT:
            i = int(g.integers(len(S2))); lang, kind = LANG_SCALE * S2[i], int(topic[i])
        k = (not k) if gk.random() < 1 / 10 else k
        res = torch.zeros(RES)
        if kick_from is not None and t >= kick_from:               # the action-echo channels come alive
            res[:8] = (1.0 if k else -1.0) + 0.1 * torch.tensor(gk.standard_normal(8)).float()
        X.append(torch.cat([lang, torch.tensor(environment.vector(raw)).float(), torch.tensor([1.0, -1.0, -1.0]), res]))
        Hd.append([hid[h] for h in HIDDEN]); K.append(kind); kick.append(k)
    return torch.stack(X), torch.tensor(Hd).long(), torch.tensor(K), torch.tensor(kick).long()


def genesis_pool(seed, S2, width):
    """GENESIS=rich: what her senses form on.  A receptor exists only for a channel that
    carries signal at genesis and the set never changes afterwards (measured s066: 84 of
    148, and a channel that woke later was never read).  So genesis draws on a broad
    sample of her world — every state toggled freely — with SPONTANEOUS ACTIVITY (noise)
    on the reserved channels, which fall silent again once she lives."""
    w, g = World(500 + seed), np.random.default_rng(3000 + seed)
    rows = []
    for _ in range(10 * WINDOW):
        w.busy, w.full, w.elsewhere, w.window = (bool(g.random() < 0.5) for _ in range(4))
        w.t = int(g.integers(PER_DAY * 7))
        raw, _ = w.step()
        lang = LANG_SCALE * S2[int(g.integers(len(S2)))] if g.random() < P_TEXT else torch.zeros(LANG)
        pres = torch.tensor(g.choice([-1.0, 1.0], 3)).float()
        res = torch.tensor(g.standard_normal(RES)).float()
        rows.append(torch.cat([lang, torch.tensor(environment.vector(raw)).float(), pres, res])[:width])
    return torch.stack(rows)


def grow(X, seed, pool=None):
    n = DAYS * PER_DAY
    leaf = PhasecyteLeaf(lambda x: x, X[:WINDOW] if pool is None else pool, seed, 0, window=WINDOW, class_cap=None,
                         manifold=True, composer=True, verbose=False)
    counts = []
    for i in range(0, n, WINDOW):
        leaf.mixed.observe(X[i:i + WINDOW], labels=None)
        leaf.sub.end_task()
        counts.append(len(leaf.mixed.classes))
    return leaf, leaf.mixed.classify(X)[0], counts


def kmeans(X, k, seed, iters=30):
    g = torch.Generator().manual_seed(seed)
    C = X[torch.randperm(len(X), generator=g)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(X, C).argmin(1)
        for j in range(k):
            if (a == j).any(): C[j] = X[a == j].mean(0)
    return C


def adjusted_rand(a, b) -> float:
    """Agreement of two groupings of the same moments, blind to how the groups are numbered."""
    a, b = a.unique(return_inverse=True)[1], b.unique(return_inverse=True)[1]
    M = torch.zeros(int(a.max()) + 1, int(b.max()) + 1, dtype=torch.float64)
    M.index_put_((a, b), torch.ones(len(a), dtype=torch.float64), accumulate=True)
    c2 = lambda x: (x * (x - 1) / 2).sum()
    s, sa, sb, tot = c2(M), c2(M.sum(1)), c2(M.sum(0)), len(a) * (len(a) - 1) / 2
    exp = sa * sb / tot
    return float((s - exp) / ((sa + sb) / 2 - exp))


def readable(cls, y, n):
    """accuracy on moments after n when each class votes its majority value before n; and the base rate."""
    tr, te, ytr, yte = cls[:n], cls[n:], y[:n], y[n:]
    vals = y.unique()
    default = vals[torch.stack([(ytr == v).sum() for v in vals]).argmax()]
    pred = torch.full_like(yte, int(default))
    for c in tr.unique():
        m = ytr[tr == c]
        pred[te == c] = vals[torch.stack([(m == v).sum() for v in vals]).argmax()]
    return (pred == yte).float().mean().item(), (yte == default).float().mean().item()


if __name__ == "__main__":
    from stage2_memory import load
    H2, group2, block2, _, _ = load("both")
    if os.environ.get("AMBIENT", "q") == "wiki":                 # §6.2 sense (sentences + questions), Wikipedia ambient text
        import birth_check
        Hw = birth_check.wiki_states()
        S2 = HiddenStateSense(torch.cat([Hw, H2]), LANG, dup=False)(Hw)
    else:
        S2 = HiddenStateSense(H2, LANG, dup=False)(H2)
    topic = (kmeans(S2, 6, 0)[None] - S2[:, None]).norm(dim=2).argmin(1)     # six kinds of text, for check 3
    n = DAYS * PER_DAY
    print(f"# germline checks  seeds={SEEDS}  {DAYS} days lived + {TEST} days unseen, {PER_DAY} moments a day", flush=True)
    R: dict[str, list] = {}
    for seed in range(SEEDS):
        X, Hd, kind, kick = stream(seed, S2, topic)
        rich = os.environ.get("GENESIS", "first") == "rich"
        pool = genesis_pool(seed, S2, X.shape[1]) if rich else None
        leaf, cls, counts = grow(X, seed, pool)
        _, cls0, _ = grow(X[:, :-RES], seed, pool[:, :-RES] if rich else None)   # the same life without the reserved channels
        R.setdefault("0 receptors (of 148)", []).append(leaf.sub.scheduler._plan.receptor_ids.numel())
        same = adjusted_rand(cls, cls0)
        _, cls_b, _ = grow(X, seed + 100, genesis_pool(seed + 100, S2, X.shape[1]) if rich else None)   # yardstick: another genesis draw
        R.setdefault("1 agreement (adjusted Rand) with / without silent channels", []).append(same)
        R.setdefault("1 agreement between two genesis draws, both with silent channels", []).append(adjusted_rand(cls, cls_b))
        R.setdefault("1 classes with silent", []).append(len(leaf.mixed.classes))
        R.setdefault("1 classes without", []).append(int(cls0.max()) + 1)
        k = len(leaf.mixed.classes)
        km = torch.cdist(X, kmeans(X[:n], k, seed)).argmin(1)
        for j, hname in enumerate(HIDDEN):
            a, b = readable(cls, Hd[:, j], n); c, _ = readable(km, Hd[:, j], n)
            R.setdefault(f"3 {hname}: her classes / k-means / base", []).append([a, c, b])
        present = kind >= 0
        a, b = readable(cls, present.long(), n); c, _ = readable(km, present.long(), n)
        R.setdefault("3 text present: her classes / k-means / base", []).append([a, c, b])
        a, b = readable(cls[present], kind[present], int(present[:n].sum())); c, _ = readable(km[present], kind[present], int(present[:n].sum()))
        R.setdefault("3 kind of text (6): her classes / k-means / base", []).append([a, c, b])
        R.setdefault("3 class count at day 13 / day 20", []).append([counts[len(counts) * 2 // 3], counts[-1]])
        # check 2: the echo channels wake half-way; control has them from the start
        for name, start in (("woke at day 10", n // 2), ("live from the start", 0)):
            Xk, Hk, _, kk = stream(seed, S2, topic, kick_from=start)
            lf, ck, _ = grow(Xk, seed, pool)
            a, b = readable(ck, kk, n)
            olds = np.mean([readable(ck, Hk[:, j], n)[0] for j in range(len(HIDDEN))])
            R.setdefault(f"2 kick readable, {name}: acc / base / old states mean / classes", []).append([a, b, olds, len(lf.mixed.classes)])
        print(f"  seed {seed} done: classes {k}, agreement with/without {same:.2f}", flush=True)
    print()
    for name, v in R.items():
        m = np.array(v, float).mean(0)
        print(f"{name:66} " + ("  ".join(f"{x:.2f}" for x in np.atleast_1d(m))))
