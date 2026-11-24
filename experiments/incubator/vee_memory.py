"""Vee's fact memory: a day table and a phasecyte that is fed at night (s066).

Day    a fact is heard ONCE.  Its read state (reflex.read: h_last + h_mean,
       through a 64-number sense) goes into the day table; recall from the
       table is a cosine to that one row.
Night  ``sleep()`` replays every fact still in the day table to ONE phasecyte
       leaf as a stream: the original wording plus NPER copies with wording
       noise borrowed from other questions (pilot_night.py: "noise" source),
       or supplied rewordings.  Earlier nights are never replayed.  A fact the
       leaf then recalls from its original wording leaves the day table; one
       it does not stays and is replayed the next night.
Recall the day table first, then the leaf; both answer only above the
       threshold that lets FALSE_RATE of known-untaught questions through.

The phasecyte is never batch-trained and nothing is frozen (Rocky s066).
Nothing of the leaf is saved: ``wake()`` rebuilds it by replaying the
childhood (the 40 pilot facts, four nights) and then the user's nights in
order — deterministic, but it grows with life; a state file is future work.

Needs the local stage-2 and pilot collections in outputs/incubator/.

Self-test (console regime: a few facts per night after the childhood):
    python3 experiments/incubator/vee_memory.py
Env: NIGHTS=20 PER=1 (facts per night) SEEDS=3 HOLD=10
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(__file__))
from trioron.pcll import PhasecyteLeaf
from trioron.senses.hidden_state import HiddenStateSense

OUT = "outputs/incubator"
USER = f"{OUT}/vee_facts"                       # .jsonl (q, answer, born) + .npz (H)


class FactMemory:
    def __init__(self, H_base, group_base, H_neg, *, seed=0, kp=64, window=50, nper=40, false_rate=0.05):
        self.sense = HiddenStateSense(H_base, kp, dup=False)
        _, g = group_base.unique(return_inverse=True)
        gm = torch.zeros(int(g.max()) + 1, H_base.shape[1]).index_add_(0, g, H_base) / torch.bincount(g)[:, None]
        self.resid = H_base - gm[g]                                 # wording residuals of other questions
        self.negS = self.sense(H_neg)
        self.seed, self.window, self.nper, self.false_rate = seed, window, nper, false_rate
        self.g = torch.Generator().manual_seed(seed)
        self.facts: list[dict] = []                                 # q, answer, born (night count when heard), kept (in the leaf)
        self.H: list[torch.Tensor] = []
        self.S = torch.zeros(0, kp)
        self.leaf, self.names, self.tau_leaf, self.night = None, None, float("inf"), 0

    # ---- day ---------------------------------------------------------------
    def teach(self, q: str, answer: str, h: torch.Tensor, **extra) -> int:
        self.facts.append({"q": q, "answer": answer, "born": self.night, "kept": False, **extra})
        self.H.append(h.float())
        self.S = torch.cat([self.S, self.sense(h.float()[None])])
        return len(self.facts) - 1

    def day(self) -> list[int]:
        return [i for i, f in enumerate(self.facts) if not f["kept"]]

    # ---- night -------------------------------------------------------------
    def sleep(self, material: dict[int, torch.Tensor] | None = None) -> dict:
        todo, t0 = self.day(), time.time()
        if not todo:
            return {"fed": 0, "kept": 0, "classes": len(self.leaf.mixed.classes) if self.leaf else 0, "s": 0.0}
        X, y = [], []
        for i in todo:
            rows = (material or {}).get(i)
            if rows is None:
                rows = self.H[i] + self.resid[torch.randint(len(self.resid), (self.nper,), generator=self.g)]
            X.append(self.sense(torch.cat([self.H[i][None], rows.float()])))
            y += [i] * (len(rows) + 1)
        X, y = torch.cat(X), torch.tensor(y)
        p = torch.randperm(len(X), generator=self.g)
        X, y = X[p], y[p]
        if self.leaf is None:
            self.leaf = PhasecyteLeaf(lambda x: x, X[:self.window], self.seed, 0, window=min(self.window, len(X)),
                                      class_cap=None, manifold=True, composer=True, verbose=False)
        n_win = max(1, round(len(X) / self.window))                 # ~window samples per boundary, no tiny tail
        for xb, yb in zip(X.tensor_split(n_win), y.tensor_split(n_win)):
            self.leaf.mixed.observe(xb, labels=[f"g{int(f)}" for f in yb])
            self.leaf.sub.end_task()
        m = self.leaf.mixed
        self.names = self.leaf.member_names()                       # each class named from its CURRENT members
        self.tau_leaf = float(torch.quantile(m.classify(self.negS)[1], 1 - self.false_rate))
        got, _ = self._leaf(self.S[todo])
        for j, i in enumerate(todo):
            self.facts[i]["kept"] = bool(got[j] == i)
        self.night += 1
        return {"fed": len(todo), "kept": int(sum(self.facts[i]["kept"] for i in todo)),
                "classes": len(m.classes), "s": time.time() - t0}

    # ---- recall ------------------------------------------------------------
    def _leaf(self, S):
        best, marg = self.leaf.mixed.classify(S)
        f = torch.where(best >= 0, self.names[best.clamp(min=0)], -1)
        return torch.where(marg > self.tau_leaf, f, -1), marg

    def recall_many(self, H) -> tuple[torch.Tensor, torch.Tensor]:
        """fact index per row (-1 = not remembered) and where (0 none, 1 day table, 2 leaf)."""
        S = self.sense(H.float())
        fact, where = torch.full((len(S),), -1), torch.zeros(len(S), dtype=torch.long)
        if self.leaf is not None:
            f, _ = self._leaf(S)
            fact, where = f, torch.where(f >= 0, 2, 0)
        day = self.day()
        if day:                                                     # the day table speaks first
            D = F.normalize(self.S[day], dim=1)
            tau = torch.quantile((F.normalize(self.negS, dim=1) @ D.T).max(1).values, 1 - self.false_rate)
            c, j = (F.normalize(S, dim=1) @ D.T).max(1)
            hit = c > tau
            fact = torch.where(hit, torch.tensor(day)[j], fact)
            where = torch.where(hit, 1, where)
        return fact, where

    def recall(self, h) -> dict | None:
        f, w = self.recall_many(torch.as_tensor(h)[None])
        return None if f[0] < 0 else {**self.facts[int(f[0])], "where": ("", "today's", "long-term")[int(w[0])]}


# ---- data ------------------------------------------------------------------
def stage2():
    from stage2_memory import load
    return load("both")


def pilot():
    """Pilot rows: H, qid, wave (-1 untaught), is_original, and qid -> (question, answer)."""
    rows = [json.loads(l) for l in open(f"{OUT}/pilot_keys.jsonl")]
    z = np.load(f"{OUT}/pilot_keys.npz")
    H = torch.cat([torch.tensor(z[p][:len(rows)].astype(np.float32)) for p in ("h_last", "h_mean")], 1)
    qa = {}
    for w in range(4):
        for l in open(f"{OUT}/pilot_in_wave{w}.tsv"):
            qid, q, a = l.rstrip("\n").split("\t"); qa[qid] = (q, a)
    return H, [r["qid"] for r in rows], torch.tensor([r["wave"] for r in rows]), torch.tensor([r["i"] < 0 for r in rows]), qa


def childhood(mem: FactMemory, held: torch.Tensor | None = None) -> dict[str, int]:
    """Four nights of 10 pilot facts, fed with the helper's rewordings."""
    H, qid, wave, orig, qa = pilot()
    ids: dict[str, int] = {}
    for w in range(4):
        material = {}
        for q in dict.fromkeys(q for q, v in zip(qid, wave) if v == w):
            m = torch.tensor([x == q for x in qid])
            ids[q] = mem.teach(*qa[q], H[(m & orig).nonzero()[0, 0]], pilot=True)
            material[ids[q]] = H[m & ~orig & (~held if held is not None else True)]
        mem.sleep(material)
    return ids


def wake(seed: int = 0) -> FactMemory:
    """The console's memory: childhood, then the user's own nights replayed in order."""
    H2, group2, block2, variant2, texts2 = stage2()
    _, _, _, _, qa = pilot()
    pq = {q for q, _ in qa.values()}
    other = ~torch.isin(block2, block2[torch.tensor([t in pq for t in texts2])].unique())
    neg = H2[other & (variant2 == 0)]
    mem = FactMemory(H2[other], group2[other], neg[torch.randperm(len(neg), generator=torch.Generator().manual_seed(seed))[:2000]], seed=seed)
    childhood(mem)
    if os.path.exists(USER + ".jsonl"):
        rows = [json.loads(l) for l in open(USER + ".jsonl")]
        Hu = torch.tensor(np.load(USER + ".npz")["H"][:len(rows)].astype(np.float32))
        last = max([r["born"] for r in rows] + [mem.night - 1])
        for n in range(mem.night, last + 2):                        # the last group may not have slept yet
            for r, h in zip(rows, Hu):
                if r["born"] == n: mem.teach(r["q"], r["answer"], h)
            if n <= last and any(r.get("slept", True) for r in rows if r["born"] == n):
                mem.sleep()
    return mem


def save_user(mem: FactMemory) -> None:
    mine = [(f, h) for f, h in zip(mem.facts, mem.H) if not f.get("pilot")]
    if not mine: return
    np.savez(USER + ".tmp.npz", H=np.stack([h.numpy() for _, h in mine]).astype(np.float16))
    os.replace(USER + ".tmp.npz", USER + ".npz")
    with open(USER + ".jsonl", "w") as f:
        for r, _ in mine:
            f.write(json.dumps({"q": r["q"], "answer": r["answer"], "born": r["born"],
                                "slept": mem.night > r["born"]}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    E = lambda k, d: type(d)(os.environ.get(k, d))
    NIGHTS, PER, SEEDS, HOLD = E("NIGHTS", 20), E("PER", 1), E("SEEDS", 3), E("HOLD", 10)
    H2, group2, block2, variant2, texts2 = stage2()
    Hp, qid, wave, orig, qa = pilot()
    pq = {q for q, _ in qa.values()}
    other = ~torch.isin(block2, block2[torch.tensor([t in pq for t in texts2])].unique())
    print(f"# vee memory self-test  childhood 40 facts, then {NIGHTS} nights x {PER} fact(s) heard once; "
          f"later facts tested on their 2 other wordings; seeds={SEEDS}", flush=True)
    out = []
    for seed in range(SEEDS):
        g = torch.Generator().manual_seed(seed)
        blocks = block2[other].unique()
        A = other & torch.isin(block2, blocks[torch.randperm(len(blocks), generator=g)[:len(blocks) // 2]])
        B = other & ~A                                              # A: sense, noise, negatives.  B: later facts
        negA = H2[A & (variant2 == 0)]
        mem = FactMemory(H2[A], group2[A], negA[torch.randperm(len(negA), generator=g)[:2000]], seed=seed)
        held = torch.zeros(len(Hp), dtype=torch.bool)
        for q in qa:
            idx = (torch.tensor([x == q for x in qid]) & ~orig).nonzero().squeeze(1)
            held[idx[torch.randperm(len(idx), generator=g)[:HOLD]]] = True
        ids = childhood(mem, held)
        pf = torch.tensor([ids.get(q, -1) for q in qid])

        def child(label):
            f, _ = mem.recall_many(Hp)
            w0 = (f == pf)[held & (wave == 0)].float().mean().item()
            al = (f == pf)[held].float().mean().item()
            fa = (f >= 0)[wave < 0].float().mean().item()
            print(f"  seed {seed} {label:22} childhood recall w0 {w0:.2f} all {al:.2f}  false recall {fa:.2f}  "
                  f"classes {len(mem.leaf.mixed.classes)}", flush=True)
            return w0, al, fa
        c0 = child("after childhood")
        gs = group2[B].unique()
        gs = gs[torch.randperm(len(gs), generator=g)]
        new, rest = gs[:NIGHTS * PER], gs[NIGHTS * PER:NIGHTS * PER + 300]
        gid, secs = {}, []
        for n in range(NIGHTS):
            for q in new[n * PER:(n + 1) * PER]:
                gid[int(q)] = mem.teach(f"q{int(q)}", "", H2[(group2 == q) & (variant2 == 0)][0])
            secs.append(mem.sleep()["s"])
        c1 = child(f"after {NIGHTS} more nights")
        test = torch.isin(group2, new) & (variant2 > 0)
        want = torch.tensor([gid[int(q)] for q in group2[test]])
        f, where = mem.recall_many(H2[test])
        kept = np.mean([mem.facts[i]["kept"] for i in gid.values()])
        fa = (mem.recall_many(H2[torch.isin(group2, rest)])[0] >= 0).float().mean().item()
        S = F.normalize(mem.sense(H2[test]), dim=1); T = F.normalize(mem.S, dim=1)   # reference: every fact in a table
        c, j = (S @ T.T).max(1)
        tau = torch.quantile((F.normalize(mem.negS, dim=1) @ T.T).max(1).values, 0.95)
        tab = ((j == want) & (c > tau)).float().mean().item()
        tab_fa = ((F.normalize(mem.sense(H2[torch.isin(group2, rest)]), dim=1) @ T.T).max(1).values > tau).float().mean().item()
        new_hit = (f == want).float().mean().item()
        print(f"  seed {seed} later facts: recalled {new_hit:.2f} (from leaf {((f == want) & (where == 2)).float().mean():.2f}, "
              f"from day table {((f == want) & (where == 1)).float().mean():.2f})  moved into leaf {kept:.2f}  "
              f"false recall {fa:.2f}  all-in-table reference {tab:.2f} (false {tab_fa:.2f})  sleep {np.mean(secs):.1f}s", flush=True)
        out.append([*c0, *c1, new_hit, kept, fa, tab, tab_fa])
    r = np.array(out).mean(0)
    print(f"\n# mean of {SEEDS}: childhood w0 {r[0]:.2f} -> {r[3]:.2f}, all {r[1]:.2f} -> {r[4]:.2f}, "
          f"false {r[2]:.2f} -> {r[5]:.2f};  later facts recalled {r[6]:.2f}, moved into leaf {r[7]:.2f}, "
          f"false recall {r[8]:.2f}, all-in-table reference {r[9]:.2f} (false {r[10]:.2f})")
