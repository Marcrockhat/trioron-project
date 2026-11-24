"""Teach Vee about HERSELF first, thoroughly (Rocky s066).

Six facts about herself (name, what she is, who made her, her matron, how she
remembers, when she was born), each in 24 helper-written wordings, are the
FIRST thing the phasecyte ever hears; the 40 pilot facts (the old childhood)
come afterwards.  Nothing in the phasecyte is changed.

  thin      1 night, 6 wordings per fact
  thorough  NIGHTS nights x PER new wordings per fact (thorough_test.py: a
            thoroughly taught fact keeps clean classes and is not overwritten)

Read after the self nights and again after the 40 later facts (PHASECYTE ALONE):
  self      6 untaught wordings per fact answered with the right self fact
  near      questions about someone else ("What is my name?") answered with a
            self fact — generalisation, reported, not counted as an error
  other     unrelated everyday questions answered with any fact — an error
  later w0  held-out wordings of the first 10 later facts (do they still fit in)

In : outputs/incubator/self_wordings.jsonl (local, helper-written)
Out: outputs/incubator/self_keys.pt (read states, cached)
Run from repo root:  python3 experiments/incubator/self_test.py
Env: SEEDS=3 NIGHTS=6 PER=4
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import torch
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(__file__))
import vee_memory as vm

OUT = "outputs/incubator"


def self_rows():
    """Wordings and their read states (read once, cached)."""
    rows = [json.loads(l) for l in open(f"{OUT}/self_wordings.jsonl")]
    p = f"{OUT}/self_keys.pt"
    h = torch.load(p) if os.path.exists(p) else {}
    todo = [r["text"] for r in rows if r["text"] not in h]
    if todo:
        from reflex import Reflex
        reflex = Reflex()
        for t in todo:
            k = reflex.read(t)
            h[t] = torch.cat([torch.tensor(k["h_last"]), torch.tensor(k["h_mean"])]).float()
        torch.save(h, p)
    return rows, h


def teach_self(mem, rows, h, nights: int, per: int) -> dict[str, int]:
    """``nights`` nights, ``per`` new wordings of every self fact per night."""
    ids: dict[str, int] = {}
    train: dict[str, list] = {}
    for r in rows:
        if r["kind"] == "train": train.setdefault(r["fact"], []).append(r)
    for n in range(nights):
        for k, rs in train.items():
            for r in rs[n * per:(n + 1) * per]:
                ids[k] = mem.teach(r["text"], r["answer"], h[r["text"]], same_as=ids.get(k), pilot=True, self_fact=True)
        mem.sleep()
    return ids


if __name__ == "__main__":
    E = lambda k, d: int(os.environ.get(k, d))
    SEEDS, NIGHTS, PER = E("SEEDS", 3), E("NIGHTS", 6), E("PER", 4)
    rows, h = self_rows()
    H2, group2, block2, variant2, texts2 = vm.stage2()
    Hp, qid, wave, orig, qa = vm.pilot()
    pq = {q for q, _ in qa.values()}
    other = ~torch.isin(block2, block2[torch.tensor([t in pq for t in texts2])].unique())
    neg = H2[other & (variant2 == 0)]
    print(f"# self test  seeds={SEEDS}  thorough = {NIGHTS} nights x {PER} wordings, thin = 1 night x 6", flush=True)
    kind = lambda k: [r for r in rows if r["kind"] == k]

    def measure(mem, ids):
        leaf = lambda rs: mem._leaf(mem.sense(torch.stack([h[r["text"]] for r in rs])))[0]
        selfids = torch.tensor(list(ids.values()))
        t = kind("test")
        right = (leaf(t) == torch.tensor([ids[r["fact"]] for r in t])).float().mean().item()
        near = torch.isin(leaf(kind("near")), selfids).float().mean().item()
        oth = (leaf(kind("other")) >= 0).float().mean().item()
        m = mem.leaf.mixed
        lab = {v: int(k[1:]) for k, v in m.label_taps.vocab.items()}
        inside = total = 0
        for c, tg in enumerate(m.tags):
            for i in selfids.tolist():
                n = sum(lab[int(x)] == i for x in tg[tg >= 0]); total += n; inside += n * int(mem.names[c] == i)
        return [right, near, oth, inside / max(total, 1), int(torch.isin(mem.names, selfids).sum())]

    res: dict[str, list] = {}
    for seed in range(SEEDS):
        g = torch.Generator().manual_seed(seed)
        held = torch.zeros(len(Hp), dtype=torch.bool)
        for q in qa:
            idx = (torch.tensor([x == q for x in qid]) & ~orig).nonzero().squeeze(1)
            held[idx[torch.randperm(len(idx), generator=g)[:10]]] = True
        for name, n, per in (("thin", 1, 6), ("thorough", NIGHTS, PER)):
            mem = vm.FactMemory(H2[other], group2[other], neg[torch.randperm(len(neg), generator=g)[:2000]], seed=seed)
            ids = teach_self(mem, rows, h, n, per)
            a = measure(mem, ids)
            pid = vm.childhood(mem, held)
            b = measure(mem, ids)
            pf = torch.tensor([pid.get(q, -1) for q in qid])
            m0 = held & (wave == 0)
            later = (mem._leaf(mem.sense(Hp[m0]))[0] == pf[m0]).float().mean().item()
            res.setdefault(name, []).append(a + b + [later, len(mem.leaf.mixed.classes)])
            print(f"  seed {seed} {name:9} after self: right {a[0]:.2f} near {a[1]:.2f} other {a[2]:.2f} purity {a[3]:.2f} classes {a[4]} | "
                  f"after 40 later facts: right {b[0]:.2f} near {b[1]:.2f} other {b[2]:.2f} purity {b[3]:.2f} | later w0 {later:.2f}", flush=True)
    print(f"\n{'teaching':9} | after self: right  near  other purity classes | after 40 later facts: right  near  other purity | later w0 | all classes")
    for name, rr in res.items():
        r = np.array(rr, float).mean(0)
        print(f"{name:9} | {r[0]:>17.2f} {r[1]:>5.2f} {r[2]:>6.2f} {r[3]:>6.2f} {r[4]:>7.1f} | {r[5]:>27.2f} {r[6]:>5.2f} {r[7]:>6.2f} {r[8]:>6.2f} | {r[10]:>8.2f} | {r[11]:>6.0f}")
