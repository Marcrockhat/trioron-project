"""Vee taught by INTERACTION (Rocky s066: "teach it with interaction too").

A life is an ordered list of events — the matron's replies to what Vee said,
and her nights.  One day:
    ask    the matron asks; Vee answers from memory (or is silent)
    teach  the matron replies to each answer: the right answer for that wording
           (a confirmation teaches the same answer again = another wording of
           the fact), or UNKNOWN = "you do not know that yet"
    sleep  the day's facts are replayed into the phasecyte
The "do not know" reply and SAYING NOTHING are LEARNED like any other fact
(many wordings, one label), not thresholds; the console can hand "do not
know" questions to the reflex.

Nothing is batch-trained; the leaf is born from the first night.  It is not
saved: every call replays the life (vee_memory.py).

    python3 experiments/incubator/vee_life.py ask   FILE    one question per line
    python3 experiments/incubator/vee_life.py teach FILE    question<TAB>answer per line ("?" = UNKNOWN, "-" = say nothing)
    python3 experiments/incubator/vee_life.py sleep
    python3 experiments/incubator/vee_life.py lesson FILE   question<TAB>the matron's reply: ask, show her misses, teach all, sleep
    python3 experiments/incubator/vee_life.py test          held-out self / near / unrelated questions (self_wordings.jsonl)
Env: LIFE=outputs/incubator/vee_life   (events .jsonl, read states .pt)
"""
from __future__ import annotations
import json, os, sys
import torch
import torch.nn.functional as F
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(__file__))
import vee_memory as vm

LIFE = os.environ.get("LIFE", "outputs/incubator/vee_life")
UNKNOWN = "(I do not know that yet.)"
QUIET = "(says nothing)"                       # a learned reply too: some things need no answer
REPLY = lambda a: {"?": UNKNOWN, "-": QUIET}.get(a, a)


def states(texts: list[str]) -> dict[str, torch.Tensor]:
    """Read states by text, cached; the reflex is loaded only for new texts."""
    p = LIFE + ".pt"
    h = torch.load(p) if os.path.exists(p) else {}
    todo = [t for t in dict.fromkeys(texts) if t not in h]
    if todo:
        from reflex import Reflex
        reflex = Reflex()
        for t in todo:
            k = reflex.read(t)
            h[t] = torch.cat([torch.tensor(k["h_last"]), torch.tensor(k["h_mean"])]).float()
        torch.save(h, p)
    return h


def events() -> list[dict]:
    return [json.loads(l) for l in open(LIFE + ".jsonl")] if os.path.exists(LIFE + ".jsonl") else []


def wake(seed: int = 0) -> vm.FactMemory:
    H2, group2, block2, variant2, _ = vm.stage2()
    neg = H2[variant2 == 0]
    mem = vm.FactMemory(H2, group2, neg[torch.randperm(len(neg), generator=torch.Generator().manual_seed(seed))[:2000]], seed=seed)
    Hp, qid, _, orig, qa = vm.pilot()                                # only to measure how close a real rewording is
    S, cos = F.normalize(mem.sense(Hp), dim=1), []
    for q in qa:
        m = torch.tensor([x == q for x in qid])
        cos.append(S[m & ~orig] @ S[(m & orig).nonzero()[0, 0]])
    mem.floor = float(torch.quantile(torch.cat(cos), 0.05))
    ev = events()
    h = states([e["q"] for e in ev if e["op"] == "teach"])
    for e in ev:
        if e["op"] == "sleep": mem.sleep(); continue
        same = next((i for i, f in enumerate(mem.facts) if f["answer"].casefold() == e["a"].casefold()), None)
        mem.teach(e["q"], e["a"], h[e["q"]], same_as=same, pilot=True)
    return mem


def answers(mem, texts):
    h = states(texts)
    return [mem.recall(h[t]) for t in texts]


if __name__ == "__main__":
    op = sys.argv[1]
    if op == "ask":
        qs = [l.strip() for l in open(sys.argv[2]) if l.strip()]
        for q, r in zip(qs, answers(wake(), qs)):
            print(f"{q}\t{'(silent)' if r is None else r['answer']}" + ("" if r is None else f"\t[{r['where']}]"))
    elif op == "teach":
        rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[2]) if l.strip()]
        states([q for q, _ in rows])
        with open(LIFE + ".jsonl", "a") as f:
            for q, a in rows:
                f.write(json.dumps({"op": "teach", "q": q, "a": REPLY(a)}, ensure_ascii=False) + "\n")
        print(f"[taught {len(rows)}]")
    elif op == "lesson":                                              # ask, show what she got wrong, reply to all, sleep
        rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[2]) if l.strip()]
        rows = [(q, REPLY(a)) for q, a in rows]
        got = answers(wake(), [q for q, _ in rows])
        ok = 0
        for (q, a), r in zip(rows, got):
            said = "(silent)" if r is None else r["answer"]
            ok += said == a or (r is None and a == UNKNOWN)
            if said != a and not (r is None and a == UNKNOWN): print(f"  {q}  ->  she said: {said}   | matron: {a}")
        print(f"[she was right on {ok} of {len(rows)} before the matron replied]")
        with open(LIFE + ".jsonl", "a") as f:
            for q, a in rows: f.write(json.dumps({"op": "teach", "q": q, "a": a}, ensure_ascii=False) + "\n")
            f.write(json.dumps({"op": "sleep"}) + "\n")
        mem = wake()
        print(f"[slept: night {mem.night}, {len(mem.facts)} facts, {len(mem.leaf.mixed.classes)} classes]")
    elif op == "sleep":
        with open(LIFE + ".jsonl", "a") as f: f.write(json.dumps({"op": "sleep"}) + "\n")
        mem = wake()
        print(f"[slept: night {mem.night}, {len(mem.facts)} facts, {len(mem.day())} still in today's memory, "
              f"{len(mem.leaf.mixed.classes)} classes]")
    elif op == "test":
        import time
        rows = [json.loads(l) for l in open("outputs/incubator/self_wordings.jsonl")]
        t0 = time.time()
        mem = wake()
        print(f"night rule {os.environ.get('NIGHT', 'all')}: life replayed in {time.time() - t0:.0f}s, {mem.night} nights, "
              f"{mem.fed_rows} samples replayed, {len(mem.leaf.mixed.classes)} classes, {len(mem.day())} facts still only in today's memory")
        for kind in ("test", "near", "other"):
            rs = [r for r in rows if r["kind"] == kind]
            got = answers(mem, [r["text"] for r in rs])
            n = len(rs)
            right = sum(g is not None and g["answer"] == r["answer"] for g, r in zip(got, rs)) / n
            dunno = sum(g is not None and g["answer"] == UNKNOWN for g in got) / n
            silent = sum(g is None for g in got) / n
            print(f"{kind:5} n={n:2}  right self fact {right:.2f}  says she does not know {dunno:.2f}  silent {silent:.2f}  "
                  f"another fact {1 - right - dunno - silent:.2f}")
