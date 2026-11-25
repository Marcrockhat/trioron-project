"""Ambient language for the embryo (docs/design/vee_development.md §6.2, §7.1):
random sentences from the offline Simple English Wikipedia, READ by the reflex
(no answer).  Resumable.  No helper tokens.

Out: outputs/incubator/wiki_sent.jsonl   one row per sentence: i, title, text
     outputs/incubator/wiki_sent.npz     h_last, h_mean [N, 1536] float16, row-aligned

Run from repo root:  python3 experiments/incubator/wiki_collect.py
Env: N=4000  PER_ARTICLE=2  THREADS=6
"""
from __future__ import annotations
import json, os, random, re, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

OUT = "outputs/incubator"
IDX, HID = f"{OUT}/wiki_sent.jsonl", f"{OUT}/wiki_sent.npz"
N, PER = int(os.environ.get("N", 4000)), int(os.environ.get("PER_ARTICLE", 2))
SKIP = re.compile(r"^(List of|Category:|Template:|\d{1,4}(s| BC)?$)|\(disambiguation\)")
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def sentences(n: int, per: int, seed: int = 0) -> list[dict]:
    """``n`` sentences, at most ``per`` from one article, articles drawn without repeats."""
    from wiki import Wiki
    w, r = Wiki(), random.Random(seed)
    rows, seen, tried = [], set(), 0
    while len(rows) < n and tried < 400000:
        tried += 1
        i = r.randrange(len(w))
        if i in seen: continue
        seen.add(i)
        e = w.entry(i)
        if e is None or SKIP.search(e.title): continue
        ss = [s.strip() for p in w.paragraphs(e) for s in SENT.split(p)]
        ss = [s for s in ss if 40 <= len(s) <= 200 and s[-1] in ".!?"]
        for s in r.sample(ss, min(per, len(ss))):
            rows.append({"i": len(rows), "title": e.title, "text": s})
    return rows[:n]


if __name__ == "__main__":
    from reflex import Reflex
    texts = sentences(N, PER)
    rows, hl, hm = [], [], []
    if os.path.exists(IDX) and os.path.exists(HID):
        rows = [json.loads(l) for l in open(IDX)]
        z = np.load(HID)
        n = min(len(rows), len(z["h_last"]))
        rows, hl, hm = rows[:n], list(z["h_last"][:n]), list(z["h_mean"][:n])
    todo = texts[len(rows):]                                   # the sampler is deterministic: resume by position
    print(f"# sentences {len(texts)}  done {len(rows)}  todo {len(todo)}", flush=True)
    reflex = Reflex(n_threads=int(os.environ.get("THREADS", 6)))

    def flush():
        np.savez(HID + ".tmp.npz", h_last=np.asarray(hl, np.float16), h_mean=np.asarray(hm, np.float16))
        os.replace(HID + ".tmp.npz", HID)
        with open(IDX + ".tmp", "w") as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(IDX + ".tmp", IDX)

    t0 = time.time()
    for n, t in enumerate(todo, 1):
        k = reflex.read(t["text"])
        rows.append(t); hl.append(k["h_last"].astype(np.float16)); hm.append(k["h_mean"].astype(np.float16))
        if n % 200 == 0:
            flush()
            print(f"  {n}/{len(todo)}  {(time.time() - t0) / n:.2f} s/text", flush=True)
    flush()
    print(f"# done: {len(rows)} stored, {time.time() - t0:.0f}s", flush=True)
