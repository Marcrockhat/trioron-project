"""Stage 1 source material: sample substantial Simple English Wikipedia
articles and keep their lead text, for the matron to write questions from.

Out: outputs/incubator/stage1_leads.jsonl          {"id","title","chars","lead"}
     outputs/incubator/stage1_leads_part{0..3}.txt  "[id] Title :: lead (<=520 chars)"
The sampler draws with replacement, so a few articles appear twice;
stage1_collect.py de-duplicates by title.

Run from repo root:  python3 experiments/incubator/stage1_leads.py
"""
from __future__ import annotations
import json, os, random, re, sys
sys.path.insert(0, os.path.dirname(__file__))
from wiki import Wiki

OUT = "outputs/incubator"
N, MIN_CHARS, PARTS = 1600, 2500, 4
SKIP = re.compile(r"^(List of|Category:|Template:|\d{1,4}(s| BC)?$)|\(disambiguation\)")


def cut(t: str, n: int = 520) -> str:
    if len(t) <= n:
        return t
    s = t[:n]; i = s.rfind(". ")
    return s[:i + 1] if i > 200 else s


if __name__ == "__main__":
    w = Wiki(); random.seed(0)
    os.makedirs(OUT, exist_ok=True)
    rows, tried = [], 0
    while len(rows) < N and tried < 400000:
        tried += 1
        e = w.entry(random.randrange(len(w)))
        if e is None or SKIP.search(e.title):
            continue
        ps = w.paragraphs(e)
        total = sum(len(p) for p in ps)
        if total < MIN_CHARS:               # notability proxy: a substantial article
            continue
        lead, n = [], 0
        for p in ps:
            if n + len(p) > 900 and lead:
                break
            lead.append(p); n += len(p)
        rows.append({"id": len(rows), "title": e.title, "chars": total, "lead": " ".join(lead)[:1100]})
    with open(f"{OUT}/stage1_leads.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    per = N // PARTS
    for k in range(PARTS):
        with open(f"{OUT}/stage1_leads_part{k}.txt", "w") as f:
            for r in rows[k * per:(k + 1) * per]:
                f.write(f"[{r['id']}] {r['title']} :: {cut(r['lead'])}\n")
    print(f"tried {tried}, kept {len(rows)}")
