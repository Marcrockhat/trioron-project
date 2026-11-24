"""Phasecyte pilot, night rewordings written by the REFLEX itself (no helper tokens).

Each taught pilot fact is heard once (its original wording).  Here the frozen
Qwen rewrites that wording many ways (sampled), then READS each rewording and
we keep the hidden states — the material a night replay would feed.

In : outputs/incubator/pilot_in_wave{0..3}.tsv
Out: outputs/incubator/pilot_qwen_para.jsonl      {"qid","i","text"}
     outputs/incubator/pilot_qwen_keys.jsonl/.npz same layout as pilot_keys.*

Run from repo root:  python3 experiments/incubator/pilot_reword.py
Env: CALLS=5 (sampled lists of 10 per fact)  TEMP=0.9  TOP_P=0.95  THREADS=6  LIMIT=0 (facts, 0 = all)
"""
from __future__ import annotations
import json, os, re, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

OUT = "outputs/incubator"
PARA, IDX, HID = f"{OUT}/pilot_qwen_para.jsonl", f"{OUT}/pilot_qwen_keys.jsonl", f"{OUT}/pilot_qwen_keys.npz"
E = lambda k, d: type(d)(os.environ.get(k, d))
CALLS, TEMP, TOP_P, LIMIT = E("CALLS", 5), E("TEMP", 0.9), E("TOP_P", 0.95), E("LIMIT", 0)
SYSTEM = "You rewrite questions. You never answer them."
ASK = ("Rewrite this question in 10 different ways. Keep the meaning exactly the same and keep every "
       "name in it. Do not answer it. One rewording per line, numbered.\n\nQuestion: {q}")


def sample(reflex, prompt: str, rng, max_tokens: int = 330) -> str:
    m = reflex.m
    reflex._head_kept = False
    m.reset(); m._ctx.kv_cache_clear()
    m.eval(reflex.tokens(prompt))
    out = []
    for _ in range(max_tokens):
        lp = reflex._logp() / TEMP
        p = np.exp(lp - lp.max()); p /= p.sum()
        o = np.argsort(-p); keep = o[:int(np.searchsorted(np.cumsum(p[o]), TOP_P)) + 1]   # nucleus
        t = int(rng.choice(keep, p=p[keep] / p[keep].sum()))
        if t in reflex.eos:
            break
        out.append(t); m.eval([t])
    return m.detokenize(out).decode("utf-8", "ignore")


def lines(text: str, original: str) -> list[str]:
    got = []
    for l in text.split("\n"):
        l = re.sub(r"^\s*(\d+[.)]|[-*])\s*", "", l).strip().strip('"')
        if len(l) > 10 and l.endswith("?") and l.lower() != original.lower():
            got.append(l)
    return got


if __name__ == "__main__":
    from reflex import Reflex, chat_prompt
    facts = [l.rstrip("\n").split("\t")[:2] for w in range(4) for l in open(f"{OUT}/pilot_in_wave{w}.tsv")]
    if LIMIT: facts = facts[:LIMIT]
    reflex = Reflex(n_threads=E("THREADS", 6))
    rng = np.random.default_rng(0)
    done: dict[str, list[str]] = {}
    if os.path.exists(PARA):
        for l in open(PARA):
            r = json.loads(l); done.setdefault(r["qid"], []).append(r["text"])
    t0 = time.time()
    with open(PARA, "a") as f:
        for n, (qid, q) in enumerate(facts, 1):
            if qid in done: continue
            seen: list[str] = []
            for _ in range(CALLS):
                for t in lines(sample(reflex, chat_prompt(ASK.format(q=q), SYSTEM), rng), q):
                    if t not in seen: seen.append(t)
            for i, t in enumerate(seen):
                f.write(json.dumps({"qid": qid, "i": i, "text": t}, ensure_ascii=False) + "\n")
            f.flush(); done[qid] = seen
            print(f"  {n}/{len(facts)} {qid}: {len(seen)} rewordings  {time.time() - t0:.0f}s", flush=True)

    rows = [json.loads(l) for l in open(PARA)]
    hl, hm = [], []
    for n, r in enumerate(rows, 1):
        k = reflex.read(r["text"])
        hl.append(k["h_last"].astype(np.float16)); hm.append(k["h_mean"].astype(np.float16))
        if n % 400 == 0: print(f"  read {n}/{len(rows)}  {time.time() - t0:.0f}s", flush=True)
    np.savez(HID, h_last=np.asarray(hl, np.float16), h_mean=np.asarray(hm, np.float16))
    with open(IDX, "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"# done: {len(rows)} rewordings of {len(done)} facts, {time.time() - t0:.0f}s", flush=True)
