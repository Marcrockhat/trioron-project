"""Stage 1 data collection: the reflex answers closed-book, we grade and keep
its confidence and hidden states.  Resumable (skips items already stored).

In : outputs/incubator/stage1_qa_part*.jsonl   {"id","q","a","alt"}
Out: outputs/incubator/stage1_answers.jsonl    one row per item (text, grade, log-probs)
     outputs/incubator/stage1_hidden.npz       h_pre, h_post [N, 1536] float16, row-aligned

Run from repo root:  python3 experiments/incubator/stage1_collect.py
Env: LIMIT=n (stop after n new items)  THREADS=6  PARTS=0,1 (default: all)
"""
from __future__ import annotations
import glob, json, os, re, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from reflex import Reflex, chat_prompt

OUT = "outputs/incubator"
ANS, HID = f"{OUT}/stage1_answers.jsonl", f"{OUT}/stage1_hidden.npz"
_ART = re.compile(r"\b(the|a|an)\b")


def norm(s: str) -> list[str]:
    s = re.sub(r"(?<=\d),(?=\d)", "", s.lower())          # 1,234 -> 1234
    return re.findall(r"[a-z0-9]+", _ART.sub(" ", s))


def grade(answer: str, golds: list[str]) -> bool:
    """Correct if any gold form appears in the answer as a whole-token run."""
    a = norm(answer)
    for g in golds:
        t = norm(g)
        if t and any(a[i:i + len(t)] == t for i in range(len(a) - len(t) + 1)):
            return True
    return False


def load_items() -> list[dict]:
    """All questions, one article per title (the lead sampler drew with
    replacement, so a few articles appear under two ids)."""
    title = {json.loads(l)["id"]: json.loads(l)["title"] for l in open(f"{OUT}/stage1_leads.jsonl")}
    parts = os.environ.get("PARTS")
    paths = ([f"{OUT}/stage1_qa_part{k}.jsonl" for k in parts.split(",")] if parts
             else sorted(glob.glob(f"{OUT}/stage1_qa_part*.jsonl")))
    items, owner = [], {}
    for p in paths:
        seen = {}
        for line in open(p):
            if not line.strip():
                continue
            r = json.loads(line)
            if owner.setdefault(title[r["id"]], r["id"]) != r["id"]:
                continue                               # duplicate article
            j = seen.get(r["id"], 0); seen[r["id"]] = j + 1
            r["qid"] = f"{r['id']}-{j}"
            items.append(r)
    return items


if __name__ == "__main__":
    items = load_items()
    done, hp, hq = {}, [], []
    if os.path.exists(ANS) and os.path.exists(HID):
        rows = [json.loads(l) for l in open(ANS)]
        z = np.load(HID)
        n = min(len(rows), len(z["h_pre"]))            # trust only row-aligned prefix
        rows = rows[:n]
        hp, hq = list(z["h_pre"][:n]), list(z["h_post"][:n])
        with open(ANS, "w") as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        done = {r["qid"] for r in rows}
    todo = [it for it in items if it["qid"] not in done]
    limit = int(os.environ.get("LIMIT", len(todo)))
    print(f"# items {len(items)}  done {len(done)}  todo {len(todo)}  running {min(limit, len(todo))}", flush=True)
    reflex = Reflex(n_threads=int(os.environ.get("THREADS", 6)))

    def flush():
        np.savez(HID + ".tmp.npz", h_pre=np.asarray(hp, np.float16), h_post=np.asarray(hq, np.float16))
        os.replace(HID + ".tmp.npz", HID)

    t0, ok, f = time.time(), 0, open(ANS, "a")
    for n, it in enumerate(todo[:limit], 1):
        a = reflex.answer(chat_prompt(it["q"]))
        correct = grade(a["text"], [it["a"]] + list(it.get("alt") or []))
        ok += correct
        f.write(json.dumps({"qid": it["qid"], "id": it["id"], "q": it["q"], "gold": it["a"],
                            "alt": it.get("alt") or [], "answer": a["text"], "correct": bool(correct),
                            "logprobs": [round(x, 4) for x in a["logprobs"]]}, ensure_ascii=False) + "\n")
        f.flush()
        hp.append(a["h_pre"].astype(np.float16)); hq.append(a["h_post"].astype(np.float16))
        if n % 50 == 0:
            flush()
            dt = time.time() - t0
            print(f"  {n}/{min(limit, len(todo))}  acc {ok / n:.3f}  {dt / n:.2f} s/item", flush=True)
    flush()
    print(f"# done: {len(hp)} stored, new-item accuracy {ok / max(1, min(limit, len(todo))):.3f}, "
          f"{time.time() - t0:.0f}s", flush=True)
