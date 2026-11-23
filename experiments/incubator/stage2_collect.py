"""Stage 2 data collection: the reflex READS each question and each paraphrase
(no answer) and we keep the hidden states a fact memory can use as a key.
Resumable (skips texts already stored).

In : outputs/incubator/stage1_answers.jsonl          the original questions (variant "q")
     outputs/incubator/stage2_para_part*_c*.jsonl    {"qid","p1","p2"} matron paraphrases
Out: outputs/incubator/stage2_keys.jsonl             one row per text: qid, variant, text
     outputs/incubator/stage2_keys.npz               h_last, h_mean [N, 1536] float16, row-aligned

Run from repo root:  python3 experiments/incubator/stage2_collect.py
Env: LIMIT=n (stop after n new texts)  THREADS=6
"""
from __future__ import annotations
import glob, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from reflex import Reflex

OUT = "outputs/incubator"
IDX, HID = f"{OUT}/stage2_keys.jsonl", f"{OUT}/stage2_keys.npz"


def load_texts() -> list[dict]:
    """Originals first, then paraphrases of known qids (bad lines are skipped)."""
    texts = [{"qid": r["qid"], "variant": "q", "text": r["q"]}
             for r in map(json.loads, open(f"{OUT}/stage1_answers.jsonl"))]
    orig = {t["qid"]: t["text"] for t in texts}
    seen = set()
    for p in sorted(glob.glob(f"{OUT}/stage2_para_part*_c*.jsonl")):
        for line in open(p):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("qid") not in orig:
                continue
            for v in ("p1", "p2"):
                t = (r.get(v) or "").strip()
                if t and t != orig[r["qid"]] and (r["qid"], v) not in seen:
                    seen.add((r["qid"], v))
                    texts.append({"qid": r["qid"], "variant": v, "text": t})
    return texts


if __name__ == "__main__":
    texts = load_texts()
    rows, hl, hm = [], [], []
    if os.path.exists(IDX) and os.path.exists(HID):
        rows = [json.loads(l) for l in open(IDX)]
        z = np.load(HID)
        n = min(len(rows), len(z["h_last"]))           # trust only the row-aligned prefix
        rows, hl, hm = rows[:n], list(z["h_last"][:n]), list(z["h_mean"][:n])
    done = {(r["qid"], r["variant"]) for r in rows}
    todo = [t for t in texts if (t["qid"], t["variant"]) not in done]
    limit = min(int(os.environ.get("LIMIT", len(todo))), len(todo))
    print(f"# texts {len(texts)}  done {len(done)}  todo {len(todo)}  running {limit}", flush=True)
    reflex = Reflex(n_threads=int(os.environ.get("THREADS", 6)))

    def flush():
        np.savez(HID + ".tmp.npz", h_last=np.asarray(hl, np.float16), h_mean=np.asarray(hm, np.float16))
        os.replace(HID + ".tmp.npz", HID)
        with open(IDX + ".tmp", "w") as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(IDX + ".tmp", IDX)

    t0 = time.time()
    for n, t in enumerate(todo[:limit], 1):
        k = reflex.read(t["text"])
        rows.append(t); hl.append(k["h_last"].astype(np.float16)); hm.append(k["h_mean"].astype(np.float16))
        if n % 200 == 0:
            flush()
            print(f"  {n}/{limit}  {(time.time() - t0) / n:.2f} s/text", flush=True)
    flush()
    print(f"# done: {len(rows)} stored, {time.time() - t0:.0f}s", flush=True)
