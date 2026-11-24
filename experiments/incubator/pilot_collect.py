"""Phasecyte pilot, data collection: the reflex READS every wording of the pilot
facts (no answer) and we keep the hidden states.  Resumable.

In : outputs/incubator/pilot_in_wave{0..3}.tsv     qid, question, answer (40 taught facts, 4 waves)
     outputs/incubator/pilot_in_untaught.tsv       20 questions never taught
     outputs/incubator/pilot_para_*.jsonl          {"qid","i","text"} matron rewordings
Out: outputs/incubator/pilot_keys.jsonl            one row per text: qid, wave (-1 = untaught), i (-1 = original), text
     outputs/incubator/pilot_keys.npz              h_last, h_mean [N, 1536] float16, row-aligned

Run from repo root:  python3 experiments/incubator/pilot_collect.py
Env: THREADS=6
"""
from __future__ import annotations
import glob, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

OUT = "outputs/incubator"
IDX, HID = f"{OUT}/pilot_keys.jsonl", f"{OUT}/pilot_keys.npz"


def load_texts() -> list[dict]:
    texts, wave = [], {}
    for w, p in [(w, f"{OUT}/pilot_in_wave{w}.tsv") for w in range(4)] + [(-1, f"{OUT}/pilot_in_untaught.tsv")]:
        for line in open(p):
            qid, q, _ = line.rstrip("\n").split("\t")
            wave[qid] = w
            texts.append({"qid": qid, "wave": w, "i": -1, "text": q})
    seen = {(t["qid"], t["text"]) for t in texts}
    for p in sorted(glob.glob(f"{OUT}/pilot_para_*.jsonl")):
        for line in open(p):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            t = (r.get("text") or "").strip()
            if r.get("qid") in wave and t and (r["qid"], t) not in seen:
                seen.add((r["qid"], t))
                texts.append({"qid": r["qid"], "wave": wave[r["qid"]], "i": int(r["i"]), "text": t})
    return texts


if __name__ == "__main__":
    from reflex import Reflex
    texts = load_texts()
    rows, hl, hm = [], [], []
    if os.path.exists(IDX) and os.path.exists(HID):
        rows = [json.loads(l) for l in open(IDX)]
        z = np.load(HID)
        n = min(len(rows), len(z["h_last"]))
        rows, hl, hm = rows[:n], list(z["h_last"][:n]), list(z["h_mean"][:n])
    done = {(r["qid"], r["text"]) for r in rows}
    todo = [t for t in texts if (t["qid"], t["text"]) not in done]
    print(f"# texts {len(texts)}  done {len(done)}  todo {len(todo)}", flush=True)
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
