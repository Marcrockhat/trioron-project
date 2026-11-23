"""Incubator console — talk to the reflex and see the organism's doubt (s064).

You ask; the reflex brain (Qwen) answers; the organism (Syneidesis) shows how
much it doubts that answer.  You can mark the answer right or wrong: every
verdict is stored and becomes training data, so you teach it by hand.

Run from repo root, in your own terminal:
    python3 experiments/incubator/console.py
Commands (anything else is a question):
    /right              the last answer was correct
    /wrong [answer]     it was wrong (optionally give the right answer)
    /look [query]       look the last question (or a query) up in offline Wikipedia
    /learn              retrain the organism on all graded data + your verdicts
    /quit

Env: SOURCE=both|h_post|h_pre  K=64  EPOCHS=30   (used when (re)training)
Files: outputs/incubator/syneidesis_stage1.pt, console_verdicts.jsonl/.npz
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(__file__))
from trioron.organisms import Syneidesis
from reflex import Reflex, chat_prompt

OUT = "outputs/incubator"
ORG = f"{OUT}/syneidesis_stage1.pt"
VJ, VH = f"{OUT}/console_verdicts.jsonl", f"{OUT}/console_verdicts.npz"


def pick(source: str, h_pre, h_post) -> torch.Tensor:
    h_pre, h_post = torch.as_tensor(h_pre).float(), torch.as_tensor(h_post).float()
    return {"h_pre": h_pre, "h_post": h_post, "both": torch.cat([h_pre, h_post], -1)}[source]


STOP = set("a an the of in on at to for is was were are be been do does did who whom whose what "
           "which when where why how many much and or with by from as it its this that".split())


def keywords(question: str) -> str:
    """Full questions search badly; keep the content words."""
    ws = [w.strip("?.,!\"'()") for w in question.split()]
    return " ".join(w for w in ws if w and w.lower() not in STOP)


def graded_data():
    """Stage-1 collection + console verdicts -> (h_pre, h_post, wrong)."""
    rows = [json.loads(l) for l in open(f"{OUT}/stage1_answers.jsonl")]
    z = np.load(f"{OUT}/stage1_hidden.npz")
    n = min(len(rows), len(z["h_pre"]))
    hp, hq = [z["h_pre"][:n].astype(np.float32)], [z["h_post"][:n].astype(np.float32)]
    wrong = [0 if r["correct"] else 1 for r in rows[:n]]
    if os.path.exists(VJ) and os.path.exists(VH):
        v = [json.loads(l) for l in open(VJ)]
        zv = np.load(VH)
        m = min(len(v), len(zv["h_pre"]))
        hp.append(zv["h_pre"][:m].astype(np.float32)); hq.append(zv["h_post"][:m].astype(np.float32))
        wrong += [0 if r["correct"] else 1 for r in v[:m]]
    return np.concatenate(hp), np.concatenate(hq), torch.tensor(wrong)


def train() -> Syneidesis:
    source = os.environ.get("SOURCE", "both")
    hp, hq, wrong = graded_data()
    t = time.time()
    o = Syneidesis(0, k=int(os.environ.get("K", 64)))
    H = pick(source, hp, hq)
    o.fit(H, wrong, epochs=int(os.environ.get("EPOCHS", 30)))
    o.source = source
    o.doubt_sorted = torch.sort(o.doubt(H)).values          # for "higher than x% of answers"
    o.save(ORG)
    print(f"[organism trained on {len(wrong)} graded answers ({source}), "
          f"{o.n_params()} params, {time.time() - t:.0f}s]")
    return o


def save_verdict(rec: dict, h_pre, h_post) -> None:
    hp, hq = [], []
    if os.path.exists(VH):
        z = np.load(VH); hp, hq = list(z["h_pre"]), list(z["h_post"])
    hp.append(h_pre.astype(np.float16)); hq.append(h_post.astype(np.float16))
    np.savez(VH, h_pre=np.asarray(hp), h_post=np.asarray(hq))
    with open(VJ, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    org = Syneidesis.load(ORG) if os.path.exists(ORG) else train()
    print("[loading the reflex brain...]", flush=True)
    reflex = Reflex()
    wiki = None
    last = None
    print("Ask a question. /right /wrong [answer] /look [query] /learn /quit")
    while True:
        try:
            line = input("\nyou> ").strip()
        except EOFError:
            break
        if not line:
            continue
        cmd, _, arg = line.partition(" ")
        if cmd == "/quit":
            break
        if cmd in ("/right", "/wrong"):
            if last is None:
                print("nothing to grade yet"); continue
            save_verdict({"q": last["q"], "answer": last["text"], "correct": cmd == "/right",
                          "gold": arg.strip(), "doubt": last["doubt"], "t": time.time()},
                         last["h_pre"], last["h_post"])
            print(f"[stored: reflex was {'right' if cmd == '/right' else 'wrong'}"
                  + (f"; correct answer: {arg.strip()}" if arg.strip() else "") + "]")
            continue
        if cmd == "/learn":
            print("[retraining, about half a minute...]", flush=True)
            org = train(); continue
        if cmd == "/look":
            if wiki is None:
                from wiki import Wiki
                wiki = Wiki()
            q = arg.strip() or keywords(last["q"] if last else "")
            hits = wiki.search(q, 1) if q else []
            if not hits:
                print("[nothing found]"); continue
            e = wiki.by_path(hits[0])
            ps = wiki.paragraphs(e)
            print(f"[{e.title}] {ps[0][:500] if ps else '(no text)'}")
            continue
        t = time.time()
        if len(reflex.tokens(chat_prompt(line))) > 480:
            print("[question too long for the reflex (limit ~400 words)]"); continue
        print("[thinking...]", end="\r", flush=True)
        try:
            a = reflex.answer(chat_prompt(line))
        except Exception as e:                       # keep the session alive
            print(f"[reflex failed: {e!r}]"); continue
        d = float(org.doubt(pick(org.source, a["h_pre"], a["h_post"]).unsqueeze(0))[0])
        pct = float((org.doubt_sorted < d).float().mean()) * 100
        conf = float(np.exp(np.mean(a["logprobs"]))) if a["logprobs"] else 0.0
        word = "low" if d < 0.35 else "medium" if d < 0.65 else "HIGH"
        print(f"reflex> {a['text']}")
        print(f"        reflex confidence {conf:.2f} | organism doubt {d:.2f} ({word}; "
              f"higher than {pct:.0f}% of graded answers) | {time.time() - t:.1f}s")
        last = {"q": line, "text": a["text"], "doubt": d, "h_pre": a["h_pre"], "h_post": a["h_post"]}


if __name__ == "__main__":
    main()
