"""Incubator console — talk to Vee: her memory, the reflex, and her doubt (s064, s066).

You ask.  If Vee remembers the fact (vee_memory.py: day table, then the
phasecyte fed at night) she answers from memory.  Otherwise the reflex brain
(Qwen) answers and Vee shows how much she doubts that answer.  You can mark
the answer right or wrong, and teach her the right answer: it is heard once,
kept in the day table, and moves into long-term memory when she sleeps.

Run from repo root, in your own terminal:
    python3 experiments/incubator/console.py
Commands (anything else is a question):
    /right              the last answer was correct
    /wrong [answer]     it was wrong (give the right answer and she learns it)
    /teach <answer>     the answer to the last question is <answer>; remember it
    /sleep              night: replay today's facts into long-term memory
    /reflex             ask the reflex the last question anyway (ignore memory)
    /facts              what she holds
    /look [query]       look the last question (or a query) up in offline Wikipedia
    /learn              retrain the organism on all graded data + your verdicts
    /quit               leave (/exit and Ctrl-D also work)

Env: SOURCE=both|h_post|h_pre  K=64  EPOCHS=30   (used when (re)training)
     MEMORY=0 runs without the fact memory (it needs the local stage-2 and pilot collections)
Files: outputs/incubator/vee_stage1.pt, console_verdicts.jsonl/.npz, vee_facts.jsonl/.npz
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(__file__))
from trioron.organisms import Vee
from reflex import Reflex, chat_prompt

OUT = "outputs/incubator"
ORG = f"{OUT}/vee_stage1.pt"
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


def train() -> Vee:
    source = os.environ.get("SOURCE", "both")
    hp, hq, wrong = graded_data()
    t = time.time()
    o = Vee(0, k=int(os.environ.get("K", 64)))
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
    org = Vee.load(ORG) if os.path.exists(ORG) else train()
    print("[loading the reflex brain...]", flush=True)
    reflex = Reflex()
    mem = None
    if os.environ.get("MEMORY", "1") != "0":
        print("[waking Vee's memory: replaying her childhood and your nights...]", flush=True)
        import vee_memory
        mem = vee_memory.wake()
        print(f"[she holds {len(mem.facts)} facts, {len(mem.day())} of them from today]")
    wiki = None
    last = None

    def teach(answer: str) -> None:
        same = next((i for i, f in enumerate(mem.facts)              # same answer = another wording of that fact
                     if not f.get("pilot") and f["answer"].casefold() == answer.casefold()), None)
        mem.teach(last["q"], answer, last["h"], same_as=same)
        vee_memory.save_user(mem)
        print(f"[heard once: \"{last['q']}\" -> {answer}. In today's memory; /sleep moves it to long-term]"
              + (f"\n[kept as another wording of: \"{mem.facts[same]['q']}\"]" if same is not None else ""))

    def ask_reflex(line: str, h) -> dict | None:
        t = time.time()
        print("[thinking...]", end="\r", flush=True)
        try:
            a = reflex.answer(chat_prompt(line))
        except Exception as e:                       # keep the session alive
            print(f"[reflex failed: {e!r}]"); return None
        d = float(org.doubt(pick(org.source, a["h_pre"], a["h_post"]).unsqueeze(0))[0])
        pct = float((org.doubt_sorted < d).float().mean()) * 100
        conf = float(np.exp(np.mean(a["logprobs"]))) if a["logprobs"] else 0.0
        word = "low" if d < 0.35 else "medium" if d < 0.65 else "HIGH"
        print(f"reflex> {a['text']}")
        print(f"        reflex confidence {conf:.2f} | organism doubt {d:.2f} ({word}; "
              f"higher than {pct:.0f}% of graded answers) | {time.time() - t:.1f}s")
        return {"q": line, "text": a["text"], "doubt": d, "h_pre": a["h_pre"], "h_post": a["h_post"], "h": h}

    print("Ask a question. /right /wrong [answer] /teach <answer> /sleep /reflex /facts /look [query] /learn /quit")
    while True:
        try:
            line = input("\nyou> ").strip()
        except EOFError:
            break
        if not line:
            continue
        cmd, _, arg = line.partition(" ")
        if cmd in ("/quit", "/exit", "/q"):
            break
        if cmd.startswith("/") and cmd not in ("/right", "/wrong", "/learn", "/look", "/teach", "/sleep", "/reflex", "/facts"):
            print("[unknown command; use /right /wrong [answer] /teach <answer> /sleep /reflex /facts /look [query] /learn /quit]"); continue
        if cmd in ("/teach", "/sleep", "/facts") and mem is None:
            print("[memory is off (MEMORY=0)]"); continue
        if cmd == "/facts":
            day = mem.day()
            print(f"[{len(mem.facts)} facts: {len(mem.facts) - len(day)} long-term, {len(day)} from today; "
                  f"{sum(not f.get('pilot') for f in mem.facts)} taught by you]")
            for i in day: print(f"   today: {mem.facts[i]['q']} -> {mem.facts[i]['answer']}")
            continue
        if cmd == "/sleep":
            r = mem.sleep()
            vee_memory.save_user(mem)
            print(f"[night: {r['fed']} facts replayed, {r['kept']} now in long-term memory"
                  + (f", {r['fed'] - r['kept']} stay in today's memory for the next night" if r["fed"] > r["kept"] else "")
                  + f"; {r['s']:.1f}s]")
            continue
        if cmd == "/teach":
            if last is None or not arg.strip():
                print("[ask the question first, then /teach <answer>]"); continue
            teach(arg.strip()); continue
        if cmd == "/reflex":
            if last is None:
                print("nothing asked yet"); continue
            last = ask_reflex(last["q"], last["h"]) or last
            continue
        if cmd in ("/right", "/wrong"):
            if last is None or "h_pre" not in last:
                print("nothing from the reflex to grade (after a remembered answer use /reflex or /teach <answer>)"); continue
            save_verdict({"q": last["q"], "answer": last["text"], "correct": cmd == "/right",
                          "gold": arg.strip(), "doubt": last["doubt"], "t": time.time()},
                         last["h_pre"], last["h_post"])
            print(f"[stored: reflex was {'right' if cmd == '/right' else 'wrong'}"
                  + (f"; correct answer: {arg.strip()}" if arg.strip() else "") + "]")
            if cmd == "/wrong" and arg.strip() and mem is not None:
                teach(arg.strip())
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
        if len(reflex.tokens(chat_prompt(line))) > 480:
            print("[question too long for the reflex (limit ~400 words)]"); continue
        h = None
        if mem is not None:
            k = reflex.read(line)
            h = torch.cat([torch.tensor(k["h_last"]), torch.tensor(k["h_mean"])]).float()
            r = mem.recall(h)
            if r is not None:
                print(f"vee> {r['answer']}")
                print(f"     from {r['where']} memory; she was taught: \"{r['q']}\"   (/reflex asks the reflex instead)")
                last = {"q": line, "h": h}
                continue
        last = ask_reflex(line, h) or last


if __name__ == "__main__":
    main()
