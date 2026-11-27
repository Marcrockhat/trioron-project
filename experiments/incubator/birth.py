"""Stage 3 — birth: the first exchange, and her hunger (docs/design/vee_development.md §16).

From birth she is addressed.  Her world goes on as in the fetal stage (the same course of
days, kicks, ambient text, her random channel if she has one); in EXCHANGES moments of a day
the matron is present and says something to her.  She replies FIRST, from her ear alone: the
reply her ear's class holds now (the name its present members give it), or nothing.  Then the
matron replies to what she actually said — the right reply for that wording — and that is
what she lives (a label on that moment, to the ear only; never batch-trained).

Hunger (Rocky s073: "wire her hunger ... once she's born and learning. Check the aidos
philosophy"; project-aidos docs/aidos_philosophy.md §4, aidos/hunger.py).  A NUMA is a right
reply, before she is told, to a wording she has not heard lately: what she learned stayed.
A MIMA is a wrong reply to such a wording on something she had already answered right: it
looked learned and went away.  Her hunger for a thing rises with the days since its last
Numa.  The day's exchanges go where she is hungriest, and a new thing is opened only when
nothing she already has is hungry (thorough before broad, rule 19).

    python3 experiments/incubator/birth.py live     DAYS further days; saved after every day; resumable
    python3 experiments/incubator/birth.py gate     the gate (§16.4) on unseen days; nothing in her changes
    python3 experiments/incubator/birth.py talk     speak to her (needs the reflex); nothing in her changes

The stage file must say ``birth``.  Log: HOME/birth_log.pt and HOME/birth_exchanges.jsonl.
Env: DAYS=30 EXCHANGES=24 DAY_BUDGET=900 TEST_DAYS=5 HORIZON=3 NEED=3 FRESH=5 WHO=matron|rocky (talk)
"""
from __future__ import annotations
import copy, json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as Fn
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(__file__))
import vee_body as vb
import fetal as F
from world import PER_DAY

E = lambda k, d: type(d)(os.environ.get(k, d))
DAYS, EXCHANGES, DAY_BUDGET, TEST = E("DAYS", 30), E("EXCHANGES", 24), E("DAY_BUDGET", 900.0), E("TEST_DAYS", 5)
HORIZON = E("HORIZON", 3.0)     # days: how fast hunger rises after a Numa (aidos/hunger.py's saturating form)
NEED = E("NEED", 3)             # a thing is fed when its Numa less its Mima reach this, and it is not hungry
FRESH = E("FRESH", 5)           # a wording not heard for this many days asks again whether it stayed
HUNGRY = 0.5                    # hunger above this = hungry (about two days without a Numa at HORIZON=3)
LOG, TALK = f"{vb.HOME}/birth_log.pt", f"{vb.HOME}/birth_exchanges.jsonl"
UNKNOWN, SILENT = "(I do not know that yet.)", "(says nothing)"
ORDER = ("name", "what", "maker", "matron", "born", "memory")      # herself first, then the matron (rule 19)


def material(name: str):
    """What the matron has to say at this stage (written by the matron in s066, read by the
    reflex then): six things about herself in 24 wordings each, 6 more wordings held back for
    the gate; and things she is not taught, to which the right reply is "I do not know"."""
    rows = [json.loads(l) for l in open(f"{vb.OUT}/self_wordings.jsonl")]
    keys = torch.load(f"{vb.OUT}/self_keys.pt", weights_only=False)
    S = vb.language_sense()(torch.stack([keys[r["text"]] for r in rows]))
    U = len(ORDER)
    lab, held, k = [], [], 0
    for r in rows:
        if r["kind"] in ("train", "test"):
            lab.append(ORDER.index(r["fact"])); held.append(r["kind"] == "test")
        else:                                               # near / other: every third is held back for the gate
            lab.append(U); held.append(k % 3 == 2); k += 1
    answers = {ORDER.index(r["fact"]): (name if r["fact"] == "name" else r["answer"]) for r in rows if r["kind"] == "train"}
    answers[U] = UNKNOWN
    return [r["text"] for r in rows], S, torch.tensor(lab), torch.tensor(held), answers


def say(body, X: torch.Tensor) -> list[int]:
    """Her reply to each moment, from the ear alone, no learning: the label its class holds now; -1 = nothing."""
    cls = body.read(X, "ear")[0]
    names = body.ear.member_names()
    return [int(names[c]) if c >= 0 else -1 for c in cls.tolist()]


def hunger(st: dict, f: int) -> float:
    last = st["ledger"][f]["last"]
    return 1.0 if last is None else 1 - math.exp(-(st["day"] - last) / HORIZON)


def fed(st: dict, f: int) -> bool:
    l = st["ledger"][f]
    return l["numa"] - l["mima"] >= NEED and hunger(st, f) < HUNGRY


def plan_day(st: dict, lab, held, g) -> list[int]:
    """Which wordings are said today.  A quarter are things she is not taught; the rest go to
    the things she has, in proportion to her hunger for each.  For each thing the wordings
    that have rested (never heard, or not for FRESH days) come first — they ask whether it
    stayed; the rest of the day's share repeats the wordings heard most lately, so the others
    go on resting."""
    U = len(ORDER)
    if all(fed(st, f) for f in st["open"]) and len(st["open"]) < U:
        st["open"].append(len(st["open"])); st["opened"].append(st["day"])
    w = np.array([hunger(st, f) + 0.05 for f in st["open"]])
    n_unknown = EXCHANGES // 4
    share = g.multinomial(EXCHANGES - n_unknown, w / w.sum())
    out = []
    for f, n in list(zip(st["open"], share)) + [(U, n_unknown)]:
        idx = ((lab == f) & ~held).nonzero().squeeze(1).tolist()
        rested = lambda i: st["day"] - st["heard"].get(i, -10 ** 6) >= FRESH
        idx.sort(key=lambda i: (0, st["heard"].get(i, -10 ** 6), g.random()) if rested(i) else (1, -st["heard"][i], g.random()))
        if f < U: n = max(n, 1) if n else 0
        out += idx[:n]
    return [out[i] for i in g.permutation(len(out))]


def live(ambient):
    body = vb.Body.load()
    ex = body.extra
    F.own_channel(body)
    texts, S, lab, held, answers = material(ex.get("name") or os.environ.get("NAME") or os.path.basename(os.path.normpath(vb.HOME)).capitalize())
    U = len(ORDER)
    if "birth" not in ex:
        ex["name"] = answers[0]
        ex["birth"] = {"day": 0, "open": [0], "opened": [0], "heard": {}, "rng": np.random.default_rng(6000 + body.seed),
                       "ledger": {f: {"numa": 0, "mima": 0, "last": None, "right": False} for f in range(U)}}
        log = {"X": [], "hidden": [], "text": [], "action": [], "said": [], "classes": [], "secs": [], "ledger": []}
        print(f"[born: she is {ex['day']} days old; her name is {ex['name']}]", flush=True)
    else:
        log = torch.load(LOG, weights_only=False)
    st = ex["birth"]
    while st["day"] < DAYS:
        t0 = time.time()
        X, Hd, T, A = F.day_of(ex["world"], ex["rng"], ex["kicks"], ambient, ex.get("noise") if F.NOISE_CH >= 0 else None)
        today = plan_day(st, lab, held, st["rng"])
        at = np.sort(st["rng"].choice(PER_DAY, len(today), replace=False))
        labels: list = [None] * PER_DAY
        for m, i in zip(at, today):                           # the matron is present and says wording i at moment m
            X[m, :vb.LANG] = vb.LANG_SCALE * S[i]; X[m, 97:100] = torch.tensor(vb.MATRON)
            T[m] = -2 - i; labels[m] = f"g{int(lab[i])}"
        said = torch.full((PER_DAY,), -9)
        right = n_ex = 0
        with open(TALK, "a") as out:
            for s in range(0, PER_DAY, vb.WINDOW):
                rep = say(body, X[s:s + vb.WINDOW])           # she replies first ...
                for m, i in zip(at, today):
                    if not s <= m < s + vb.WINDOW: continue
                    r, want = rep[m - s], int(lab[i])
                    ok = r == want or (want == U and r == -1)  # saying nothing to what she is not taught is right too
                    said[m] = r; right += ok; n_ex += 1
                    fresh = st["day"] - st["heard"].get(i, -10 ** 6) >= FRESH
                    if want < U and fresh:                     # did it stay?  (the ledger)
                        l = st["ledger"][want]
                        if ok: l["numa"] += 1; l["last"] = st["day"]; l["right"] = True
                        elif l["right"]: l["mima"] += 1
                    st["heard"][i] = st["day"]
                    out.write(json.dumps({"day": st["day"] + 1, "moment": int(m), "matron": texts[i], "she": answers.get(r, SILENT),
                                          "right": bool(ok), "matron_replies": answers[want], "fresh": bool(fresh)}, ensure_ascii=False) + "\n")
                body.live(X[s:s + vb.WINDOW], labels=labels[s:s + vb.WINDOW])     # ... then lives what the matron replied
        ex["day"] += 1; st["day"] += 1
        for key, v in (("X", X), ("hidden", Hd), ("text", T), ("action", A), ("said", said)): log[key].append(v)
        log["classes"].append(dict(body.classes)); log["secs"].append(time.time() - t0); log["ledger"].append(copy.deepcopy(st["ledger"]))
        body.save(); torch.save(log, LOG + ".tmp"); os.replace(LOG + ".tmp", LOG)
        net = {ORDER[f]: st["ledger"][f]["numa"] - st["ledger"][f]["mima"] for f in st["open"]}
        hung = max(st["open"], key=lambda f: hunger(st, f))
        print(f"  day {st['day']:3}: right before told {right:2} of {n_ex}; has {[ORDER[f] for f in st['open']]}; net (Numa - Mima) {net}; "
              f"hungriest {ORDER[hung]} {hunger(st, hung):.2f}; classes {body.classes}, {log['secs'][-1]:.1f} s", flush=True)
        if log["secs"][-1] > DAY_BUDGET:
            raise SystemExit(f"ALARM: day {st['day']} took {log['secs'][-1]:.1f} s, over the budget of {DAY_BUDGET:.0f} s; stopped, state kept")


def gate(ambient) -> bool:
    body = vb.Body.load()
    F.own_channel(body)
    st, U = body.extra["birth"], len(ORDER)
    texts, S, lab, held, answers = material(body.extra["name"])
    F.CORRIDOR = body.extra["world"].corridor
    Xp = F.probe(ambient, body.seed)[0]                       # days she never lives
    has = torch.tensor([int(l) in st["open"] or int(l) == U for l in lab])
    test = (held & has).nonzero().squeeze(1)
    g = np.random.default_rng(9900 + body.seed)
    at = torch.tensor(g.choice(len(Xp), len(test), replace=False))
    Q = Xp[at].clone(); Q[:, :vb.LANG] = vb.LANG_SCALE * S[test]; Q[:, 97:100] = torch.tensor(vb.MATRON)
    she = torch.tensor(say(body, Q)); want = lab[test]
    taught = (~held & has).nonzero().squeeze(1)               # control: the mean of what she was told, per reply (cosine, sharp numbers)
    ids = lab[taught].unique()
    mu = torch.stack([S[taught][lab[taught] == i].mean(0) for i in ids])
    ctl = ids[(Fn.normalize(S[test], dim=1) @ Fn.normalize(mu, dim=1).T).argmax(1)]
    print(f"# birth gate: {st['day']} days since birth ({body.extra['day']} days old); she has {[ORDER[f] for f in st['open']]} "
          f"(opened on days {st['opened']}); classes {body.classes}; {len(test)} wordings she has never heard, on {TEST} unseen days")
    ok = True
    def row(name, m, right_she, right_ctl, bar, gated=True):
        nonlocal ok
        if not bool(m.any()): print(f"{name:44} (none)"); return
        a, c = right_she[m].float().mean().item(), right_ctl[m].float().mean().item()
        v = a >= bar and a >= c - 0.10
        if gated: ok &= v
        print(f"{name:44} hers {a:.2f}  class mean {c:.2f}  n {int(m.sum()):3}   " + (('pass' if v else 'FAIL') + f" (>= {bar:.2f}, and not more than .10 under the control)" if gated else "(reported)"))
    fact = want < U
    row("1 replies right, wordings never heard", fact, she == want, ctl == want, 0.80)
    row("2 her earliest memory (her name)", want == 0, she == want, ctl == want, 0.80)
    row("3 not taught: says so or says nothing", ~fact, (she == U) | (she == -1), ctl == U, 0.80)
    for f in st["open"]:
        row(f"  {ORDER[f]}", want == f, she == want, ctl == want, 0, gated=False)
    silent = (she[fact] == -1).float().mean().item() if bool(fact.any()) else float("nan")
    print(f"  of the things she has: silent {silent:.2f}, says she does not know {(she[fact] == U).float().mean().item():.2f}")
    led = st["ledger"]
    numa, mima = sum(led[f]["numa"] for f in st["open"]), sum(led[f]["mima"] for f in st["open"])
    v = numa - mima > 0 and all(led[f]["numa"] - led[f]["mima"] > 0 for f in st["open"][:-1]); ok &= v
    print(f"4 what she learned stayed (her life, not this test): Numa {numa}, Mima {mima}, net {numa - mima:+d}; per thing "
          f"{ {ORDER[f]: (led[f]['numa'], led[f]['mima']) for f in st['open']} }; hunger now { {ORDER[f]: round(hunger(st, f), 2) for f in st['open']} }   "
          f"{'pass' if v else 'FAIL'} (net above zero for every thing but the newest)")
    log = torch.load(LOG, weights_only=False)
    said, T = torch.cat(log["said"][-5:]), torch.cat(log["text"][-5:])
    m = said != -9
    wantl = lab[(-2 - T[m]).long()]
    print(f"  in her last 5 days: right before told {((said[m] == wantl) | ((wantl == U) & (said[m] == -1))).float().mean().item():.2f} of {int(m.sum())} exchanges")
    print(f"  cost: mean day {np.mean(log['secs']):.1f} s (last 5: {np.mean(log['secs'][-5:]):.1f}); classes {log['classes'][0]} -> {log['classes'][-1]}")
    print("\n# 5 nothing lost: the fetal gate, read on her as she is now")
    v = F.gate(ambient); ok &= v
    print(f"\nBIRTH GATE: {'all criteria pass' if ok else 'NOT passed'} — for Rocky to read; no stage is promoted by code")
    return bool(ok)


def talk(ambient):
    """Speak to her.  She replies from her ear; nothing in her changes.  What she does not
    know is put to the reflex and looked up in the library, and shown beside her reply."""
    from reflex import Reflex, chat_prompt
    from wiki import Wiki
    from console import keywords
    body = vb.Body.load()
    _, _, _, _, answers = material(body.extra["name"])
    sense, reflex, wiki = vb.language_sense(), Reflex(), Wiki()
    who = vb.ROCKY if os.environ.get("WHO", "rocky") == "rocky" else vb.MATRON
    world = copy.deepcopy(body.extra["world"])
    print(f"[{body.extra['name']}, {body.extra['day']} days old. Say something; an empty line leaves.]")
    while True:
        try: line = input("\nyou> ").strip()
        except EOFError: break
        if not line: break
        k = reflex.read(line)
        h = torch.cat([torch.tensor(k["h_last"]), torch.tensor(k["h_mean"])]).float()
        x = vb.moment(sense(h[None])[0], world.step()[0], who, action=0)
        r = say(body, x[None])[0]
        print(f"{body.extra['name'].lower()}> {answers.get(r, SILENT)}")
        if r == -1 or answers.get(r) == UNKNOWN:
            print(f"   the reflex says: {reflex.answer(chat_prompt(line))['text']}")
            hits = wiki.search(keywords(line), 1)
            if hits:
                e = wiki.by_path(hits[0]); ps = wiki.paragraphs(e)
                print(f"   the library, under '{e.title}': {ps[0][:300] if ps else '(no text)'}")


if __name__ == "__main__":
    vb.require_stage("birth")
    ambient = vb.language_sense()(vb.wiki_states())
    {"live": live, "gate": gate, "talk": talk}[sys.argv[1]](ambient)
