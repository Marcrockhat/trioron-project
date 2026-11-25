"""Stage 1 — the embryo (docs/design/vee_development.md §7).  Her senses form and she
finds the regularities of her world by herself: nobody addresses her, nothing is
labelled, she cannot act.

    python3 experiments/incubator/embryo.py live     genesis (if she does not exist) and DAYS days of the scripted
                                                     world; saved after every simulated day; resumable
    python3 experiments/incubator/embryo.py gate     measure the gate (§7.3) on TEST_DAYS further days she does
                                                     not live; nothing in her changes

The stage file (vee_body.HOME/stage) must say ``embryo``.
World: world.py (fully simulated).  Ambient language: in P_TEXT of the moments a random
Wikipedia sentence is present (outputs/incubator/wiki_sent.*, read by the reflex).
Log: HOME/embryo_log.pt — every moment's input, hidden world state (for the gate only;
never in her input), sentence index, and per-day class count and wall time.
Alarm (stop, state kept): a simulated day takes more than DAY_BUDGET seconds (an absolute
budget, matron's rule 9).  The class count is reported, not gated (Rocky s067: it need not
settle now; cost and classes in use are watched).

Set at genesis only, then carried by her body for life (s068, plan §11):
    CORRIDOR=0.6   her world varies inside a corridor (world.py); 0 = Vee's world
    BLUR=1         her organs start blurred and sharpen slowly (blur.py); 0 = Vee's organs;
                   2 = blurred, and a silent reading ("none") is a state of its own

Env: DAYS=30 TEST_DAYS=5 P_TEXT=0.5 SEED=0 DAY_BUDGET=30  VEE_HOME=outputs/incubator/vee
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(__file__))
import vee_body as vb
from germline_checks import adjusted_rand, kmeans, readable
from world import HIDDEN, PER_DAY, World

E = lambda k, d: type(d)(os.environ.get(k, d))
DAYS, TEST, P_TEXT, SEED = E("DAYS", 30), E("TEST_DAYS", 5), E("P_TEXT", 0.5), E("SEED", 0)
CORRIDOR, BLUR, DAY_BUDGET = E("CORRIDOR", 0.0), E("BLUR", 0), E("DAY_BUDGET", 30.0)
LOG = f"{vb.HOME}/embryo_log.pt"


def day_of(world: World, g: np.random.Generator, ambient: torch.Tensor):
    """One simulated day: inputs [96, 148], hidden states [96, 5], sentence index [96] (-1 = no text)."""
    X, Hd, T = [], [], []
    for _ in range(PER_DAY):
        raw, hid = world.step()
        i = int(g.integers(len(ambient))) if g.random() < P_TEXT else -1
        X.append(vb.moment(ambient[i] if i >= 0 else None, raw))
        Hd.append([hid[h] for h in HIDDEN]); T.append(i)
    return torch.stack(X), torch.tensor(Hd).long(), torch.tensor(T)


def live(ambient):
    if os.path.exists(f"{vb.HOME}/body.pt"):
        body, log = vb.Body.load(), torch.load(LOG, weights_only=False)
    else:
        body = vb.Body(SEED, vb.genesis_pool(SEED, ambient, p_text=P_TEXT, corridor=CORRIDOR), blur={0: False, 1: True, 2: "none"}[BLUR])
        body.extra = {"world": World(SEED, corridor=CORRIDOR), "rng": np.random.default_rng(1000 + SEED), "day": 0}
        log = {"X": [], "hidden": [], "text": [], "classes": [], "ear": [], "secs": [], "names": []}
        print(f"[genesis: {body.receptors} receptors of {vb.WIDTH}; world corridor {CORRIDOR}; organs {('sharp', 'blurred', 'blurred, with none')[BLUR]}]", flush=True)
    ex = body.extra
    while ex["day"] < DAYS:
        t0 = time.time()
        X, Hd, T = day_of(ex["world"], ex["rng"], ambient)
        for i in range(0, PER_DAY, vb.WINDOW):
            body.live(X[i:i + vb.WINDOW])
        ex["day"] += 1
        log["X"].append(X); log["hidden"].append(Hd); log["text"].append(T)
        log["classes"].append(body.classes["soma"]); log["ear"].append(body.classes["ear"]); log["secs"].append(time.time() - t0)
        if ex["day"] in (20, DAYS): log["names"].append((ex["day"], probe_names(body, ambient)))
        body.save(); torch.save(log, LOG + ".tmp"); os.replace(LOG + ".tmp", LOG)
        print(f"  day {ex['day']:3}: classes {body.classes}, {log['secs'][-1]:.1f} s{sharpness(body)}", flush=True)
        if log["secs"][-1] > DAY_BUDGET:
            raise SystemExit(f"ALARM: day {ex['day']} took {log['secs'][-1]:.1f} s, over the budget of {DAY_BUDGET:.0f} s; stopped, state kept")
    return body, log


def sharpness(body) -> str:
    if body.blur is None: return ""
    return ", steps taken " + str({k: len(b.history) for k, b in body.blur.items()})


def probe(ambient, corridor: float = 0.0):
    """A fixed set of moments she never lives (another world run), for stability and the gate.
    The rare state is made to occur: she is elsewhere from the start of the third day
    (it then lasts as long as the world lets it), so home / elsewhere can be scored."""
    w, g = World(9000 + SEED, corridor=corridor), np.random.default_rng(9500 + SEED)
    days = []
    for d in range(TEST):
        if d == 2: w.elsewhere = True
        days.append(day_of(w, g, ambient))
    return torch.cat([d[0] for d in days]), torch.cat([d[1] for d in days]), torch.cat([d[2] for d in days])


def probe_names(body, ambient) -> list[str]:
    names = body.class_names()
    return [names[int(c)] if c >= 0 else "-" for c in body.read(probe(ambient, body.extra["world"].corridor)[0])[0]]


def gate(ambient):
    body, log = vb.Body.load(), torch.load(LOG, weights_only=False)
    X, Hd = torch.cat(log["X"]), torch.cat(log["hidden"])
    Xp, Hp, Tp = probe(ambient, body.extra["world"].corridor)
    n, k = len(X), body.classes["soma"]
    cls = body.read(torch.cat([X, Xp]))[0]
    km = torch.cdist(torch.cat([X, Xp])[:, vb.LANG:], kmeans(X[:, vb.LANG:], k, SEED)).argmin(1)   # control on the numbers the soma reads
    Hall = torch.cat([Hd, Hp])
    print(f"# embryo gate: {body.extra['day']} days lived, {TEST} unseen days; soma {k} classes, ear {body.classes['ear']}; receptors {body.receptors}; "
          f"{vb.WINDOW} moments a sitting, text in {P_TEXT} of moments; world corridor {body.extra['world'].corridor}")
    if body.blur is not None:
        import environment
        for o, b in body.blur.items():
            name = (lambda j: j) if o == "ear" else (lambda j: environment.NAMES[j - 64 + vb.LANG] if j < 33 else f"channel {j + vb.LANG}")
            st = b.steps[b.varies]
            print(f"blur, {o}: {len(b.history)} steps taken; readings at 2 / 3 / more steps: {int((st == 2).sum())} / {int((st == 3).sum())} / {int((st > 3).sum())}; "
                  f"sharpened (sitting, reading, steps): {[(s, name(j), k) for s, j, k in b.history]}")
    print(f"{'state':22} {'her classes':>11} {'k-means':>8} {'base':>6}   verdict (clearly above base, and not more than .05 under k-means)")
    ok = True
    for j, h in enumerate(HIDDEN):
        a, b = readable(cls, Hall[:, j], n); c, _ = readable(km, Hall[:, j], n)
        v = a > b + 0.05 and a >= c - 0.05
        if h == "night":                                          # Rocky s067: night is not meaningful for her yet
            print(f"{h:22} {a:>11.2f} {c:>8.2f} {b:>6.2f}   (reported, not gated — Rocky s067)"); continue
        ok &= v
        print(f"{h:22} {a:>11.2f} {c:>8.2f} {b:>6.2f}   {'pass' if v else 'FAIL'}")
    win = Hall[:, HIDDEN.index("window")] == 1                    # night, seen through the open window only
    nw = int(win[:n].sum())
    a, b = readable(cls[win], Hall[win][:, 0], nw); c, _ = readable(km[win], Hall[win][:, 0], nw)
    print(f"{'night, window open':22} {a:>11.2f} {c:>8.2f} {b:>6.2f}   (reported, not in the gate as written)")
    present = (torch.cat([torch.cat(log["text"]), Tp]) >= 0).long()
    ear = body.read(torch.cat([X, Xp]), "ear")[0]
    a, b = readable(ear, present, n)
    print(f"{'text present (ear)':22} {a:>11.2f} {'':>8} {b:>6.2f}   (reported)")
    cc = log["classes"]; third = cc[len(cc) * 2 // 3 - 1]
    grow = cc[-1] / third - 1
    print(f"class count (reported, not gated — Rocky s067): {third} at day {len(cc) * 2 // 3} -> {cc[-1]} at day {len(cc)} = {grow:+.0%}; ear {log['ear'][-1]}")
    if len(log["names"]) == 2:
        (d0, a0), (d1, a1) = log["names"]
        same = np.mean([x == y for x, y in zip(a0, a1)])
        u = {s: i for i, s in enumerate(dict.fromkeys(a0 + a1))}
        ari = adjusted_rand(torch.tensor([u[s] for s in a0]), torch.tensor([u[s] for s in a1]))
        print(f"stability (reported; Rocky s067 accepted 'same grouping' in place of 'same class', no bar set): probe moments day {d0} -> day {d1}: "
              f"same grouping (adjusted Rand) {ari:.2f}; same class {same:.2f}")
    else:
        print("stability: not measured (needs day 20 and the last day)")
    used = cls[n:].unique().numel(); top3 = torch.bincount(cls[n:].clamp(min=0)).topk(min(3, k)).values.sum().item() / len(Xp)
    print(f"classes used on unseen days: {used} of {k}; share of unseen moments in her 3 biggest classes {top3:.2f}")
    print(f"silent channels: all {vb.RES} reserved channels read exactly zero through the embryo: {bool((X[:, vb.WIDTH - vb.RES:] == 0).all())}")
    live = cls[max(0, n - 10 * PER_DAY):n].unique().numel()
    print(f"classes she used in her last 10 days: {live} of {k}")
    print(f"mean day {np.mean(log['secs']):.1f} s (first 5: {np.mean(log['secs'][:5]):.1f}, last 5: {np.mean(log['secs'][-5:]):.1f})")
    print(f"\nGATE: {'all criteria pass' if ok else 'NOT passed'} — for Rocky to read; no stage is promoted by code")


if __name__ == "__main__":
    vb.require_stage("embryo")
    ambient = vb.language_sense()(vb.wiki_states())
    {"live": live, "gate": gate}[sys.argv[1]](ambient)
