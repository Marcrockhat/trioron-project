"""Stage 2a — fetal: body and first kicks (docs/design/vee_development.md §8).
Her action echo comes alive, a sense of change (kin) is born, and kicks fire by
themselves about one moment in six.  She does not choose (2b is not designed).
Nobody addresses her; nothing is labelled.  All consequences are scripted (world.py).

    python3 experiments/incubator/fetal.py live    DAYS further days; saved after every simulated day; resumable
    python3 experiments/incubator/fetal.py gate    the gate (§8.6) on TEST_DAYS days she does not live; nothing changes

An organism with a fourth organ (trioron/organs/fourth.py) is read twice (Rocky s072: "gate
reads both"): from the working organs alone, and with the held children of the fourth organ
added.  The gate is passed when one of the two readings passes every row.
The relative gate (Rocky s072: "put a probe when the gate passed and then the fetal state is
passed"): with PROBE=n the gate is read every n days while living; at the first reading that
passes, a copy is kept, living stops and the organism is HELD for Rocky.  No stage is promoted.

The stage file (vee_body.HOME/stage) must say ``fetal``.  She continues from the saved
embryo (body.pt): same world, same ambient text stream.
Log: HOME/fetal_log.pt — every moment's input, hidden world state, sentence index, kick,
and per-day class counts and wall time.
Alarm (stop, state kept): a simulated day takes more than DAY_BUDGET seconds (Rocky s067:
let her finish; the first alarm, twice day 5's 0.3 s, stopped her on day 17 at 2.1 s).

Env: DAYS=30 TEST_DAYS=5 P_TEXT=0.5 P_KICK=0.1667 DAY_BUDGET=30
     PROBE=0       read the gate every PROBE days while living and stop at the first pass (0 = off)
     NOISE_CH=-1   a probe for COPIES only (Rocky s072: "add one factor that only filled by random
                   numbers"): that input channel (a spare one, 140..147) carries a fresh random number
                   every moment, from a stream of its own, so the world and the kicks are unchanged
"""
from __future__ import annotations
import contextlib, io, math, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(__file__))
import vee_body as vb
from germline_checks import kmeans, readable
from world import ACTIONS, HIDDEN, PER_DAY, World
from trioron.pcll import N_QUANTA

E = lambda k, d: type(d)(os.environ.get(k, d))
DAYS, TEST, P_TEXT, P_KICK = E("DAYS", 30), E("TEST_DAYS", 5), E("P_TEXT", 0.5), E("P_KICK", 1 / 6)
DAY_BUDGET = E("DAY_BUDGET", 30.0)
NOISE_CH = E("NOISE_CH", -1)
PROBE = E("PROBE", 0)
assert NOISE_CH < 0 or "/copies/" in os.path.abspath(vb.HOME), "the random channel is tried on copies only (rule 27)"
CORRIDOR = 0.0              # set from her own world in gate(): unseen days come from the same kind of world
LOG = f"{vb.HOME}/fetal_log.pt"
EMBRYO = {"busy": 0.99, "full": 0.95, "elsewhere": 1.00, "window": 1.00}     # Vee's embryo gate values (plan §7.3a)
EAR_EMBRYO = 0.75
_own = f"{vb.HOME}/embryo_values.json"      # another organism's own embryo gate values, written by hand from her gate log
if os.path.exists(_own):
    import json
    EMBRYO = json.load(open(_own)); EAR_EMBRYO = EMBRYO.pop("ear")
# what each acting kick really moves, as columns of the kin organ's 35 change numbers
# (33 environment readings, then held, marks); environment.NAMES gives the order
MOVES = {"work": {8, 10}, "hold": {5, 6, 26, 33}, "let go": {5, 6, 26, 33}, "mark": {34}, "wipe": {34}}


def day_of(world: World, g: np.random.Generator, k: np.random.Generator, ambient: torch.Tensor, noise: np.random.Generator | None = None):
    """One simulated day with spontaneous kicks: inputs [96, 148], hidden [96, 5], sentence [96], kick [96]."""
    X, Hd, T, A = [], [], [], []
    for _ in range(PER_DAY):
        ok = [a for a in range(1, len(ACTIONS)) if world.can(a)]
        a = int(k.choice(ok)) if k.random() < P_KICK else 0
        raw, hid = world.step(a)
        i = int(g.integers(len(ambient))) if g.random() < P_TEXT else -1
        X.append(vb.moment(ambient[i] if i >= 0 else None, raw, action=a))
        if noise is not None: X[-1][NOISE_CH] = float(noise.standard_normal())
        Hd.append([hid[h] for h in HIDDEN]); T.append(i); A.append(a)
    return torch.stack(X), torch.tensor(Hd).long(), torch.tensor(T), torch.tensor(A)


def live(ambient):
    body = vb.Body.load()
    ex = body.extra
    if "kin" not in body.organs():
        body.grow_kin(vb.kin_pool(body.seed, corridor=ex["world"].corridor))
        ex["kicks"], ex["fetal_day"] = np.random.default_rng(4000 + body.seed), 0
        log = {"X": [], "hidden": [], "text": [], "action": [], "classes": [], "secs": []}
        print(f"[kin is born: {body.receptors['kin']} receptors of 43; she is {ex['day']} days old]", flush=True)
    else:
        log = torch.load(LOG, weights_only=False)
    while ex["fetal_day"] < DAYS:
        t0 = time.time()
        if NOISE_CH >= 0 and "noise" not in ex: ex["noise"] = np.random.default_rng(5000 + body.seed)
        X, Hd, T, A = day_of(ex["world"], ex["rng"], ex["kicks"], ambient, ex.get("noise") if NOISE_CH >= 0 else None)
        for i in range(0, PER_DAY, vb.WINDOW):
            body.live(X[i:i + vb.WINDOW])
        ex["day"] += 1; ex["fetal_day"] += 1
        for key, v in (("X", X), ("hidden", Hd), ("text", T), ("action", A)): log[key].append(v)
        log["classes"].append(dict(body.classes)); log["secs"].append(time.time() - t0)
        body.save(); torch.save(log, LOG + ".tmp"); os.replace(LOG + ".tmp", LOG)
        print(f"  fetal day {ex['fetal_day']:3}: classes {body.classes}, {log['secs'][-1]:.1f} s", flush=True)
        if log["secs"][-1] > DAY_BUDGET:
            raise SystemExit(f"ALARM: fetal day {ex['fetal_day']} took {log['secs'][-1]:.1f} s, over the budget of {DAY_BUDGET:.0f} s; stopped, state kept")
        if PROBE and ex["fetal_day"] % PROBE == 0:               # the relative gate: read on a saved copy, nothing of her changes
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                passed = gate(ambient)
            open(f"{vb.HOME}/gate_fetal_day{ex['fetal_day']}.log", "w").write(out.getvalue())
            print(f"  gate read at fetal day {ex['fetal_day']}: {'PASSED' if passed else 'not passed'}", flush=True)
            if passed:
                body.save(f"body_fetal_gate_day{ex['fetal_day']}")
                print(f"HELD at fetal day {ex['fetal_day']}: the gate is passed; for Rocky to read, no stage is promoted by code", flush=True)
                return

def probe(ambient, seed):
    """Days she never lives (another world run, with kicks); she is elsewhere from the third day."""
    w, g, k = World(9100 + seed, corridor=CORRIDOR), np.random.default_rng(9600 + seed), np.random.default_rng(9700 + seed)
    nz = np.random.default_rng(9800 + seed) if NOISE_CH >= 0 else None
    days = []
    for d in range(TEST):
        if d == 2: w.elsewhere = True
        days.append(day_of(w, g, k, ambient, nz))
    return [torch.cat([d[j] for d in days]) for j in range(4)]


def recall_per_value(pred, y):
    return float(np.mean([(pred[y == v] == v).float().mean().item() for v in y.unique()]))


def vote(cls_tr, y_tr, cls_te, n_values):
    """Each class votes its majority value on the lived moments; predictions for the unseen ones."""
    default = int(torch.bincount(y_tr, minlength=n_values).argmax())
    pred = torch.full((len(cls_te),), default)
    for c in cls_tr.unique():
        pred[cls_te == c] = int(torch.bincount(y_tr[cls_tr == c], minlength=n_values).argmax())
    return pred


def reader(body, fine: bool):
    """organ, numbers as the organ sees them -> a class per moment.  ``fine``: the class, then the
    held children of the fourth organ the moment falls to (each distinct path is its own class)."""
    ids: dict = {}
    def read(organ, P):
        m = body.organs()[organ].mixed
        best = m.classify(P)[0]
        court = getattr(m, "court", None)
        if not fine or court is None: return best
        Z = torch.exp(1j * 2 * math.pi * m.pockets_of(P) / N_QUANTA)
        table = ids.setdefault(organ, {})
        return torch.tensor([table.setdefault(s, len(table)) if s else -1 for s in court.refine([c.name for c in m.classes], best, Z)])
    return read


def gate(ambient) -> bool:
    body = vb.Body.load()
    ok = _gate(ambient, body, False)
    if any(getattr(l.mixed, "court", None) is not None for l in body.organs().values()):
        print("\n# the same gate, read with the fourth organ's held children added")
        fine = _gate(ambient, body, True)
        for k, l in body.organs().items():
            c = l.mixed.court
            print(f"  fourth organ, {k}: {c.size()}; heard {c.counts['heard']}, held {c.counts['held']}, settled and carried out {c.counts['carried']}, "
                  f"asleep (failed / unfed / void) {c.counts['failed']} / {c.counts['starved']} / {c.counts['void']}, gone {c.counts['gone']}, "
                  f"woken {c.counts['woken']} (settled {c.counts['woken_numa']}), inner born / settled / dropped {c.counts['inner']} / {c.counts['inner_numa']} / {c.counts['inner_dropped']}")
        print(f"\nGATE, either reading: working organs {'pass' if ok else 'NOT passed'}; with the fourth organ {'pass' if fine else 'NOT passed'}")
        ok = ok or fine
    return ok


def _gate(ambient, body, fine: bool) -> bool:
    global CORRIDOR
    log = torch.load(LOG, weights_only=False)
    read = reader(body, fine)
    organ = lambda k, M: read(k, body.seen(k, body.part(k, M)))    # from whole moments (ear, body sense)
    CORRIDOR = body.extra["world"].corridor
    X, Hd, T, A = (torch.cat(log[k]) for k in ("X", "hidden", "text", "action"))
    Xp, Hp, Tp, Ap = probe(ambient, body.seed)
    n = len(X)
    K, Kp = vb.changes(X), vb.changes(Xp)                         # what kin reads
    class kin:                                                    # her sense of change, read as she sees it (through her blur, if any)
        classes = body.kin.mixed.classes
        classify = staticmethod(lambda M: (read("kin", body.seen("kin", M)),))
    kk = len(kin.classes)
    C = kmeans(K, kk, body.seed)
    km = lambda M: torch.cdist(M, C).argmin(1)
    cls, clsp = kin.classify(K)[0], kin.classify(Kp)[0]
    ok = True
    print(f"# fetal gate: {body.extra['fetal_day']} fetal days lived ({body.extra['day']} days old), {TEST} unseen days; classes {body.classes}; "
          f"receptors {body.receptors}; kicks {int((A > 0).sum())} lived")

    # 1 she feels her kicks: which action was just done (mean recall over the six actions; chance 1/6)
    a = recall_per_value(vote(cls, A, clsp, 6), Ap); c = recall_per_value(vote(km(K), A, km(Kp), 6), Ap)
    v = a > 1 / 6 + 0.1 and a >= c - 0.05; ok &= v
    print(f"1 feels her kicks (which action, mean recall over 6):  hers {a:.2f}  k-means {c:.2f}  chance .17   {'pass' if v else 'FAIL'}")

    # 2 she knows what a kick moves: echo shown, changes hidden -> the class she falls in; that class's
    #   mean change over the lived moments it claimed points at a reading
    mu, sd = K[:, 8:].mean(0), K[:, 8:].std(0) + 1e-6          # a reading 'moves' = unlike its usual change (the clock always moves)
    Kh = Kp.clone(); Kh[:, 8:] = 0
    ch = kin.classify(Kh)[0]
    right = 0
    for an, cols in MOVES.items():
        rows = (Ap == ACTIONS.index(an)).nonzero().squeeze(1)
        hit = []
        for r in rows:
            m = cls == ch[r]
            hit.append(bool(m.any()) and int(((K[m][:, 8:].mean(0) - mu) / sd).abs().argmax()) in cols)
        acc = float(np.mean(hit)) if hit else float("nan")
        right += acc > 0.5
        print(f"2 knows what '{an}' moves (changes hidden): {acc:.2f} of {len(rows)} kicks point at the right reading")
    v = right >= 4; ok &= v
    print(f"2 right for {right} of 5 acting kicks   {'pass' if v else 'FAIL'} (>= 4)")

    # 3 her doing or the world's, ECHO HIDDEN: a CPU rise (her work / the world turning busy), a memory rise (her hold / memory filling)
    def events(Hx, Ax):
        up = lambda j: torch.cat([torch.zeros(1, dtype=torch.bool), (Hx[1:, j] == 1) & (Hx[:-1, j] == 0)])
        out = {}
        for name, act, j in (("CPU rise", "work", HIDDEN.index("busy")), ("memory rise", "hold", HIDDEN.index("full"))):
            mine, world = Ax == ACTIONS.index(act), up(j)
            keep = mine ^ world                                    # leave out the moments where both happened
            out[name] = (keep, mine[keep].long())
        return out
    Ke, Kpe = K.clone(), Kp.clone()
    Ke[:, :8] = 0; Kpe[:, :8] = 0; Ke[:, 41:] = 0; Kpe[:, 41:] = 0  # echo and the change of held / marks hidden
    ce, cpe = kin.classify(Ke)[0], kin.classify(Kpe)[0]
    ev, evp = events(Hd, A), events(Hp, Ap)
    for name in ev:
        (m, y), (mp, yp) = ev[name], evp[name]
        a = (vote(ce[m], y, cpe[mp], 2) == yp).float().mean().item()
        c = (vote(km(Ke)[m], y, km(Kpe)[mp], 2) == yp).float().mean().item()
        base = max(yp.float().mean().item(), 1 - yp.float().mean().item())
        v = a > max(base, 0.5) + 0.05; ok &= v
        print(f"3 {name}: hers or the world's, echo hidden ({int(mp.sum())} unseen events, {int(yp.sum())} hers):  hers {a:.2f}  k-means {c:.2f}  base {base:.2f}   {'pass' if v else 'FAIL'}")

    # lasting and fading (reported): holding anything / any mark, read from the body sense
    soma, somap = organ("soma", X), organ("soma", Xp)
    for name, col in (("holding something", 106), ("a mark exists", 107)):
        y, yp = (X[:, col] > -1).long(), (Xp[:, col] > -1).long()
        a = (vote(soma, y, somap, 2) == yp).float().mean().item()
        print(f"  lasting (reported): {name}, read from the body sense {a:.2f} (base {max(yp.float().mean().item(), 1 - yp.float().mean().item()):.2f})")

    # nothing lost: the embryo's readings from the body sense, the ear's from the ear
    sall, Hall = torch.cat([soma, somap]), torch.cat([Hd, Hp])
    for j, h in enumerate(HIDDEN):
        if h == "night": continue
        a, b = readable(sall, Hall[:, j], n)
        v = a >= EMBRYO[h] - 0.05; ok &= v
        print(f"4 nothing lost: {h:10} {a:.2f} (embryo {EMBRYO[h]:.2f}, base {b:.2f})   {'pass' if v else 'FAIL'}")
    ear = organ("ear", torch.cat([X, Xp]))
    a, b = readable(ear, (torch.cat([T, Tp]) >= 0).long(), n)
    v = a >= EAR_EMBRYO - 0.05; ok &= v
    print(f"4 the ear: text present {a:.2f} (embryo {EAR_EMBRYO:.2f}, base {b:.2f})   {'pass' if v else 'FAIL'}")
    used = {k: organ(k, X[-10 * PER_DAY:]).unique().numel() for k in ("ear", "soma")}
    used["kin"] = cls[-10 * PER_DAY:].unique().numel()
    print(f"cost (reported): classes {log['classes'][0]} on fetal day 1 -> {log['classes'][-1]}; used in her last 10 days {used}; "
          f"mean day {np.mean(log['secs']):.1f} s (first 5: {np.mean(log['secs'][:5]):.1f}, last 5: {np.mean(log['secs'][-5:]):.1f})")
    print(f"\nGATE: {'all criteria pass' if ok else 'NOT passed'} — for Rocky to read; no stage is promoted by code")
    return bool(ok)


if __name__ == "__main__":
    vb.require_stage("fetal")
    ambient = vb.language_sense()(vb.wiki_states())
    {"live": live, "gate": gate}[sys.argv[1]](ambient)
