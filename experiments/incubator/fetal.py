"""Stage 2a — fetal: body and first kicks (docs/design/vee_development.md §8).
Her action echo comes alive, a sense of change (kin) is born, and kicks fire by
themselves about one moment in six.  She does not choose (2b is not designed).
Nobody addresses her; nothing is labelled.  All consequences are scripted (world.py).

    python3 experiments/incubator/fetal.py live    DAYS further days; saved after every simulated day; resumable
    python3 experiments/incubator/fetal.py gate    the gate (§8.6) on TEST_DAYS days she does not live; nothing changes

The stage file (vee_body.HOME/stage) must say ``fetal``.  She continues from the saved
embryo (body.pt): same world, same ambient text stream.
Log: HOME/fetal_log.pt — every moment's input, hidden world state, sentence index, kick,
and per-day class counts and wall time.
Alarm (stop, state kept): a day takes more than twice the fifth fetal day's time.

Env: DAYS=30 TEST_DAYS=5 P_TEXT=0.5 P_KICK=0.1667
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(__file__))
import vee_body as vb
from germline_checks import kmeans, readable
from world import ACTIONS, HIDDEN, PER_DAY, World

E = lambda k, d: type(d)(os.environ.get(k, d))
DAYS, TEST, P_TEXT, P_KICK = E("DAYS", 30), E("TEST_DAYS", 5), E("P_TEXT", 0.5), E("P_KICK", 1 / 6)
LOG = f"{vb.HOME}/fetal_log.pt"
EMBRYO = {"busy": 0.99, "full": 0.95, "elsewhere": 1.00, "window": 1.00}     # her embryo gate values (plan §7.3a)
EAR_EMBRYO = 0.75
# what each acting kick really moves, as columns of the kin organ's 35 change numbers
# (33 environment readings, then held, marks); environment.NAMES gives the order
MOVES = {"work": {8, 10}, "hold": {5, 6, 26, 33}, "let go": {5, 6, 26, 33}, "mark": {34}, "wipe": {34}}


def day_of(world: World, g: np.random.Generator, k: np.random.Generator, ambient: torch.Tensor):
    """One simulated day with spontaneous kicks: inputs [96, 148], hidden [96, 5], sentence [96], kick [96]."""
    X, Hd, T, A = [], [], [], []
    for _ in range(PER_DAY):
        ok = [a for a in range(1, len(ACTIONS)) if world.can(a)]
        a = int(k.choice(ok)) if k.random() < P_KICK else 0
        raw, hid = world.step(a)
        i = int(g.integers(len(ambient))) if g.random() < P_TEXT else -1
        X.append(vb.moment(ambient[i] if i >= 0 else None, raw, action=a))
        Hd.append([hid[h] for h in HIDDEN]); T.append(i); A.append(a)
    return torch.stack(X), torch.tensor(Hd).long(), torch.tensor(T), torch.tensor(A)


def live(ambient):
    body = vb.Body.load()
    ex = body.extra
    if "kin" not in body.organs():
        body.grow_kin(vb.kin_pool(body.seed))
        ex["kicks"], ex["fetal_day"] = np.random.default_rng(4000 + body.seed), 0
        log = {"X": [], "hidden": [], "text": [], "action": [], "classes": [], "secs": []}
        print(f"[kin is born: {body.receptors['kin']} receptors of 43; she is {ex['day']} days old]", flush=True)
    else:
        log = torch.load(LOG, weights_only=False)
    while ex["fetal_day"] < DAYS:
        t0 = time.time()
        X, Hd, T, A = day_of(ex["world"], ex["rng"], ex["kicks"], ambient)
        for i in range(0, PER_DAY, vb.WINDOW):
            body.live(X[i:i + vb.WINDOW])
        ex["day"] += 1; ex["fetal_day"] += 1
        for key, v in (("X", X), ("hidden", Hd), ("text", T), ("action", A)): log[key].append(v)
        log["classes"].append(dict(body.classes)); log["secs"].append(time.time() - t0)
        body.save(); torch.save(log, LOG + ".tmp"); os.replace(LOG + ".tmp", LOG)
        print(f"  fetal day {ex['fetal_day']:3}: classes {body.classes}, {log['secs'][-1]:.1f} s", flush=True)
        if ex["fetal_day"] > 5 and log["secs"][-1] > 2 * max(log["secs"][4], 1.0):
            raise SystemExit(f"ALARM: fetal day {ex['fetal_day']} took {log['secs'][-1]:.1f} s, more than twice day 5 ({log['secs'][4]:.1f} s); stopped, state kept")


def probe(ambient, seed):
    """Days she never lives (another world run, with kicks); she is elsewhere from the third day."""
    w, g, k = World(9100 + seed), np.random.default_rng(9600 + seed), np.random.default_rng(9700 + seed)
    days = []
    for d in range(TEST):
        if d == 2: w.elsewhere = True
        days.append(day_of(w, g, k, ambient))
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


def gate(ambient):
    body, log = vb.Body.load(), torch.load(LOG, weights_only=False)
    X, Hd, T, A = (torch.cat(log[k]) for k in ("X", "hidden", "text", "action"))
    Xp, Hp, Tp, Ap = probe(ambient, body.seed)
    n = len(X)
    K, Kp = vb.changes(X), vb.changes(Xp)                         # what kin reads
    kin = body.kin.mixed
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
    soma, somap = body.read(X)[0], body.read(Xp)[0]
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
    ear = body.read(torch.cat([X, Xp]), "ear")[0]
    a, b = readable(ear, (torch.cat([T, Tp]) >= 0).long(), n)
    v = a >= EAR_EMBRYO - 0.05; ok &= v
    print(f"4 the ear: text present {a:.2f} (embryo {EAR_EMBRYO:.2f}, base {b:.2f})   {'pass' if v else 'FAIL'}")
    used = {k: body.read(X[-10 * PER_DAY:], k)[0].unique().numel() for k in ("ear", "soma")}
    used["kin"] = cls[-10 * PER_DAY:].unique().numel()
    print(f"cost (reported): classes {log['classes'][0]} on fetal day 1 -> {log['classes'][-1]}; used in her last 10 days {used}; "
          f"mean day {np.mean(log['secs']):.1f} s (first 5: {np.mean(log['secs'][:5]):.1f}, last 5: {np.mean(log['secs'][-5:]):.1f})")
    print(f"\nGATE: {'all criteria pass' if ok else 'NOT passed'} — for Rocky to read; no stage is promoted by code")


if __name__ == "__main__":
    vb.require_stage("fetal")
    ambient = vb.language_sense()(vb.wiki_states())
    {"live": live, "gate": gate}[sys.argv[1]](ambient)
