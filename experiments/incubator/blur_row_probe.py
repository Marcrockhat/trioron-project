"""Why a blurred organism never sharpens, and what her blur leaves of the gate's row 3 (s070;
Rocky: "they should have their sense sharpened gradually and there must be reason why it
never triggered").

READ-ONLY: her saved body and fetal log are loaded and measured; nothing is saved.
One organism, one seed.

  A  the sharpening trigger: an organ sharpens when its class count has stood still for
     3 + classes // 8 sittings in a row (vee_body._mature).  How long it ever stood still.
  B  gate row 3 (her doing or the world's, echo hidden), on the gate's own events:
     what the two kinds of event look like through her blur, and the best any reader of
     that view could score (each distinct seen pattern votes its majority on the lived
     events), at her present steps, at every change reading one or more rungs sharper,
     and unblurred.

    VEE_HOME=outputs/incubator/weiss python3 experiments/incubator/blur_row_probe.py
"""
from __future__ import annotations
import copy, os, sys
import torch
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(__file__))
import vee_body as vb
import fetal
from world import ACTIONS, HIDDEN, PER_DAY


def longest_still(counts):
    best = run = 0
    for a, b in zip(counts, counts[1:]):
        run = run + 1 if a == b else 0
        best = max(best, run)
    return best


def pattern_vote(S_tr, y_tr, S_te):
    """Each distinct seen pattern votes its majority on the lived events."""
    key = lambda r: tuple(round(float(v), 5) for v in r)
    votes: dict = {}
    for r, y in zip(S_tr, y_tr.tolist()):
        votes.setdefault(key(r), [0, 0])[y] += 1
    default = int(y_tr.float().mean() >= 0.5)
    pred = torch.tensor([(lambda v: default if v is None else int(v[1] >= v[0]))(votes.get(key(r))) for r in S_te])
    return pred, votes


def main():
    body, log = vb.Body.load(), torch.load(fetal.LOG, weights_only=False)
    fetal.CORRIDOR = body.extra["world"].corridor
    print(f"# {vb.HOME}: {body.extra['fetal_day']} fetal days, {body.extra['day']} days old; classes {body.classes}; read only")

    print("\nA  the sharpening trigger (2 sittings a day)")
    for k in body.organs():
        m, b = body.mature[k], body.blur[k]
        n = len(body.organs()[k].mixed.classes)
        daily = [d[k] for d in log["classes"]]
        still = longest_still(daily)
        same_days = sum(a == b_ for a, b_ in zip(daily, daily[1:]))
        print(f"   {k:5} classes {n:5}: needs {vb.PATIENCE_BASE + n // vb.PATIENCE_PER:4} still sittings in a row; "
              f"still now {m['quiet']:3}; sittings lived {m['sittings']}; steps taken {len(b.history)} {b.history}")
        print(f"         by her daily log: class count equal to the day before on {same_days} of {len(daily) - 1} days; "
              f"longest run of such days {still} (about {2 * still} sittings)")
        print(f"         readings that can sharpen {int(b.varies.sum())} of {len(b.varies)}; steps now {torch.bincount(b.steps[b.varies]).tolist() if bool(b.varies.any()) else []} (index = steps)")

    X, Hd, A = (torch.cat(log[k]) for k in ("X", "hidden", "action"))
    ambient = vb.language_sense()(vb.wiki_states())
    Xp, Hp, Tp, Ap = fetal.probe(ambient, body.seed)
    K, Kp = vb.changes(X), vb.changes(Xp)
    Ke, Kpe = K.clone(), Kp.clone()
    Ke[:, :8] = 0; Kpe[:, :8] = 0; Ke[:, 41:] = 0; Kpe[:, 41:] = 0

    def events(Hx, Ax):
        up = lambda j: torch.cat([torch.zeros(1, dtype=torch.bool), (Hx[1:, j] == 1) & (Hx[:-1, j] == 0)])
        out = {}
        for name, act, j in (("CPU rise", "work", HIDDEN.index("busy")), ("memory rise", "hold", HIDDEN.index("full"))):
            mine, world = Ax == ACTIONS.index(act), up(j)
            keep = mine ^ world
            out[name] = (keep, mine[keep].long())
        return out

    ev, evp = events(Hd, A), events(Hp, Ap)
    blur = body.blur["kin"]
    views = [("as she sees now", blur)]
    for n in (3, 5, 7):
        b = copy.deepcopy(blur)
        b.steps = torch.where(b._is_change(), torch.maximum(b.steps, torch.tensor(n)), b.steps)
        views.append((f"every change at {n} steps", b))
    mix = body.kin.mixed
    print("\nB  gate row 3 through her blur (lived events vote, unseen events are scored)")
    for name in ev:
        (m, y), (mp, yp) = ev[name], evp[name]
        base = max(yp.float().mean().item(), 1 - yp.float().mean().item())
        print(f"\n   {name}: lived {int(m.sum())} events ({int(y.sum())} hers), unseen {int(mp.sum())} ({int(yp.sum())} hers), base {base:.2f}")
        d = Ke[m][y == 1].mean(0) - Ke[m][y == 0].mean(0)
        top = d.abs().argsort(descending=True)[:5]
        print("     full numbers, the readings that differ most between hers and the world's (kin column: hers mean / world's mean, range she was born with):")
        for j in top.tolist():
            print(f"       col {j:2}: {Ke[m][y == 1][:, j].mean():+.3f} / {Ke[m][y == 0][:, j].mean():+.3f}   range ±{float(blur.hi[j]):.3f}, "
                  f"'changed' from {float(blur.hi[j]) / 3:.3f}; steps now {int(blur.steps[j])}; can sharpen {bool(blur.varies[j])}")
        for label, b in views:
            S, Sp = b.see(Ke[m]), b.see(Kpe[mp])
            pred, votes = pattern_vote(S, y, Sp)
            both = sum(1 for v in votes.values() if v[0] and v[1])
            amb = sum(min(v) for v in votes.values())
            print(f"     {label:26}: {len(votes):4} distinct patterns on the lived events, {both} of them hold both kinds "
                  f"({amb} of {int(m.sum())} lived events cannot be told apart); best reader of this view on the unseen: {float((pred == yp).float().mean()):.2f}")
        pred, votes = pattern_vote(Ke[m], y, Kpe[mp])
        print(f"     {'unblurred':26}: {len(votes):4} distinct patterns; exact-pattern reader {float((pred == yp).float().mean()):.2f} (k-means in the gate reads these numbers)")
        ce, cpe = mix.classify(blur.see(Ke[m]))[0], mix.classify(blur.see(Kpe[mp]))[0]
        tab = {}
        for c, yy in zip(ce.tolist(), y.tolist()):
            tab.setdefault(c, [0, 0])[yy] += 1
        a = (fetal.vote(ce, y, cpe, 2) == yp).float().mean().item()
        print(f"     her classes: the lived events fall in {len(tab)} classes; (world's, hers) per class, largest first: "
              f"{sorted(tab.values(), key=lambda v: -sum(v))[:6]}; her score {a:.2f}")


if __name__ == "__main__":
    main()
