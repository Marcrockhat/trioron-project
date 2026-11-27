"""The Numa rule for class growth, tried on a COPY (plan §13.5; Rocky s071: "build it and run
the copy trial").  Never on a living organism (rule 27): the copy has its own home.

    VEE_HOME=<copy home> python3 experiments/incubator/numa_probe.py setup <source home> <fetal day> [plain|witness|fourth|still]
        the source's body_fetal_day<N>.pt becomes the copy's body.pt, its fetal log is cut to
        N days, and the rule is switched on in every organ ("plain" = left off: a check that
        the copy lives the source's own days again; "witness" = the rule off, but every
        division's split dimension is written down; "fourth" = the fourth organ as every
        organ's court, no rule inside the working organs: plan §14; "still" = the control for
        "fourth": the same court, but nothing ever settles, so no division is carried out)
    VEE_HOME=<copy home> python3 experiments/incubator/numa_probe.py dims <input channel>
        which of the body sense's columns answers to that input channel, and how many
        divisions (born / settled / taken back / on trial) split on it (read-only)
    VEE_HOME=<copy home> python3 experiments/incubator/numa_probe.py ledger
        the ledger of every organ, as it stands (read-only)

Then live and read the gate with fetal.py as for any organism.
"""
from __future__ import annotations
import os, shutil, sys
import torch
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(__file__))
import vee_body as vb
from trioron.organs import FourthOrgan, NumaLedger


def setup(src: str, day: int, mode: str) -> None:
    plain = mode == "plain"
    assert os.path.abspath(src) != os.path.abspath(vb.HOME) and "/copies/" in os.path.abspath(vb.HOME), "a copy lives under copies/"
    assert not os.path.exists(f"{vb.HOME}/body.pt"), "this home is already in use"
    os.makedirs(vb.HOME, exist_ok=True)
    for f in ("stage", "sense.pt", "embryo_values.json"):
        if os.path.exists(f"{src}/{f}"): shutil.copy(f"{src}/{f}", f"{vb.HOME}/{f}")
    shutil.copy(f"{src}/body_fetal_day{day}.pt", f"{vb.HOME}/body.pt")
    log = torch.load(f"{src}/fetal_log.pt", weights_only=False)
    torch.save({k: v[:day] for k, v in log.items()}, f"{vb.HOME}/fetal_log.pt")
    body = vb.Body.load()
    assert body.extra["fetal_day"] == day
    if not plain:
        for leaf in body.organs().values():
            if mode == "fourth": leaf.mixed.court = FourthOrgan()
            elif mode == "still": leaf.mixed.court = FourthOrgan(batches=10 ** 9)
            else: leaf.mixed.numa = NumaLedger(witness=mode == "witness")
        body.save()
    print(f"copy of {src} at fetal day {day} in {vb.HOME}; classes {body.classes}; the rule is {'off' if plain else 'off, divisions witnessed' if mode == 'witness' else 'in the fourth organ' if mode == 'fourth' else 'in a fourth organ that settles nothing (control)' if mode == 'still' else 'on'}")


def ledger() -> None:
    body = vb.Body.load()
    print(f"# ledger at fetal day {body.extra['fetal_day']}; classes {body.classes}")
    for name, leaf in body.organs().items():
        led = getattr(leaf.mixed, "numa", None)
        if led is None:
            print(f"  {name}: the rule is off"); continue
        c = led.counts
        tb = led.taken_back
        late = sum(1 for t in tb if t.get("why", "starved" if t["period"] - t["born"] >= led.trial else "failed") == "starved")
        seen, ref = getattr(led, "seen", 0), getattr(led, "refused", 0)
        age = sorted(leaf.mixed.periods - q["born"] for q in led.pairs.values())
        fed = sum(1 for t in tb if min(t["world"]) > 0)
        print(f"  {name}: born {c['born']}, Numa {c['numa']}, Mima {c['mima']}, gone {c['gone']}, on trial {len(led.pairs)}, net {led.net}; "
              f"of the Mima: {late} went unfed, {len(tb) - late} failed a batch or lost their column; {fed} had both children fed by the world; "
              f"members held {sum(len(b) for b in leaf.mixed.bufs)}; world rows refused {ref} of {seen}; "
              f"sittings on trial (median / longest) {age[len(age) // 2] if age else 0} / {age[-1] if age else 0}")


def dims(ch: int) -> None:
    """The column of the body sense that answers to input channel ``ch``, found by moving that
    channel alone; then the divisions that split on it."""
    body = vb.Body.load()
    m, led = body.soma.mixed, getattr(body.soma.mixed, "numa", None)
    X = torch.cat(torch.load(f"{vb.HOME}/fetal_log.pt", weights_only=False)["X"][-2:]).clone()   # real moments
    q = []
    for v in (0.0, 1.0):                           # the channel silent, then high: its own column answers in every moment
        X[:, ch] = v
        q.append(m.pockets_of(body.seen("soma", body.part("soma", X))))
    moved = (q[0] != q[1]).all(0).nonzero().flatten().tolist()
    width = m.bufs[0].shape[1]
    print(f"# input channel {ch} moves column(s) {moved} of the body sense's {width} (fetal day {body.extra['fetal_day']}, classes {body.classes})")
    court = getattr(m, "court", None)
    if court is not None:
        n = lambda ev: sum(1 for e in court.log if e[1] == ev)
        on = lambda ev: sum(1 for e in court.log if e[1] == ev and e[3] in moved)
        print("  fourth organ, pairs on that column / all: " + ", ".join(f"{ev} {on(ev)} / {n(ev)}" for ev in ("held", "woken", "numa", "failed", "starved", "void")))
        return
    born = led.dims["born"]
    on = lambda L: sum(1 for d in L if d in moved)
    tb = [t["dim"] for t in led.taken_back]
    trial = [p["dim"] for p in led.pairs.values()]
    print(f"  divisions born {len(born)}, on that column {on(born)}; settled {len(led.dims['numa'])}, on it {on(led.dims['numa'])}; "
          f"taken back {len(tb)}, on it {on(tb)}; on trial {len(trial)}, on it {on(trial)}")
    top = torch.bincount(torch.tensor(born), minlength=width).topk(min(8, width))
    print("  the columns most divided on: " + ", ".join(f"{int(i)} ({int(n)})" for n, i in zip(top.values, top.indices)))


if __name__ == "__main__":
    if sys.argv[1] == "setup":
        setup(sys.argv[2], int(sys.argv[3]), sys.argv[4] if len(sys.argv) > 4 else "")
    elif sys.argv[1] == "dims":
        dims(int(sys.argv[2]))
    else:
        ledger()
