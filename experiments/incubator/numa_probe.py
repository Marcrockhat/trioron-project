"""The Numa rule for class growth, tried on a COPY (plan §13.5; Rocky s071: "build it and run
the copy trial").  Never on a living organism (rule 27): the copy has its own home.

    VEE_HOME=<copy home> python3 experiments/incubator/numa_probe.py setup <source home> <fetal day> [plain]
        the source's body_fetal_day<N>.pt becomes the copy's body.pt, its fetal log is cut to
        N days, and the rule is switched on in every organ ("plain" = left off: a check that
        the copy lives the source's own days again)
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
from trioron.pcll.numa import NumaLedger


def setup(src: str, day: int, plain: bool) -> None:
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
            leaf.mixed.numa = NumaLedger()
        body.save()
    print(f"copy of {src} at fetal day {day} in {vb.HOME}; classes {body.classes}; the rule is {'off' if plain else 'on'}")


def ledger() -> None:
    body = vb.Body.load()
    print(f"# ledger at fetal day {body.extra['fetal_day']}; classes {body.classes}")
    for name, leaf in body.organs().items():
        led = getattr(leaf.mixed, "numa", None)
        if led is None:
            print(f"  {name}: the rule is off"); continue
        c = led.counts
        tb = led.taken_back
        late = sum(1 for t in tb if t["period"] - t["born"] >= led.trial)
        fed = sum(1 for t in tb if min(t["world"]) > 0)
        print(f"  {name}: born {c['born']}, Numa {c['numa']}, Mima {c['mima']}, gone {c['gone']}, on trial {len(led.pairs)}, net {led.net}; "
              f"of the Mima: {late} ran out of time, {len(tb) - late} failed a batch; {fed} had both children fed by the world; "
              f"members held {sum(len(b) for b in leaf.mixed.bufs)}")


if __name__ == "__main__":
    if sys.argv[1] == "setup":
        setup(sys.argv[2], int(sys.argv[3]), len(sys.argv) > 4 and sys.argv[4] == "plain")
    else:
        ledger()
