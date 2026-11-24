"""Does a correction separate two near facts — and how many wordings does it take?
(Rocky s066: answering the cat's name for the dog is generalisation for a young
organism; the test is whether teaching separates them.)

Two near facts: "name of my cat -> Miso" and, as the correction, "name of my
dog -> Rex".  Each is taught in K wordings.  Test = four sentence forms never
taught, for both animals.  A third animal (hamster) is never taught.

  K matched    both animals are taught in the same K sentence forms
  1 unmatched  cat in one form, dog in another (the s066 console case: she
               separated by sentence form, not by animal)

Stages:  table  both facts only in the day table (never slept)
         mixed  cat slept, then the dog is taught (cat in the leaf, dog in the table)
         leaf   both slept (the phasecyte only)

Run from repo root:  python3 experiments/incubator/correction_test.py
Env: SEEDS=3
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(__file__))
import vee_memory as vm

TRAIN = ["What is the name of my {}?", "What is my {} called?", "What do I call my {}?",
         "Tell me my {}'s name.", "Which name did I give my {}?", "My {} is named what?"]
TEST = ["How is my {} named?", "What name does my {} have?", "Do you know my {}'s name?",
        "Remind me what my {} is called."]
NAME = {"cat": "Miso", "dog": "Rex"}
CONDS = [("1 unmatched", [0], [1]), ("1 matched", [0], [0]), ("2 matched", [0, 1], [0, 1]),
         ("3 matched", [0, 1, 2], [0, 1, 2]), ("4 matched", [0, 1, 2, 3], [0, 1, 2, 3]),
         ("6 matched", list(range(6)), list(range(6)))]

if __name__ == "__main__":
    from reflex import Reflex
    reflex = Reflex()
    h = {}
    for a in ("cat", "dog", "hamster"):
        for f in TRAIN + TEST:
            k = reflex.read(f.format(a))
            h[f.format(a)] = torch.cat([torch.tensor(k["h_last"]), torch.tensor(k["h_mean"])]).float()
    del reflex
    seeds = int(os.environ.get("SEEDS", 3))
    print(f"# correction test  seeds={seeds}  test = {len(TEST)} untaught sentence forms per animal", flush=True)

    def teach(mem, animal, forms):
        i = None
        for f in forms:
            i = mem.teach(TRAIN[f].format(animal), NAME[animal], h[TRAIN[f].format(animal)], same_as=i)

    def score(mem):
        """right, wrong animal, no answer over the cat+dog test forms; hamster answered with a taught name."""
        out = np.zeros(4)
        for a in ("cat", "dog"):
            for f in TEST:
                r = mem.recall(h[f.format(a)])
                out[2 if r is None else 0 if r["answer"] == NAME[a] else 1] += 1 / (2 * len(TEST))
        out[3] = np.mean([(r := mem.recall(h[f.format("hamster")])) is not None and r["answer"] in NAME.values()
                          for f in TEST])
        return out

    res: dict[tuple, list] = {}
    for seed in range(seeds):
        for name, cf, df in CONDS:
            m = vm.wake(seed, user=False)
            teach(m, "cat", cf); teach(m, "dog", df)
            res.setdefault((name, "table"), []).append(score(m))
            m = vm.wake(seed, user=False)
            teach(m, "cat", cf); m.sleep(); teach(m, "dog", df)
            res.setdefault((name, "mixed"), []).append(score(m))
            m.sleep()
            res.setdefault((name, "leaf"), []).append(score(m))
            print(f"  seed {seed} {name:12} " + "  ".join(
                f"{st} right {res[(name, st)][-1][0]:.2f} wrong {res[(name, st)][-1][1]:.2f}" for st in ("table", "mixed", "leaf")), flush=True)
    print(f"\n{'wordings per fact':18} {'stage':6} {'right':>6} {'wrong animal':>13} {'no answer':>10} {'hamster gets a name':>20}")
    for (name, st), rows in res.items():
        r = np.array(rows).mean(0)
        print(f"{name:18} {st:6} {r[0]:>6.2f} {r[1]:>13.2f} {r[2]:>10.2f} {r[3]:>20.2f}")
