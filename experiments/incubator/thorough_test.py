"""Learn ONE fact thoroughly, then the near one from it (Rocky s066).

correction_test.py taught "cat -> Miso" in a few wordings on one night, then
"dog -> Rex" the same way; the phasecyte never held the cat as one class (its
41 samples scattered over old classes) and gave both animals one name.  Here
the cat is taught over several nights, a few new wordings a night, before the
dog appears.  Nothing in the phasecyte is changed.

  thin cat, thin dog          1 night each, 6 wordings (the correction test)
  thorough cat, thin dog      cat: NIGHTS nights x PER new wordings; dog: 1 night, 6 wordings
  thorough cat, thorough dog  both NIGHTS x PER

Test = 4 sentence forms never taught, answered by the PHASECYTE ALONE (the day
table would answer first and hide it).  Also reported: how many classes are
named for each animal, and how much of each animal's tagged samples sit in a
class named for it (purity).

Run from repo root:  python3 experiments/incubator/thorough_test.py
Env: SEEDS=3 NIGHTS=6 PER=4
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(__file__))
import vee_memory as vm
from correction_test import NAME, TEST, TRAIN

MORE = ["What did I name my {}?", "My {}'s name is what?", "Can you tell me what my {} is called?",
        "What's my {}'s name?", "I have a {}; what is its name?", "Say the name of my {}.",
        "By what name does my {} go?", "What should you call my {}?", "Do you recall what I call my {}?",
        "The {} I own is called what?", "Give me the name of my {}.", "What name does my {} answer to?",
        "Which name belongs to my {}?", "My {} goes by what name?", "Name my {} for me.",
        "What is the {} of mine called?", "Tell me what I named my {}.", "What do people call my {}?"]
FORMS = TRAIN + MORE                                                 # 24 taught sentence forms

if __name__ == "__main__":
    from reflex import Reflex
    reflex = Reflex()
    h = {}
    for a in ("cat", "dog", "hamster"):
        for f in FORMS + TEST:
            k = reflex.read(f.format(a))
            h[f.format(a)] = torch.cat([torch.tensor(k["h_last"]), torch.tensor(k["h_mean"])]).float()
    del reflex
    E = lambda k, d: int(os.environ.get(k, d))
    SEEDS, NIGHTS, PER = E("SEEDS", 3), E("NIGHTS", 6), E("PER", 4)
    print(f"# thorough test  seeds={SEEDS}  thorough = {NIGHTS} nights x {PER} new wordings; thin = 1 night x 6", flush=True)
    ids: dict[str, int] = {}

    def teach(mem, a, thorough):
        for n in range(NIGHTS if thorough else 1):
            for f in (FORMS[n * PER:(n + 1) * PER] if thorough else FORMS[:6]):
                ids[a] = mem.teach(f.format(a), NAME[a], h[f.format(a)], same_as=ids.get(a))
            mem.sleep()

    def measure(mem):
        out = []
        for a in ("cat", "dog"):                                     # the phasecyte alone (the day table is not asked)
            f = mem._leaf(mem.sense(torch.stack([h[t.format(a)] for t in TEST])))[0]
            out.append((f == ids.get(a, -9)).float().mean().item())
        f = mem._leaf(mem.sense(torch.stack([h[t.format("hamster")] for t in TEST])))[0]
        out.append(torch.isin(f, torch.tensor(list(ids.values()))).float().mean().item())
        m = mem.leaf.mixed
        lab = {v: int(k[1:]) for k, v in m.label_taps.vocab.items()}
        for a in ("cat", "dog"):
            if a not in ids: out += [0, 0]; continue
            inside = total = 0
            for c, t in enumerate(m.tags):
                n = sum(lab[int(x)] == ids[a] for x in t[t >= 0])
                total += n; inside += n * int(mem.names[c] == ids[a])
            out += [int((mem.names == ids[a]).sum()), inside / max(total, 1)]
        return out

    res: dict[str, list] = {}
    for seed in range(SEEDS):
        for name, tc, td in (("thin cat, thin dog", 0, 0), ("thorough cat, thin dog", 1, 0), ("thorough cat, thorough dog", 1, 1)):
            mem = vm.wake(seed, user=False); ids.clear()
            teach(mem, "cat", tc)
            before = measure(mem)
            teach(mem, "dog", td)
            r = [before[0], before[3], before[4]] + measure(mem)
            res.setdefault(name, []).append(r)
            print(f"  seed {seed} {name:27} cat alone: right {r[0]:.2f} classes {r[1]} purity {r[2]:.2f} | after dog: "
                  f"cat {r[3]:.2f} dog {r[4]:.2f} hamster named {r[5]:.2f}  cat classes {r[6]} purity {r[7]:.2f}  "
                  f"dog classes {r[8]} purity {r[9]:.2f}", flush=True)
    print(f"\n{'teaching':27} | cat alone: right classes purity | after dog: cat right  dog right  hamster named | cat classes purity | dog classes purity")
    for name, rows in res.items():
        r = np.array(rows, float).mean(0)
        print(f"{name:27} | {r[0]:>16.2f} {r[1]:>7.1f} {r[2]:>6.2f} | {r[3]:>20.2f} {r[4]:>10.2f} {r[5]:>14.2f} | {r[6]:>11.1f} {r[7]:>6.2f} | {r[8]:>11.1f} {r[9]:>6.2f}")
