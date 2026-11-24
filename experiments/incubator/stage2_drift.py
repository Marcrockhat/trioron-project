"""Stage 2 drift test — do stored facts survive when the key leaf keeps learning?
(docs/design/incubator_matron.md §3, "corrected items stay corrected after N later lessons")

Keys are stored EMBEDDED (e numbers per fact).  If the key leaf changes after a
fact is stored, the stored key goes stale: that is real forgetting.

Protocol, per split (same held-out articles and negatives as stage2_memory.py):
  lessons    the training articles, cut into LESSONS equal parts, taught in order,
             each lesson trained on its own data only (no replay)
  old facts  half the taught questions, stored after lesson 0 with the keys of that moment
  new facts  the other half, stored after the last lesson with the final keys
  recall     right fact nearest AND above the 1 %-false-override threshold; queries
             are always embedded by the CURRENT key leaf
Conditions (each for a LINEAR trioron leaf and the QUAD leaf, same sense, same loss):
  frozen          key leaf never trained after lesson 0 (no drift, no further learning)
  free            keeps learning, nothing held
  lock uniform s  epigenetic lock, lambda = 1 on every cell, pull strength s
  lock saliency s epigenetic lock, lambda from |w.g| saliency gathered in lesson 0
                  (normalised to mean 1 over computing cells), pull strength s
  re-key          the free leaf, but old facts re-embedded with the final keys
                  (needs the raw states kept: 3,072 numbers per fact) — reference

Run from repo root:  python3 experiments/incubator/stage2_drift.py
Env: SEEDS=1 LESSONS=5 EPOCHS=60 K=256 E=32 STRENGTHS=1,30,1000 TEST_FRAC=0.3
"""
from __future__ import annotations
import copy, os, sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(__file__))
from stage2_memory import load, split
from trioron.learning.epigenetic_lock import (accumulate_saliency, anchor, ewc_penalty,
                                              refresh_lambda, set_lambda)
from trioron.organisms.base import chunked, leaf, train_contrastive
from trioron.senses.hidden_state import HiddenStateSense


def recall_rate(Eq, Kmem, gmem, group, pos, neg, f=0.01) -> float:
    """Queries Eq (current keys, all rows) against stored keys Kmem."""
    sp, ip = (Eq[pos] @ Kmem.T).max(1)
    sn = (Eq[neg] @ Kmem.T).max(1).values
    tau = torch.quantile(sn, 1 - f)
    return ((gmem[ip] == group[pos]) & (sp > tau)).float().mean().item()


if __name__ == "__main__":
    H, group, block, variant, _ = load(os.environ.get("STATE", "both"))
    seeds = range(int(os.environ.get("SEEDS", 1)))
    L, epochs = int(os.environ.get("LESSONS", 5)), int(os.environ.get("EPOCHS", 60))
    k, e = int(os.environ.get("K", 256)), int(os.environ.get("E", 32))
    strengths = [float(x) for x in os.environ.get("STRENGTHS", "1,30,1000").split(",")]
    frac = float(os.environ.get("TEST_FRAC", 0.3))
    print(f"# stage 2 drift  questions={len(H) // 3}  seeds={list(seeds)} lessons={L} epochs/lesson={epochs} "
          f"k={k} e={e} strengths={strengths}", flush=True)
    res: dict[str, dict[str, list]] = {}
    for seed in seeds:
        m = split(group, block, variant, seed, frac)
        g = torch.Generator().manual_seed(1000 + seed)
        arts = torch.unique(block[m["train"]])
        lesson = [torch.isin(block, a) for a in arts[torch.randperm(len(arts), generator=g)].chunk(L)]
        tq = torch.unique(group[m["memory"]])
        tq = tq[torch.randperm(len(tq), generator=g)]
        old_q, new_q = tq[:len(tq) // 2], tq[len(tq) // 2:]
        sets = {}
        for name, qs in (("old", old_q), ("new", new_q)):
            inq = torch.isin(group, qs)
            sets[name] = (inq & m["memory"], inq & m["pos"])
        neg = m["neg_hard"] | m["neg_easy"]
        sense = HiddenStateSense(H[lesson[0]], k, dup=False)
        P = sense(H)
        for arm, nonlinear in (("linear", False), ("quad", True)):
            Z = torch.cat([P, P], 1) if nonlinear else P          # dup only matters to the quad
            key = lambda sub: F.normalize(chunked(sub, Z), dim=1)

            def teach(sub, t, penalty=None, gather=False):
                a = sub.arena
                train_contrastive(sub, sub.trainable_tensors(), Z[lesson[t]], group[lesson[t]],
                                  block[lesson[t]], seed * 100 + t, epochs, blocks_per_batch=48,
                                  penalty=penalty,
                                  after_backward=(lambda: accumulate_saliency(a)) if gather else None)
                with torch.no_grad(): sub(Z[:1])                    # drop grad-bearing activations

            base = leaf(Z.shape[1], e, seed, nonlinear=nonlinear)
            teach(base, 0, gather=True)
            E0 = key(base)
            K_old, g_old = E0[sets["old"][0]], group[sets["old"][0]]

            def score(sub, tag, rekey=False):
                E = key(sub)
                Ko = E[sets["old"][0]] if rekey else K_old
                old = recall_rate(E, Ko, g_old, group, sets["old"][1], neg)
                new = recall_rate(E, E[sets["new"][0]], group[sets["new"][0]], group, sets["new"][1], neg)
                stale = (E[sets["old"][0]] * K_old).sum(1).mean().item()   # cos(old key now, as stored)
                return old, new, stale

            def record(tag, curve, new, stale):
                d = res.setdefault(f"{arm:6} {tag}", {"curve": [], "new": [], "stale": []})
                d["curve"].append(curve); d["new"].append(new); d["stale"].append(stale)
                print(f"  seed {seed} {arm:6} {tag:22} old after each lesson "
                      + " ".join(f"{x:.3f}" for x in curve) + f" | new {new:.3f} | key cos {stale:.3f}", flush=True)

            o, n_, s_ = score(base, "frozen")
            record("frozen", [o] * L, n_, s_)
            conds = [("free", None, 0.0)] + [(f"lock {mode} s={s:g}", mode, s)
                                              for mode in ("uniform", "saliency") for s in strengths]
            for tag, mode, s in conds:
                sub = copy.deepcopy(base)
                a = sub.arena
                pen = None
                if mode:
                    anchor(a)
                    if mode == "uniform":
                        set_lambda(a, torch.ones_like(a.node_lambda), "absolute")
                    else:
                        refresh_lambda(a)
                        lam = a.node_lambda.clone()
                        live = lam > 2 * lam.min()
                        set_lambda(a, lam / lam[live].mean(), "absolute")
                    pen = lambda a=a, s=s: s * ewc_penalty(a)
                curve = [score(base, tag)[0]]
                for t in range(1, L):
                    teach(sub, t, penalty=pen)
                    o, n_, s_ = score(sub, tag)
                    curve.append(o)
                record(tag, curve, n_, s_)
                if tag == "free":
                    o, n_, s_ = score(sub, tag, rekey=True)
                    record("re-key (reference)", [curve[0]] + [float("nan")] * (L - 2) + [o], n_, 1.0)
    print(f"\n{'condition':30} {'old @start':>11} {'old @end':>13} {'new @end':>13} {'key cos':>8}")
    for name, d in res.items():
        c = np.array(d["curve"]); sd = lambda x: np.std(x, ddof=1) if len(x) > 1 else 0.0
        print(f"{name:30} {c[:, 0].mean():>11.3f} {c[:, -1].mean():>7.3f}±{sd(c[:, -1]):.3f} "
              f"{np.mean(d['new']):>7.3f}±{sd(d['new']):.3f} {np.mean(d['stale']):>8.3f}")
