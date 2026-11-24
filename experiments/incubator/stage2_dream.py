"""Stage 2 dream test — shadow copy by day, re-key at night, under a DOMAIN SHIFT.
(follow-up to stage2_drift.py, where later lessons had nothing new to teach)

Domains: articles clustered (cosine k-means, fixed seed) on the reflex's mean
read-state, LESSONS clusters (roughly places / people / history / concepts / other).
One lesson = one domain, taught in a random order, own data only, no replay.
The sense (PCA-whiten) is fitted on lesson 0 only and never refitted.

  old facts  taught questions of lesson 0's domain, stored after lesson 0
  new facts  taught questions of every later domain
  memory     old and new facts TOGETHER; one threshold (1 % of negatives through)

Conditions (LINEAR trioron leaf and QUAD leaf, same sense, same loss):
  frozen            key leaf stops after lesson 0.  Also what the organism runs
                    BY DAY in the shadow scheme (live leaf frozen, a copy learns)
  free, stale       leaf keeps learning; old keys left as stored
  free + dream      same leaf; at night every stored question is re-read and re-keyed
                    (= shadow copy swapped in + re-key)
  lock s, stale     epigenetic lock (uniform lambda, pull s) while learning; no re-key
  lock s + dream    lock while learning, then re-key
  joint (ceiling)   one leaf trained on all domains at once, same number of passes

Run from repo root:  python3 experiments/incubator/stage2_dream.py
Env: SEEDS=5 LESSONS=5 EPOCHS=60 K=256 E=32 STRENGTHS=1,30 TEST_FRAC=0.3
"""
from __future__ import annotations
import copy, os, sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(__file__))
from stage2_memory import load, split
from trioron.learning.epigenetic_lock import anchor, ewc_penalty, set_lambda
from trioron.organisms.base import chunked, leaf, train_contrastive
from trioron.senses.hidden_state import HiddenStateSense


def domains(Hmean, block, L, seed=0):
    """Domain id per row: cosine k-means over each article's mean read-state."""
    arts, inv = torch.unique(block, return_inverse=True)
    M = torch.zeros(len(arts), Hmean.shape[1]).index_add_(0, inv, Hmean) / torch.bincount(inv)[:, None]
    M = F.normalize(M - M.mean(0), dim=1)
    C = M[torch.randperm(len(arts), generator=torch.Generator().manual_seed(seed))[:L]]
    for _ in range(50):
        a = (M @ C.T).argmax(1)
        C = F.normalize(torch.stack([M[a == j].mean(0) for j in range(L)]), dim=1)
    return a[inv]


def score(E, Kmem, gmem, group, pos_old, pos_new, neg, f=0.01):
    """(old hit, old top1, new hit, new top1) against the joint memory."""
    tau = torch.quantile((E[neg] @ Kmem.T).max(1).values, 1 - f)
    out = []
    for pos in (pos_old, pos_new):
        s, i = (E[pos] @ Kmem.T).max(1)
        right = gmem[i] == group[pos]
        out += [(right & (s > tau)).float().mean().item(), right.float().mean().item()]
    return out


if __name__ == "__main__":
    H, group, block, variant, _ = load("both")
    seeds = range(int(os.environ.get("SEEDS", 5)))
    L, epochs = int(os.environ.get("LESSONS", 5)), int(os.environ.get("EPOCHS", 60))
    k, e = int(os.environ.get("K", 256)), int(os.environ.get("E", 32))
    strengths = [float(x) for x in os.environ.get("STRENGTHS", "1,30").split(",")]
    frac = float(os.environ.get("TEST_FRAC", 0.3))
    dom = domains(H[:, H.shape[1] // 2:], block, L)
    print(f"# stage 2 dream  questions={len(H) // 3}  seeds={list(seeds)} lessons={L} epochs/lesson={epochs} "
          f"k={k} e={e} strengths={strengths}  domain rows={torch.bincount(dom).tolist()}", flush=True)
    res: dict[str, list] = {}
    for seed in seeds:
        m = split(group, block, variant, seed, frac)
        order = torch.randperm(L, generator=torch.Generator().manual_seed(1000 + seed))
        lesson = [m["train"] & (dom == d) for d in order]
        first = dom == order[0]
        mem_old, mem_new = m["memory"] & first, m["memory"] & ~first
        pos_old, pos_new = m["pos"] & first, m["pos"] & ~first
        neg = m["neg_hard"] | m["neg_easy"]
        gmem = torch.cat([group[mem_old], group[mem_new]])
        print(f"  seed {seed}: first domain {int(order[0])}  lesson rows {[int(x.sum()) for x in lesson]}  "
              f"facts old {int(mem_old.sum())} new {int(mem_new.sum())}", flush=True)
        P = HiddenStateSense(H[lesson[0]], k, dup=False)(H)
        for arm, nonlinear in (("linear", False), ("quad", True)):
            Z = torch.cat([P, P], 1) if nonlinear else P          # dup only matters to the quad
            key = lambda sub: F.normalize(chunked(sub, Z), dim=1)

            def teach(sub, rows, s, penalty=None):
                train_contrastive(sub, sub.trainable_tensors(), Z[rows], group[rows], block[rows], s, epochs,
                                  blocks_per_batch=48, penalty=penalty)
                with torch.no_grad(): sub(Z[:1])                    # drop grad-bearing activations

            def record(tag, E, K_old):
                r = score(E, torch.cat([K_old, E[mem_new]]), gmem, group, pos_old, pos_new, neg)
                res.setdefault(f"{arm:6} {tag}", []).append(r)
                print(f"    {arm:6} {tag:22} old {r[0]:.3f} (top1 {r[1]:.3f})  new {r[2]:.3f} (top1 {r[3]:.3f})", flush=True)

            base = leaf(Z.shape[1], e, seed, nonlinear=nonlinear)
            teach(base, lesson[0], seed * 100)
            E0 = key(base)
            K_old = E0[mem_old]
            record("frozen (= by day)", E0, K_old)
            for tag, s in [("free", 0.0)] + [(f"lock s={s:g}", s) for s in strengths]:
                sub = copy.deepcopy(base)
                pen = None
                if s:
                    a = sub.arena
                    anchor(a); set_lambda(a, torch.ones_like(a.node_lambda), "absolute")
                    pen = lambda a=a, s=s: s * ewc_penalty(a)
                for t in range(1, L):
                    teach(sub, lesson[t], seed * 100 + t, pen)
                E = key(sub)
                record(f"{tag}, stale", E, K_old)
                record(f"{tag} + dream", E, E[mem_old])
            joint = leaf(Z.shape[1], e, seed, nonlinear=nonlinear)
            teach(joint, m["train"], seed * 100)
            E = key(joint)
            record("joint (ceiling)", E, E[mem_old])
    print(f"\n{'condition':30} {'old hit':>12} {'old top1':>9} {'new hit':>12} {'new top1':>9}")
    for name, rows in res.items():
        r = np.array(rows); sd = r.std(0, ddof=1) if len(r) > 1 else np.zeros(4)
        mu = r.mean(0)
        print(f"{name:30} {mu[0]:>6.3f}±{sd[0]:.3f} {mu[1]:>9.3f} {mu[2]:>6.3f}±{sd[2]:.3f} {mu[3]:>9.3f}")
