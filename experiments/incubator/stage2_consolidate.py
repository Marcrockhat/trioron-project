"""Stage 2 consolidation test — is a fact cheaper in WEIGHTS than in the table?
(Rocky s065: where could growth earn its place.)

The table stores one learned key per fact (h numbers), written in one shot from
the question's original wording.  Consolidation (a dream step) instead TRAINS
the stored set into a network: every fact is a class, and each fact owns the h
incoming weights of its output cell.  Same h numbers per fact; the network also
gets to learn what separates these particular facts from each other.

Memory: the crowded set (taught held-out facts + every training-article
question, ~2,160 facts).  Tested as in stage2_memory.py: the two unseen
rewordings of each taught fact must recall it; siblings and untaught questions
must not; hit = right fact AND above the threshold that lets 1 % of negatives through.

Consolidation data: training-article facts in their 3 wordings (kept lesson
material); taught facts in their ONE stored wording plus 3 copies with borrowed
variation (the wording residual of a random training question added on).

Arms, for each h (numbers per fact):
  table              linear key e=h, key of the original wording stored (baseline)
  rows only          key leaf frozen; the table rows become trainable prototypes
                     (cosine classifier, started from the stored keys)
  rows + trunk       same, the linear key leaf trains too
  trioron leaf       Seeded quad leaf, h interior cells, one OUTPUT CELL PER FACT
                     (score = top logit / top probability)

Run from repo root:  python3 experiments/incubator/stage2_consolidate.py
Env: SEEDS=3 K=256 HS=8,16,32 EPOCHS=60 KEY_EPOCHS=30 TEST_FRAC=0.3
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(__file__))
from stage2_memory import load, split
from trioron.organisms.base import chunked, leaf, train_contrastive, train_leaf
from trioron.senses.hidden_state import HiddenStateSense


def hits(score, pred, cls, m, f=0.01):
    """score/pred per row (best stored fact and how sure); cls = true class or -1."""
    neg = score[m["neg_hard"] | m["neg_easy"]]
    tau = torch.quantile(neg, 1 - f)
    right = pred[m["pos"]] == cls[m["pos"]]
    return (right & (score[m["pos"]] > tau)).float().mean().item(), right.float().mean().item()


def train_rows(trunk, proto, X, y, seed, epochs, train_trunk, scale=20.0, lr=1e-3):
    params = [proto] + (list(trunk.parameters()) if train_trunk else [])
    for p in params: p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=lr)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for bi in torch.randperm(len(y), generator=g).split(256):
            opt.zero_grad()
            logits = scale * F.normalize(trunk(X[bi]), dim=1) @ F.normalize(proto, dim=1).T
            F.cross_entropy(logits, y[bi]).backward()
            opt.step()
    for p in params: p.requires_grad_(False)


if __name__ == "__main__":
    H, group, block, variant, _ = load("both")
    seeds = range(int(os.environ.get("SEEDS", 3)))
    k, epochs = int(os.environ.get("K", 256)), int(os.environ.get("EPOCHS", 60))
    key_epochs, frac = int(os.environ.get("KEY_EPOCHS", 30)), float(os.environ.get("TEST_FRAC", 0.3))
    hs = [int(x) for x in os.environ.get("HS", "8,16,32").split(",")]
    print(f"# stage 2 consolidate  questions={len(H) // 3} seeds={list(seeds)} k={k} h={hs} epochs={epochs}", flush=True)
    res: dict[str, list] = {}
    npar: dict[str, int] = {}
    for seed in seeds:
        m = split(group, block, variant, seed, frac)
        tr = m["train"]
        P = HiddenStateSense(H[tr], k, dup=False)(H)
        mem = m["memory"] | (tr & (variant == 0))                 # the crowded memory
        n = int(mem.sum())
        cls = torch.full((int(group.max()) + 1,), -1)
        cls[group[mem]] = torch.arange(n)
        cls = cls[group]                                          # class of every row (-1 = not stored)
        # consolidation set: lesson facts x3 wordings, taught facts x1 + 3 borrowed-variation copies
        g = torch.Generator().manual_seed(seed)
        gm = torch.zeros(int(group.max()) + 1, P.shape[1]).index_add_(0, group, P) / 3
        resid = (P - gm[group])[tr]
        taught = m["memory"].nonzero().squeeze(1)
        aug = P[taught].repeat(3, 1) + resid[torch.randint(len(resid), (3 * len(taught),), generator=g)]
        X = torch.cat([P[tr], P[taught], aug])
        y = torch.cat([cls[tr], cls[taught], cls[taught].repeat(3)])
        if seed == 0:
            print(f"# split 0: facts in memory {n}  consolidation rows {len(y)}  pos {int(m['pos'].sum())}", flush=True)

        def record(name, score, pred, params):
            res.setdefault(name, []).append(hits(score, pred, cls, m)); npar[name] = params
            print(f"  seed {seed} {name:34} hit {res[name][-1][0]:.3f} top1 {res[name][-1][1]:.3f}", flush=True)

        for h in hs:
            torch.manual_seed(seed)
            lin = torch.nn.Linear(k, h)
            train_contrastive(lin, list(lin.parameters()), P[tr], group[tr], block[tr], seed, key_epochs)
            with torch.no_grad(): E = F.normalize(lin(P), dim=1)
            s, i = (E @ E[mem].T).max(1)
            record(f"h={h:<3} table", s, i, n * h)
            for name, tt in (("rows only", False), ("rows + trunk", True)):
                torch.manual_seed(seed)
                trunk = torch.nn.Linear(k, h); trunk.load_state_dict(lin.state_dict())
                proto = E[mem].clone()
                train_rows(trunk, proto, X, y, seed, epochs, tt)
                with torch.no_grad():
                    s, i = (F.normalize(trunk(P), dim=1) @ F.normalize(proto, dim=1).T).max(1)
                record(f"h={h:<3} {name}", s, i, n * h)
            Z, Xq = torch.cat([P, P], 1), torch.cat([X, X], 1)
            sub = leaf(Z.shape[1], n, seed, interior=h, max_bytes=4_000_000)
            train_leaf(sub, Xq, y, seed, epochs)
            logit = chunked(sub, Z)
            s, i = logit.max(1)
            pars = int((sub.arena.edge_weight != 0).sum())           # live weights (n_params counts allocated slots)
            record(f"h={h:<3} trioron leaf, top logit", s, i, pars)
            record(f"h={h:<3} trioron leaf, top prob", logit.softmax(1).max(1).values, i, pars)
    print(f"\n{'arm':40} {'numbers':>8} {'hit@1%':>13} {'top1':>7}")
    for name, rows in res.items():
        r = np.array(rows); sd = r.std(0, ddof=1) if len(r) > 1 else np.zeros(2)
        print(f"{name:40} {npar[name]:>8} {r[:, 0].mean():>7.3f}±{sd[0]:.3f} {r[:, 1].mean():>7.3f}")
