"""Stage 1 gate — awareness: does the organism predict the reflex's errors
better than the reflex's own confidence?  (docs/design/incubator_matron.md §3)

In : outputs/incubator/stage1_answers.jsonl + stage1_hidden.npz (stage1_collect.py)
Arms (all scored by AUROC for "the reflex is wrong" on held-out ARTICLES):
  baselines   mean / min / first-token log-prob of the reflex's answer
  linear      logistic regression on the raw 1536-d hidden state (reference:
              what a plain linear readout gets)
  organism    Syneidesis (PCA-k sense + one quad leaf) on h_pre, h_post, both
  combined    organism doubt + mean log-prob (2-feature logistic)
GATE: best organism arm > best log-prob baseline, n>=3 seeds, spread reported.

Run from repo root:  python3 experiments/incubator/stage1_awareness.py
Env: SEEDS=3 KS=64 EPOCHS=30 TEST_FRAC=0.3
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import torch
sys.path.insert(0, os.getcwd())
from trioron.organisms import Syneidesis

OUT = "outputs/incubator"


def auroc(score: torch.Tensor, y: torch.Tensor) -> float:
    """P(score_pos > score_neg), ties count half (rank formula)."""
    s = score.double(); y = y.bool()
    order = torch.argsort(s)
    ranks = torch.empty_like(s); ranks[order] = torch.arange(1, len(s) + 1, dtype=torch.double)
    for v in torch.unique(s):                       # average ranks over ties
        m = s == v
        if m.sum() > 1: ranks[m] = ranks[m].mean()
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def logistic(X, y, Xt, seed, wd=1e-2, epochs=300):
    """Full-batch logistic regression (Adam, L2); returns test scores."""
    torch.manual_seed(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    X, Xt = (X - mu) / sd, (Xt - mu) / sd
    lin = torch.nn.Linear(X.shape[1], 1)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2, weight_decay=wd)
    for _ in range(epochs):
        opt.zero_grad()
        torch.nn.functional.binary_cross_entropy_with_logits(lin(X).squeeze(1), y.float()).backward()
        opt.step()
    with torch.no_grad():
        return lin(Xt).squeeze(1)


if __name__ == "__main__":
    rows = [json.loads(l) for l in open(f"{OUT}/stage1_answers.jsonl")]
    z = np.load(f"{OUT}/stage1_hidden.npz")
    n = min(len(rows), len(z["h_pre"])); rows = rows[:n]
    Hpre = torch.tensor(z["h_pre"][:n].astype(np.float32))
    Hpost = torch.tensor(z["h_post"][:n].astype(np.float32))
    wrong = torch.tensor([0 if r["correct"] else 1 for r in rows])
    lp = [r["logprobs"] or [-20.0] for r in rows]
    conf = {"lp_mean": torch.tensor([float(np.mean(x)) for x in lp]),
            "lp_min": torch.tensor([float(np.min(x)) for x in lp]),
            "lp_first": torch.tensor([float(x[0]) for x in lp])}
    art = torch.tensor([r["id"] for r in rows])
    seeds = range(int(os.environ.get("SEEDS", 3)))
    ks = [int(x) for x in os.environ.get("KS", "64").split(",")]
    epochs = int(os.environ.get("EPOCHS", 30))
    frac = float(os.environ.get("TEST_FRAC", 0.3))
    print(f"# stage 1 awareness  items={n}  reflex accuracy={1 - wrong.float().mean():.3f}  "
          f"seeds={list(seeds)} ks={ks} epochs={epochs}")
    res: dict[str, list[float]] = {}
    add = lambda k, v: res.setdefault(k, []).append(v)
    for seed in seeds:
        g = torch.Generator().manual_seed(seed)
        ids = torch.unique(art)
        test_ids = ids[torch.randperm(len(ids), generator=g)[:int(len(ids) * frac)]]
        te = torch.isin(art, test_ids); tr = ~te
        yt = wrong[te]
        for name, c in conf.items():
            add(name, auroc(-c[te], yt))                   # low confidence => wrong
        add("linear h_pre", auroc(logistic(Hpre[tr], wrong[tr], Hpre[te], seed), yt))
        add("linear h_post", auroc(logistic(Hpost[tr], wrong[tr], Hpost[te], seed), yt))
        for k in ks:
            for tag, H in (("h_pre", Hpre), ("h_post", Hpost), ("both", torch.cat([Hpre, Hpost], 1))):
                o = Syneidesis(seed, k=k).fit(H[tr], wrong[tr], epochs=epochs)
                d = o.doubt(H[te])
                add(f"organism {tag} k={k}", auroc(d, yt))
                if tag == "h_post":
                    dtr = o.doubt(H[tr])
                    Xtr = torch.stack([dtr, conf["lp_mean"][tr]], 1)
                    Xte = torch.stack([d, conf["lp_mean"][te]], 1)
                    add(f"combined h_post k={k} + lp_mean",
                        auroc(logistic(Xtr, wrong[tr], Xte, seed, wd=0.0), yt))
                    add(f"  (params k={k})", float(o.n_params()))
    print(f"{'arm':38} {'AUROC mean±std':>16}   per-seed")
    for k, v in res.items():
        t = torch.tensor(v)
        if k.startswith("  (params"):
            print(f"{k:38} {int(t[0]):>16}")
        else:
            print(f"{k:38} {t.mean():.3f}±{t.std() if len(v) > 1 else 0:.3f}      "
                  + " ".join(f"{x:.3f}" for x in v))
    base = max(torch.tensor(res[k]).mean().item() for k in conf)
    org = {k: torch.tensor(v).mean().item() for k, v in res.items() if k.startswith("organism")}
    best = max(org, key=org.get)
    print(f"GATE: best organism arm [{best}] {org[best]:.3f} vs best log-prob baseline {base:.3f} "
          f"-> {'PASS' if org[best] > base else 'FAIL'} (margin {org[best] - base:+.3f})")
