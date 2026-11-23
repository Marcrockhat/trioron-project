"""Stage 1 controls (s064) — is the gate pass real?

10 article-level splits, fixed arm (Vee on [h_pre, h_post], k=64):
  PAIRED organism - lp_min      same split, same test items
  linear on same PCA-64         does the quad leaf add anything over linear?
  no-brain control              writer id + question length + answer length only
                                (the four question writers differ in difficulty)
  combined                      organism doubt + the reflex's log-probs
  within writers                organism vs lp_min inside each writer pair

Run from repo root:  python3 experiments/incubator/stage1_controls.py
"""
import json, os, sys, numpy as np, torch
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "experiments/incubator")
from stage1_awareness import auroc, logistic
from trioron.organisms import Vee
from trioron.senses import HiddenStateSense
OUT = "outputs/incubator"
rows = [json.loads(l) for l in open(f"{OUT}/stage1_answers.jsonl")]
z = np.load(f"{OUT}/stage1_hidden.npz")
Hpre = torch.tensor(z["h_pre"].astype(np.float32)); Hpost = torch.tensor(z["h_post"].astype(np.float32))
B = torch.cat([Hpre, Hpost], 1)
wrong = torch.tensor([0 if r["correct"] else 1 for r in rows])
lpmin = torch.tensor([float(np.min(r["logprobs"] or [-20.0])) for r in rows])
lpmean = torch.tensor([float(np.mean(r["logprobs"] or [-20.0])) for r in rows])
art = torch.tensor([r["id"] for r in rows]); part = art // 400
qlen = torch.tensor([float(len(r["q"])) for r in rows]); alen = torch.tensor([float(len(r["logprobs"])) for r in rows])
R = {}
add = lambda k, v: R.setdefault(k, []).append(v)
for seed in range(10):
    g = torch.Generator().manual_seed(100 + seed)
    ids = torch.unique(art); te = torch.isin(art, ids[torch.randperm(len(ids), generator=g)[:int(len(ids) * .3)]]); tr = ~te
    yt = wrong[te]
    o = Vee(seed, k=64).fit(B[tr], wrong[tr], epochs=30); d = o.doubt(B[te]); dtr = o.doubt(B[tr])
    a_org, a_lp = auroc(d, yt), auroc(-lpmin[te], yt)
    add("organism both k=64", a_org); add("lp_min", a_lp); add("PAIRED organism - lp_min", a_org - a_lp)
    s = HiddenStateSense(B[tr], 64, dup=False)
    add("linear on same PCA-64 (both)", auroc(logistic(s(B[tr]), wrong[tr], s(B[te]), seed), yt))
    meta = lambda m: torch.stack([part[m].float(), qlen[m], alen[m]], 1)
    add("no-brain control: writer+q length+answer length", auroc(logistic(meta(tr), wrong[tr], meta(te), seed), yt))
    X = lambda m, dd: torch.stack([dd, lpmin[m], lpmean[m]], 1)
    add("combined organism + lp_min + lp_mean", auroc(logistic(X(tr, dtr), wrong[tr], X(te, d), seed, wd=0.0), yt))
    for name, sel in (("writers 0-1", part < 2), ("writers 2-3", part >= 2)):
        m = sel[te]
        add(f"  within {name}: organism", auroc(d[m], yt[m])); add(f"  within {name}: lp_min", auroc(-lpmin[te][m], yt[m]))
print(f"items {len(rows)}  accuracy by writer:", [round(1 - wrong[part == p].float().mean().item(), 3) for p in range(4)])
for k, v in R.items():
    t = torch.tensor(v); print(f"{k:52} {t.mean():+.3f}±{t.std():.3f}" + (f"   wins {int((t > 0).sum())}/10" if k.startswith("PAIRED") else ""))
