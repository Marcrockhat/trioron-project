"""Stage 2 gate — fact memory: does the organism's learned key catch a taught
fact asked in other words, without hijacking other questions, better than a
nearest-neighbour cache on the same hidden states?  (docs/design/incubator_matron.md §3)

In : outputs/incubator/stage2_keys.{jsonl,npz} (stage2_collect.py), stage1_answers.jsonl
Protocol, per split (articles held out; the key space is learned on the rest):
  taught     one question from each of half the held-out two-question articles;
             its ORIGINAL wording is stored in memory
  positives  the two paraphrases of each taught question (must recall that fact)
  negatives  hard: the other question of the same article, in all 3 wordings
             easy: every question of the untaught held-out articles, 3 wordings
  crowded    memory also holds every training-article question (retention under load)
Scores: top1 = right fact is nearest (no threshold); hit@f = right fact recalled
at the threshold that lets a fraction f of negatives through (false override).
Arms: nearest-neighbour cache on raw / centred / whitened states, and on
projections as small as the organism's key (PCA, whitened, random); a learned
LINEAR key; the organism's quad key leaf (Vee.learn_keys).  Same loss, same data.

Run from repo root:  python3 experiments/incubator/stage2_memory.py
Env: SEEDS=3 K=256 E=32 EPOCHS=30 TEST_FRAC=0.3 STATE=both|last|mean
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.getcwd())
from trioron.organisms import Vee
from trioron.organisms.base import train_contrastive
from trioron.senses.hidden_state import HiddenStateSense

OUT = "outputs/incubator"
VAR = ("q", "p1", "p2")


def load(state: str):
    """Rows for questions that have all three wordings.  Returns H [N, d],
    group (question index), block (article id), variant (0/1/2), texts."""
    art = {r["qid"]: r["id"] for r in map(json.loads, open(f"{OUT}/stage1_answers.jsonl"))}
    idx = [json.loads(l) for l in open(f"{OUT}/stage2_keys.jsonl")]
    z = np.load(f"{OUT}/stage2_keys.npz")
    n = min(len(idx), len(z["h_last"])); idx = idx[:n]
    have: dict[str, dict[str, int]] = {}
    for i, r in enumerate(idx):
        have.setdefault(r["qid"], {})[r["variant"]] = i
    qids = [q for q, v in have.items() if all(x in v for x in VAR)]
    rows = [have[q][v] for q in qids for v in VAR]
    parts = {"last": ["h_last"], "mean": ["h_mean"], "both": ["h_last", "h_mean"]}[state]
    H = torch.cat([torch.tensor(z[p][rows].astype(np.float32)) for p in parts], 1)
    group = torch.arange(len(qids)).repeat_interleave(3)
    block = torch.tensor([art[q] for q in qids]).repeat_interleave(3)
    variant = torch.arange(3).repeat(len(qids))
    return H, group, block, variant, [idx[i]["text"] for i in rows]


def split(group, block, variant, seed, frac):
    """Masks over rows: train, memory (taught originals), pos, neg_hard, neg_easy."""
    g = torch.Generator().manual_seed(seed)
    arts = torch.unique(block)
    test = arts[torch.randperm(len(arts), generator=g)[:int(len(arts) * frac)]]
    te = torch.isin(block, test)
    taught_art = test[torch.randperm(len(test), generator=g)[:len(test) // 2]]
    in_taught = torch.isin(block, taught_art)
    taught_q = torch.zeros(int(group.max()) + 1, dtype=torch.bool)
    for a in taught_art.tolist():                       # one question per taught article
        qs = torch.unique(group[block == a])
        if len(qs) >= 2:                                # needs a sibling to be a hard test
            taught_q[qs[torch.randint(len(qs), (1,), generator=g)]] = True
    tq = taught_q[group]
    return {"train": ~te, "memory": tq & (variant == 0), "pos": tq & (variant > 0),
            "neg_hard": in_taught & ~tq, "neg_easy": te & ~in_taught}


def evaluate(E, group, m, crowded: bool) -> dict:
    """E: [N, e] keys for every row (any arm).  Cosine nearest neighbour."""
    E = F.normalize(E, dim=1)
    mem = m["memory"] | (m["train"] & (torch.arange(len(E)) % 3 == 0) if crowded else False)
    K, kg = E[mem], group[mem]

    def nearest(mask):
        s, i = (E[mask] @ K.T).max(1)
        return s, kg[i]
    sp, gp = nearest(m["pos"])
    right = gp == group[m["pos"]]
    sh, _ = nearest(m["neg_hard"]); se, _ = nearest(m["neg_easy"])
    neg = torch.cat([sh, se])
    out = {"top1": right.float().mean().item(), "n_mem": int(mem.sum())}
    for f in (0.01, 0.05):
        tau = torch.quantile(neg, 1 - f)
        out[f"hit@{f:.0%}"] = (right & (sp > tau)).float().mean().item()
        out[f"hardshare@{f:.0%}"] = ((sh > tau).sum() / max(1, (neg > tau).sum())).item()
    return out


if __name__ == "__main__":
    state = os.environ.get("STATE", "both")
    H, group, block, variant, texts = load(state)
    seeds = range(int(os.environ.get("SEEDS", 3)))
    k, e = int(os.environ.get("K", 256)), int(os.environ.get("E", 32))
    epochs, frac = int(os.environ.get("EPOCHS", 30)), float(os.environ.get("TEST_FRAC", 0.3))
    print(f"# stage 2 memory  questions={len(H) // 3}  articles={len(torch.unique(block))}  "
          f"state={state} d={H.shape[1]}  seeds={list(seeds)} k={k} e={e} epochs={epochs}")
    res: dict[str, dict[str, list[float]]] = {}
    for seed in seeds:
        m = split(group, block, variant, seed, frac)
        tr = m["train"]
        if seed == 0:
            print(f"# split 0: train rows {int(tr.sum())}  taught {int(m['memory'].sum())}  "
                  f"pos {int(m['pos'].sum())}  neg hard {int(m['neg_hard'].sum())}  easy {int(m['neg_easy'].sum())}")
        sense = HiddenStateSense(H[tr], k, dup=False)
        Z = sense(H)
        C = H - H[tr].mean(0)
        arms = {"NN raw": H, "NN centred": C, f"NN whitened k={k}": Z}
        # caches at the organism's key size (e numbers per fact)
        arms[f"NN PCA-{e} of centred"] = C @ torch.linalg.svd(C[tr], full_matrices=False)[2][:e].T
        arms[f"NN whitened k={e}"] = HiddenStateSense(H[tr], e, dup=False)(H)
        arms[f"NN random projection {e}"] = C @ torch.randn(C.shape[1], e, generator=torch.Generator().manual_seed(seed))
        torch.manual_seed(seed)
        lin = torch.nn.Linear(Z.shape[1], e)
        train_contrastive(lin, list(lin.parameters()), Z[tr], group[tr], block[tr], seed, epochs)
        with torch.no_grad(): arms[f"linear key k={k} e={e}"] = lin(Z)
        vee = Vee(seed).learn_keys(H[tr], group[tr], block[tr], k=k, e=e, epochs=epochs)
        arms[f"organism key k={k} e={e}"] = vee.key(H)
        n_par = {f"linear key k={k} e={e}": sum(p.numel() for p in lin.parameters()),
                 f"organism key k={k} e={e}": vee.n_params()}
        for name, E in arms.items():
            for crowded in (False, True):
                r = evaluate(E, group, m, crowded)
                d = res.setdefault(name + (" | crowded" if crowded else ""), {})
                for kk, v in r.items(): d.setdefault(kk, []).append(v)
                if name in n_par: d["params"] = [n_par[name]]
        print(f"  seed {seed} done", flush=True)
    cols = ("top1", "hit@1%", "hit@5%", "hardshare@1%")
    print(f"{'arm':40} {'mem':>5} " + " ".join(f"{c:>13}" for c in cols) + "  params")
    for name, d in res.items():
        cell = lambda c: f"{np.mean(d[c]):.3f}±{np.std(d[c], ddof=1) if len(d[c]) > 1 else 0:.3f}"
        print(f"{name:40} {int(np.mean(d['n_mem'])):>5} " + " ".join(f"{cell(c):>13}" for c in cols)
              + (f"  {d['params'][0]}" if "params" in d else ""))
    small_arms = {f"NN PCA-{e} of centred", f"NN whitened k={e}", f"NN random projection {e}"}
    for tag in ("", " | crowded"):
        of = lambda small: max(np.mean(d["hit@1%"]) for n_, d in res.items() if n_.startswith("NN")
                               and n_.endswith("crowded") == bool(tag) and (n_.split(" |")[0] in small_arms) == small)
        nn, nn_small = of(False), of(True)
        org = np.mean(res[f"organism key k={k} e={e}{tag}"]["hit@1%"])
        lin_ = np.mean(res[f"linear key k={k} e={e}{tag}"]["hit@1%"])
        print(f"GATE{tag}: organism hit@1% {org:.3f} vs best NN cache {nn:.3f} -> "
              f"{'PASS' if org > nn else 'FAIL'} ({org - nn:+.3f}); vs best cache at {e} numbers/fact {nn_small:.3f} "
              f"({org - nn_small:+.3f}); vs linear key {lin_:.3f} ({org - lin_:+.3f})")
