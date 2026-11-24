"""Stage 2 consolidation, readout check — does the trioron leaf lose because of
the quad leaf or because of the softmax readout?  (Rocky s066.)

stage2_consolidate.py changed two things at once in its leaf arm: the trunk
(quad cells instead of a linear key) and the readout (softmax over one output
cell per fact instead of a cosine to a stored row).  Here the readout is held
at cosine and only the trunk varies.  Same splits, memory and scoring.

Arms, for each h (numbers per fact):
  table, linear             linear key e=h, stored wording's key (reference)
  leaf softmax, top prob    the s065 leaf arm again (reference)
  same leaf, cosine codes   THE SAME trained weights; the fact's row is the
                            h-number interior code of its stored wording and
                            the score is the cosine to it (readout swap only)
  table, quad               h quad cells trained like the linear key
                            (contrastive on training articles), row = code
  quad rows + trunk         that trunk and its rows trained on the
                            consolidation set with the cosine classifier
                            (= "rows + trunk" with quad cells)

Run from repo root:  python3 experiments/incubator/stage2_consolidate_cos.py
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
from stage2_consolidate import hits, train_rows
from trioron.learning.manifold import get_interior_ids
from trioron.organisms.base import chunked, leaf, train_contrastive, train_leaf
from trioron.senses.hidden_state import HiddenStateSense


class Codes(torch.nn.Module):
    """A leaf read at its interior cells (h numbers per row), as a trunk."""

    def __init__(self, sub):
        super().__init__()
        self.sub, self.ids = sub, get_interior_ids(sub.arena).long()

    def forward(self, Z):
        self.sub(Z)
        return self.sub.live_activations[:, self.ids]      # grad-bearing

    def parameters(self):
        return self.sub.trainable_tensors()

    @torch.no_grad()
    def all(self, Z, n=4000):
        return torch.cat([self(Z[i:i + n]).detach() for i in range(0, len(Z), n)])


if __name__ == "__main__":
    H, group, block, variant, _ = load("both")
    seeds = range(int(os.environ.get("SEEDS", 3)))
    k, epochs = int(os.environ.get("K", 256)), int(os.environ.get("EPOCHS", 60))
    key_epochs, frac = int(os.environ.get("KEY_EPOCHS", 30)), float(os.environ.get("TEST_FRAC", 0.3))
    hs = [int(x) for x in os.environ.get("HS", "8,16,32").split(",")]
    print(f"# stage 2 consolidate, cosine readout  questions={len(H) // 3} seeds={list(seeds)} k={k} h={hs} epochs={epochs}", flush=True)
    res: dict[str, list] = {}
    for seed in seeds:
        m = split(group, block, variant, seed, frac)
        tr = m["train"]
        P = HiddenStateSense(H[tr], k, dup=False)(H)
        mem = m["memory"] | (tr & (variant == 0))
        n = int(mem.sum())
        cls = torch.full((int(group.max()) + 1,), -1)
        cls[group[mem]] = torch.arange(n)
        cls = cls[group]
        g = torch.Generator().manual_seed(seed)
        gm = torch.zeros(int(group.max()) + 1, P.shape[1]).index_add_(0, group, P) / 3
        resid = (P - gm[group])[tr]
        taught = m["memory"].nonzero().squeeze(1)
        aug = P[taught].repeat(3, 1) + resid[torch.randint(len(resid), (3 * len(taught),), generator=g)]
        X = torch.cat([P[tr], P[taught], aug])
        y = torch.cat([cls[tr], cls[taught], cls[taught].repeat(3)])
        Z, Xq = torch.cat([P, P], 1), torch.cat([X, X], 1)      # fan-in duplicated across the two branches

        def record(name, score, pred):
            res.setdefault(name, []).append(hits(score, pred, cls, m))
            print(f"  seed {seed} {name:34} hit {res[name][-1][0]:.3f} top1 {res[name][-1][1]:.3f}", flush=True)

        def cosine(E):
            E = F.normalize(E, dim=1)
            return (E @ E[mem].T).max(1)

        for h in hs:
            torch.manual_seed(seed)
            lin = torch.nn.Linear(k, h)
            train_contrastive(lin, list(lin.parameters()), P[tr], group[tr], block[tr], seed, key_epochs)
            with torch.no_grad(): record(f"h={h:<3} table, linear", *cosine(lin(P)))

            sub = leaf(Z.shape[1], n, seed, interior=h, max_bytes=4_000_000)
            train_leaf(sub, Xq, y, seed, epochs)
            logit = chunked(sub, Z)
            record(f"h={h:<3} leaf softmax, top prob", logit.softmax(1).max(1).values, logit.argmax(1))
            record(f"h={h:<3} same leaf, cosine codes", *cosine(Codes(sub).all(Z)))

            trunk = Codes(leaf(Z.shape[1], 2, seed, interior=h, max_bytes=4_000_000))
            train_contrastive(trunk, trunk.parameters(), Z[tr], group[tr], block[tr], seed, key_epochs)
            E = trunk.all(Z)
            record(f"h={h:<3} table, quad", *cosine(E))
            proto = F.normalize(E[mem], dim=1).clone()
            train_rows(trunk, proto, Xq, y, seed, epochs, True)
            with torch.no_grad():
                s, i = (F.normalize(trunk.all(Z), dim=1) @ F.normalize(proto, dim=1).T).max(1)
            record(f"h={h:<3} quad rows + trunk", s, i)
    print(f"\n{'arm':40} {'hit@1%':>13} {'top1':>7}")
    for name, rows in res.items():
        r = np.array(rows); sd = r.std(0, ddof=1) if len(r) > 1 else np.zeros(2)
        print(f"{name:40} {r[:, 0].mean():>7.3f}±{sd[0]:.3f} {r[:, 1].mean():>7.3f}")
