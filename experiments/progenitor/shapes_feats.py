"""Cache front-end features per split: outputs/data/shapes/feat_<front>_<split>.pt (fp16)."""
import os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.progenitor import frontend as FE, shapes as SH
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "6")))
FRONTS = {"ds": FE.dense_stereo, "col": FE.colour_block, "bd": FE.boundary_block, "cn": FE.corner_block}
def feats(front, split):
    p = os.path.join(SH.OUT, f"feat_{front}_{split}.pt")
    if os.path.exists(p): return torch.load(p).float()
    X, _, _ = SH.load(split); t = time.time(); Z = FE.batched(FRONTS[front], X); torch.save(Z.half(), p)
    print(f"  cached {front}/{split} {tuple(Z.shape)} in {time.time()-t:.0f}s", flush=True); return Z
def grouped(split, canon=False):
    """grouping.describe features per split, cached (fp16): feat_grp[_canon]_<split>.pt"""
    from experiments.progenitor import grouping as G
    p = os.path.join(SH.OUT, f"feat_grp{'_canon' if canon else ''}_{split}.pt")
    if os.path.exists(p): return {k: v.float() for k, v in torch.load(p).items()}
    X, _, _ = SH.load(split); t = time.time(); D, _ = G.describe(X, canon=canon); torch.save({k: v.half() for k, v in D.items()}, p)
    print(f"  grouped{'-canon' if canon else ''} {split} {tuple(D['silhouette'].shape)} in {time.time()-t:.0f}s", flush=True); return {k: v.float() for k, v in D.items()}
def dsc(split): return torch.cat([feats("ds", split), feats("col", split)], 1)
if __name__ == "__main__":
    for f in FRONTS:
        for s in SH.SPLITS: feats(f, s)
