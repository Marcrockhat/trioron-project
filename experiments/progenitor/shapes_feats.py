"""Cache front-end features per split: outputs/data/shapes/feat_<front>_<split>.pt (fp16)."""
import os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.progenitor import frontend as FE, shapes as SH
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "6")))
FRONTS = {"ds": FE.dense_stereo, "col": FE.colour_block}
def feats(front, split):
    p = os.path.join(SH.OUT, f"feat_{front}_{split}.pt")
    if os.path.exists(p): return torch.load(p).float()
    X, _, _ = SH.load(split); t = time.time(); Z = FE.batched(FRONTS[front], X); torch.save(Z.half(), p)
    print(f"  cached {front}/{split} {tuple(Z.shape)} in {time.time()-t:.0f}s", flush=True); return Z
def dsc(split): return torch.cat([feats("ds", split), feats("col", split)], 1)
if __name__ == "__main__":
    for f in FRONTS:
        for s in SH.SPLITS: feats(f, s)
