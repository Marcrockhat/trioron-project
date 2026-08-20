"""s054: shape-world-trained nest as a PRE-FEEDER organ for CIFAR (Rocky's ask).

Trains the s053 nest (shape 103 / whole 205 / fill 292 leaves, joint 8 ep on the
shape-world train split, seed 0) once, freezes it, and runs CIFAR-100 through it:
per leaf, the 48-d interior H-code + the class logits (5/5/4) are the organ's
output = 48*3 + 14 = 158-d per image.  The organ keeps its NATIVE calibration:
CIFAR features are standardized with the SHAPE-WORLD train statistics before
entering the leaves (a transplanted retina, not a re-fit one).

Cache: outputs/data/cifar/feat_pre_<split>.pt (fp16); organ weights are not kept
(deterministic retrain: torch.manual_seed(PRE_SEED) + fixed data order).
Consumed by cifar_continual.py streams "pre" -> readers mono+pre / nest+pre.
Run standalone to build the cache: python3 experiments/progenitor/shape_prefeeder.py
Sanity printed: organ's shape/fill acc on shapes test_fresh (expect ~.75/.80).
"""
import os, sys, time, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.progenitor import shapes as SH, shapes_feats as SF
from trioron.bases.seeded import Seeded
from trioron.core import Envelope, construct
from trioron.phenotype import default_dispatch_table
from trioron.learning.manifold import get_interior_ids
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "6")))
PRE_SEED = int(os.environ.get("PRE_SEED", "0")); PRE_EPOCHS = int(os.environ.get("PRE_EPOCHS", "8"))

def _streams(GD, WH):
    return {"shape": torch.cat([GD["silhouette"], GD["frame"], GD["flags"]], 1), "whole": WH,
            "fill": torch.cat([GD["ctex"], GD["cstereo"], GD["flags"]], 1)}

def _shape_streams(split):
    GD = SF.grouped(split, canon="scale"); WH = torch.cat([SF.feats(k, split) for k in ("bd", "col", "cn")], 1)
    return _streams(GD, WH)

class Organ:
    """The trained-on-shapes nest, frozen; __call__(streams dict) -> [N, 158] (3x48 H + 14 logits)."""
    N_OUT = {"shape": 5, "whole": 5, "fill": 4}
    def __init__(self):
        tr = _shape_streams("train")
        self.std = {k: (Z.mean(0), Z.std(0) + 1e-6) for k, Z in tr.items()}   # SHAPE-WORLD calibration
        self.leaves = {}
        for k, Z in tr.items():
            torch.manual_seed(PRE_SEED)
            sub = construct(base=Seeded(Z.shape[1], self.N_OUT[k], interior_cells=48, nonlinear=True), envelope=Envelope(max_parameter_bytes=400_000),
                            dispatch_table=default_dispatch_table(), capacity=Z.shape[1] + 48 + self.N_OUT[k] + 8 + int(os.environ.get("ORGAN_CAP_EXTRA", "0")), sparsity_k=0)   # ORGAN_CAP_EXTRA: spare cells for grafts (motion_absorb)
            sub.compile(); sub.prepare_training(); self.leaves[k] = sub
        self._train(tr)
        for sub in self.leaves.values():
            for p in sub.trainable_tensors(): p.requires_grad_(False)
        self.h_ids = {k: get_interior_ids(sub.arena).long() for k, sub in self.leaves.items()}
    def _norm(self, k, Z): mu, sd = self.std[k]; return ((Z - mu) / sd).float()
    def _train(self, tr):
        ys = SH.load("train")[1]; y = {"shape": ys["y_shape"], "whole": ys["y_shape"], "fill": ys["y_fill"]}
        Zn = {k: self._norm(k, Z) for k, Z in tr.items()}
        params = [p for sub in self.leaves.values() for p in sub.trainable_tensors()]
        opt = torch.optim.Adam(params, lr=1e-3); g = torch.Generator().manual_seed(PRE_SEED); t = time.time()
        for ep in range(PRE_EPOCHS):
            for bi in torch.randperm(len(ys["y_shape"]), generator=g).split(256):
                opt.zero_grad(); sum(F.cross_entropy(self.leaves[k](Zn[k][bi]), y[k][bi]) for k in self.leaves).backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        print(f"  organ trained ({PRE_EPOCHS} ep, seed {PRE_SEED}) in {time.time()-t:.0f}s", flush=True)
    def __call__(self, streams, chunk=4000):
        outs = []
        with torch.no_grad():
            for k in ("shape", "whole", "fill"):
                sub = self.leaves[k]; Z = self._norm(k, streams[k]); hs, lg = [], []
                for i in range(0, len(Z), chunk):
                    o = sub(Z[i:i + chunk]); hs.append(sub.last_activations[:, self.h_ids[k]]); lg.append(o)
                outs += [torch.cat(hs), torch.cat(lg)]
        return torch.cat(outs, 1)   # [N, 48+5 + 48+5 + 48+4] = 158
    def sanity(self):
        te = _shape_streams("test_fresh"); ys = SH.load("test_fresh")[1]
        with torch.no_grad():
            sh = F.log_softmax(self.leaves["shape"](self._norm("shape", te["shape"])), 1) + F.log_softmax(self.leaves["whole"](self._norm("whole", te["whole"])), 1)
            fl = self.leaves["fill"](self._norm("fill", te["fill"]))
        print(f"  organ sanity (shapes test_fresh): shape {float((sh.argmax(1)==ys['y_shape']).float().mean()):.3f} fill {float((fl.argmax(1)==ys['y_fill']).float().mean()):.3f}", flush=True)

CIFAR_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "outputs", "data", "cifar")

def cifar_pre(split):
    """Cached [N,158] pre-feeder features for a CIFAR split ('train'|'test').
    Reads the cifar_continual.py feature caches directly (cifar_continual runs on import)."""
    p = os.path.join(CIFAR_OUT, f"feat_pre_{split}.pt")
    if os.path.exists(p): return torch.load(p).float()
    global _ORGAN
    if "_ORGAN" not in globals():
        _ORGAN = Organ(); _ORGAN.sanity()
    GD = {k: v.float() for k, v in torch.load(os.path.join(CIFAR_OUT, f"feat_grp_canon_{split}.pt")).items()}
    WH = torch.cat([torch.load(os.path.join(CIFAR_OUT, f"feat_{k}_{split}.pt")).float() for k in ("bd", "col", "cn")], 1)
    t = time.time(); Z = _ORGAN(_streams(GD, WH)); torch.save(Z.half(), p)
    print(f"  cached pre/{split} {tuple(Z.shape)} in {time.time()-t:.0f}s", flush=True); return Z

if __name__ == "__main__":
    for sp in ("train", "test"): cifar_pre(sp)
