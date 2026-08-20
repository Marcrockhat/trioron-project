"""Motion leaf (s055, motion arc step 4): pure-trioron Seeded leaves on the fixed
motion primitive, trained on the moving-shape world (motion.py splits).

Streams (cached outputs/data/motion/feat_<stream>_<split>.pt, fp16):
  motion   : motion_front.motion_full  (cross-spectrum phasors 450 + energy 25 + population read-out 181 = 656)
  static   : s052 dense_stereo of the MID frame (800) -- magnitude-only = motion-blind CONTROL
  sil_col  : s053 colour/Otsu grouping on the mid frame -> largest body -> scale-canon silhouette -> boundary block (92)
  sil_mot  : common-fate grouping v3 (motion_diag.motion_group) -> same descriptor (92)
Leaves (Seeded(d, n_out, 48, nonlinear), joint 8 ep Adam 1e-3, n=SEEDS):
  velocity : motion -> y_vel (17-way)   vs  static -> y_vel (expect chance .06)
  shape    : sil_col -> y_shape (3-way) vs sil_mot -> y_shape vs sil_col++sil_mot  -- shape-from-motion-grouping:
             on photo backgrounds colour grouping fails, so the motion-grouped silhouette should carry the shape.
Eval on test (mixed), test_photo, test_flat; shape additionally on moving-only packets.
Env: SEEDS, EPOCHS, FEATS_ONLY, STREAMS (comma list to (re)build).
"""
import os, sys, time, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.progenitor import motion as MO, motion_front as MF, frontend as FE, grouping as GR, motion_diag as MD
from trioron.bases.seeded import Seeded
from trioron.core import Envelope, construct
from trioron.phenotype import default_dispatch_table
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "6")))
SEEDS = int(os.environ.get("SEEDS", "3")); EPOCHS = int(os.environ.get("EPOCHS", "8")); HIDDEN = 48
SPLITS = ["train", "test", "test_photo", "test_flat"]
def log(*a): print(*a, flush=True)

def _sil(masks):
    return FE.boundary_block(torch.stack([GR.canon_mask(m) for m in masks])[:, None].expand(-1, 3, -1, -1).contiguous())
def _build(stream, split):
    d = MO.load(split); Xf = MO.as_float(d["X"]); mid = Xf.shape[1] // 2
    if stream == "motion": return MF.batched(MF.motion_full, Xf, 250)
    if stream == "static": return FE.batched(FE.dense_stereo, Xf[:, mid])
    if stream == "sil_col":
        out = []
        for i in range(0, len(Xf), 1000):
            gl, _, _ = GR.groups(Xf[i:i + 1000, mid]); out.append(_sil([g[0]["mask"] if g else torch.zeros(MO.S, MO.S, dtype=torch.bool) for g in gl]))
        return torch.cat(out)
    if stream == "sil_mot": return torch.cat([_sil(MD.motion_group(Xf[i:i + 500])) for i in range(0, len(Xf), 500)])
    raise KeyError(stream)
def feats(stream, split):
    p = os.path.join(MO.OUT, f"feat_{stream}_{split}.pt")
    if not os.path.exists(p) or stream in os.environ.get("STREAMS", "").split(","):
        t = time.time(); Z = _build(stream, split); torch.save(Z.half(), p); log(f"  built {stream}/{split} {tuple(Z.shape)} in {time.time() - t:.0f}s")
    return torch.load(p).float()

class Std:
    def __init__(self, Z): self.mu, self.sd = Z.mean(0), Z.std(0) + 1e-6
    def __call__(self, Z): return (Z - self.mu) / self.sd

def leaf(d, n_out, seed):
    torch.manual_seed(seed)
    sub = construct(base=Seeded(d, n_out, interior_cells=HIDDEN, nonlinear=True), envelope=Envelope(max_parameter_bytes=400_000), dispatch_table=default_dispatch_table(),
                    capacity=max(d + HIDDEN + n_out + 8, (d * HIDDEN + HIDDEN * n_out) * 5 // (4 * 64) + 8), sparsity_k=0)
    sub.compile(); sub.prepare_training(); return sub
def train_eval(Ztr, ytr, tests, n_out, seed):
    sub = leaf(Ztr.shape[1], n_out, seed); params = sub.trainable_tensors(); opt = torch.optim.Adam(params, lr=1e-3); g = torch.Generator().manual_seed(seed)
    for ep in range(EPOCHS):
        for bi in torch.randperm(len(ytr), generator=g).split(256):
            opt.zero_grad(); F.cross_entropy(sub(Ztr[bi]), ytr[bi]).backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
    out = {}
    with torch.no_grad():
        for nm, (Z, y, m) in tests.items():
            pred = torch.cat([sub(Z[i:i + 4000]) for i in range(0, len(Z), 4000)]).argmax(1)
            out[nm] = float((pred == y)[m].float().mean())
    return out, sum(p.numel() for p in params)

if __name__ == "__main__":
    Y = {sp: MO.load(sp)["ys"] for sp in SPLITS}
    streams = os.environ.get("STREAMS", "motion,static,sil_col,sil_mot").split(",")
    Z = {s: {sp: feats(s, sp) for sp in SPLITS} for s in ["motion", "static", "sil_col", "sil_mot"]}
    if os.environ.get("FEATS_ONLY"): sys.exit(0)
    for s in Z:
        st = Std(Z[s]["train"]); Z[s] = {sp: st(Z[s][sp]) for sp in SPLITS}
    Z["sil_both"] = {sp: torch.cat([Z["sil_col"][sp], Z["sil_mot"][sp]], 1) for sp in SPLITS}
    Z["motion+sil_mot"] = {sp: torch.cat([Z["motion"][sp], Z["sil_mot"][sp]], 1) for sp in SPLITS}
    ALL = lambda sp: torch.ones(len(Y[sp]["y_vel"]), dtype=torch.bool); MOV = lambda sp: Y[sp]["y_vel"] > 0
    jobs = [("velocity", "motion", "y_vel", 17, ALL), ("velocity", "static", "y_vel", 17, ALL), ("velocity", "motion+sil_mot", "y_vel", 17, ALL),
            ("shape", "sil_col", "y_shape", 3, ALL), ("shape", "sil_mot", "y_shape", 3, ALL), ("shape", "sil_both", "y_shape", 3, ALL), ("shape", "static", "y_shape", 3, ALL), ("shape", "motion", "y_shape", 3, ALL)]
    log(f"MOTION LEAF s055: seeds {SEEDS} epochs {EPOCHS} hidden {HIDDEN}; train n={len(Y['train']['y_vel'])}")
    for task, s, yk, n_out, _ in jobs:
        res = []
        for seed in range(SEEDS):
            t0 = time.time()
            tests = {f"{sp}{'/mov' if mv else ''}": (Z[s][sp], Y[sp][yk], (MOV if mv else ALL)(sp)) for sp in SPLITS[1:] for mv in ((False, True) if task == "shape" else (False,))}
            r, npar = train_eval(Z[s]["train"], Y["train"][yk], tests, n_out, seed); res.append(r)
        keys = list(res[0]); m = {k: torch.tensor([r[k] for r in res]) for k in keys}
        log(f"  {task:8s} {s:16s} d={Z[s]['train'].shape[1]:4d} {npar/1e3:5.1f}K | " + " | ".join(f"{k} {m[k].mean():.3f}±{m[k].std() if SEEDS > 1 else 0:.3f}" for k in keys) + f"  [{time.time() - t0:.0f}s/seed]")
