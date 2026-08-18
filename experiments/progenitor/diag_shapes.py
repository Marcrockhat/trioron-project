"""s053 shape-world recognition probe. Question: on synthetic shapes with many
images per class and separately controlled nuisance factors (colour, skew,
line thickness, zoom in/out, crop, flip, fill texture, count), does the
s052 fixed front end + one <=50K trioron leaf recognise images it has NEVER
seen -- fresh draws, held-out (shape,fill) COMBINATIONS, extreme zoom, cropped,
iso-luminant, multi-object -- and which front-end primitive carries which factor?

Train: N_TRAIN single-object images (seed 1), excluding combos
  HELD = {(triangle,dotted),(square,striped),(circle,outline)}.
Test sets (all unseen images): fresh (same dist, seed 2) | held-out combos
  (seed 3, only=HELD) | zoom-out r<=5 / zoom-in r>=14 / cropped / iso-luminant
  subsets of a big fresh draw (seed 4) | multi-object (seed 5, maxk=3):
  the shape leaf is asked which shapes are present via a 5-way sigmoid head
  trained on a separate multi-object train set (seed 6); + blur level 0/1/2,
  focus-gradient and per-object depth-of-field subsets.
Data: outputs/data/shapes/*.pt (build once: python3 experiments/progenitor/shapes.py build).
Fronts: dense+stereo 800 | +colour 900 | colour only 100 | raw 3072 (ref).
Env: N_TRAIN (20000), SEEDS (3), EPOCHS (8), HIDDEN (48).
Run: OMP_NUM_THREADS=6 python3 experiments/progenitor/diag_shapes.py
"""
import os, sys, time, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.progenitor import frontend as FE
from experiments.progenitor import shapes as SH
from experiments.progenitor import cifar_eye_nest as B
from trioron.bases.seeded import Seeded
from trioron.core import Envelope, construct
from trioron.phenotype import default_dispatch_table
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "6")))
N_TRAIN = int(os.environ.get("N_TRAIN", "20000")); SEEDS = int(os.environ.get("SEEDS", "3"))
EPOCHS = int(os.environ.get("EPOCHS", "8")); HIDDEN = int(os.environ.get("HIDDEN", "48"))
HELD = {(1, 2), (2, 1), (0, 3)}
t0 = time.time()
def log(*a): print(*a, flush=True)
log(f"SHAPE PROBE s053 n_train={N_TRAIN} seeds={SEEDS} epochs={EPOCHS} hidden={HIDDEN} held={sorted(HELD)}")
Xtr, ytr, _ = SH.load("train"); Xtr, ytr = Xtr[:N_TRAIN], {k: v[:N_TRAIN] for k, v in ytr.items()}
Xfr, yfr, _ = SH.load("test_fresh"); Xho, yho, _ = SH.load("test_held"); Xbig, ybig, _ = SH.load("test_stress")
Xmtr, ymtr, _ = SH.load("train_multi"); Xmtr, ymtr = Xmtr[:N_TRAIN], {k: v[:N_TRAIN] for k, v in ymtr.items()}
Xmte, ymte, _ = SH.load("test_multi")
log(f"  loaded in {time.time()-t0:.0f}s; train shape hist {torch.bincount(ytr['y_shape'],minlength=5).tolist()} fill hist {torch.bincount(ytr['y_fill'],minlength=4).tolist()}")
nc = ybig["y_crop"] == 0
subsets = {"zoom-out r<=5": (ybig["y_scale"] <= 5) & nc, "zoom-in r>=14": (ybig["y_scale"] >= 14) & nc, "cropped": ybig["y_crop"] == 1,
           "iso-luminant": (ybig["y_iso"] == 1) & nc, "outline": (ybig["y_fill"] == 3) & nc,
           "sharp": (ybig["y_blur"] == 0) & nc, "blur-mild": (ybig["y_blur"] == 1) & nc, "blur-strong": (ybig["y_blur"] == 2) & nc, "focus-gradient": (ybig["y_focus"] == 2) & nc}
log("  subset sizes: " + ", ".join(f"{k}={int(v.sum())}" for k, v in subsets.items()))

def leaf(X, y, n_out, seed, loss="ce"):
    torch.manual_seed(seed)
    sub = construct(base=Seeded(X.shape[1], n_out, interior_cells=HIDDEN, nonlinear=True), envelope=Envelope(max_parameter_bytes=400_000),
                    dispatch_table=default_dispatch_table(), capacity=X.shape[1] + HIDDEN + n_out + 8, sparsity_k=0)
    sub.compile(); sub.prepare_training(); opt = torch.optim.Adam(sub.trainable_tensors(), lr=1e-3)
    g = torch.Generator().manual_seed(seed + 1)
    for ep in range(EPOCHS):
        perm = torch.randperm(len(X), generator=g)
        for i in range(0, len(X), 256):
            idx = perm[i:i + 256]; opt.zero_grad(); out = sub(X[idx])
            l = F.cross_entropy(out, y[idx]) if loss == "ce" else F.binary_cross_entropy_with_logits(out, y[idx])
            l.backward(); torch.nn.utils.clip_grad_norm_(sub.trainable_tensors(), 1.0); sub.zero_dormant_grads(); opt.step()
    return sub
def acc(sub, X, y): return float((B.logits_of(sub, X).argmax(1) == y).float().mean())

fronts = {"dense+stereo 800": FE.dense_stereo, "dense+stereo+colour 900": FE.dense_stereo_colour, "colour block 100": FE.colour_block, "raw 3072 (OVER CAP)": FE.raw}
for name, fn in fronts.items():
    Ztr = FE.batched(fn, Xtr); mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-6; nz = lambda Z: (Z - mu) / sd; Ztr = nz(Ztr)
    Zfr, Zho, Zbig = nz(FE.batched(fn, Xfr)), nz(FE.batched(fn, Xho)), nz(FE.batched(fn, Xbig))
    rows = {k: [] for k in ["fresh", "held-out combos", *subsets, "fill (fresh)", "multi set-acc", "multi 1-obj", "multi depth-of-field"]}
    for s in range(SEEDS):
        sub = leaf(Ztr, ytr["y_shape"], 5, s)
        rows["fresh"].append(acc(sub, Zfr, yfr["y_shape"])); rows["held-out combos"].append(acc(sub, Zho, yho["y_shape"]))
        for k, m in subsets.items(): rows[k].append(acc(sub, Zbig[m], ybig["y_shape"][m]))
        subf = leaf(Ztr, ytr["y_fill"], 4, s); rows["fill (fresh)"].append(acc(subf, Zfr, yfr["y_fill"]))
        if s == 0:   # multi-object: separate multi-label leaf (train on multi-object draws), same front end
            Zm = nz(FE.batched(fn, Xmtr)); Zmt = nz(FE.batched(fn, Xmte))
            subm = leaf(Zm, ymtr["y_set"], 5, s, loss="bce"); P = (B.logits_of(subm, Zmt) > 0).float()
            rows["multi set-acc"].append(float((P == ymte["y_set"]).all(1).float().mean()))
            one = ymte["y_count"] == 1; rows["multi 1-obj"].append(float((P[one] == ymte["y_set"][one]).all(1).float().mean()))
            dof = ymte["y_focus"] == 1; rows["multi depth-of-field"].append(float((P[dof] == ymte["y_set"][dof]).all(1).float().mean()))
    def f(v): v = torch.tensor(v); return f"{v.mean():.3f}±{v.std():.3f}" if len(v) > 1 else f"{v.mean():.3f}"
    log(f"  {name:>26s} d={Ztr.shape[1]:4d} | " + " | ".join(f"{k} {f(v)}" for k, v in rows.items()) + f"   [{time.time()-t0:.0f}s]")
log("chance: shape 0.20 (fields 2/5 -> field-vs-geom prior 0.6 hard); fill 0.25; multi set-acc chance ~0.03")
