"""s053 (b): per-factor leaves + router (the survivor recipe: SPLIT by factor, arbitrate).
Leaves (<=50K each; train on 16K of train): shape <- canon silhouette+frame+flags (103) |
whole <- whole-image bd+col+cn (205; the field/texture reader) | fill <- ctex+flags (220) |
hue <- bcolour (6-way) | iso <- bcolour (2-way) | blur <- edge+ctex (3-way) | count <- grouping.
Router (on the other 4K): [shape logits, whole logits, flags, frame scale] -> 5-way.
Controls: each leaf alone; uniform mix of shape+whole logits (= absorb); single 311 leaf.
Multi-object: per body shape+fill leaves -> set of shapes / set of (shape,fill) pairs.
n=SEEDS for the shape rows."""
import os, sys, time, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.progenitor import shapes as SH, shapes_feats as SF, grouping as G, cifar_eye_nest as B
from trioron.bases.seeded import Seeded
from trioron.core import Envelope, construct
from trioron.phenotype import default_dispatch_table
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "6")))
EPOCHS = int(os.environ.get("EPOCHS", "8")); R_EP = int(os.environ.get("ROUTER_EPOCHS", "80")); MULTI = os.environ.get("MULTI", "1") == "1"; SEEDS = int(os.environ.get("SEEDS", "3")); CANON = "scale"; t0 = time.time()
def leaf(X, y, n_out, seed, hidden=48, epochs=None):
    torch.manual_seed(seed)
    sub = construct(base=Seeded(X.shape[1], n_out, interior_cells=hidden, nonlinear=True), envelope=Envelope(max_parameter_bytes=400_000),
                    dispatch_table=default_dispatch_table(), capacity=X.shape[1] + hidden + n_out + 8, sparsity_k=0)
    sub.compile(); sub.prepare_training(); opt = torch.optim.Adam(sub.trainable_tensors(), lr=1e-3); g = torch.Generator().manual_seed(seed + 1)
    for ep in range(epochs or EPOCHS):
        for idx in torch.randperm(len(X), generator=g).split(256):
            opt.zero_grad(); F.cross_entropy(sub(X[idx]), y[idx]).backward(); torch.nn.utils.clip_grad_norm_(sub.trainable_tensors(), 1.0); sub.zero_dormant_grads(); opt.step()
    return sub
class Std:
    def __init__(self, Z): self.mu, self.sd = Z.mean(0), Z.std(0) + 1e-6
    def __call__(self, Z): return ((Z - self.mu) / self.sd).float()
SPL = ("train", "test_fresh", "test_held", "test_stress", "test_multi")
Y = {sp: SH.load(sp)[1] for sp in SPL}; GD = {sp: SF.grouped(sp, canon=CANON) for sp in SPL}
W = ["bd", "col", "cn"]; WH = {sp: torch.cat([SF.feats(k, sp) for k in W], 1) for sp in SPL}
STREAMS = {"shape": lambda sp: torch.cat([GD[sp]["silhouette"], GD[sp]["frame"], GD[sp]["flags"]], 1), "whole": lambda sp: WH[sp],
           "fill": lambda sp: torch.cat([GD[sp]["ctex"], GD[sp]["flags"]], 1), "hue": lambda sp: GD[sp]["bcolour"], "iso": lambda sp: GD[sp]["bcolour"],
           "blur": lambda sp: torch.cat([GD[sp]["edge"], GD[sp]["ctex"]], 1),
           "single311": lambda sp: torch.cat([GD[sp]["silhouette"], GD[sp]["colour"], GD[sp]["frame"], GD[sp]["flags"], WH[sp]], 1)}
TARGET = {"shape": ("y_shape", 5), "whole": ("y_shape", 5), "fill": ("y_fill", 4), "hue": ("y_hue", 6), "iso": ("y_iso", 2), "blur": ("y_blur", 3), "single311": ("y_shape", 5)}
N = len(Y["train"]["y_shape"]); perm = torch.randperm(N, generator=torch.Generator().manual_seed(0)); L_IDX, R_IDX = perm[:16000], perm[16000:]
yst = Y["test_stress"]; nc = yst["y_crop"] == 0
sl = {"small r<5": (yst["y_scale"] < 5) & nc & (yst["y_shape"] < 3), "cropped": yst["y_crop"] == 1, "iso": (yst["y_iso"] == 1) & nc, "blur2": (yst["y_blur"] == 2) & nc}
def acc(p, y): return float((p == y).float().mean())
def f(v): v = torch.tensor(v); return f"{v.mean():.3f}±{v.std():.3f}" if len(v) > 1 else f"{v.mean():.3f}"
rows = {}
def add(name, sp_preds):   # sp_preds: dict split->pred tensor
    r = rows.setdefault(name, {k: [] for k in ["fresh", "geo 3-way", "held-out", *sl]})
    pf, ph, ps = sp_preds["test_fresh"], sp_preds["test_held"], sp_preds["test_stress"]; geo = Y["test_fresh"]["y_shape"] < 3
    r["fresh"].append(acc(pf, Y["test_fresh"]["y_shape"])); r["geo 3-way"].append(acc(pf[geo], Y["test_fresh"]["y_shape"][geo])); r["held-out"].append(acc(ph, Y["test_held"]["y_shape"]))
    for k, m in sl.items(): r[k].append(acc(ps[m], yst["y_shape"][m]))
factor_rows = {}
for s in range(SEEDS):
    leaves, std, logits = {}, {}, {}
    for name, fn in STREAMS.items():
        yk, nout = TARGET[name]; std[name] = Std(fn("train")[L_IDX]); leaves[name] = leaf(std[name](fn("train")[L_IDX]), Y["train"][yk][L_IDX], nout, s)
        logits[name] = {sp: B.logits_of(leaves[name], std[name](fn(sp))) for sp in SPL}
        if name in ("fill", "hue", "iso", "blur"):
            factor_rows.setdefault(name, []).append((acc(logits[name]["test_fresh"].argmax(1), Y["test_fresh"][yk]), acc(logits[name]["test_held"].argmax(1), Y["test_held"][yk])))
    for name in ("shape", "whole", "single311"): add(f"leaf {name} alone", {sp: logits[name][sp].argmax(1) for sp in SPL})
    add("uniform mix shape+whole (absorb)", {sp: (F.log_softmax(logits["shape"][sp], 1) + F.log_softmax(logits["whole"][sp], 1)).argmax(1) for sp in SPL})
    # router: arbitration leaf on the leaves' logits + grouping context, trained on R_IDX
    def rin(sp, idx=None):
        z = torch.cat([F.log_softmax(logits["shape"][sp], 1), F.log_softmax(logits["whole"][sp], 1), GD[sp]["flags"], GD[sp]["frame"][:, 2:4] / 16], 1)
        return z if idx is None else z[idx]
    rstd = Std(rin("train", R_IDX)); router = leaf(rstd(rin("train", R_IDX)), Y["train"]["y_shape"][R_IDX], 5, s, hidden=16, epochs=R_EP)
    add("ROUTER over shape+whole leaves", {sp: B.logits_of(router, rstd(rin(sp))).argmax(1) for sp in SPL})
    # + fill/colour/blur logits as router context (does knowing the other factors help arbitrate shape?)
    def rin2(sp, idx=None):
        z = torch.cat([rin(sp), F.log_softmax(logits["fill"][sp], 1), F.log_softmax(logits["iso"][sp], 1), F.log_softmax(logits["blur"][sp], 1)], 1)
        return z if idx is None else z[idx]
    r2std = Std(rin2("train", R_IDX)); router2 = leaf(r2std(rin2("train", R_IDX)), Y["train"]["y_shape"][R_IDX], 5, s, hidden=16, epochs=R_EP)
    # linear router sanity bar (torch logistic regression on the same input)
    torch.manual_seed(s); lin = torch.nn.Linear(rin("train", R_IDX).shape[1], 5); opt = torch.optim.Adam(lin.parameters(), 1e-2); Zr = rstd(rin("train", R_IDX)); yr = Y["train"]["y_shape"][R_IDX]
    for ep in range(200):
        for idx in torch.randperm(len(Zr)).split(256): opt.zero_grad(); F.cross_entropy(lin(Zr[idx]), yr[idx]).backward(); opt.step()
    with torch.no_grad(): add("linear router (ref)", {sp: lin(rstd(rin(sp))).argmax(1) for sp in SPL})
    add("ROUTER + fill/iso/blur context", {sp: B.logits_of(router2, r2std(rin2(sp))).argmax(1) for sp in SPL})
    print(f"  seed {s} done [{time.time()-t0:.0f}s]", flush=True)
    if s == 0 and MULTI:   # multi-object: per body shape+fill leaves
        Xm, ym, mm = SH.load("test_multi"); gl, _, _ = G.groups(Xm); PG = G.describe_groups(Xm, gl, canon=CANON)
        pw = logits["whole"]["test_multi"].argmax(1); pred_set = torch.zeros(len(Xm), 5); pair_ok = []
        Dm = SF.grouped("test_multi", canon=CANON)   # largest-group extras only; per-body ctex needs per-body crops -> compute here
        for i, d in enumerate(PG):
            gt_pairs = {(o["shape"], o["fill"]) for o in mm[i]["objects"]}
            if d is None or bool(d["is_field"].any()): pred_set[i, pw[i]] = 1; pair_ok.append(gt_pairs == {(int(pw[i]), 0)}); continue
            z = std["shape"](torch.cat([d["silhouette"], d["frame"], d["flags"]], 1))
            with torch.no_grad(): psh = leaves["shape"](z).argmax(1)
            for c in psh.tolist(): pred_set[i, c] = 1
            # per-body fill: ctex on each body's canon crop
            gs = gl[i][:4]; ys_ = []
            for g in gs:
                box = G.canon_box(g["mask"]); yimg = G.FE.Y(Xm[i:i + 1])[0]; yi = yimg * g["interior"].float() + (~g["interior"]).float() * (yimg[g["interior"]].mean() if g["interior"].any() else 0)
                ys_.append(G.canon_crop(yi[None], box)[0] if box is not None else yimg)
            ct = G.ctex_pool(torch.stack(ys_)); zf = std["fill"](torch.cat([ct, d["flags"]], 1))
            with torch.no_grad(): pfl = leaves["fill"](zf).argmax(1)
            pair_ok.append({(int(a), int(b)) for a, b in zip(psh.tolist(), pfl.tolist())} == gt_pairs)
        ok = (pred_set == ym["y_set"]).all(1).float(); po = torch.tensor(pair_ok).float()
        for tag, m in (("all", torch.ones(len(ok), dtype=torch.bool)), ("no-overlap", ym["y_overlap"] == 0), ("k=2", ym["y_count"] == 2), ("k=3", ym["y_count"] == 3)):
            print(f"  multi [{tag}]: shape-set acc {float(ok[m].mean()):.3f} | (shape,fill)-pair-set acc {float(po[m].mean()):.3f}", flush=True)
        n = Dm["flags"][:, 0].long().clamp(max=3); n = torch.where(Dm["flags"][:, 1] > 0, torch.ones_like(n), n); print(f"  count primitive: exact {acc(n, ym['y_count']):.3f} (no-overlap {acc(n[ym['y_overlap']==0], ym['y_count'][ym['y_overlap']==0]):.3f})")
print("factor leaves (fresh / held-out): " + " | ".join(f"{k} {f([a for a,_ in v])} / {f([b for _,b in v])}" for k, v in factor_rows.items()))
for name, r in rows.items(): print(f"  {name:>34s} | " + " | ".join(f"{k} {f(v)}" for k, v in r.items()), flush=True)
