"""s053: grouping BEFORE describing. Streams from grouping.describe (largest group):
silhouette-only boundary block (92) | group colour (3) | frame (7: cx,cy,scale-major,
scale-minor,orient,elongation,fill-fraction) | flags (4: n_objects,has_field,touches,fillfrac)
| interior-only cepstra (600).  Compare with the whole-image bd+col+corner 205 baseline and
with both together.  Shape leaf + fill leaf; fresh / geometric 3-way / held-out combos /
small / cropped / iso / blur2; n=3.  Also grouping's own count-primitive: n_objects vs y_count
on test_multi (P_N without a leaf)."""
import os, sys, time, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.progenitor import shapes as SH, shapes_feats as SF, grouping as G, cifar_eye_nest as B
from trioron.bases.seeded import Seeded
from trioron.core import Envelope, construct
from trioron.phenotype import default_dispatch_table
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "6")))
EPOCHS = int(os.environ.get("EPOCHS", "8")); SEEDS = int(os.environ.get("SEEDS", "3")); t0 = time.time()
def leaf(X, y, n_out, seed, hidden=48):
    torch.manual_seed(seed)
    sub = construct(base=Seeded(X.shape[1], n_out, interior_cells=hidden, nonlinear=True), envelope=Envelope(max_parameter_bytes=400_000),
                    dispatch_table=default_dispatch_table(), capacity=X.shape[1] + hidden + n_out + 8, sparsity_k=0)
    sub.compile(); sub.prepare_training(); opt = torch.optim.Adam(sub.trainable_tensors(), lr=1e-3); g = torch.Generator().manual_seed(seed + 1)
    for ep in range(EPOCHS):
        for idx in torch.randperm(len(X), generator=g).split(256):
            opt.zero_grad(); F.cross_entropy(sub(X[idx]), y[idx]).backward(); torch.nn.utils.clip_grad_norm_(sub.trainable_tensors(), 1.0); sub.zero_dormant_grads(); opt.step()
    return sub
def grouped(split):
    p = os.path.join(SH.OUT, f"feat_grp_{split}.pt")
    if os.path.exists(p): return torch.load(p)
    X, _, _ = SH.load(split); t = time.time(); D, _ = G.describe(X); D = {k: v.half() for k, v in D.items()}; torch.save(D, p)
    print(f"  grouped {split} in {time.time()-t:.0f}s", flush=True); return D
SPL = ("train", "test_fresh", "test_held", "test_stress")
_, ytr, _ = SH.load("train"); _, yfr, _ = SH.load("test_fresh"); _, yho, _ = SH.load("test_held"); _, yst, _ = SH.load("test_stress")
GD = {sp: {k: v.float() for k, v in grouped(sp).items()} for sp in SPL}
def stream(sp, keys):
    parts = []
    for k in keys:
        if k in ("silhouette", "colour", "frame", "flags", "interior"): parts.append(GD[sp][k])
        else: parts.append(SF.feats(k, sp))
    return torch.cat(parts, 1)
combos = {"whole-image bd+col+corner 205": ["bd", "col", "cn"],
          "grouped silhouette 92": ["silhouette"],
          "grouped sil+colour+frame+flags 106": ["silhouette", "colour", "frame", "flags"],
          "grouped all (+interior) 706": ["silhouette", "colour", "frame", "flags", "interior"],
          "grouped 106 + whole 205 = 311": ["silhouette", "colour", "frame", "flags", "bd", "col", "cn"]}
nc = yst["y_crop"] == 0; geo = yfr["y_shape"] < 3
sub_st = {"small r<5": (yst["y_scale"] < 5) & nc & (yst["y_shape"] < 3), "cropped": yst["y_crop"] == 1, "iso": (yst["y_iso"] == 1) & nc, "blur2": (yst["y_blur"] == 2) & nc}
def f(v): v = torch.tensor(v); return f"{v.mean():.3f}±{v.std():.3f}"
for name, keys in combos.items():
    Z = {sp: stream(sp, keys) for sp in SPL}; mu, sd = Z["train"].mean(0), Z["train"].std(0) + 1e-6; Z = {k: (v - mu) / sd for k, v in Z.items()}
    rows = {k: [] for k in ["fresh", "geo 3-way", "held-out", *sub_st, "fill fresh", "fill held-out"]}
    for s in range(SEEDS):
        sub = leaf(Z["train"], ytr["y_shape"], 5, s); pf = B.logits_of(sub, Z["test_fresh"]); ps = B.logits_of(sub, Z["test_stress"]).argmax(1)
        rows["fresh"].append(float((pf.argmax(1) == yfr["y_shape"]).float().mean())); rows["geo 3-way"].append(float((pf[geo][:, :3].argmax(1) == yfr["y_shape"][geo]).float().mean()))
        rows["held-out"].append(float((B.logits_of(sub, Z["test_held"]).argmax(1) == yho["y_shape"]).float().mean()))
        for k, m in sub_st.items(): rows[k].append(float((ps[m] == yst["y_shape"][m]).float().mean()))
        subf = leaf(Z["train"], ytr["y_fill"], 4, s); rows["fill fresh"].append(float((B.logits_of(subf, Z["test_fresh"]).argmax(1) == yfr["y_fill"]).float().mean()))
        rows["fill held-out"].append(float((B.logits_of(subf, Z["test_held"]).argmax(1) == yho["y_fill"]).float().mean()))
    print(f"  {name:>36s} | " + " | ".join(f"{k} {f(v)}" for k, v in rows.items()) + f"  [{time.time()-t0:.0f}s]", flush=True)
# grouping as the count primitive (no leaf): n_objects vs y_count on test_multi
Xm, ym, _ = SH.load("test_multi"); Dm, _ = G.describe(Xm); n = Dm["flags"][:, 0].long().clamp(max=3); n = torch.where(Dm["flags"][:, 1] > 0, torch.ones_like(n), n)
print(f"  count primitive (grouping only) test_multi: exact {float((n==ym['y_count']).float().mean()):.3f}; per true count " + ", ".join(f"{k}:{float((n[ym['y_count']==k]==k).float().mean()):.2f}" for k in (1, 2, 3)))
