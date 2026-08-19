"""s053: multi-object read by grouping. Shape leaf trained on single-object grouped stream
(sil+colour+frame+flags 106) reads EACH group of a multi-object image; predicted set = union
(field groups -> the whole-image 205 leaf's field decision). Set-accuracy on test_multi vs the
whole-image multi-label sigmoid leaf (BCE on train_multi) on 205 / 311 streams; slices: count,
overlap tag, depth-of-field focus."""
import os, sys, time, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.progenitor import shapes as SH, shapes_feats as SF, grouping as G, cifar_eye_nest as B
from trioron.bases.seeded import Seeded
from trioron.core import Envelope, construct
from trioron.phenotype import default_dispatch_table
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "6")))
EPOCHS = int(os.environ.get("EPOCHS", "8")); t0 = time.time()
def leaf(X, y, n_out, seed, loss="ce", hidden=48):
    torch.manual_seed(seed)
    sub = construct(base=Seeded(X.shape[1], n_out, interior_cells=hidden, nonlinear=True), envelope=Envelope(max_parameter_bytes=400_000),
                    dispatch_table=default_dispatch_table(), capacity=X.shape[1] + hidden + n_out + 8, sparsity_k=0)
    sub.compile(); sub.prepare_training(); opt = torch.optim.Adam(sub.trainable_tensors(), lr=1e-3); g = torch.Generator().manual_seed(seed + 1)
    for ep in range(EPOCHS):
        for idx in torch.randperm(len(X), generator=g).split(256):
            opt.zero_grad(); out = sub(X[idx]); l = F.cross_entropy(out, y[idx]) if loss == "ce" else F.binary_cross_entropy_with_logits(out, y[idx])
            l.backward(); torch.nn.utils.clip_grad_norm_(sub.trainable_tensors(), 1.0); sub.zero_dormant_grads(); opt.step()
    return sub
CANON = os.environ.get("CANON", "0") == "1"
def grp(split): return SF.grouped(split, canon=CANON)
_, ytr, _ = SH.load("train"); Xm, ym, _ = SH.load("test_multi"); _, ymtr, _ = SH.load("train_multi")
Dtr = grp("train"); K = ["silhouette", "colour", "frame", "flags"]
Ztr = torch.cat([Dtr[k] for k in K], 1); mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-6
shape_leaf = leaf((Ztr - mu) / sd, ytr["y_shape"], 5, 0)
# whole-image field/shape leaf on 205 for field images
W = ["bd", "col", "cn"]; Wtr = torch.cat([SF.feats(k, "train") for k in W], 1); wmu, wsd = Wtr.mean(0), Wtr.std(0) + 1e-6
whole_leaf = leaf((Wtr - wmu) / wsd, ytr["y_shape"], 5, 0)
print(f"  leaves trained [{time.time()-t0:.0f}s]", flush=True)
# per-group read on test_multi
gl, _, _ = G.groups(Xm); PG = G.describe_groups(Xm, gl, canon=CANON); Wm = (torch.cat([SF.feats(k, "test_multi") for k in W], 1) - wmu) / wsd
pw = B.logits_of(whole_leaf, Wm).argmax(1)
pred = torch.zeros(len(Xm), 5)
for i, d in enumerate(PG):
    if d is None: pred[i, pw[i]] = 1; continue
    if bool(d["is_field"].any()): pred[i, pw[i]] = 1; continue
    z = ((torch.cat([d[k] for k in K], 1) - mu) / sd).float()
    with torch.no_grad(): p = shape_leaf(z).argmax(1)
    for c in p.tolist(): pred[i, c] = 1
def report(name, P):
    ok = (P == ym["y_set"]).all(1).float()
    sl = {"all": torch.ones(len(ok), dtype=torch.bool), "k=1": ym["y_count"] == 1, "k=2": ym["y_count"] == 2, "k=3": ym["y_count"] == 3,
          "no-overlap": ym["y_overlap"] == 0, "overlap": ym["y_overlap"] == 1, "depth-of-field": ym["y_focus"] == 1}
    print(f"  {name:>34s} set-acc | " + " | ".join(f"{k} {float(ok[m].mean()):.3f}" for k, m in sl.items()) + f"  [{time.time()-t0:.0f}s]", flush=True)
report(f"grouping per-group read (106+205){' CANON' if CANON else ''}", pred)
# single-object rows for the same shape leaf (canon effect on small / cropped / held-out)
_, yfr, _ = SH.load("test_fresh"); _, yho, _ = SH.load("test_held"); _, yst, _ = SH.load("test_stress")
for sp, yy in (("test_fresh", yfr), ("test_held", yho), ("test_stress", yst)):
    D = grp(sp); z = (torch.cat([D[k] for k in K], 1) - mu) / sd; pr = B.logits_of(shape_leaf, z).argmax(1)
    if sp == "test_stress":
        nc = yst["y_crop"] == 0; sl = {"small r<5": (yst["y_scale"] < 5) & nc & (yst["y_shape"] < 3), "cropped": yst["y_crop"] == 1, "zoom-in r>=14": (yst["y_scale"] >= 14) & nc}
        print("  single-object shape leaf on grouped 106" + (" CANON" if CANON else "") + ": " + " | ".join(f"{k} {float((pr[m]==yy['y_shape'][m]).float().mean()):.3f}" for k, m in sl.items()), flush=True)
    else: print(f"  single-object shape leaf on grouped 106{' CANON' if CANON else ''}: {sp} {float((pr==yy['y_shape']).float().mean()):.3f}", flush=True)
# baselines: whole-image multi-label sigmoid leaf trained on train_multi
for name, keys in (("whole-image BCE 205", W), ("whole+grouped-largest BCE 311", None)):
    if keys is None:
        Dm_tr = grp("train_multi"); Dm_te = grp("test_multi")
        A = torch.cat([torch.cat([Dm_tr[k] for k in K], 1)] + [SF.feats(k, "train_multi") for k in W], 1); Bt = torch.cat([torch.cat([Dm_te[k] for k in K], 1)] + [SF.feats(k, "test_multi") for k in W], 1)
    else:
        A = torch.cat([SF.feats(k, "train_multi") for k in keys], 1); Bt = torch.cat([SF.feats(k, "test_multi") for k in keys], 1)
    m_, s_ = A.mean(0), A.std(0) + 1e-6; sub = leaf((A - m_) / s_, ymtr["y_set"], 5, 0, loss="bce"); report(name, (B.logits_of(sub, (Bt - m_) / s_) > 0).float())
