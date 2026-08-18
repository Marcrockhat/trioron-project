"""s053 follow-up on cached features (dense+stereo+colour 900): confusion, where
held-out combos go, accuracy per (shape,fill) / scale bin / vis bin, and whether the
ceiling is the leaf (hidden 48 vs 256 over-cap ref, 1-NN) or the front end."""
import os, sys, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.progenitor import shapes as SH, shapes_feats as SF, cifar_eye_nest as B
from trioron.bases.seeded import Seeded
from trioron.core import Envelope, construct
from trioron.phenotype import default_dispatch_table
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "6")))
EPOCHS = int(os.environ.get("EPOCHS", "8"))
def leaf(X, y, n_out, seed, hidden):
    torch.manual_seed(seed)
    sub = construct(base=Seeded(X.shape[1], n_out, interior_cells=hidden, nonlinear=True), envelope=Envelope(max_parameter_bytes=4_000_000),
                    dispatch_table=default_dispatch_table(), capacity=X.shape[1] + hidden + n_out + 8, sparsity_k=0)
    sub.compile(); sub.prepare_training(); opt = torch.optim.Adam(sub.trainable_tensors(), lr=1e-3); g = torch.Generator().manual_seed(seed + 1)
    for ep in range(EPOCHS):
        perm = torch.randperm(len(X), generator=g)
        for i in range(0, len(X), 256):
            idx = perm[i:i + 256]; opt.zero_grad(); l = F.cross_entropy(sub(X[idx]), y[idx]); l.backward()
            torch.nn.utils.clip_grad_norm_(sub.trainable_tensors(), 1.0); sub.zero_dormant_grads(); opt.step()
    return sub
_, ytr, _ = SH.load("train"); _, yfr, _ = SH.load("test_fresh"); _, yho, _ = SH.load("test_held"); _, yst, _ = SH.load("test_stress")
Ztr = SF.dsc("train"); mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-6; nz = lambda Z: (Z - mu) / sd
Ztr, Zfr, Zho, Zst = nz(Ztr), nz(SF.dsc("test_fresh")), nz(SF.dsc("test_held")), nz(SF.dsc("test_stress"))
def cm(p, y, C=5): return torch.zeros(C, C).index_put_((y, p), torch.ones(len(y)), accumulate=True).long()
import sys as _s
if os.environ.get("TAIL"): pass
sub = leaf(Ztr, ytr["y_shape"], 5, 0, 48); pf = B.logits_of(sub, Zfr).argmax(1); ph = B.logits_of(sub, Zho).argmax(1); ps = B.logits_of(sub, Zst).argmax(1)
print("hidden 48 fresh acc", float((pf == yfr["y_shape"]).float().mean()))
print("confusion fresh (rows true circ/tri/squ/stripes/dots):"); print(cm(pf, yfr["y_shape"]).tolist())
print("held-out combos -> predicted (rows: circle-outline, triangle-dotted, square-striped):")
for sh, fl in [(0, 3), (1, 2), (2, 1)]:
    m = (yho["y_shape"] == sh) & (yho["y_fill"] == fl); print(f"  {SH.NAMES[sh]}-{SH.FILLS[fl]}: {torch.bincount(ph[m], minlength=5).tolist()}")
print("fresh acc per (shape,fill):")
for sh in range(3):
    print("  " + SH.NAMES[sh] + ": " + ", ".join(f"{SH.FILLS[fl]} {float((pf[(yfr['y_shape']==sh)&(yfr['y_fill']==fl)]==sh).float().mean()):.2f}" for fl in range(4) if (sh, fl) not in SH.HELD))
print("stress acc per scale bin (uncropped, geometric only):")
g = (yst["y_shape"] < 3) & (yst["y_crop"] == 0)
for lo, hi in [(3, 5), (5, 8), (8, 11), (11, 14), (14, 19)]:
    m = g & (yst["y_scale"] >= lo) & (yst["y_scale"] < hi); print(f"  r[{lo},{hi}) n={int(m.sum())} acc {float((ps[m]==yst['y_shape'][m]).float().mean()):.3f}")
print("stress acc per vis bin (cropped):")
for lo, hi in [(0.3, 0.5), (0.5, 0.7), (0.7, 1.01)]:
    m = (yst["y_crop"] == 1) & (yst["y_vis"] >= lo) & (yst["y_vis"] < hi); print(f"  vis[{lo},{hi}) n={int(m.sum())} acc {float((ps[m]==yst['y_shape'][m]).float().mean()):.3f}")
# ceiling: leaf capacity vs front end (plain torch MLP as the over-cap reference)
def mlp(X, y, h, seed=0, epochs=EPOCHS):
    torch.manual_seed(seed); net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], h), torch.nn.ReLU(), torch.nn.Linear(h, h), torch.nn.ReLU(), torch.nn.Linear(h, 5))
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    for ep in range(epochs):
        for idx in torch.randperm(len(X)).split(256): opt.zero_grad(); F.cross_entropy(net(X[idx]), y[idx]).backward(); opt.step()
    return net
for h in (256, 1024):
    net = mlp(Ztr, ytr["y_shape"], h)
    with torch.no_grad(): print(f"MLP 2x{h} (OVER CAP ref) fresh {float((net(Zfr).argmax(1)==yfr['y_shape']).float().mean()):.3f} held {float((net(Zho).argmax(1)==yho['y_shape']).float().mean()):.3f} stress-small(r<5) {float((net(Zst[(yst['y_scale']<5)&(yst['y_crop']==0)]).argmax(1)==yst['y_shape'][(yst['y_scale']<5)&(yst['y_crop']==0)]).float().mean()):.3f}")
d = torch.cdist(Zfr, Ztr); nn = d.argmin(1); print("1-NN fresh acc", float((ytr["y_shape"][nn] == yfr["y_shape"]).float().mean()))
# same-leaf multi-task: shape+fill jointly (2 heads = 9 outputs) -- does joint supervision factor them?
y2 = ytr["y_shape"] * 4 + ytr["y_fill"]; s3 = leaf(Ztr, y2, 20, 0, 48); L = B.logits_of(s3, Zho).view(-1, 5, 4)
psh = L.logsumexp(2).argmax(1); print("joint (shape x fill) leaf: held-out shape acc via marginal", float((psh == yho["y_shape"]).float().mean()), " fresh", float((B.logits_of(s3, Zfr).view(-1,5,4).logsumexp(2).argmax(1)==yfr['y_shape']).float().mean()))
