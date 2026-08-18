"""s053: does a boundary-orientation primitive (frontend.boundary_block, 92-d) give the
leaf the silhouette the cepstral front end lacks?  ds+col 900 | +boundary 992 | boundary
alone 92 | boundary + colour 192.  Rows: fresh, geometric-only (3-way among circ/tri/squ),
held-out combos, small r<5, cropped, iso; n=3 seeds; + kNN and 2x256 MLP refs on +boundary."""
import os, sys, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.progenitor import shapes as SH, shapes_feats as SF, cifar_eye_nest as B
from trioron.bases.seeded import Seeded
from trioron.core import Envelope, construct
from trioron.phenotype import default_dispatch_table
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "6")))
EPOCHS = int(os.environ.get("EPOCHS", "8")); SEEDS = int(os.environ.get("SEEDS", "3"))
def leaf(X, y, n_out, seed, hidden=48):
    torch.manual_seed(seed)
    sub = construct(base=Seeded(X.shape[1], n_out, interior_cells=hidden, nonlinear=True), envelope=Envelope(max_parameter_bytes=400_000),
                    dispatch_table=default_dispatch_table(), capacity=X.shape[1] + hidden + n_out + 8, sparsity_k=0)
    sub.compile(); sub.prepare_training(); opt = torch.optim.Adam(sub.trainable_tensors(), lr=1e-3); g = torch.Generator().manual_seed(seed + 1)
    for ep in range(EPOCHS):
        for idx in torch.randperm(len(X), generator=g).split(256):
            opt.zero_grad(); F.cross_entropy(sub(X[idx]), y[idx]).backward(); torch.nn.utils.clip_grad_norm_(sub.trainable_tensors(), 1.0); sub.zero_dormant_grads(); opt.step()
    return sub
_, ytr, _ = SH.load("train"); _, yfr, _ = SH.load("test_fresh"); _, yho, _ = SH.load("test_held"); _, yst, _ = SH.load("test_stress")
ALL = {"ds+col 900": ["ds", "col"], "ds+col+boundary 992": ["ds", "col", "bd"], "boundary 92": ["bd"], "boundary+col 192": ["bd", "col"],
       "corner 13": ["cn"], "boundary+col+corner 205": ["bd", "col", "cn"], "ds+col+bd+corner 1005": ["ds", "col", "bd", "cn"]}
combos = {k: ALL[k] for k in os.environ.get("COMBOS", ",".join(ALL)).split(",")}
nc = yst["y_crop"] == 0; geo = yfr["y_shape"] < 3
sub_st = {"small r<5": (yst["y_scale"] < 5) & nc & (yst["y_shape"] < 3), "cropped": yst["y_crop"] == 1, "iso": (yst["y_iso"] == 1) & nc, "blur-strong": (yst["y_blur"] == 2) & nc}
def f(v): v = torch.tensor(v); return f"{v.mean():.3f}±{v.std():.3f}"
for name, fs in combos.items():
    Z = {sp: torch.cat([SF.feats(k, sp) for k in fs], 1) for sp in ("train", "test_fresh", "test_held", "test_stress")}
    mu, sd = Z["train"].mean(0), Z["train"].std(0) + 1e-6; Z = {k: (v - mu) / sd for k, v in Z.items()}
    rows = {k: [] for k in ["fresh", "geometric 3-way", "held-out", *sub_st]}
    for s in range(SEEDS):
        sub = leaf(Z["train"], ytr["y_shape"], 5, s); pf = B.logits_of(sub, Z["test_fresh"]); ps = B.logits_of(sub, Z["test_stress"]).argmax(1)
        rows["fresh"].append(float((pf.argmax(1) == yfr["y_shape"]).float().mean()))
        rows["geometric 3-way"].append(float((pf[geo][:, :3].argmax(1) == yfr["y_shape"][geo]).float().mean()))
        rows["held-out"].append(float((B.logits_of(sub, Z["test_held"]).argmax(1) == yho["y_shape"]).float().mean()))
        for k, m in sub_st.items(): rows[k].append(float((ps[m] == yst["y_shape"][m]).float().mean()))
    print(f"  {name:>22s} | " + " | ".join(f"{k} {f(v)}" for k, v in rows.items()), flush=True)
    if name in ("ds+col+boundary 992", "boundary+col+corner 205"):
        nn = torch.cdist(Z["test_fresh"], Z["train"]).argmin(1); print(f"      1-NN fresh {float((ytr['y_shape'][nn]==yfr['y_shape']).float().mean()):.3f}", flush=True)
        torch.manual_seed(0); net = torch.nn.Sequential(torch.nn.Linear(Z["train"].shape[1], 256), torch.nn.ReLU(), torch.nn.Linear(256, 256), torch.nn.ReLU(), torch.nn.Linear(256, 5)); opt = torch.optim.Adam(net.parameters(), 1e-3)
        for ep in range(EPOCHS):
            for idx in torch.randperm(20000).split(256): opt.zero_grad(); F.cross_entropy(net(Z["train"][idx]), ytr["y_shape"][idx]).backward(); opt.step()
        with torch.no_grad(): print(f"      MLP 2x256 (OVER CAP ref) fresh {float((net(Z['test_fresh']).argmax(1)==yfr['y_shape']).float().mean()):.3f} held {float((net(Z['test_held']).argmax(1)==yho['y_shape']).float().mean()):.3f}", flush=True)
