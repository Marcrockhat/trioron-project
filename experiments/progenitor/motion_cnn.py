"""CNN reference bar (s058, NEXT-0): how do ordinary conv nets do on the SAME moving-shape
packets the Kinopsis leaves see?  The motion arc had no CNN numbers; Rocky's target sentence
"linear machinery + correct primitives ~ convolution layers" is untested without this.

Arms (all ~40-50K params to match the leaves, 8 ep Adam 1e-3 bs 256, n=SEEDS):
  cnn2d  : raw packet, T*3 = 18 channels stacked -> 3x conv3x3 (24/48/64) + pool -> GAP -> head
  cnn3d  : raw packet as [3,T,S,S]          -> 3x conv3d (16/32/48)            -> GAP -> head
  cnn2d_big    : cnn2d_strong with channels 32/64/64 (~70K, over the leaf budget) -- capacity check
  cnn2d_strong : frames + frame DIFFERENCES (33 ch), BatchNorm, flatten instead of GAP (keeps position/phase)
  mlp656 : the leaves' own primitive (motion_front.motion_full, 656-d, standardised) -> MLP-48.
           Same shape as the trioron leaf but plain torch: separates primitive from architecture.
Tasks: velocity (y_vel 17-way, all packets) and shape (y_shape 3-way; reported on all and on
moving-only packets, like motion_leaf / kinopsis).  Splits test / test_photo / test_flat.
Reference bar only (s054 scope: readers stay pure trioron).  Env: SEEDS, EPOCHS, ARMS.
"""
import os, sys, time, torch
import torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.progenitor import motion as MO, motion_leaf as ML
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "6")))
SEEDS = int(os.environ.get("SEEDS", "3")); EPOCHS = int(os.environ.get("EPOCHS", "8"))
ARMS = os.environ.get("ARMS", "cnn2d,cnn3d,mlp656").split(",")
SPLITS = ["train", "test", "test_photo", "test_flat"]
def log(*a): print(*a, flush=True)

class CNN2D(nn.Module):
    def __init__(self, n_out, T=6):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(3 * T, 24, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                               nn.Conv2d(24, 48, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                               nn.Conv2d(48, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.h = nn.Linear(64, n_out)
    def forward(self, x): return self.h(self.f(x.flatten(1, 2)))          # x [N,T,3,S,S]
class CNN3D(nn.Module):
    def __init__(self, n_out):
        super().__init__()
        self.f = nn.Sequential(nn.Conv3d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool3d((1, 2, 2)),
                               nn.Conv3d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool3d((2, 2, 2)),
                               nn.Conv3d(32, 48, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool3d(1), nn.Flatten())
        self.h = nn.Linear(48, n_out)
    def forward(self, x): return self.h(self.f(x.transpose(1, 2)))        # -> [N,3,T,S,S]
class CNN2DStrong(nn.Module):
    """Fairer raw-packet arm: frames + frame differences as channels, BatchNorm, flatten (keeps position/phase) instead of GAP."""
    def __init__(self, n_out, T=6, ch=(24, 32, 16)):
        super().__init__()
        c = 3 * T + 3 * (T - 1); a, b, d = ch
        self.f = nn.Sequential(nn.Conv2d(c, a, 3, padding=1), nn.BatchNorm2d(a), nn.ReLU(), nn.MaxPool2d(2),
                               nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(), nn.MaxPool2d(2),
                               nn.Conv2d(b, d, 3, padding=1), nn.BatchNorm2d(d), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten())
        self.h = nn.Linear(d * 4 * 4, n_out)
    def forward(self, x):
        return self.h(self.f(torch.cat([x.flatten(1, 2), (x[:, 1:] - x[:, :-1]).flatten(1, 2)], 1)))
class MLP(nn.Module):
    def __init__(self, d, n_out, hid=48):
        super().__init__(); self.f = nn.Sequential(nn.Linear(d, hid), nn.ReLU(), nn.Linear(hid, n_out))
    def forward(self, x): return self.f(x)

def make(arm, n_out, d=None):
    return {"cnn2d": lambda: CNN2D(n_out), "cnn3d": lambda: CNN3D(n_out), "cnn2d_strong": lambda: CNN2DStrong(n_out), "cnn2d_big": lambda: CNN2DStrong(n_out, ch=(32, 64, 64)), "mlp656": lambda: MLP(d, n_out)}[arm]()

def run(arm, Xtr, ytr, tests, n_out, seed):
    torch.manual_seed(seed); net = make(arm, n_out, Xtr.shape[1] if Xtr.dim() == 2 else None)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3); g = torch.Generator().manual_seed(seed)
    prep = (lambda b: b) if Xtr.dim() == 2 else MO.as_float
    for ep in range(EPOCHS):
        net.train()
        for bi in torch.randperm(len(ytr), generator=g).split(256):
            opt.zero_grad(); F.cross_entropy(net(prep(Xtr[bi])), ytr[bi]).backward(); opt.step()
    net.eval(); out = {}
    with torch.no_grad():
        for nm, (X, y, m) in tests.items():
            pred = torch.cat([net(prep(X[i:i + 500])) for i in range(0, len(X), 500)]).argmax(1)
            out[nm] = float((pred == y)[m].float().mean())
    return out, sum(p.numel() for p in net.parameters())

if __name__ == "__main__":
    D = {sp: MO.load(sp) for sp in SPLITS}; Y = {sp: D[sp]["ys"] for sp in SPLITS}; X = {sp: D[sp]["X"] for sp in SPLITS}
    ALL = lambda sp: torch.ones(len(Y[sp]["y_vel"]), dtype=torch.bool); MOV = lambda sp: Y[sp]["y_vel"] > 0
    inputs = {"cnn2d": X, "cnn3d": X, "cnn2d_strong": X, "cnn2d_big": X}
    if "mlp656" in ARMS:
        Zm = {sp: ML.feats("motion", sp) for sp in SPLITS}; st = ML.Std(Zm["train"]); inputs["mlp656"] = {sp: st(Zm[sp]) for sp in SPLITS}
    log(f"MOTION CNN BAR s058: seeds {SEEDS} epochs {EPOCHS} arms {ARMS}; train n={len(Y['train']['y_vel'])}; T={X['train'].shape[1]}")
    log("  reference (Kinopsis s057, n=3): velocity .655 | shape moving-only photo .653 flat .870 mixed .766; msil voter alone photo .658")
    for task, yk, n_out in [("velocity", "y_vel", 17), ("shape", "y_shape", 3)]:
        for arm in ARMS:
            res = []; t0 = time.time()
            for seed in range(SEEDS):
                tests = {f"{sp}{'/mov' if mv else ''}": (inputs[arm][sp], Y[sp][yk], (MOV if mv else ALL)(sp)) for sp in SPLITS[1:] for mv in ((False, True) if task == "shape" else (False,))}
                r, npar = run(arm, inputs[arm]["train"], Y["train"][yk], tests, n_out, seed); res.append(r)
            m = {k: torch.tensor([r[k] for r in res]) for k in res[0]}
            log(f"  {task:8s} {arm:7s} {npar/1e3:5.1f}K | " + " | ".join(f"{k} {m[k].mean():.3f}±{m[k].std() if SEEDS > 1 else 0:.3f}" for k in m) + f"  [{(time.time() - t0) / SEEDS:.0f}s/seed]")
