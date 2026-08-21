"""Routed arbitration over heterogeneous leaves (s056 NEXT-0; Rocky: "build the router").

The uniform log-softmax vote dilutes a strong specialist (s054 nest+pre; s056 msil voter
photo .52 vs .66 alone).  A tiny router weights the leaves PER PACKET from evidence the
organism already has -- never the bg-kind / blur labels:
  evidence e (11-d, label-free):
    motion   : log gate ratio (motion_diag.motion_gate statistic), log mean / max temporal-
               difference energy (colour), decoded speed, mean patch coherence
    colour grouping : border-bg distance spread (std of the border distance map = scenery
               clutter), Otsu threshold / border noise floor (separation), colour-mask area,
               border-touch flag, motion-mask area
  + per-leaf confidence: max softmax prob and entropy of each voter (2 x K)
  router = softmax(W e' + b) over the K voters, motion voter hard-masked when its mask is empty (fires-only-when-moving), (K x (11 + 2K) + K params ~ 60), applied as
  per-packet weights on the voters' log-softmax; trained alone (leaves frozen) on the
  moving-world train split + a FROZEN REPLAY of each packet (T copies of the mid frame: the
  "nothing moves" regime, self-generated, no old-world data; FROZEN=0 disables) with CE of the
  weighted vote (300 Adam steps, batch 256).
Voters: the s053 organ's shape + whole leaves (colour-grouped, frozen) + msil (motion-
grouped shape leaf, trained per seed, then frozen).  The fill leaf is not a shape voter.
Arms: uniform vote | learned-temperature (per-leaf scalar, packet-independent) | router |
ORACLE (per packet pick any voter that is right: upper bound) | msil alone.
Scored on moving test (mixed/photo/flat, moving packets) + the OLD shape world (static
images -> T identical frames: gate off, motion mask empty; the router must fall back to
the colour leaves -> forgetting stays 0 by construction, but a bad router could hurt).
Env: SEEDS, EPOCHS (donor), STEPS (router, 300).
"""
import os, sys, time, math, torch, numpy as np
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.progenitor import motion as MO, motion_front as MF, motion_diag as MD, motion_leaf as ML, frontend as FE, grouping as GR, shapes as SH
from experiments.progenitor.motion_absorb import moving_streams, _shape_streams, nest_logits, acc
from experiments.progenitor.shape_prefeeder import Organ
import experiments.progenitor.shape_prefeeder as SP
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "6")))
SEEDS = int(os.environ.get("SEEDS", "3")); EPOCHS = int(os.environ.get("EPOCHS", "8")); STEPS = int(os.environ.get("STEPS", "300"))
SPLITS = ["train", "test", "test_photo", "test_flat"]
def log(*a): print(*a, flush=True)

def evidence(Xf, mmask):
    """label-free per-packet evidence [N,11] from the packet Xf [N,T,3,S,S] and the motion mask."""
    N, T = Xf.shape[:2]; C = MD._chans(Xf); mid = T // 2
    e = F.avg_pool2d(torch.sqrt(((C[:, 1:] - C[:, :-1]) ** 2).sum(2)).amax(1)[:, None], 3, 1, 1)[:, 0].flatten(1)
    ratio = (e.quantile(0.99, 1) + 1e-4) / (e.median(1).values + 1e-4)   # eps: T identical frames (old static world) give 0/0
    v, coh, E, _ = MF.velocity_map(Xf); spd = MF.decode_velocity(Xf).norm(dim=1)
    fg, d = GR.foreground(Xf[:, mid]); border = torch.cat([d[:, 0, :], d[:, -1, :], d[:, :, 0], d[:, :, -1]], 1)
    ots = torch.tensor([GR._otsu(di.flatten()) for di in d]); floor = border.quantile(0.10, dim=1).clamp(min=1e-3)
    touch = (fg[:, 0].any(1) | fg[:, -1].any(1) | fg[:, :, 0].any(1) | fg[:, :, -1].any(1)).float()
    return torch.stack([torch.log(ratio), torch.log(e.mean(1) + 1e-6), torch.log(e.amax(1) + 1e-6), spd, coh.flatten(1).mean(1),
                        border.std(1), torch.log(ots / floor), fg.float().flatten(1).mean(1), touch, mmask.float().flatten(1).mean(1), torch.log(d.flatten(1).mean(1) + 1e-6)], 1)

def evidence_split(split):
    p = os.path.join(MO.OUT, f"feat_evid_{split}.pt")
    if os.path.exists(p): return torch.load(p).float()
    d = MO.load(split); Xf = MO.as_float(d["X"]); t = time.time(); out = []
    for i in range(0, len(Xf), 500): out.append(evidence(Xf[i:i + 500], MD.motion_group(Xf[i:i + 500])))
    Z = torch.cat(out); torch.save(Z.half(), p); log(f"  evidence {split} in {time.time() - t:.0f}s"); return Z
def evidence_frozen(split="train"):
    """frozen replay: T identical copies of each packet's mid frame (self-generated; no old-world data) -> the 'nothing moves' regime."""
    p = os.path.join(MO.OUT, f"feat_evid_frozen_{split}.pt")
    if os.path.exists(p): return torch.load(p).float()
    d = MO.load(split); X = MO.as_float(d["X"]); mid = X.shape[1] // 2; Xf = X[:, mid][:, None].expand(-1, X.shape[1], -1, -1, -1).contiguous(); t = time.time(); out = []
    for i in range(0, len(Xf), 500): out.append(evidence(Xf[i:i + 500], MD.motion_group(Xf[i:i + 500])))
    Z = torch.cat(out); torch.save(Z.half(), p); log(f"  evidence frozen/{split} in {time.time() - t:.0f}s"); return Z
def evidence_old(split="test_fresh"):
    p = os.path.join(MO.OUT, f"feat_evid_old_{split}.pt")
    if os.path.exists(p): return torch.load(p).float()
    X = SH.load(split)[0]; Xf = X[:, None].expand(-1, MO.T_DEFAULT, -1, -1, -1).contiguous(); t = time.time(); out = []
    for i in range(0, len(Xf), 500): out.append(evidence(Xf[i:i + 500], MD.motion_group(Xf[i:i + 500])))
    Z = torch.cat(out); torch.save(Z.half(), p); log(f"  evidence old/{split} in {time.time() - t:.0f}s"); return Z

def conf(L):   # per voter: max prob, entropy
    p = F.softmax(L, 1); return torch.stack([p.max(1).values, -(p * torch.log(p + 1e-8)).sum(1)], 1)
class Router(torch.nn.Module):
    """softmax over voters; the MOTION voter is hard-masked when its motion mask is empty (fires-only-when-moving):
    a leaf that has never seen a class cannot be allowed to vote against it from an empty input."""
    def __init__(self, d, K, gate_col=None, motion_idx=None): super().__init__(); self.lin = torch.nn.Linear(d, K); torch.nn.init.zeros_(self.lin.weight); torch.nn.init.zeros_(self.lin.bias); self.gc, self.mi = gate_col, motion_idx
    def forward(self, e):
        z = self.lin(e)
        if self.gc is not None:
            off = e[:, self.gc] <= 0; z = z.clone(); z[off, self.mi] = -1e4
        return F.softmax(z, 1)
def vote(Ls, w):   # Ls list of [N,C] logits, w [N,K] -> [N,C]
    return sum(w[:, k:k + 1] * F.log_softmax(L, 1) for k, L in enumerate(Ls))

if __name__ == "__main__":
    log(f"MOTION ROUTER s056: seeds {SEEDS} donor epochs {EPOCHS} router steps {STEPS}")
    Y = {sp: MO.load(sp)["ys"] for sp in SPLITS}; S = {sp: moving_streams(sp) for sp in SPLITS}; EV = {sp: evidence_split(sp) for sp in SPLITS}
    MOV = {sp: Y[sp]["y_vel"] > 0 for sp in SPLITS}
    old = _shape_streams("test_fresh"); y_old = SH.load("test_fresh")[1]["y_shape"]; EV_old = evidence_old()
    # old world has no motion mask -> msil stream = descriptor of an empty mask: build it once (zeros silhouette, frame, flags as describe() gives)
    Xo = SH.load("test_fresh")[0]; gl0 = [[] for _ in range(len(Xo))]; D0, _ = GR.describe(Xo, gl=gl0, canon="scale", extras=False); old["mshape"] = torch.cat([D0["silhouette"], D0["frame"], D0["flags"]], 1)
    EV_fr = evidence_frozen("train"); FROZEN = os.environ.get("FROZEN", "1") == "1"
    frz = {k: S["train"][k] for k in ("shape", "whole")}; frz["mshape"] = old["mshape"][:1].expand(len(S["train"]["shape"]), -1).contiguous()   # empty-mask descriptor
    mu_e, sd_e = EV["train"].mean(0), EV["train"].std(0) + 1e-6; ne = lambda e: (e - mu_e) / sd_e
    rows = []
    for seed in range(SEEDS):
        SP.PRE_SEED = seed; os.environ["PRE_SEED"] = str(seed); organ = Organ(); leaves = organ.leaves; norm = organ._norm
        Ztr = norm("shape", S["train"]["mshape"]); msil = ML.leaf(Ztr.shape[1], 5, seed); params = msil.trainable_tensors(); opt = torch.optim.Adam(params, lr=1e-3); g = torch.Generator().manual_seed(seed)
        for ep in range(EPOCHS):
            for bi in torch.randperm(len(Ztr), generator=g).split(256):
                opt.zero_grad(); F.cross_entropy(msil(Ztr[bi]), Y["train"]["y_shape"][bi]).backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        for p_ in params: p_.requires_grad_(False)
        def voters(Sd):
            L = nest_logits(leaves, norm, Sd, keys=("shape", "whole"))
            with torch.no_grad(): Lm = torch.cat([msil(norm("shape", Sd["mshape"][i:i + 4000])) for i in range(0, len(Sd["mshape"]), 4000)])
            return [L["shape"], L["whole"], Lm]
        V = {sp: voters(S[sp]) for sp in SPLITS}; V["old"] = voters(old); V["frozen"] = voters(frz)
        feat = lambda sp, e: torch.cat([ne(e), *[conf(L) for L in V[sp]], e[:, 9:10]], 1)   # last col = RAW motion-mask area (gate)
        Xr = {sp: feat(sp, EV[sp]) for sp in SPLITS}; Xr["old"] = feat("old", EV_old); Xr["frozen"] = feat("frozen", EV_fr)
        if FROZEN:   # router training set = moving packets + their frozen replay (same labels)
            V["train"] = [torch.cat([a, b]) for a, b in zip(V["train"], V["frozen"])]; Xr["train"] = torch.cat([Xr["train"], Xr["frozen"]]); Y_rt = torch.cat([Y["train"]["y_shape"]] * 2)
        else: Y_rt = Y["train"]["y_shape"]
        K = 3; router = Router(Xr["train"].shape[1], K, gate_col=Xr["train"].shape[1] - 1, motion_idx=2); temp = torch.nn.Parameter(torch.zeros(K))
        gated = lambda x: (lambda w: w * torch.where(x[:, -1:] > 0, torch.ones(1), torch.tensor([[1., 1., 0.]])) / (w * torch.where(x[:, -1:] > 0, torch.ones(1), torch.tensor([[1., 1., 0.]]))).sum(1, keepdim=True))(torch.full((len(x), K), 1 / K))
        def fit(params, wfn, steps=STEPS):
            opt = torch.optim.Adam(params, lr=1e-2); g = torch.Generator().manual_seed(seed); y = Y_rt
            for _ in range(steps):
                bi = torch.randint(len(y), (256,), generator=g); opt.zero_grad(); F.cross_entropy(vote([L[bi] for L in V["train"]], wfn(Xr["train"][bi])), y[bi]).backward(); opt.step()
        fit([temp], lambda x: F.softmax(temp, 0)[None].expand(len(x), -1)); fit(list(router.parameters()), router)
        arms = {"uniform": lambda x: torch.full((len(x), K), 1 / K), "uniform+gate": gated, "temperature": lambda x: F.softmax(temp, 0)[None].expand(len(x), -1), "router": router,
                "msil alone": lambda x: torch.tensor([[0., 0., 1.]]).expand(len(x), -1), "colour only": lambda x: torch.tensor([[.5, .5, 0.]]).expand(len(x), -1)}
        with torch.no_grad():
            for nm, wfn in arms.items():
                r = dict(arm=nm, seed=seed, old=acc(vote(V["old"], wfn(Xr["old"])).argmax(1), y_old))
                for sp in SPLITS[1:]: r[sp] = acc(vote(V[sp], wfn(Xr[sp])).argmax(1), Y[sp]["y_shape"], MOV[sp])
                rows.append(r); log(f"  s{seed} {nm:12s} | mixed {r['test']:.3f} photo {r['test_photo']:.3f} flat {r['test_flat']:.3f} | old world {r['old']:.3f}")
            r = dict(arm="oracle", seed=seed, old=float(torch.stack([L.argmax(1) == y_old for L in V["old"]]).any(0).float().mean()))
            for sp in SPLITS[1:]: r[sp] = float(torch.stack([L.argmax(1) == Y[sp]["y_shape"] for L in V[sp]]).any(0)[MOV[sp]].float().mean())
            rows.append(r); log(f"  s{seed} {'oracle':12s} | mixed {r['test']:.3f} photo {r['test_photo']:.3f} flat {r['test_flat']:.3f} | old world {r['old']:.3f}")
            w = router(Xr["test"]); log(f"      router mean weight (shape, whole, msil): photo {w[Y['test']['y_bgkind'] == 1].mean(0).tolist()} flat {w[Y['test']['y_bgkind'] == 0].mean(0).tolist()} static {w[~MOV['test']].mean(0).tolist()} old-world {router(Xr['old']).mean(0).tolist()}")
            log(f"      router params {sum(p.numel() for p in router.parameters())}; |W| by evidence dim: {[round(float(a), 2) for a in router.lin.weight.abs().sum(0)]}")
    log("\n== summary (mean±std over seeds; moving packets) ==")
    for arm in dict.fromkeys(r["arm"] for r in rows):
        rs = [r for r in rows if r["arm"] == arm]; m = lambda k: torch.tensor([r[k] for r in rs])
        log(f"  {arm:12s} mixed {m('test').mean():.3f}±{m('test').std():.3f} photo {m('test_photo').mean():.3f}±{m('test_photo').std():.3f} flat {m('test_flat').mean():.3f}±{m('test_flat').std():.3f} | old world {m('old').mean():.3f}±{m('old').std():.3f}")
