"""Absorb the motion leaf into the s053 nest (s056, motion arc step 4b; Rocky: option 1).

Recipient = the s053 shape-world nest (shape 103 / whole 205 / fill 292 leaves, trained
jointly on the STATIC shape world, seed PRE_SEED; `shape_prefeeder.Organ`), frozen =
the pre-trained organ.  It meets the MOVING world (motion.py) through its own streams
computed on the packet's MID frame with its native shape-world standardization.
Donors (trained on the moving world, 8 ep, per seed):
  vel      : motion_full 656 -> y_vel 17       (the motion leaf)
  msil     : shape stream (103: silhouette+frame+flags) from the MOTION-GROUPED body (v4) -> 5-way
  csil     : the SAME shape stream from the colour-grouped body, new world -> 5-way (a true SIBLING
             of the nest's shape leaf: same input width/semantics, same head) -> graftable
  both_in  : [colour shape 103 ++ motion shape 103] -> 5-way (motion silhouette as INPUT, s054 lesson)
Arms (shape = 5-way argmax over the nest's log-softmax vote; velocity from the vel leaf):
  zero-shot     : frozen nest on the moving world                  (transfer baseline)
  +vel          : paste-and-go: nest + vel leaf; shape untouched    (velocity acquired, zero forgetting by construction)
  +msil voter   : nest + msil leaf as a 4th equal voter
  +both_in voter: nest + both_in leaf as a 4th voter
  msil only     : msil leaf alone (what the motion silhouette carries)
  graft+settle  : absorb(nest.shape, csil) head-merged (Protocol B) then HEAD-ONLY settle (200 Adam 1e-3
                  steps, soma frozen) of all three nest leaves on SETTLE_N labelled moving packets
  settle naive / settle+replay : settle without graft; 'naive' = moving-world labels only (3 of 5 shapes -> forgets the
                  field classes), '+replay' = the s054 recipe: class-balanced full-cov archive samples of every OLD class mixed in
  finetune      : all nest params trained on the moving world (bound: best moving, worst forgetting)
Scored on moving test (mixed / photo / flat; moving packets only, shapes 0-2 present) and on the OLD
shape world (shapes test_fresh): forgetting = old-world shape acc before - after.
Env: SEEDS, EPOCHS, SETTLE_N (default 2000), SETTLE_STEPS (200).
"""
import os, sys, time, copy, math, torch, numpy as np
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scipy import ndimage as ndi
from experiments.progenitor import motion as MO, motion_front as MF, motion_diag as MD, motion_leaf as ML, frontend as FE, grouping as GR, shapes as SH, shapes_feats as SF
from experiments.progenitor.shape_prefeeder import Organ, _shape_streams
from trioron.api import absorb
from trioron.learning.manifold import get_interior_ids
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "6")))
SEEDS = int(os.environ.get("SEEDS", "3")); EPOCHS = int(os.environ.get("EPOCHS", "8")); SETTLE_N = int(os.environ.get("SETTLE_N", "2000")); SETTLE_STEPS = int(os.environ.get("SETTLE_STEPS", "200"))
SPLITS = ["train", "test", "test_photo", "test_flat"]
def log(*a): print(*a, flush=True)

def _groups_from_masks(X, masks):
    """grouping.groups-compatible group lists from given body masks (one body per image)."""
    gl = []; ek = GR._disk(1)
    for i in range(len(X)):
        sil = masks[i].numpy().astype(bool); area = int(sil.sum())
        if area < GR.MIN_AREA: gl.append([]); continue
        inter = ndi.binary_erosion(sil, structure=ek, border_value=0); bnd = sil & ~inter
        yy, xx = np.nonzero(sil); cy, cx = yy.mean(), xx.mean(); cov = np.cov(np.stack([xx - cx, yy - cy])) if area > 2 else np.eye(2)
        ev, evec = np.linalg.eigh(cov + 1e-6 * np.eye(2)); sc = np.sqrt(np.maximum(ev, 1e-6)); ang = math.atan2(evec[1, 1], evec[0, 1]) % math.pi
        col = X[i][:, torch.from_numpy(sil)].mean(1)
        gl.append([dict(mask=torch.from_numpy(sil), interior=torch.from_numpy(inter), boundary=torch.from_numpy(bnd), rawfg=torch.from_numpy(sil), area=area, is_field=False,
                        touches=bool(sil[0].any() or sil[-1].any() or sil[:, 0].any() or sil[:, -1].any()), colour=col,
                        frame=torch.tensor([cx, cy, sc[1], sc[0], ang, sc[1] / max(sc[0], 1e-3), 1.0], dtype=torch.float))])
    return gl

def moving_streams(split):
    """nest streams (shape/whole/fill) on the mid frame via COLOUR grouping + the shape stream via MOTION grouping; cached."""
    p = os.path.join(MO.OUT, f"feat_nest_{split}.pt")
    if os.path.exists(p): return {k: v.float() for k, v in torch.load(p).items()}
    d = MO.load(split); Xf = MO.as_float(d["X"]); mid = Xf.shape[1] // 2; Xm = Xf[:, mid]; t = time.time(); out = {"shape": [], "whole": [], "fill": [], "mshape": []}
    for i in range(0, len(Xm), 1000):
        D, _ = GR.describe(Xm[i:i + 1000], canon="scale"); WH = torch.cat([FE.boundary_block(Xm[i:i + 1000]), FE.colour_block(Xm[i:i + 1000]), FE.corner_block(Xm[i:i + 1000])], 1)
        out["shape"].append(torch.cat([D["silhouette"], D["frame"], D["flags"]], 1)); out["whole"].append(WH); out["fill"].append(torch.cat([D["ctex"], D["cstereo"], D["flags"]], 1))
        gl = _groups_from_masks(Xm[i:i + 1000], MD.motion_group(Xf[i:i + 1000])); Dm, _ = GR.describe(Xm[i:i + 1000], gl=gl, canon="scale", extras=False)
        out["mshape"].append(torch.cat([Dm["silhouette"], Dm["frame"], Dm["flags"]], 1))
    out = {k: torch.cat(v) for k, v in out.items()}; torch.save({k: v.half() for k, v in out.items()}, p); log(f"  nest streams {split} in {time.time() - t:.0f}s"); return out

def nest_logits(leaves, norm, S, keys=("shape", "whole", "fill")):
    with torch.no_grad(): return {k: torch.cat([leaves[k](norm(k, S[k][i:i + 4000])) for i in range(0, len(S[k]), 4000)]) for k in keys}
def shape_vote(L, extra=()):
    v = F.log_softmax(L["shape"], 1) + F.log_softmax(L["whole"], 1)
    for e in extra: v = v + F.log_softmax(e, 1)
    return v.argmax(1)
def acc(pred, y, m=None): return float((pred == y)[m if m is not None else slice(None)].float().mean())

if __name__ == "__main__":
    log(f"MOTION ABSORB s056: seeds {SEEDS} epochs {EPOCHS} settle_n {SETTLE_N} settle_steps {SETTLE_STEPS}")
    Y = {sp: MO.load(sp)["ys"] for sp in SPLITS}; S = {sp: moving_streams(sp) for sp in SPLITS}
    MOV = {sp: Y[sp]["y_vel"] > 0 for sp in SPLITS}
    old = _shape_streams("test_fresh"); y_old = SH.load("test_fresh")[1]["y_shape"]
    Zm = {sp: ML.feats("motion", sp) for sp in SPLITS}; stdm = ML.Std(Zm["train"]); Zm = {sp: stdm(Zm[sp]) for sp in SPLITS}
    rows = []
    for seed in range(SEEDS):
        os.environ["PRE_SEED"] = str(seed); import experiments.progenitor.shape_prefeeder as SP; SP.PRE_SEED = seed
        organ = Organ(); organ.sanity(); leaves = organ.leaves; norm = organ._norm
        base_old = acc(shape_vote(nest_logits(leaves, norm, old)), y_old)
        def score(name, extra_fn=None, lv=None, leaves_=None):
            lv_ = leaves_ or leaves; r = dict(arm=name, seed=seed, old=acc(shape_vote(nest_logits(lv_, norm, old)), y_old))
            r["forget"] = base_old - r["old"]
            for sp in SPLITS[1:]:
                L = nest_logits(lv_, norm, S[sp]); ex = extra_fn(sp) if extra_fn else ()
                r[sp] = acc(shape_vote(L, ex), Y[sp]["y_shape"], MOV[sp])
            r["vel"] = lv if lv is not None else float("nan"); rows.append(r)
            log(f"  s{seed} {name:20s} | moving shape: mixed {r['test']:.3f} photo {r['test_photo']:.3f} flat {r['test_flat']:.3f} | old world {r['old']:.3f} (forget {r['forget']:+.3f}) | vel {r['vel']:.3f}")
        score("zero-shot")
        # donors
        vel_sub = ML.leaf(Zm["train"].shape[1], 17, seed); ML.EPOCHS = EPOCHS
        tests = {sp: (Zm[sp], Y[sp]["y_vel"], torch.ones(len(Y[sp]["y_vel"]), dtype=torch.bool)) for sp in SPLITS[1:]}
        rv, _ = ML.train_eval(Zm["train"], Y["train"]["y_vel"], tests, 17, seed)
        score("+vel", lv=rv["test"])
        def donor(stream_tr, y, d_out):
            sub = ML.leaf(stream_tr.shape[1], d_out, seed); params = sub.trainable_tensors(); opt = torch.optim.Adam(params, lr=1e-3); g = torch.Generator().manual_seed(seed)
            for ep in range(EPOCHS):
                for bi in torch.randperm(len(y), generator=g).split(256):
                    opt.zero_grad(); F.cross_entropy(sub(stream_tr[bi]), y[bi]).backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            return sub
        msil = donor(norm("shape", S["train"]["mshape"]), Y["train"]["y_shape"], 5)
        ev = lambda sub, k: (lambda sp: (torch.cat([sub(norm("shape", S[sp][k][i:i + 4000])) for i in range(0, len(S[sp][k]), 4000)]).detach(),))
        score("+msil voter", extra_fn=ev(msil, "mshape"), lv=rv["test"])
        for sp in SPLITS[1:]: log(f"      msil alone {sp}: {acc(ev(msil, 'mshape')(sp)[0].argmax(1), Y[sp]['y_shape'], MOV[sp]):.3f}")
        both_tr = torch.cat([norm("shape", S["train"]["shape"]), norm("shape", S["train"]["mshape"])], 1); both = donor(both_tr, Y["train"]["y_shape"], 5)
        evb = lambda sp: (torch.cat([both(torch.cat([norm("shape", S[sp]["shape"][i:i + 4000]), norm("shape", S[sp]["mshape"][i:i + 4000])], 1)) for i in range(0, len(S[sp]["shape"]), 4000)]).detach(),)
        score("+both_in voter", extra_fn=evb, lv=rv["test"])
        for sp in SPLITS[1:]: log(f"      both_in alone {sp}: {acc(evb(sp)[0].argmax(1), Y[sp]['y_shape'], MOV[sp]):.3f}")
        csil = donor(norm("shape", S["train"]["shape"]), Y["train"]["y_shape"], 5)
        # settle helpers (head-only, all three nest leaves, on SETTLE_N labelled moving packets)
        g = torch.Generator().manual_seed(100 + seed); idx = torch.randperm(len(Y["train"]["y_shape"]), generator=g)[:SETTLE_N]
        # full-cov manifold archive of the OLD world per leaf/class (s054 recipe): mu + rank-32 sampler in the leaf's normalized stream
        old_tr = _shape_streams("train"); y_old_tr = SH.load("train")[1]; arch = {}
        for k in leaves:
            Zk = norm(k, old_tr[k]); yk = y_old_tr["y_shape" if k != "fill" else "y_fill"]; arch[k] = {}
            for c in yk.unique().tolist():
                Zc = Zk[yk == c]; mu = Zc.mean(0); cov = torch.cov(Zc.T) + 0.1 * torch.eye(Zc.shape[1]); ev, V = torch.linalg.eigh(cov); ev = ev.clamp(min=1e-8)
                arch[k][c] = (mu, V[:, -32:] * ev[-32:].sqrt(), float(ev[:-32].mean()) ** 0.5 if Zc.shape[1] > 32 else 0.0)
        def settle(lv_, head_only=True, steps=SETTLE_STEPS, replay=False):
            for k, sub in lv_.items():
                for p_ in sub.trainable_tensors(): p_.requires_grad_(True)
                a = sub.arena; out_ids = sub.scheduler._plan.output_ids.long(); hc = torch.zeros(a.capacity, dtype=torch.bool); hc[out_ids] = True; he = torch.isin(a.edge_dst.long(), out_ids)
                opt = torch.optim.Adam([a.bias, a.edge_weight], lr=1e-3); yk = Y["train"]["y_shape" if k != "fill" else "y_fill"]; Zk = norm(k, S["train"][k])
                gg = torch.Generator().manual_seed(seed)
                for _ in range(steps):
                    bi = idx[torch.randint(len(idx), (128,), generator=gg)]; xb, yb = Zk[bi], yk[bi]
                    if replay:   # class-balanced archive samples of EVERY old class (the s054 settle), same count as the new batch
                        per = max(2, 128 // len(arch[k])); xs, ys_ = [], []
                        for c, (mu, A, rs) in arch[k].items():
                            smp = mu + torch.randn(per, A.shape[1], generator=gg) @ A.t()
                            if rs > 0: smp = smp + rs * torch.randn(per, mu.numel(), generator=gg)
                            xs.append(smp); ys_.append(torch.full((per,), c, dtype=torch.long))
                        xb = torch.cat([xb, *xs]); yb = torch.cat([yb, *ys_])
                    opt.zero_grad(); F.cross_entropy(sub(xb), yb).backward()
                    if head_only: a.bias.grad[~hc] = 0.0; a.edge_weight.grad[~he] = 0.0
                    sub.zero_dormant_grads(); opt.step()
                for p_ in sub.trainable_tensors(): p_.requires_grad_(False)
        lv2 = {k: copy.deepcopy(v) for k, v in leaves.items()}; settle(lv2); score("settle naive", leaves_=lv2, lv=rv["test"])
        lv2 = {k: copy.deepcopy(v) for k, v in leaves.items()}; settle(lv2, replay=True); score("settle+replay", leaves_=lv2, lv=rv["test"])
        lv3 = {k: copy.deepcopy(v) for k, v in leaves.items()}
        for p_ in lv3["shape"].trainable_tensors(): p_.requires_grad_(True)
        res = absorb(lv3["shape"], csil, freeze=False)[0]; log(f"      graft: {res}")
        for p_ in lv3["shape"].trainable_tensors(): p_.requires_grad_(False)
        score("graft (no settle)", leaves_=lv3, lv=rv["test"]); lv3b = {k: copy.deepcopy(v) for k, v in lv3.items()}
        settle(lv3b); score("graft+settle naive", leaves_=lv3b, lv=rv["test"]); settle(lv3, replay=True); score("graft+settle+replay", leaves_=lv3, lv=rv["test"])
        lv4 = {k: copy.deepcopy(v) for k, v in leaves.items()}; settle(lv4, head_only=False, steps=EPOCHS * len(Y["train"]["y_shape"]) // 128); score("finetune", leaves_=lv4, lv=rv["test"])
    log("\n== summary (mean±std over seeds; moving packets only) ==")
    for arm in dict.fromkeys(r["arm"] for r in rows):
        rs = [r for r in rows if r["arm"] == arm]; m = lambda k: torch.tensor([r[k] for r in rs])
        log(f"  {arm:20s} mixed {m('test').mean():.3f}±{m('test').std():.3f} photo {m('test_photo').mean():.3f}±{m('test_photo').std():.3f} flat {m('test_flat').mean():.3f}±{m('test_flat').std():.3f} | old {m('old').mean():.3f} forget {m('forget').mean():+.3f}±{m('forget').std():.3f} | vel {m('vel').mean():.3f}")
