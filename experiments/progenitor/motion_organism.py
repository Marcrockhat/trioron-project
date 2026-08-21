"""ONE organism (s057, NEXT-0/2 of s056; Rocky: "focus on 2(a)"): the s053 organ + the motion-
silhouette leaf + the velocity leaf + the s056 evidence router, built by a CONTINUAL SCHEDULE and
carried as a single saveable object.

Schedule (per seed):
  stage 1  OLD WORLD     : shape_prefeeder.Organ (shape/whole/fill leaves, static shape world)      -> frozen
  stage 2  MOVING WORLD  : (a) optional replay-settle of the organ heads (s054 recipe: class-balanced
                           full-cov archive samples of every old class mixed into each batch);
                           (b) msil leaf acquired on the motion-grouped shape stream (8 ep), frozen;
                           (c) router (motion_router.Router, 57 params) fitted alone on the moving train
                           packets + their FROZEN REPLAY, voters frozen; motion voter hard-gated.
  stage 3  VELOCITY      : vel leaf on motion_full (17-way), paste-and-go (separate head, no interaction).
Scored AFTER EVERY STAGE on moving test (mixed/photo/flat; moving packets) + old world + velocity, so
forgetting is tracked per stage.  Arms (= how stage 2 arbitrates):
  uniform        : equal log-softmax vote (the motion_absorb "+msil voter" arm)            [reference]
  router         : evidence router, no settle                                               [s056 router, now inside the organism]
  settle+router  : replay-settle the organ heads first, then msil + router on the settled voters
  settle+uniform : replay-settle, equal vote                                                [motion_absorb settle+replay arm]
One object: MotionOrganism.save(path) / load(path) (torch pickle); size reported.  Inference API:
  shape(streams, evidence) -> [N,5] routed log-prob;  velocity(Zmotion) -> [N,17] logits.
Env: SEEDS, EPOCHS (leaves), STEPS (router, 300), SETTLE_N (2000), SETTLE_STEPS (200).
"""
import os, sys, time, copy, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.progenitor import motion as MO, motion_leaf as ML, grouping as GR, shapes as SH
from experiments.progenitor.motion_absorb import moving_streams, _shape_streams, nest_logits, acc
from experiments.progenitor.motion_router import evidence_split, evidence_frozen, evidence_old, conf, Router, vote
from experiments.progenitor.shape_prefeeder import Organ
import experiments.progenitor.shape_prefeeder as SP
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "6")))
SEEDS = int(os.environ.get("SEEDS", "3")); EPOCHS = int(os.environ.get("EPOCHS", "8")); STEPS = int(os.environ.get("STEPS", "300"))
SETTLE_N = int(os.environ.get("SETTLE_N", "2000")); SETTLE_STEPS = int(os.environ.get("SETTLE_STEPS", "200"))
SPLITS = ["train", "test", "test_photo", "test_flat"]; VOTERS = ("shape", "whole", "msil")
def log(*a): print(*a, flush=True)

def _train_leaf(sub, Z, y, seed, epochs):
    params = sub.trainable_tensors(); opt = torch.optim.Adam(params, lr=1e-3); g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for bi in torch.randperm(len(y), generator=g).split(256):
            opt.zero_grad(); F.cross_entropy(sub(Z[bi]), y[bi]).backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
    for p in params: p.requires_grad_(False)
    return sub
def _chunked(sub, Z, n=4000):
    with torch.no_grad(): return torch.cat([sub(Z[i:i + n]) for i in range(0, len(Z), n)])

class MotionOrganism:
    """organ (3 leaves) + msil + vel + router + all calibration stats, one object."""
    def __init__(self, seed):
        self.seed = seed; self.msil = self.vel = self.router = None; self.stdm = None; self.ev_mu = self.ev_sd = None; self.settled = False
    # ── stage 1: old world ──
    def stage_old(self):
        SP.PRE_SEED = self.seed; os.environ["PRE_SEED"] = str(self.seed); organ = Organ()
        self.leaves, self.std = organ.leaves, organ.std
        with torch.no_grad():   # drop the grad-bearing last_activations left by training (deepcopy/pickle need leaves only)
            for k, sub in self.leaves.items(): sub(torch.zeros(1, self.std[k][0].numel()))
        # empty-mask descriptor for the msil stream when nothing moves (static images -> gate off); stored with the organism
        X0 = SH.load("test_fresh")[0][:1]; D0, _ = GR.describe(X0, gl=[[]], canon="scale", extras=False)
        self.empty_mshape = torch.cat([D0["silhouette"], D0["frame"], D0["flags"]], 1)[0]
        return self
    def norm(self, k, Z): mu, sd = self.std[k]; return ((Z - mu) / sd).float()
    # ── stage 2a: replay settle of the organ heads (s054 recipe) ──
    def settle_heads(self, S_tr, Y_tr, old_tr, y_old_tr, n=SETTLE_N, steps=SETTLE_STEPS):
        g = torch.Generator().manual_seed(100 + self.seed); idx = torch.randperm(len(Y_tr["y_shape"]), generator=g)[:n]
        for k, sub in self.leaves.items():
            yk_old = y_old_tr["y_shape" if k != "fill" else "y_fill"]; Zo = self.norm(k, old_tr[k]); arch = {}
            for c in yk_old.unique().tolist():
                Zc = Zo[yk_old == c]; mu = Zc.mean(0); cov = torch.cov(Zc.T) + 0.1 * torch.eye(Zc.shape[1]); ev, V = torch.linalg.eigh(cov); ev = ev.clamp(min=1e-8)
                arch[c] = (mu, V[:, -32:] * ev[-32:].sqrt(), float(ev[:-32].mean()) ** 0.5)
            for p in sub.trainable_tensors(): p.requires_grad_(True)
            a = sub.arena; out_ids = sub.scheduler._plan.output_ids.long(); hc = torch.zeros(a.capacity, dtype=torch.bool); hc[out_ids] = True; he = torch.isin(a.edge_dst.long(), out_ids)
            opt = torch.optim.Adam([a.bias, a.edge_weight], lr=1e-3); yk = Y_tr["y_shape" if k != "fill" else "y_fill"]; Zk = self.norm(k, S_tr[k]); gg = torch.Generator().manual_seed(self.seed)
            per = max(2, 128 // len(arch))
            for _ in range(steps):
                bi = idx[torch.randint(len(idx), (128,), generator=gg)]; xs, ys = [Zk[bi]], [yk[bi]]
                for c, (mu, A, rs) in arch.items():
                    xs.append(mu + torch.randn(per, A.shape[1], generator=gg) @ A.t() + rs * torch.randn(per, mu.numel(), generator=gg)); ys.append(torch.full((per,), c, dtype=torch.long))
                opt.zero_grad(); F.cross_entropy(sub(torch.cat(xs)), torch.cat(ys)).backward(); a.bias.grad[~hc] = 0.0; a.edge_weight.grad[~he] = 0.0; sub.zero_dormant_grads(); opt.step()
            for p in sub.trainable_tensors(): p.requires_grad_(False)
        self.settled = True; return self
    # ── stage 2b/c: msil leaf + router ──
    def acquire_msil(self, S_tr, y_tr, epochs=EPOCHS):
        Z = self.norm("shape", S_tr["mshape"]); self.msil = _train_leaf(ML.leaf(Z.shape[1], 5, self.seed), Z, y_tr, self.seed, epochs); return self
    def voters(self, S):
        L = nest_logits(self.leaves, self.norm, S, keys=("shape", "whole"))
        m = S["mshape"] if "mshape" in S else self.empty_mshape.expand(len(S["shape"]), -1)
        return [L["shape"], L["whole"], _chunked(self.msil, self.norm("shape", m)) if self.msil is not None else torch.zeros_like(L["shape"])]
    def _feat(self, Ls, E):
        return torch.cat([(E - self.ev_mu) / self.ev_sd, *[conf(L) for L in Ls], E[:, 9:10]], 1)   # last col = raw motion-mask area (hard gate)
    def fit_router(self, S_tr, E_tr, E_frz, y_tr, steps=STEPS):
        self.ev_mu, self.ev_sd = E_tr.mean(0), E_tr.std(0) + 1e-6
        Vm = self.voters(S_tr); Vf = self.voters({k: S_tr[k] for k in ("shape", "whole")})   # frozen replay: mid frame only, empty motion mask
        V = [torch.cat([a, b]) for a, b in zip(Vm, Vf)]; X = torch.cat([self._feat(Vm, E_tr), self._feat(Vf, E_frz)]); y = torch.cat([y_tr, y_tr])
        self.router = Router(X.shape[1], 3, gate_col=X.shape[1] - 1, motion_idx=2); opt = torch.optim.Adam(self.router.parameters(), lr=1e-2); g = torch.Generator().manual_seed(self.seed)
        for _ in range(steps):
            bi = torch.randint(len(y), (256,), generator=g); opt.zero_grad(); F.cross_entropy(vote([L[bi] for L in V], self.router(X[bi])), y[bi]).backward(); opt.step()
        for p in self.router.parameters(): p.requires_grad_(False)
        return self
    # ── stage 3: velocity ──
    def acquire_vel(self, Zm_tr, y_vel, epochs=EPOCHS):
        self.stdm = ML.Std(Zm_tr); self.vel = _train_leaf(ML.leaf(Zm_tr.shape[1], 17, self.seed), self.stdm(Zm_tr), y_vel, self.seed, epochs); return self
    # ── inference ──
    def shape(self, S, E=None, mode="router"):
        Ls = self.voters(S)
        if mode == "router" and self.router is not None and E is not None:
            with torch.no_grad(): return vote(Ls, self.router(self._feat(Ls, E)))
        w = torch.tensor([[1., 1., 1.]]) if self.msil is not None else torch.tensor([[1., 1., 0.]])
        if "mshape" not in S: w = torch.tensor([[1., 1., 0.]])   # nothing moves: msil silent
        return vote(Ls, w.expand(len(Ls[0]), -1))
    def velocity(self, Zm): return _chunked(self.vel, self.stdm(Zm))
    def save(self, path): torch.save(self, path); return os.path.getsize(path)
    @staticmethod
    def load(path): return torch.load(path, weights_only=False)
    def n_params(self):
        subs = list(self.leaves.values()) + [s for s in (self.msil, self.vel) if s is not None]
        return sum(int(p.numel()) for s in subs for p in s.trainable_tensors()) + (sum(p.numel() for p in self.router.parameters()) if self.router else 0)

if __name__ == "__main__":
    log(f"MOTION ORGANISM s057: seeds {SEEDS} epochs {EPOCHS} router steps {STEPS} settle_n {SETTLE_N} settle_steps {SETTLE_STEPS}")
    Y = {sp: MO.load(sp)["ys"] for sp in SPLITS}; S = {sp: moving_streams(sp) for sp in SPLITS}; EV = {sp: evidence_split(sp) for sp in SPLITS}; MOV = {sp: Y[sp]["y_vel"] > 0 for sp in SPLITS}
    old = _shape_streams("test_fresh"); y_old = SH.load("test_fresh")[1]["y_shape"]; EV_old = evidence_old(); EV_fr = evidence_frozen("train")
    old_tr = _shape_streams("train"); y_old_tr = SH.load("train")[1]
    Zm = {sp: ML.feats("motion", sp) for sp in SPLITS}
    rows = []
    def score(org, arm, stage, mode, base_old):
        with torch.no_grad():
            r = dict(arm=arm, stage=stage, seed=org.seed, old=acc(org.shape(old, EV_old, mode).argmax(1), y_old)); r["forget"] = base_old - r["old"]
            for sp in SPLITS[1:]: r[sp] = acc(org.shape(S[sp], EV[sp], mode).argmax(1), Y[sp]["y_shape"], MOV[sp])
            r["vel"] = acc(org.velocity(Zm["test"]).argmax(1), Y["test"]["y_vel"]) if org.vel is not None else float("nan")
        rows.append(r); log(f"  s{org.seed} {arm:15s} {stage:9s} | mixed {r['test']:.3f} photo {r['test_photo']:.3f} flat {r['test_flat']:.3f} | old {r['old']:.3f} (forget {r['forget']:+.3f}) | vel {r['vel']:.3f}"); return r
    for seed in range(SEEDS):
        t0 = time.time(); base = MotionOrganism(seed).stage_old(); base_old = acc(base.shape(old, None, "uniform").argmax(1), y_old)
        log(f"  s{seed} stage 1 old world: shape {base_old:.3f} (uniform vote of the organ); zero-shot moving: " + " ".join(f"{sp} {acc(base.shape(S[sp], None, 'uniform').argmax(1), Y[sp]['y_shape'], MOV[sp]):.3f}" for sp in SPLITS[1:]))
        for arm in ("uniform", "router", "settle+uniform", "settle+router"):
            org = copy.deepcopy(base); mode = "router" if "router" in arm else "uniform"
            if arm.startswith("settle"): org.settle_heads(S["train"], Y["train"], old_tr, y_old_tr)
            org.acquire_msil(S["train"], Y["train"]["y_shape"])
            if mode == "router": org.fit_router(S["train"], EV["train"], EV_fr, Y["train"]["y_shape"])
            score(org, arm, "stage 2", mode, base_old)
            org.acquire_vel(Zm["train"], Y["train"]["y_vel"]); r = score(org, arm, "stage 3", mode, base_old)
            if arm == "router":
                p = os.path.join(MO.OUT, f"motion_organism_s{seed}.pt"); sz = org.save(p); org2 = MotionOrganism.load(p)
                with torch.no_grad(): same = torch.equal(org2.shape(S["test"], EV["test"]).argmax(1), org.shape(S["test"], EV["test"]).argmax(1))
                log(f"      saved {p} {sz / 1024:.0f} KiB, {org.n_params()} trainable params; reload identical predictions: {same}; router mean w (shape,whole,msil) photo {org.router(org._feat(org.voters(S['test_photo']), EV['test_photo']))[MOV['test_photo']].mean(0).tolist()}")
        log(f"  seed {seed} done in {time.time() - t0:.0f}s")
    log("\n== summary (mean±std over seeds; moving packets) ==")
    for key in dict.fromkeys((r["arm"], r["stage"]) for r in rows):
        rs = [r for r in rows if (r["arm"], r["stage"]) == key]; m = lambda k: torch.tensor([r[k] for r in rs])
        log(f"  {key[0]:15s} {key[1]:8s} mixed {m('test').mean():.3f}±{m('test').std():.3f} photo {m('test_photo').mean():.3f}±{m('test_photo').std():.3f} flat {m('test_flat').mean():.3f}±{m('test_flat').std():.3f} | old {m('old').mean():.3f} forget {m('forget').mean():+.3f}±{m('forget').std():.3f} | vel {m('vel').mean():.3f}")
