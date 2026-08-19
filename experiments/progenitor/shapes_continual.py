"""s053 continual / compositional stream on the shape world (Rocky's metric: shared heads,
full-softmax, NO task id at test).

Classes = (shape, fill) pairs: 3 geometric shapes x 4 fills = 12 (HELD 3 never trained) + 2
fields = 14 trained-able, 11 trained.  Two shared heads: shape (5-way) + fill (4-way).
STREAM (5 tasks): T1 solid {circle,triangle,square} | T2 fields {stripes,dots} |
  T3 striped {circle,triangle} | T4 dotted {circle,square} | T5 outline {triangle,square}
  -> learn shapes, then learn fills on the same shapes; the held-out combos test whether the
  organism reads (shape,fill) it never saw by factors.
READERS: mono = ONE leaf on the 311-d stream (shape+fill heads) |
         nest = shape leaf (canon silhouette+frame+flags 103) + whole leaf (205) summed for shape,
                fill leaf (ctex+flags 220) — the (b) split, each leaf protected separately.
ARMS (protection): none | lambda (soft λ anchor, |w·g| saliency) | credit (hard lock, LOCK_RATE 1.0) |
         replay (manifold pseudo-rehearsal over the FIXED descriptors, per pair class) | all |
         replay+lambda (no hard lock) | all-soft (all with the DEFAULT lock rate 0.078).
BARS: cnn-seq = the 242K CNN fine-tuned task after task (forgetting bar).
METRICS at stream end on test_fresh: shape / fill / pair acc; per-task pair-acc right after
training vs at end -> forgetting; test_held: shape / fill / pair (compositional, never trained).
Env: SEEDS (0,1,2) READERS ARMS EPOCHS(8) STRENGTH(1e3) REPLAY_BS(32) CNN(1)
Run: OMP_NUM_THREADS=6 python3 experiments/progenitor/shapes_continual.py
"""
import os, sys, time, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.progenitor import shapes as SH, shapes_feats as SF, cifar_eye_nest as B
from trioron.bases.seeded import Seeded
from trioron.core import Envelope, construct
from trioron.core.arena import Arena
from trioron.phenotype import default_dispatch_table
from trioron.learning.credit import CreditTracker, CreditConfig
from trioron.learning import epigenetic_lock as epi
from trioron.learning.manifold import ManifoldArchive, ManifoldConfig
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "6")))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]; EPOCHS = int(os.environ.get("EPOCHS", "8"))
STRENGTH = float(os.environ.get("STRENGTH", "1e3")); REPLAY_BS = int(os.environ.get("REPLAY_BS", "32")); REPLAY_W = float(os.environ.get("REPLAY_W", "1.0"))
READERS = os.environ.get("READERS", "mono,nest").split(","); ARMS = os.environ.get("ARMS", "none,lambda,credit,replay,all").split(","); RUN_CNN = os.environ.get("CNN", "1") == "1"
t0 = time.time()
def log(*a): print(*a, flush=True)
# ── data: pair classes ─────────────────────────────────────────────────────
def pair_id(ys): return torch.where(ys["y_shape"] < 3, ys["y_shape"] * 4 + ys["y_fill"], 12 + (ys["y_shape"] - 3))
TASKS = [("T1 solid", [0, 4, 8]), ("T2 fields", [12, 13]), ("T3 striped", [1, 5]), ("T4 dotted", [2, 10]), ("T5 outline", [7, 11])]   # HELD: 6 (tri-dot), 9 (sq-stri), 3 (circ-out)
def sf_of(pid): return torch.where(pid >= 12, pid - 12 + 3, pid // 4), torch.where(pid >= 12, torch.zeros_like(pid), pid % 4)
Y = {sp: SH.load(sp)[1] for sp in ("train", "test_fresh", "test_held")}; P = {sp: pair_id(Y[sp]) for sp in Y}
GD = {sp: SF.grouped(sp, canon="scale") for sp in Y}; WH = {sp: torch.cat([SF.feats(k, sp) for k in ("bd", "col", "cn")], 1) for sp in Y}
STREAMS = {"mono": lambda sp: torch.cat([GD[sp]["silhouette"], GD[sp]["colour"], GD[sp]["frame"], GD[sp]["flags"], WH[sp]], 1),
           "shape": lambda sp: torch.cat([GD[sp]["silhouette"], GD[sp]["frame"], GD[sp]["flags"]], 1), "whole": lambda sp: WH[sp],
           "fill": lambda sp: torch.cat([GD[sp]["ctex"], GD[sp]["flags"]], 1)}
class Std:
    def __init__(self, Z): self.mu, self.sd = Z.mean(0), Z.std(0) + 1e-6
    def __call__(self, Z): return ((Z - self.mu) / self.sd).float()
STD = {k: Std(fn("train")) for k, fn in STREAMS.items()}   # sensor calibration on the (fixed, germline) front end
X = {k: {sp: STD[k](fn(sp)) for sp in Y} for k, fn in STREAMS.items()}
# ── one protected leaf ─────────────────────────────────────────────────────
class Leaf:
    def __init__(self, d, n_out, seed, arm, hidden=48):
        torch.manual_seed(seed)
        self.sub = construct(base=Seeded(d, n_out, interior_cells=hidden, nonlinear=True), envelope=Envelope(max_parameter_bytes=400_000),
                             dispatch_table=default_dispatch_table(), capacity=d + hidden + n_out + 8, sparsity_k=0)
        self.sub.compile(); self.sub.prepare_training(); self.a = self.sub.arena; self.arm = arm
        self.lam = arm in ("lambda", "all", "replay+lambda", "all-soft"); self.cred = arm in ("credit", "all", "all-soft"); self.rep = arm in ("replay", "all", "replay+lambda", "all-soft")
        rate = float(os.environ.get("LOCK_RATE", "0.078" if arm == "all-soft" else "1.0"))
        self.credit = CreditTracker(self.a, CreditConfig(consecutive_tasks=1, theta_e=0.30, g_min=1e-3, lock_base_rate=rate, engagement_decay=0.3)) if self.cred else None
        self.archive = ManifoldArchive(Arena(Envelope(), capacity=32), ManifoldConfig(replay_steps_per_class=1), full_cov=False) if self.rep else None
        self.locked = 0
    def __call__(self, Z): return self.sub(Z)
    def logits(self, Z): return B.logits_of(self.sub, Z)
    def penalty(self): return STRENGTH * epi.ewc_penalty(self.a) if self.lam else 0.0
    def replay(self):   # -> (samples, pair ids) or None
        if not self.rep: return None
        out = self.archive.replay_batches(REPLAY_BS)
        if not out: return None
        return torch.cat([s for s, _ in out]), torch.cat([torch.full((len(s),), c, dtype=torch.long) for s, c in out])
    def after_backward(self):
        if self.cred: self.credit.update_utility()
        if self.lam: epi.accumulate_saliency(self.a)
        self.sub.zero_dormant_grads()
    def after_step(self):
        if self.cred:
            act = self.sub.last_activations
            if act is not None: self.credit.update_engagement(act)
    def boundary(self, Ztask, ptask):
        if self.lam: epi.refresh_lambda(self.a); epi.anchor(self.a)
        if self.cred: self.locked += len(self.credit.consolidate())
        if self.rep:
            for c in ptask.unique().tolist(): self.archive.update_class(c, Ztask[ptask == c])
            self.archive.finalize_all()
# ── readers ────────────────────────────────────────────────────────────────
class Mono:
    def __init__(self, seed, arm): self.L = Leaf(X["mono"]["train"].shape[1], 9, seed, arm); self.leaves = {"mono": self.L}; self.streams = {"mono": "mono"}
    def heads(self, Zd):   # Zd: dict stream->batch
        o = self.L(Zd["mono"]); return o[:, :5], o[:, 5:]
    def loss_of(self, Zd, ysh, yfl): sh, fl = self.heads(Zd); return F.cross_entropy(sh, ysh) + F.cross_entropy(fl, yfl)
    def predict(self, sp):
        with torch.no_grad(): o = self.L.logits(X["mono"][sp]); return o[:, :5].argmax(1), o[:, 5:].argmax(1)
class Nest:
    def __init__(self, seed, arm):
        self.leaves = {"shape": Leaf(X["shape"]["train"].shape[1], 5, seed, arm), "whole": Leaf(X["whole"]["train"].shape[1], 5, seed, arm), "fill": Leaf(X["fill"]["train"].shape[1], 4, seed, arm)}
        self.streams = {"shape": "shape", "whole": "whole", "fill": "fill"}
    def loss_of(self, Zd, ysh, yfl):
        return F.cross_entropy(self.leaves["shape"](Zd["shape"]), ysh) + F.cross_entropy(self.leaves["whole"](Zd["whole"]), ysh) + F.cross_entropy(self.leaves["fill"](Zd["fill"]), yfl)
    def predict(self, sp):
        with torch.no_grad():
            sh = F.log_softmax(self.leaves["shape"].logits(X["shape"][sp]), 1) + F.log_softmax(self.leaves["whole"].logits(X["whole"][sp]), 1)
            return sh.argmax(1), self.leaves["fill"].logits(X["fill"][sp]).argmax(1)
def run(reader_name, arm, seed):
    R = Mono(seed, arm) if reader_name == "mono" else Nest(seed, arm)
    ptr = P["train"]; acc_after = []
    def pair_acc(sp, classes):
        m = torch.isin(P[sp], torch.tensor(classes)); psh, pfl = R.predict(sp); ysh, yfl = sf_of(P[sp])
        return float(((psh == ysh) & (pfl == yfl))[m].float().mean())
    for t, (name, cls) in enumerate(TASKS):
        m = torch.isin(ptr, torch.tensor(cls)); idx = torch.nonzero(m).squeeze(1); ysh, yfl = sf_of(ptr)
        params = [p for L in R.leaves.values() for p in L.sub.trainable_tensors()]; opt = torch.optim.Adam(params, lr=1e-3); g = torch.Generator().manual_seed(seed * 100 + t)
        for ep in range(EPOCHS):
            for bi in idx[torch.randperm(len(idx), generator=g)].split(256):
                opt.zero_grad(); Zd = {nm: X[st]["train"][bi] for nm, st in R.streams.items()}
                loss = R.loss_of(Zd, ysh[bi], yfl[bi]) + sum(L.penalty() for L in R.leaves.values())
                for nm, L in R.leaves.items():   # manifold replay of past pair-classes, per leaf, in its own descriptor space
                    rp = L.replay()
                    if rp is not None:
                        rs, rp_ = rp; rsh, rfl = sf_of(rp_); out = L(rs)
                        if nm == "mono": loss = loss + REPLAY_W * (F.cross_entropy(out[:, :5], rsh) + F.cross_entropy(out[:, 5:], rfl))
                        elif nm == "fill": loss = loss + REPLAY_W * F.cross_entropy(out, rfl)
                        else: loss = loss + REPLAY_W * F.cross_entropy(out, rsh)
                loss.backward()
                for L in R.leaves.values(): L.after_backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
                for L in R.leaves.values(): L.after_step()
        acc_after.append(pair_acc("test_fresh", cls))
        for nm, L in R.leaves.items(): L.boundary(X[R.streams[nm]]["train"][idx], ptr[idx])
    trained = sorted({c for _, cls in TASKS for c in cls}); psh, pfl = R.predict("test_fresh"); ysh, yfl = sf_of(P["test_fresh"]); m = torch.isin(P["test_fresh"], torch.tensor(trained))
    res = dict(shape=float((psh == ysh)[m].float().mean()), fill=float((pfl == yfl)[m].float().mean()), pair=float(((psh == ysh) & (pfl == yfl))[m].float().mean()),
               forget=sum(acc_after[t] - pair_acc("test_fresh", cls) for t, (_, cls) in enumerate(TASKS)) / len(TASKS), acq=sum(acc_after) / len(acc_after),
               t1_end=pair_acc("test_fresh", TASKS[0][1]), locked=sum(L.locked for L in R.leaves.values()))
    hsh, hfl = R.predict("test_held"); ysh_h, yfl_h = sf_of(P["test_held"])
    res.update(held_shape=float((hsh == ysh_h).float().mean()), held_fill=float((hfl == yfl_h).float().mean()), held_pair=float(((hsh == ysh_h) & (hfl == yfl_h)).float().mean()))
    return res
def cnn_seq(seed):
    import torch.nn as nn
    Xtr, ytr, _ = SH.load("train"); Xfr, yfr, _ = SH.load("test_fresh"); Xho, yho, _ = SH.load("test_held"); ptr = pair_id(ytr)
    torch.manual_seed(seed)
    def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU())
    net = nn.Sequential(blk(3, 32), nn.MaxPool2d(2), blk(32, 64), nn.MaxPool2d(2), blk(64, 128), nn.MaxPool2d(2), blk(128, 128), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 9))
    def pred(Xs):
        net.eval()
        with torch.no_grad(): o = torch.cat([net(Xs[i:i + 1000]) for i in range(0, len(Xs), 1000)])
        return o[:, :5].argmax(1), o[:, 5:].argmax(1)
    def pair_acc(Xs, ys, classes):
        pid = pair_id(ys); m = torch.isin(pid, torch.tensor(classes)); psh, pfl = pred(Xs); return float(((psh == ys["y_shape"]) & (pfl == ys["y_fill"]))[m].float().mean())
    acc_after = []
    for t, (name, cls) in enumerate(TASKS):
        idx = torch.nonzero(torch.isin(ptr, torch.tensor(cls))).squeeze(1); opt = torch.optim.Adam(net.parameters(), 1e-3); g = torch.Generator().manual_seed(seed * 100 + t)
        for ep in range(EPOCHS):
            net.train()
            for bi in idx[torch.randperm(len(idx), generator=g)].split(128):
                opt.zero_grad(); o = net(Xtr[bi]); (F.cross_entropy(o[:, :5], ytr["y_shape"][bi]) + F.cross_entropy(o[:, 5:], ytr["y_fill"][bi])).backward(); opt.step()
        acc_after.append(pair_acc(Xfr, yfr, cls))
    trained = sorted({c for _, cls in TASKS for c in cls}); psh, pfl = pred(Xfr); m = torch.isin(pair_id(yfr), torch.tensor(trained))
    hsh, hfl = pred(Xho)
    return dict(shape=float((psh == yfr["y_shape"])[m].float().mean()), fill=float((pfl == yfr["y_fill"])[m].float().mean()), pair=float(((psh == yfr["y_shape"]) & (pfl == yfr["y_fill"]))[m].float().mean()),
                forget=sum(acc_after[t] - pair_acc(Xfr, yfr, cls) for t, (_, cls) in enumerate(TASKS)) / len(TASKS), acq=sum(acc_after) / len(acc_after), t1_end=pair_acc(Xfr, yfr, TASKS[0][1]), locked=0,
                held_shape=float((hsh == yho["y_shape"]).float().mean()), held_fill=float((hfl == yho["y_fill"]).float().mean()), held_pair=float(((hsh == yho["y_shape"]) & (hfl == yho["y_fill"])).float().mean()))
def fmt(rs):
    keys = ["shape", "fill", "pair", "forget", "acq", "t1_end", "held_shape", "held_fill", "held_pair", "locked"]; T = {k: torch.tensor([float(r[k]) for r in rs]) for k in keys}
    return " | ".join(f"{k} {T[k].mean():.3f}±{T[k].std():.3f}" if len(rs) > 1 else f"{k} {T[k].mean():.3f}" for k in keys)
log(f"SHAPE CONTINUAL s053: tasks {[n for n,_ in TASKS]}; seeds {SEEDS}; epochs {EPOCHS}; strength {STRENGTH}; replay_bs {REPLAY_BS}")
log("metrics on test_fresh over the 11 trained pair-classes (full-softmax, no task id): shape/fill/pair acc at end; forget = mean(pair acc right after task - at end); t1_end = T1 pair acc at end; held_* = never-trained combos")
for rd in READERS:
    for arm in ARMS:
        rs = [run(rd, arm, s) for s in SEEDS]; log(f"  {rd:>4s} {arm:>7s} | " + fmt(rs) + f"   [{time.time()-t0:.0f}s]")
if RUN_CNN:
    rs = [cnn_seq(s) for s in SEEDS]; log(f"  cnn-seq (242K, no protection) | " + fmt(rs) + f"   [{time.time()-t0:.0f}s]")
