"""s054: the s053 shape-world continual recipe carried to split-CIFAR-100 (handoff NEXT-1).
Rocky's metric: ONE shared 100-way head, full-softmax, NO task id at test.

STREAM: 10 tasks x 10 fine classes in index order (same split as the archived
  bench_2_0_cifar_continual), 8 epochs/task, Adam 1e-3, batch 256.  STREAM=joint =
  one task with all 100 classes (static reference for each reader).
FRONT END: fixed primitives from s052/s053 (frontend.py + grouping.py), cached at
  outputs/data/cifar/feat_<front>_<split>.pt: ds 800 (dense cepstra + synced stereo),
  col 100, bd 92 (boundary orientation), cn 13 (corners), grp (grouping.describe,
  scale canon: silhouette 92 / colour 3 / frame 7 / flags 4 / ctex 216 / cstereo 72 ...).
READERS (every leaf = Seeded(d,100,48,nonlinear) <= 50 K params):
  mono    = one leaf on grouped(sil+col+frame+flags 106) + whole(bd+col+cn 205) = 311  (s053 mono)
  mono-ds = one leaf on ds+col 900                                                (s052 CIFAR front end)
  nest    = shape(sil+frame+flags 103) + whole(205) + fill(ctex+cstereo+flags 292) + ds(800) leaves,
            prediction = sum of per-leaf log-softmax (the s053 nest + the CIFAR texture leaf)
  nest3   = the s053 nest exactly (shape + whole + fill, no ds leaf)
  mono+pre / nest+pre / pre-only = + the shape-world-trained PRE-FEEDER organ (shape_prefeeder.py,
            158-d frozen H-codes+logits of the s053 nest run on CIFAR): mono 311+158=469-d leaf;
            nest + 5th leaf on the 158-d stream; pre-only = one leaf on the organ alone
ARMS: none | replay (diag manifold sketch) | replay-full (full-cov, sample_full rank FULL_RANK=32) |
      full+credit-soft (full-cov + credit lock rate 0.078)  -- the s053 operating point |
      full+settle (full-cov + post-task HEAD-ONLY settle: SETTLE_STEPS=200 Adam 1e-3 steps on class-balanced
      full-cov samples of every archived class; soma frozen -- the archived CIFAR bench's load-bearing fix, per leaf).
Replay budget per step: all past classes, per-class batch = clamp(REPLAY_TOTAL // n_past, 2, 32);
  full-cov samplers are eigendecomposed ONCE per task boundary (cached) -- sample_full's per-call
  eigh is fine for 14 classes x 311-d but not for 90 x 800-d.
BAR: cnn-seq = the s053 242 K CNN fine-tuned task after task (no protection).
METRICS at stream end on the 10 K test set: full (100-way argmax), task (argmax restricted to
  the sample's own 10-class task), forget = mean over tasks of (full acc on task t right after
  training it - at end), acq = mean of the right-after accs, t1_end.
Env: SEEDS (0,1,2) READERS (none = CNN bar only) ARMS EPOCHS(8) STREAM(split|joint) REPLAY_TOTAL(256) FULL_RANK(32) CNN(1)
     LOCK_RATE FEATS_ONLY(0: just build the caches)
Run: OMP_NUM_THREADS=6 python3 experiments/progenitor/cifar_continual.py
"""
import os, sys, time, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.progenitor import frontend as FE, cifar_eye_nest as B
from trioron.bases.seeded import Seeded
from trioron.core import Envelope, construct
from trioron.core.arena import Arena
from trioron.core.state import CellState
from trioron.phenotype import default_dispatch_table
from trioron.learning.credit import CreditTracker, CreditConfig
from trioron.learning.manifold import ManifoldArchive, ManifoldConfig
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "6")))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]; EPOCHS = int(os.environ.get("EPOCHS", "8"))
REPLAY_TOTAL = int(os.environ.get("REPLAY_TOTAL", "256")); SETTLE_STEPS = int(os.environ.get("SETTLE_STEPS", "200")); FULL_RANK = int(os.environ.get("FULL_RANK", "32")) or None
READERS = os.environ.get("READERS", "mono,mono-ds,nest").split(","); ARMS = os.environ.get("ARMS", "none,replay,replay-full,full+credit-soft").split(",")
RUN_CNN = os.environ.get("CNN", "1") == "1"; STREAM = os.environ.get("STREAM", "split")
OUT = os.path.join(B.DATA, "cifar"); os.makedirs(OUT, exist_ok=True)
t0 = time.time()
def log(*a): print(*a, flush=True)
# ── data + cached front end ────────────────────────────────────────────────
RAW = {"train": B.load_cifar100(True), "test": B.load_cifar100(False)}
Y = {sp: RAW[sp][1] for sp in RAW}
FRONTS = {"ds": FE.dense_stereo, "col": FE.colour_block, "bd": FE.boundary_block, "cn": FE.corner_block}
def feats(front, split):
    p = os.path.join(OUT, f"feat_{front}_{split}.pt")
    if os.path.exists(p): return torch.load(p).float()
    t = time.time(); Z = FE.batched(FRONTS[front], RAW[split][0]); torch.save(Z.half(), p)
    log(f"  cached {front}/{split} {tuple(Z.shape)} in {time.time()-t:.0f}s"); return Z
def grouped(split, chunk=2500):
    p = os.path.join(OUT, f"feat_grp_canon_{split}.pt")
    if os.path.exists(p): return {k: v.float() for k, v in torch.load(p).items()}
    from experiments.progenitor import grouping as G
    X = RAW[split][0]; t = time.time(); parts = []
    for i in range(0, len(X), chunk):
        D, _ = G.describe(X[i:i + chunk], canon="scale"); parts.append({k: v.half() for k, v in D.items()})
        log(f"    grouped {split} {i + len(D['silhouette'])}/{len(X)} [{time.time()-t:.0f}s]")
    D = {k: torch.cat([q[k] for q in parts]) for k in parts[0]}; torch.save(D, p)
    log(f"  grouped {split} {tuple(D['silhouette'].shape)} in {time.time()-t:.0f}s"); return {k: v.float() for k, v in D.items()}
GD = {sp: grouped(sp) for sp in Y}; WH = {sp: torch.cat([feats(k, sp) for k in ("bd", "col", "cn")], 1) for sp in Y}; DS = {sp: torch.cat([feats("ds", sp), feats("col", sp)], 1) for sp in Y}
if os.environ.get("FEATS_ONLY", "0") == "1":
    log(f"grouping on CIFAR: objects/img {GD['train']['flags'][:, 0].mean():.2f}, field-flag {GD['train']['flags'][:, 1].mean():.2f}, touches {GD['train']['flags'][:, 2].mean():.2f}, fill-frac {GD['train']['flags'][:, 3].mean():.2f}"); sys.exit(0)
def _pre(sp):
    from experiments.progenitor.shape_prefeeder import cifar_pre
    return cifar_pre(sp)
STREAMS = {"mono": lambda sp: torch.cat([GD[sp]["silhouette"], GD[sp]["colour"], GD[sp]["frame"], GD[sp]["flags"], WH[sp]], 1),
           "mono+pre": lambda sp: torch.cat([GD[sp]["silhouette"], GD[sp]["colour"], GD[sp]["frame"], GD[sp]["flags"], WH[sp], _pre(sp)], 1),
           "pre": _pre,
           "ds": lambda sp: DS[sp],
           "shape": lambda sp: torch.cat([GD[sp]["silhouette"], GD[sp]["frame"], GD[sp]["flags"]], 1), "whole": lambda sp: WH[sp],
           "fill": lambda sp: torch.cat([GD[sp]["ctex"], GD[sp]["cstereo"], GD[sp]["flags"]], 1)}
class Std:
    def __init__(self, Z): self.mu, self.sd = Z.mean(0), Z.std(0) + 1e-6
    def __call__(self, Z): return ((Z - self.mu) / self.sd).float()
STD = {k: Std(fn("train")) for k, fn in STREAMS.items()}
NEEDED = {"mono": ["mono"], "mono-ds": ["ds"], "nest": ["shape", "whole", "fill", "ds"], "nest3": ["shape", "whole", "fill"],
          "mono+pre": ["mono+pre"], "pre-only": ["pre"], "nest+pre": ["shape", "whole", "fill", "ds", "pre"]}
X = {k: {sp: STD[k](fn(sp)) for sp in Y} for k, fn in STREAMS.items() if any(k in NEEDED.get(r, []) for r in READERS)}
GFLAGS = GD["train"]["flags"].mean(0).tolist(); del GD, WH, DS, STD   # memory: several processes share 7 GB
if not RUN_CNN: RAW = {sp: (None, RAW[sp][1]) for sp in RAW}
NCLS = 100
TASKS = [(f"T{i+1} {i*10}-{i*10+9}", list(range(i * 10, i * 10 + 10))) for i in range(10)] if STREAM == "split" else [("joint 0-99", list(range(100)))]
TASK_OF = torch.tensor([c // 10 for c in range(100)])
# ── one protected leaf ─────────────────────────────────────────────────────
class Leaf:
    def __init__(self, d, n_out, seed, arm, hidden=48):
        torch.manual_seed(seed)
        self.sub = construct(base=Seeded(d, n_out, interior_cells=hidden, nonlinear=True), envelope=Envelope(max_parameter_bytes=400_000),
                             dispatch_table=default_dispatch_table(), capacity=d + hidden + n_out + 8, sparsity_k=0)
        self.sub.compile(); self.sub.prepare_training(); self.a = self.sub.arena; self.arm = arm
        self.cred = arm in ("full+credit-soft",); self.rep = arm in ("replay", "replay-full", "full+credit-soft", "full+settle"); self.full = arm in ("replay-full", "full+credit-soft", "full+settle")
        self.settle = arm == "full+settle"   # post-task HEAD-ONLY re-calibration on balanced full-cov samples of every archived class
        out_ids = self.sub.scheduler._plan.output_ids.long(); self.head_cells = torch.zeros(self.a.capacity, dtype=torch.bool); self.head_cells[out_ids] = True
        self.head_edges = torch.isin(self.a.edge_dst.long(), out_ids)
        rate = float(os.environ.get("LOCK_RATE", "0.078"))
        self.credit = CreditTracker(self.a, CreditConfig(consecutive_tasks=1, theta_e=0.30, g_min=1e-3, lock_base_rate=rate, engagement_decay=0.3)) if self.cred else None
        self.archive = ManifoldArchive(Arena(Envelope(), capacity=NCLS + 8), ManifoldConfig(replay_steps_per_class=1), full_cov=self.full) if self.rep else None
        self.samplers = {}   # class -> (mu, A [d,k], resid_sd): full-cov sampler cached at the boundary
        self.locked = 0; self.n_params = sum(p.numel() for p in self.sub.trainable_tensors())
    def __call__(self, Z): return self.sub(Z)
    def logits(self, Z): return B.logits_of(self.sub, Z)
    def replay(self):   # -> (samples, class ids) or None; all past classes, budget REPLAY_TOTAL per step
        if not self.rep: return None
        past = [c for c, astro in self.archive._astrocytes.items() if self.archive.arena.state[astro.cell_id] == CellState.DORMANT]
        if not past: return None
        bs = int(min(32, max(2, REPLAY_TOTAL // len(past)))); xs, cs = [], []
        for c in past:
            if self.full:
                mu, A, rs = self.samplers[c]; s = mu + torch.randn(bs, A.shape[1]) @ A.t()
                if rs > 0: s = s + rs * torch.randn(bs, mu.numel())
            else: s = self.archive._astrocytes[c].sample(bs)
            xs.append(s); cs.append(torch.full((bs,), c, dtype=torch.long))
        return torch.cat(xs), torch.cat(cs)
    def after_backward(self):
        if self.cred: self.credit.update_utility()
        self.sub.zero_dormant_grads()
    def after_step(self):
        if self.cred:
            act = self.sub.last_activations
            if act is not None: self.credit.update_engagement(act)
    def boundary(self, Ztask, ytask):
        if self.cred: self.locked += len(self.credit.consolidate())
        if self.rep:
            for c in ytask.unique().tolist(): self.archive.update_class(c, Ztask[ytask == c])
            self.archive.finalize_all()
            if self.full:
                for c in ytask.unique().tolist():
                    astro = self.archive._astrocytes[c]; cov = astro._regularized_cov(0.1); evals, evecs = torch.linalg.eigh(cov); evals = evals.clamp(min=1e-8)
                    k = min(FULL_RANK or cov.shape[0], cov.shape[0]); A = evecs[:, -k:] * evals[-k:].sqrt(); resid = float(evals[:-k].mean()) ** 0.5 if cov.shape[0] > k else 0.0
                    self.samplers[c] = (astro.mu, A, resid)
            if self.settle: self.settle_head()
    def settle_head(self, n_steps=SETTLE_STEPS, lr=1e-3):
        past = list(self.samplers); bs = int(min(32, max(2, REPLAY_TOTAL // len(past)))); opt = torch.optim.Adam([self.a.bias, self.a.edge_weight], lr=lr)
        for _ in range(n_steps):
            xs, cs = [], []
            for c in past:
                mu, A, rs = self.samplers[c]; s = mu + torch.randn(bs, A.shape[1]) @ A.t()
                if rs > 0: s = s + rs * torch.randn(bs, mu.numel())
                xs.append(s); cs.append(torch.full((bs,), c, dtype=torch.long))
            opt.zero_grad(); F.cross_entropy(self(torch.cat(xs)), torch.cat(cs)).backward()
            self.a.bias.grad[~self.head_cells] = 0.0; self.a.edge_weight.grad[~self.head_edges] = 0.0; self.sub.zero_dormant_grads(); opt.step()
# ── readers ────────────────────────────────────────────────────────────────
class Reader:
    def __init__(self, streams, seed, arm):
        self.streams = {s: s for s in streams}; self.leaves = {s: Leaf(X[s]["train"].shape[1], NCLS, seed, arm) for s in streams}
    def loss_of(self, Zd, y): return sum(F.cross_entropy(L(Zd[nm]), y) for nm, L in self.leaves.items())
    def scores(self, sp):
        with torch.no_grad():
            if len(self.leaves) == 1: return next(iter(self.leaves.values())).logits(X[next(iter(self.leaves))][sp])
            return sum(F.log_softmax(L.logits(X[nm][sp]), 1) for nm, L in self.leaves.items())
READER_STREAMS = NEEDED
def evaluate(S, y):
    full = S.argmax(1); mask = TASK_OF[None, :] == TASK_OF[y][:, None]; task = S.masked_fill(~mask, -1e9).argmax(1)
    return full == y, task == y
def run(reader_name, arm, seed):
    R = Reader(READER_STREAMS[reader_name], seed, arm); ytr, yte = Y["train"], Y["test"]; acc_after = []
    def task_acc(cls):
        ok, _ = evaluate(R.scores("test"), yte); m = torch.isin(yte, torch.tensor(cls)); return float(ok[m].float().mean())
    for t, (name, cls) in enumerate(TASKS):
        idx = torch.nonzero(torch.isin(ytr, torch.tensor(cls))).squeeze(1)
        params = [p for L in R.leaves.values() for p in L.sub.trainable_tensors()]; opt = torch.optim.Adam(params, lr=1e-3); g = torch.Generator().manual_seed(seed * 100 + t)
        for ep in range(EPOCHS):
            for bi in idx[torch.randperm(len(idx), generator=g)].split(256):
                opt.zero_grad(); Zd = {nm: X[st]["train"][bi] for nm, st in R.streams.items()}
                loss = R.loss_of(Zd, ytr[bi])
                for nm, L in R.leaves.items():
                    rp = L.replay()
                    if rp is not None: rs, rc = rp; loss = loss + F.cross_entropy(L(rs), rc)
                loss.backward()
                for L in R.leaves.values(): L.after_backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
                for L in R.leaves.values(): L.after_step()
        acc_after.append(task_acc(cls))
        for nm, L in R.leaves.items(): L.boundary(X[R.streams[nm]]["train"][idx], ytr[idx])
        if STREAM == "split" and t in (0, 4, 9): log(f"      [{reader_name} {arm} s{seed}] after {name}: acc {acc_after[-1]:.3f} T1 now {task_acc(TASKS[0][1]):.3f} [{time.time()-t0:.0f}s]")
    okf, okt = evaluate(R.scores("test"), yte)
    return dict(full=float(okf.float().mean()), task=float(okt.float().mean()), forget=sum(acc_after[t] - task_acc(cls) for t, (_, cls) in enumerate(TASKS)) / len(TASKS),
                acq=sum(acc_after) / len(acc_after), t1_end=task_acc(TASKS[0][1]), locked=sum(L.locked for L in R.leaves.values()), params=sum(L.n_params for L in R.leaves.values()))
def cnn_seq(seed):
    import torch.nn as nn
    Xtr, ytr = RAW["train"]; Xte, yte = RAW["test"]
    torch.manual_seed(seed)
    def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU())
    net = nn.Sequential(blk(3, 32), nn.MaxPool2d(2), blk(32, 64), nn.MaxPool2d(2), blk(64, 128), nn.MaxPool2d(2), blk(128, 128), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, NCLS))
    def scores():
        net.eval()
        with torch.no_grad(): return torch.cat([net(Xte[i:i + 1000]) for i in range(0, len(Xte), 1000)])
    def task_acc(cls):
        ok, _ = evaluate(scores(), yte); m = torch.isin(yte, torch.tensor(cls)); return float(ok[m].float().mean())
    acc_after = []
    for t, (name, cls) in enumerate(TASKS):
        idx = torch.nonzero(torch.isin(ytr, torch.tensor(cls))).squeeze(1); opt = torch.optim.Adam(net.parameters(), 1e-3); g = torch.Generator().manual_seed(seed * 100 + t)
        for ep in range(EPOCHS):
            net.train()
            for bi in idx[torch.randperm(len(idx), generator=g)].split(128):
                opt.zero_grad(); F.cross_entropy(net(Xtr[bi]), ytr[bi]).backward(); opt.step()
        acc_after.append(task_acc(cls))
    okf, okt = evaluate(scores(), yte)
    return dict(full=float(okf.float().mean()), task=float(okt.float().mean()), forget=sum(acc_after[t] - task_acc(cls) for t, (_, cls) in enumerate(TASKS)) / len(TASKS),
                acq=sum(acc_after) / len(acc_after), t1_end=task_acc(TASKS[0][1]), locked=0, params=sum(p.numel() for p in net.parameters()))
def fmt(rs):
    keys = ["full", "task", "forget", "acq", "t1_end", "locked", "params"]; T = {k: torch.tensor([float(r[k]) for r in rs]) for k in keys}
    return " | ".join(f"{k} {T[k].mean():.3f}±{T[k].std():.3f}" if len(rs) > 1 and k not in ("params",) else f"{k} {T[k].mean():.0f}" if k == "params" else f"{k} {T[k].mean():.3f}" for k in keys)
log(f"CIFAR CONTINUAL s054: stream {STREAM} tasks {len(TASKS)}; seeds {SEEDS}; epochs {EPOCHS}; replay_total {REPLAY_TOTAL}; full_rank {FULL_RANK}; readers {READERS}; arms {ARMS}")
log(f"grouping on CIFAR (train): objects/img {GFLAGS[0]:.2f}, field-flag {GFLAGS[1]:.2f}, touches {GFLAGS[2]:.2f}, fill-frac {GFLAGS[3]:.2f}")
log("metrics on the 10K test set at stream end: full = 100-way argmax (chance .01); task = argmax within the sample's own 10-class task (chance .10); forget = mean(full acc on task t right after it - at end); t1_end = T1 full acc at end")
for rd in READERS:
    if rd not in NEEDED: continue   # READERS=none -> bar only
    for arm in ARMS:
        rs = [run(rd, arm, s) for s in SEEDS]; log(f"  {rd:>7s} {arm:>16s} | " + fmt(rs) + f"   [{time.time()-t0:.0f}s]")
if RUN_CNN:
    rs = [cnn_seq(s) for s in SEEDS]; log(f"  cnn-seq (242K, no protection) | " + fmt(rs) + f"   [{time.time()-t0:.0f}s]")
