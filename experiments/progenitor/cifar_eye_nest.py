"""Retina-pooled Phasecyte -> trioron nest on CIFAR-100, continual 20-task.
Design: docs/design/retina_phasecyte.md (approved s051). Bar: pure-trioron
blindman v2b full 0.309 / task 0.606 (5-way superclass-restricted).

Organism (per seed):
  for each fixation f (FIX = 1|5|9, `pcll.eye.fixations`):
     Eye(f) -> sense_P (fovea+parafovea, Y/RG/BY, DoG ON/OFF)  -> Phasecyte leaf P_f
             -> sense_M (pooled fovea/parafovea + periphery, Y)  -> Phasecyte leaf M_f
     T_f = trioron leaf (|P_f|+|M_f| -> HIDDEN quad -> 100), DREAM-distilled at
           the end of the stream from the P_f/M_f per-class sketches (no stored
           images; every class seen so far => stationary joint per leaf).
  level 2 (FIX>1): ABSORB = sum_f logits_f (== head-merged graft, s051);
                   NEST   = trioron router over centre-M pockets picks one f
                            (router trained on centre-M PSEUDO pockets, label =
                            per-class best fixation — no stored data);
                   SOFT   = router-softmax weighted sum.
  A0 null (FIX=0): flat luminance 32x32 (input_shape=(32,32): the existing
                   redundancy retina), one leaf, one T.

Stream: 20 CIFAR-100 superclasses in index order, 5 fine classes each; every
leaf observes every task (leaves are per FIXATION, not per domain); windows
of WINDOW images; microsaccade jitter ±JITTER px during wake, off at eval.

Env: EYE_FIX (5), EYE_SEED (0), EYE_FRAC (1.0 per-task subsample),
     EYE_WINDOW (1000), EYE_JITTER (0; templates need a still eye), EYE_HIDDEN (48), EYE_PSEUDO (300),
     EYE_EPOCHS (8), EYE_CAP (128), EYE_TASKS (20; smoke subsetting).
Run: python3 -m experiments.progenitor.cifar_eye_nest
"""
from __future__ import annotations

import os
import pickle
import time
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F
from torchvision import datasets as tvd

from trioron.bases.seeded import Seeded
from trioron.core import Envelope, construct
from trioron.core.receptor import N_QUANTA
from trioron.pcll import PhasecyteLeaf
from trioron.pcll.eye import Eye, fixations
from trioron.phenotype import default_dispatch_table

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "outputs", "data")
FIX = int(os.environ.get("EYE_FIX", "5"))
SEED = int(os.environ.get("EYE_SEED", "0"))
FRAC = float(os.environ.get("EYE_FRAC", "1.0"))
WINDOW = int(os.environ.get("EYE_WINDOW", "1000"))
JITTER = int(os.environ.get("EYE_JITTER", "0"))   # templates need a still eye (s051)
HIDDEN = int(os.environ.get("EYE_HIDDEN", "48"))
PSEUDO = int(os.environ.get("EYE_PSEUDO", "300"))
EPOCHS = int(os.environ.get("EYE_EPOCHS", "8"))
CAP = int(os.environ.get("EYE_CAP", "128"))
N_TASKS = int(os.environ.get("EYE_TASKS", "20"))
RECTIFY = os.environ.get("EYE_RECTIFY", "0") == "1"   # signed DoG default (q=0 mask)
MIN_FRAC = float(os.environ.get("EYE_MIN_FRAC", "0.0"))  # sketch purity floor for pseudo labels
PURITY_POW = float(os.environ.get("EYE_PURITY_POW", "3.0"))  # weight comp^pow
MIN_SKETCH = 8
BAR = (0.309, 0.606)


# ── data ─────────────────────────────────────────────────────────────

def load_cifar100(train: bool):
    ds = tvd.CIFAR100(root=DATA, train=train, download=False)
    X = torch.from_numpy(ds.data).float().div_(255.0).permute(0, 3, 1, 2).contiguous()
    return X, torch.tensor(ds.targets, dtype=torch.long)


def fine_to_coarse() -> torch.Tensor:
    with open(os.path.join(DATA, "cifar-100-python", "train"), "rb") as f:
        d = pickle.load(f, encoding="latin1")
    fine, coarse = torch.tensor(d["fine_labels"]), torch.tensor(d["coarse_labels"])
    m = torch.full((100,), -1, dtype=torch.long)
    m[fine] = coarse
    return m


def luminance_flat(x: torch.Tensor) -> torch.Tensor:
    return (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).reshape(len(x), -1)


# ── organism ─────────────────────────────────────────────────────────

class FixationOrgan:
    """One fixation: an Eye + its P and M Phasecyte leaves (+ later T)."""

    def __init__(self, centre, gen_pool, seed, idx, jitter_gen):
        self.eye = Eye(centre, jitter=JITTER, generator=jitter_gen, rectify=RECTIFY)
        self.idx = idx
        self.leaves: Dict[str, PhasecyteLeaf] = {}
        for k, stream in enumerate(("P", "M")):
            sense = self.eye.sense_P if stream == "P" else self.eye.sense_M
            self.leaves[stream] = PhasecyteLeaf(
                sense, gen_pool, seed, 10 * idx + k, window=WINDOW,
                class_cap=CAP, manifold=True, composer=True,
                input_shape=None, capacity=8192, verbose=(idx == 0))
        self.T = None
        self.width = 0
        self.mu = None       # sketch-derived per-receptor standardizer
        self.sd = None       # (fitted on the PSEUDO set — no real data)

    def standardize(self, Q: torch.Tensor) -> torch.Tensor:
        return Q if self.mu is None else (Q - self.mu) / self.sd

    def senses(self):
        return {"P": self.eye.sense_P, "M": self.eye.sense_M}

    def observe(self, X, labels):
        for stream, leaf in self.leaves.items():
            leaf.observe(self.senses()[stream](X), labels)

    @torch.no_grad()
    def pockets(self, X, *, chunk=2000) -> torch.Tensor:
        """[N, |P|+|M|] canonical pockets / N_QUANTA (jitter off)."""
        j, self.eye.jitter = self.eye.jitter, 0
        out = []
        for i in range(0, len(X), chunk):
            xb = X[i:i + chunk]
            out.append(torch.cat([self.leaves[s].mixed.pockets_of(self.senses()[s](xb))
                                  for s in ("P", "M")], dim=1) / N_QUANTA)
        self.eye.jitter = j
        return torch.cat(out)


class FlatOrgan(FixationOrgan):
    """A0 null: flat luminance sheet, one leaf with the redundancy retina."""

    def __init__(self, gen_pool, seed):
        self.idx = 0
        self.T = None; self.width = 0; self.mu = None; self.sd = None
        self.leaves = {"Y": PhasecyteLeaf(
            luminance_flat, gen_pool, seed, 0, window=WINDOW, class_cap=CAP,
            manifold=True, composer=True, input_shape=(32, 32),
            capacity=8192, verbose=True)}
        self.T = None

    def senses(self):
        return {"Y": luminance_flat}

    @torch.no_grad()
    def pockets(self, X, *, chunk=2000):
        return torch.cat([self.leaves["Y"].mixed.pockets_of(luminance_flat(X[i:i + chunk]))
                          for i in range(0, len(X), chunk)]) / N_QUANTA


def sketch_table(leaf: PhasecyteLeaf, min_frac: float = 0.0):
    """[(sketch, {global label: fraction})] for every internal class with a
    usable manifold sketch. Labels come from the leaf's own label taps
    (composition, not majority): a real class that is a MINORITY member
    of several internal classes still contributes pseudo samples under
    its own label (majority naming lost 6/25 classes, s051 diag)."""
    out = []
    taps = leaf.mixed.label_taps
    for cls in leaf.mixed.classes:
        astro = leaf.mixed.manifold.sketches.get(cls.name)
        if astro is None or astro._n < MIN_SKETCH:
            continue
        comp = taps.composition_of(cls.name) if taps else {}
        comp = {int(k[1:]): v for k, v in comp.items() if v >= min_frac}
        if comp:
            out.append((astro, comp))
    return out


def per_class_pseudo(table, classes: Sequence[int], n_per: int,
                     gen: torch.Generator) -> Dict[int, torch.Tensor]:
    """For each global class g: n_per pseudo pockets drawn from the
    internal-class sketches that contain g, in proportion to g's mass in
    each (composition-weighted mixture)."""
    out: Dict[int, torch.Tensor] = {}
    for g in classes:
        w = torch.tensor([comp.get(g, 0.0) ** PURITY_POW * astro._n
                          for astro, comp in table])
        if w.sum() <= 0:
            continue
        counts = torch.multinomial(w / w.sum(), n_per, replacement=True,
                                   generator=gen).bincount(minlength=len(table))
        parts = [table[i][0].sample(int(c)).clamp(0, N_QUANTA) / N_QUANTA
                 for i, c in enumerate(counts.tolist()) if c > 0]
        out[g] = torch.cat(parts)[torch.randperm(n_per, generator=gen)]
    return out


def dream_pseudo(organ: FixationOrgan, classes: Sequence[int], n_per: int,
                 seed: int):
    """Joint pseudo set over ALL streams: per class, independent samples per
    stream (P/M correlation is not in the sketches; noted in the design),
    paired by position. Returns X [N, W], y [N] (global labels), skipped."""
    gen = torch.Generator().manual_seed(seed)
    tabs = {s: per_class_pseudo(sketch_table(l, MIN_FRAC), classes, n_per, gen)
            for s, l in organ.leaves.items()}
    Xs, ys, skipped = [], [], 0
    for g in classes:
        if any(g not in t for t in tabs.values()):
            skipped += 1
            continue
        Xs.append(torch.cat([tabs[s][g] for s in organ.leaves], dim=1))
        ys.append(torch.full((n_per,), int(g), dtype=torch.long))
    return torch.cat(Xs), torch.cat(ys), skipped


def train_sub(X, y, n_out, *, hidden, seed, epochs=EPOCHS, lr=1e-3, batch=256,
              tag=""):
    torch.manual_seed(seed)
    sub = construct(base=Seeded(X.shape[1], n_out, interior_cells=hidden,
                                nonlinear=True),
                    envelope=Envelope(max_parameter_bytes=400_000),
                    dispatch_table=default_dispatch_table(),
                    capacity=X.shape[1] + hidden + n_out + 8, sparsity_k=0)
    sub.compile(); sub.prepare_training()
    opt = torch.optim.Adam(sub.trainable_tensors(), lr=lr)
    g = torch.Generator().manual_seed(seed + 1)
    for ep in range(epochs):
        perm = torch.randperm(len(X), generator=g); tot = 0.0
        for i in range(0, len(X), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = F.cross_entropy(sub(X[idx]), y[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sub.trainable_tensors(), 1.0)
            sub.zero_dormant_grads(); opt.step()
            tot += float(loss.detach()) * len(idx)
        if ep in (0, epochs - 1):
            print(f"    [dream {tag}] ep{ep} loss={tot / len(X):.3f} "
                  f"({len(X)} pseudo)", flush=True)
    return sub


def n_params(sub) -> int:
    a = sub.arena
    return int(a.alive.sum()) + int(a.edge_cursor)


@torch.no_grad()
def logits_of(sub, X, chunk=4000):
    return torch.cat([sub(X[i:i + chunk]) for i in range(0, len(X), chunk)])


def acc_pair(logits, y, f2c):
    full = float((logits.argmax(1) == y).float().mean())
    mask = f2c.unsqueeze(0) == f2c[y].unsqueeze(1)          # [N, 100]
    task = float((logits.masked_fill(~mask, -1e9).argmax(1) == y).float().mean())
    return full, task


# ── main ─────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print(f"CIFAR EYE NEST (s051) fix={FIX} seed={SEED} frac={FRAC} window={WINDOW} "
          f"jitter={JITTER} hidden={HIDDEN} pseudo={PSEUDO}/class epochs={EPOCHS} "
          f"cap={CAP} tasks={N_TASKS} rectify={RECTIFY}", flush=True)
    Xtr, ytr = load_cifar100(True)
    Xte, yte = load_cifar100(False)
    f2c = fine_to_coarse()
    tasks = [(f2c == t).nonzero().squeeze(1).tolist() for t in range(N_TASKS)]
    if N_TASKS < 20:
        keep = torch.isin(yte, torch.tensor(sum(tasks, [])))
        Xte, yte = Xte[keep], yte[keep]
    all_classes = sorted(sum(tasks, []))
    g = torch.Generator().manual_seed(SEED)

    # genesis pool = first task's images (what the eye first opens on)
    m0 = torch.isin(ytr, torch.tensor(tasks[0]))
    pool = Xtr[m0]
    jit_gen = torch.Generator().manual_seed(SEED + 77)
    if FIX == 0:
        organs = [FlatOrgan(pool, SEED)]
    else:
        organs = [FixationOrgan(c, pool, SEED, i, jit_gen)
                  for i, c in enumerate(fixations(FIX))]
    print(f"[genesis] {len(organs)} fixation organs, "
          f"{sum(len(o.leaves) for o in organs)} phasecyte leaves  "
          f"t={time.time() - t0:.0f}s", flush=True)

    # the continual stream
    for ti, cls in enumerate(tasks):
        m = torch.isin(ytr, torch.tensor(cls))
        idx = m.nonzero().squeeze(1)
        idx = idx[torch.randperm(len(idx), generator=g)]
        if FRAC < 1.0:
            idx = idx[:max(1, int(FRAC * len(idx)))]
        X, labs = Xtr[idx], [f"g{int(v):03d}" for v in ytr[idx]]
        for w0 in range(0, len(X), WINDOW):
            for o in organs:
                o.observe(X[w0:w0 + WINDOW], labs[w0:w0 + WINDOW])
        lp = organs[0].leaves[next(iter(organs[0].leaves))]
        print(f"  [task {ti + 1:>2d}/{len(tasks)} coarse {ti}] classes(leaf0)="
              f"{len(lp.mixed.classes)} spawned={len(lp.mixed.specs)} "
              f"t={time.time() - t0:.0f}s", flush=True)

    # dream the trioron leaves (end of stream; sketches persisted throughout)
    print(f"[dream] {len(organs)} trioron leaves  t={time.time() - t0:.0f}s", flush=True)
    for o in organs:
        Xp, yp, sk = dream_pseudo(o, all_classes, PSEUDO, 5000 + SEED + o.idx)
        # scale: pockets sit at ~0.5±0.07 after the per-sample frame; a
        # gradient leaf on that barely learns (diag s051: 0.046 raw vs 0.202
        # standardized on REAL pockets). Standardizer from the pseudo set.
        o.mu, o.sd = Xp.mean(0), Xp.std(0) + 1e-3
        Xp = o.standardize(Xp)
        o.T = train_sub(Xp, yp, 100, hidden=HIDDEN, seed=6000 + SEED + o.idx,
                        tag=f"f{o.idx} skipped={sk}")
        o.width = Xp.shape[1]

    # eval
    print(f"[eval] test={len(Xte)}  t={time.time() - t0:.0f}s", flush=True)
    L = []
    for o in organs:
        P = o.standardize(o.pockets(Xte))
        lg = logits_of(o.T, P)
        L.append(lg)
        fa, ta = acc_pair(lg, yte, f2c)
        print(f"  fixation {o.idx} centre={getattr(o.eye, 'centre', 'flat') if hasattr(o, 'eye') else 'flat'} "
              f"width={o.width} params={n_params(o.T)}: full={fa:.4f} task={ta:.4f}",
              flush=True)
    # phasecyte-alone diagnostic on organ 0's first leaf (matched filter)
    o0 = organs[0]; s0 = next(iter(o0.leaves)); leaf0 = o0.leaves[s0]
    _, Ec = leaf0.evidence(o0.senses()[s0], Xte)
    nm = leaf0.names()
    pc_full = float((nm[Ec.argmax(1)] == yte).float().mean())
    print(f"  phasecyte leaf0 ({s0}) alone: full={pc_full:.4f} "
          f"classes={len(leaf0.mixed.classes)}", flush=True)

    total_params = sum(n_params(o.T) for o in organs)
    if len(organs) == 1:
        fa, ta = acc_pair(L[0], yte, f2c)
        print(f"\n[seed {SEED} fix={FIX}] SINGLE full={fa:.4f} task={ta:.4f} "
              f"params={total_params}  bar {BAR}  elapsed={time.time() - t0:.0f}s",
              flush=True)
        return

    S = torch.stack(L)                                    # [F, N, 100]
    fa_abs, ta_abs = acc_pair(S.sum(0), yte, f2c)

    # router: centre-M pseudo pockets -> best fixation per class
    centre = organs[0]
    tabM = per_class_pseudo(sketch_table(centre.leaves["M"], MIN_FRAC), all_classes, PSEUDO,
                            torch.Generator().manual_seed(9000 + SEED))
    Xr, yr = [], []
    per_class_best = {}
    for gcls in all_classes:
        # per-class best fixation on that class's own pseudo set
        accs = []
        for o in organs:
            Xp, yp, _ = dream_pseudo(o, [gcls], 100, 7000 + gcls + o.idx)
            accs.append(float((logits_of(o.T, o.standardize(Xp)).argmax(1) == yp)
                              .float().mean()))
        best = int(torch.tensor(accs).argmax())
        per_class_best[gcls] = best
        q = tabM.get(gcls)
        if q is None:
            continue
        Xr.append(q); yr.append(torch.full((len(q),), best, dtype=torch.long))
    Xr, yr = torch.cat(Xr), torch.cat(yr)
    r_mu, r_sd = Xr.mean(0), Xr.std(0) + 1e-3
    router = train_sub((Xr - r_mu) / r_sd, yr, len(organs), hidden=32,
                       seed=8000 + SEED, tag="router")
    j, centre.eye.jitter = centre.eye.jitter, 0
    PM = torch.cat([centre.leaves["M"].mixed.pockets_of(
        centre.eye.sense_M(Xte[i:i + 2000])) for i in range(0, len(Xte), 2000)]) / N_QUANTA
    centre.eye.jitter = j
    R = logits_of(router, (PM - r_mu) / r_sd)             # [N, F]
    hard = S[R.argmax(1), torch.arange(len(yte))]         # [N, 100]
    fa_nest, ta_nest = acc_pair(hard, yte, f2c)
    w = torch.softmax(R, dim=1).T.unsqueeze(-1)           # [F, N, 1]
    fa_soft, ta_soft = acc_pair((S * w).sum(0), yte, f2c)
    hist = torch.bincount(R.argmax(1), minlength=len(organs)).tolist()
    best_hist = torch.bincount(torch.tensor(list(per_class_best.values())),
                               minlength=len(organs)).tolist()
    print(f"\n[seed {SEED} fix={FIX}] ABSORB full={fa_abs:.4f} task={ta_abs:.4f} | "
          f"NEST full={fa_nest:.4f} task={ta_nest:.4f} | "
          f"SOFT full={fa_soft:.4f} task={ta_soft:.4f}\n"
          f"  params: leaves={total_params} router={n_params(router)}  "
          f"route_hist={hist} per_class_best_hist={best_hist}  bar {BAR}  "
          f"elapsed={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
