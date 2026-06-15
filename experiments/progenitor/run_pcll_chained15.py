"""PCLL organism on bench-15 (chained-15) — s036 REBUILD.

s034 ran first contact with four design divergences from the
progenitor-council architecture (s035 audit; Rocky's correction,
s036): no positional substrate / no retinal compression, the bare
CLASS_CAP=128, the council's discrimination arm (composer) never
enabled, and CONV unreachable. This rebuild closes them with the
substrate machinery shipped in s036:

  * BODY GEOMETRY — input_shape=(28, 28) declares the retina; genesis
    imposes (x, y, scale) positions on every perception cell.
  * RETINAL COMPRESSION — at the first sitting, constant columns
    starve (census, as before) and adjacent redundant columns MERGE
    into pooled region sensors with imposed positions (retina.py;
    merge floor 1 − GAIN_D, evidence floor MIN_MEMBERS — derived from
    division's own law, no new constants). Downstream machinery reads
    the compressed positional set.
  * COUNCIL ON — composer=True: the discrimination arm trials
    LINEAR/TANH/DENDRITE genome candidates and pays the winning gene's
    council group. CONV is live as promotion-by-spatial-reuse: a
    LINEAR winner on a touching sensor pair that independently
    re-carves at ≥2 same-offset positions spawns a weight-tied conv
    lineage (composer.conv_reuse / spawn_conv).
  * CAP ARMS — Rocky's s036 ruling: bench BOTH class_cap=128 (the
    historical backstop) and uncapped (PCLL15_CAP=0), since 128 was
    never derived.

PROTOCOL (single-pass honest, Rocky's s036 ruling): one gradient-free
streaming pass per task, labels via the D22 carrier at 100% coverage,
class-sequential task order. Reported alongside the legacy gradient
stack (0.958/0.551, 8 epochs/task, ~8K params) WITH the budget
difference stated — the architecture fix is what makes the comparison
fair, not compute matching. Do not quote the pair as compute-matched.

Readout is organism-internal end to end (class x label count
majority; task-aware = argmax restricted to the task's labels' own
classes). Eval on held-out test only.

s034 first-contact record (raw arm, pre-rebuild, seed 0):
task_aware 0.552 / full 0.172 / 128 classes (tiled to cap). The lcn
projection arms tiled to cap inside 1-2 tasks (kept for parity:
PCLL15_SENSE=lcn, no geometry, disfavored).

Run: python3 -m experiments.progenitor.run_pcll_chained15
Env: PCLL15_SEEDS (default 1), PCLL15_MANIFOLD (default 1),
     PCLL15_WINDOW (default 1000), PCLL15_SENSE (raw|gabor|rff|conv|lcn),
     RFF_DIM (default 512), RFF_BANDWIDTH (default 0.25),
     PCLL15_CAP (default 128; 0 = uncapped),
     PCLL15_COMPOSER (default 1).
"""
from __future__ import annotations

import math
import os
import time
from typing import List, Optional

import torch
import torch.nn.functional as F

from trioron.core import construct
from trioron.core.receptor import N_QUANTA
from trioron.legacy.donorkit.bench_chained_15task import build_lcn_mask
from trioron.legacy.donorkit.datasets import (DatasetBundle,
                                              chained_15_specs,
                                              build_task_views)
from trioron.pcll import (MixedStreamController, PerceptionGenesis,
                          germline_base)

DATA_ROOT = os.environ.get(
    "AXIS6_DATA_ROOT",
    os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), "outputs", "data"))
WINDOW = int(os.environ.get("PCLL15_WINDOW", "1000"))
SEEDS = int(os.environ.get("PCLL15_SEEDS", "1"))
MANIFOLD = os.environ.get("PCLL15_MANIFOLD", "1") == "1"
SENSE = os.environ.get("PCLL15_SENSE", "raw")
CAP = int(os.environ.get("PCLL15_CAP", "128"))          # 0 = uncapped
COMPOSER = os.environ.get("PCLL15_COMPOSER", "1") == "1"
N_TASKS = int(os.environ.get("PCLL15_TASKS", "15"))     # smoke subsetting
FRAC = float(os.environ.get("PCLL15_FRAC", "1.0"))      # per-task subsample
                                                        # (Rocky s036: fewer
                                                        # samples = fewer
                                                        # meetings = less tiling)
L0_WIDTH, LCN_SIGMA = 128, 0.10
CONV_C, CONV_K, CONV_POOL = 12, 5, 4   # fixed-conv benchmark geometry
# converged perception design (s037 handoff): complex quadrature Gabor
# bank at low/mid scales, pooled, EMITTING THE ENERGY CHANNEL. Energy
# |conv_even + i·conv_odd| is the invariant+shift-robust workhorse —
# bit-exact under background swap (Rocky's binding criterion: a human
# reads the digit even if you swap the background; inverted-test 0.553).
# Phase has the higher oracle ceiling (0.597) but is polarity-sensitive
# (inverted-test 0.000) AND is an angle that the per-sample contrast
# quantizer cannot carry faithfully — phase wants direct phasor injection,
# a pipeline change, deferred. Energy is a non-negative magnitude → the
# receptor quantizes it natively, exactly like raw pixels / conv.
GABOR_LAMBDAS = (4.0, 8.0)             # low/mid scales (high freq = noise)
GABOR_THETAS = (0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4)
GABOR_K, GABOR_POOL = 11, 4            # ksize fits σ=0.5·λ_max=4; pool 4px
# Random Fourier Features ON the Gabor energy (Gemma's holographic
# wave-vector idea, Rocky s038): Ψ(r) = sin(2π·W·r + b). r = Gabor energy
# (kept so the front-end stays background-swap invariant — plain RFF on raw
# pixels would be shift/polarity fragile, like a global FFT). W = random
# wave-vectors (each row a plane-wave k); b = random phase offset. This is
# the projection-THEN-wrap that we never combined: lcn projected without
# wrapping, the receptor wraps without projecting. Per-sample L2-normalize
# r first so the pre-sine argument has a controlled scale (r̂·W ~ N(0,σ_W)),
# σ_W = RFF_BANDWIDTH sets the sine frequency (small = smooth, less likely
# to over-fragment). sin is bounded [−1,1] → the receptor quantizes natively.
RFF_DIM = int(os.environ.get("RFF_DIM", "512"))
RFF_BANDWIDTH = float(os.environ.get("RFF_BANDWIDTH", "0.25"))
SHAPE = (28, 28) if SENSE == "raw" else None   # body geometry (raw only)


def _gabor_quadrature_bank() -> tuple[torch.Tensor, torch.Tensor]:
    """Complex quadrature bank: matched even (cos) / odd (sin) Gabor pairs.
    energy = sqrt(conv_even² + conv_odd²) is phase-magnitude → invariant to
    a global sign flip of the input (background swap) and, after pooling,
    shift-tolerant. Both members zero-DC + L1-normalized so the energy map
    is gain-comparable across filters. Returns [(C,1,k,k), (C,1,k,k)]."""
    half = GABOR_K // 2
    yy, xx = torch.meshgrid(torch.arange(-half, half + 1).float(),
                            torch.arange(-half, half + 1).float(),
                            indexing="ij")
    even, odd = [], []
    for lam in GABOR_LAMBDAS:
        sigma = 0.5 * lam
        for theta in GABOR_THETAS:
            x_t = xx * math.cos(theta) + yy * math.sin(theta)
            y_t = -xx * math.sin(theta) + yy * math.cos(theta)
            env = torch.exp(-(x_t ** 2 + y_t ** 2) / (2 * sigma ** 2))
            ge = env * torch.cos(2 * math.pi * x_t / lam)
            go = env * torch.sin(2 * math.pi * x_t / lam)
            ge = ge - ge.mean()                 # zero DC (cos has a DC term)
            go = go - go.mean()                 # sin is ~odd, kept for parity
            even.append(ge / ge.abs().sum().clamp_min(1e-6))
            odd.append(go / go.abs().sum().clamp_min(1e-6))
    return (torch.stack(even).unsqueeze(1),     # [C,1,k,k]
            torch.stack(odd).unsqueeze(1))


def _build_gabor_sense():
    """The converged Gabor energy front-end (s037) as a reusable closure —
    no learned weights, no gradients. Returns x[N,784] → energy[N, C*49]."""
    We, Wo = _gabor_quadrature_bank()
    pad = GABOR_K // 2

    def gabor_sense(x: torch.Tensor) -> torch.Tensor:
        img = x.view(-1, 1, 28, 28)
        # REFLECT pad (not zero): makes background-swap invariance exact at
        # the border too — conv of a reflected constant is sum(kernel)·c = 0
        # everywhere (zero-DC kernels), so conv(1−x) = −conv(x) and the
        # energy is bit-exact under inversion. Zero-pad breaks this at edges.
        img = F.pad(img, (pad, pad, pad, pad), mode="reflect")
        ce = F.conv2d(img, We)                     # even (cos) response
        co = F.conv2d(img, Wo)                     # odd  (sin) response
        energy = torch.sqrt(ce * ce + co * co + 1e-12)   # [N,C,28,28]
        energy = F.avg_pool2d(energy, GABOR_POOL)        # [N,C,7,7]
        return energy.flatten(1)                         # [N, C*49]
    return gabor_sense


def make_sense(seed: int):
    """raw: identity — genesis faces the world (default). conv: a FIXED
    (random, gradient-free) convolution + ReLU + avg-pool spatial
    feature transform — the s036 control (Rocky): bolt a simple conv
    in FRONT of the phasor module to test whether the matched-filter
    classification works on convolved features. If it does, the phasor
    machinery is sound and the only gap is that conv doesn't FORM
    natively (cell spawning). The conv is a fixed sensory transform
    (like the legacy LCN but convolutional), no training, no gradients
    — it stays inside PCLL's gradient-free claim. lcn: legacy-parity
    frozen retinotopic projection (disfavored — tiles to cap)."""
    if SENSE == "raw":
        return lambda x: x
    if SENSE == "gabor":
        # the converged perception front-end (s037). Oracle upper bound:
        # energy 0.553 vs raw 0.249 (30-way), bit-exact under background
        # swap. Organism (s038, seed 0): task-aware 0.690 / full 0.304.
        return _build_gabor_sense()
    if SENSE == "rff":
        # holographic wave-vectors ON the Gabor energy: Ψ(r)=sin(2π·W·r+b).
        # projection-then-wrap (Gemma s038). energy base keeps invariance.
        base = _build_gabor_sense()
        in_dim = base(torch.zeros(1, 784)).shape[1]
        g = torch.Generator().manual_seed(30_000 + seed)
        W = torch.randn(in_dim, RFF_DIM, generator=g) * RFF_BANDWIDTH
        b = torch.rand(RFF_DIM, generator=g) * 2 * math.pi

        def rff_sense(x: torch.Tensor) -> torch.Tensor:
            r = F.normalize(base(x), dim=1)        # unit wave amplitude
            return torch.sin(2 * math.pi * (r @ W) + b)   # [N, RFF_DIM]
        return rff_sense
    if SENSE == "conv":
        g = torch.Generator().manual_seed(20_000 + seed)
        W = torch.randn(CONV_C, 1, CONV_K, CONV_K, generator=g)
        W = W / W.flatten(1).norm(dim=1).view(CONV_C, 1, 1, 1)  # unit kernels

        def conv_sense(x: torch.Tensor) -> torch.Tensor:
            img = x.view(-1, 1, 28, 28)
            f = F.relu(F.conv2d(img, W, padding=CONV_K // 2))   # [N,C,28,28]
            f = F.avg_pool2d(f, CONV_POOL)                      # [N,C,7,7]
            return f.flatten(1)                                 # [N, C*49]
        return conv_sense
    g = torch.Generator().manual_seed(10_000 + seed)
    W = torch.randn(784, L0_WIDTH, generator=g) / math.sqrt(784.0)
    W = W * build_lcn_mask(784, L0_WIDTH, LCN_SIGMA).t()
    return lambda x: x @ W


def names_of(mixed) -> torch.Tensor:
    """Per-class global label from the internal count majority
    (-1 = unnamed). Organism-internal — no eval-side mapping."""
    out = []
    for c in mixed.classes:
        comp = (mixed.label_taps.composition_of(c.name)
                if mixed.label_taps else {})
        out.append(int(max(comp, key=comp.get)[1:]) if comp else -1)
    return torch.tensor(out, dtype=torch.long)


def evidence_both(mixed, sense, X: torch.Tensor):
    """Matched-filter evidence under TWO readouts on the SAME templates,
    computed in one forward pass (pockets_of runs the substrate — the
    expensive part — once per chunk):

      plain    E = Σ_f Re(Z·T̄)                 — the shipped generative filter
      centered E = Σ_f Re((Z−μ)·(T−μ)̄), μ = mean_k T[k]

    The CENTERED filter subtracts the COMMON-MODE phasor that every class
    template shares (the average image/background structure). s039 oracle:
    removing it ~doubles raw-pixel accuracy (0.249→0.496) — the real
    superposition is this shared component in the GENERATIVE readout, not
    the input layout (the band-offset probe falsified spatial separation).
    Readout-only: no retrain, the organism state is identical."""
    T = mixed.templates()
    mu = T.mean(0, keepdim=True)
    Tc = T - mu
    Ep, Ec = [], []
    for i in range(0, len(X), 2000):
        q = mixed.pockets_of(sense(X[i:i + 2000]))
        Z = torch.exp(1j * 2 * math.pi * q / N_QUANTA)
        Ep.append(mixed._evidence(Z, T))
        Ec.append(mixed._evidence(Z - mu, Tc))
    return torch.cat(Ep), torch.cat(Ec)


def _score(E: torch.Tensor, name: torch.Tensor, eval_views, specs):
    """(task_aware, full, per_task) for one evidence matrix — the readout
    is the only thing that varies, so name/views/specs are shared."""
    yt = torch.cat([v.labels_global for v in eval_views])
    full = float((name[E.argmax(1)] == yt).float().mean())
    per_task, off = [], 0
    for view, spec in zip(eval_views, specs):
        n = len(view.labels_global)
        Et = E[off:off + n]
        off += n
        cand = torch.isin(name, torch.tensor(spec.global_classes)) \
            .nonzero().squeeze(1)
        if not len(cand):
            per_task.append(0.0)
            continue
        pred = name[cand[Et[:, cand].argmax(1)]]
        per_task.append(
            float((pred == view.labels_global).float().mean()))
    return sum(per_task) / len(per_task), full, per_task


def run_seed(seed: int):
    bundle = DatasetBundle(["mnist", "fashion_mnist", "emnist_letters"],
                           root=DATA_ROOT, n_holdout_per_dataset=0)
    specs = chained_15_specs()[:N_TASKS]
    train_views = build_task_views(bundle, specs, split="train")
    eval_views = build_task_views(bundle, specs, split="test")
    sense = make_sense(seed)

    torch.manual_seed(seed)
    sub = construct(germline_base,
                    capacity=512 if SENSE == "lcn" else 8192)
    pg = PerceptionGenesis(sub, input_shape=SHAPE)
    t0 = time.time()

    # GENESIS on a MIXED period (Rocky, s036): pass ALL data types in one
    # genesis period so the progenitor grows CORRECT perceptions for the
    # whole heterogeneous stream — not starving columns that are dead in
    # task 1 (MNIST 0/1) but carry signal in Fashion / EMNIST. Re-genesis
    # is only needed when input DIMENSIONS change (they don't here: every
    # task is 784); perception then keeps adapting via ongoing sensation
    # growth as the stream continues. This window is PERCEPTION PRIMING
    # only (unlabeled, drawn from the union); the LABELLED continual stream
    # below is the real v0.2.2 sequential order, unchanged.
    g0 = torch.Generator().manual_seed(seed)
    pool = torch.cat([v.images for v in train_views])
    gidx = torch.randperm(len(pool), generator=g0)[:WINDOW]
    genesis_window = sense(pool[gidx])
    pg.feed(genesis_window)
    rep = sub.end_task()
    n_rec = int(sub.scheduler._plan.receptor_ids.numel())
    print(f"  [genesis/mixed] starved={len(rep.starved)} "
          f"merged={len(rep.merged)}->{len(rep.regions)} regions  "
          f"receptors {genesis_window.shape[1]}->{n_rec}", flush=True)
    mixed = MixedStreamController(
        sub, stress=pg.router, adopt=pg.controller,
        manifold=MANIFOLD, composer=COMPOSER,
        class_cap=CAP if CAP > 0 else None)

    # the real continual stream — v0.2.2 task order, labelled, one pass
    for ti, view in enumerate(train_views):
        g = torch.Generator().manual_seed(100 * seed + ti)
        order = torch.randperm(len(view.labels_global), generator=g)
        if FRAC < 1.0:                       # subsample the task stream
            order = order[:max(1, int(FRAC * len(order)))]
        X = view.images[order]
        y = view.labels_global[order]
        labs = [f"g{int(v):02d}" for v in y]
        for w0 in range(0, len(X), WINDOW):
            xw = sense(X[w0:w0 + WINDOW])
            lw = labs[w0:w0 + WINDOW]
            mixed.observe(xw, labels=lw)
            sub.end_task()
        n_conv = sum(1 for s in mixed.specs if s.gene == "conv")
        print(f"  [task {ti + 1:>2d}/15 {view.name}] "
              f"classes={len(mixed.classes)} "
              f"spawned={len(mixed.specs)} (conv {n_conv}) "
              f"t={time.time() - t0:.0f}s", flush=True)

    name = names_of(mixed)
    # full + task-aware under BOTH readouts on the union of all 15 test
    # sets — readout-only A/B on the identical trained organism.
    Xt = torch.cat([v.images for v in eval_views])
    Ep, Ec = evidence_both(mixed, sense, Xt)
    ta_p, full_p, pt_p = _score(Ep, name, eval_views, specs)
    ta_c, full_c, pt_c = _score(Ec, name, eval_views, specs)
    unnamed = int((name < 0).sum())
    print(f"  per-task plain   : " + " ".join(f"{a:.2f}" for a in pt_p))
    print(f"  per-task centered: " + " ".join(f"{a:.2f}" for a in pt_c))
    print(f"[seed {seed}] PLAIN    task_aware={ta_p:.3f}  full={full_p:.3f}")
    print(f"[seed {seed}] CENTERED task_aware={ta_c:.3f}  full={full_c:.3f}  "
          f"(Δ ta {ta_c - ta_p:+.3f}, full {full_c - full_p:+.3f})")
    print(f"[seed {seed}] classes={len(mixed.classes)} ({unnamed} unnamed)  "
          f"elapsed={time.time() - t0:.0f}s", flush=True)
    return ta_c, full_c, len(mixed.classes)


def main() -> None:
    print(f"PCLL organism on chained-15 (s036 rebuild) — single pass, "
          f"no gradients, labels 100% (D22), manifold="
          f"{'on' if MANIFOLD else 'off'}, composer="
          f"{'on' if COMPOSER else 'off'}, "
          f"cap={CAP if CAP > 0 else 'UNCAPPED'}, "
          f"sense={SENSE}, window={WINDOW}, seeds={SEEDS}")
    tas, fulls = [], []
    for seed in range(SEEDS):
        ta, full, _ = run_seed(seed)
        tas.append(ta)
        fulls.append(full)
    if SEEDS > 1:
        import statistics
        print(f"\nmean task_aware {statistics.mean(tas):.3f} "
              f"± {statistics.stdev(tas):.3f}  "
              f"full {statistics.mean(fulls):.3f} "
              f"± {statistics.stdev(fulls):.3f}")


if __name__ == "__main__":
    main()
