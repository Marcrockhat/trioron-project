"""M4 pre-probe — the relational testbed REDONE with the REAL gene set
(design mixed_stream_growth.md §4 [D14]; handoff s033 NEXT 1).

The s031 probe (run_composer_growth) used a phenotype set that is not
the genome (gcos/sinc/tan/standalone-quad — the s032 correction): its
vote-book claim does not transfer. This probe carries its mechanism
loop with the THREE corrections the design approved:

  1. GENOME-CONSTRAINED candidates, computed exactly as the substrate
     phenotypes compute them on phase inputs a = 2π(c+½):
       LINEAR  : w₀c_i + w₁c_j           (sum/diff)        frame [-1, 1]
       TANH    : tanh(2.5(w₀c_i + w₁c_j)) (sum/diff)        frame [-1, 1]
       DENDRITE: c_i² + c_j²  — two branches, one input each, σ(z)=z+z²
                 with edge weight w = −1/(2π): σ(−(c+½)) = c² − ¼
                 exactly (the tied linear term cancels)     frame [0, ½]
     ATTENTION/CONV/RECURRENT have no scalar form — deferred [D14].
  2. IMPORTANCE-GATED WIRING (failure-mode-4 fix, part 1): candidate
     dim pairs draw only from importance evidence. PROBE FINDING: the
     genesis-window margin (first-sitting perception_importance) ranks
     informative dims above noise (f0/f1 ≈ 2.8-3.8 vs f2/f3 ≈ 0.7-2.2)
     but an absolute K=3 cut starves the trial (only f1 clears; no
     pairs). The design's own criterion is per-CLASS: "noise columns
     have no lock-in margin IN ANY CLASS" — so the wired set is dims on
     which ANY current class buffer coheres (R > IMPORTANCE_R). Blobs
     pin f0/f1; nothing ever coheres on noise. Genesis importance
     remains the seed/reported ranking.
  3. FUTURE-DEPOSIT SETTLEMENT (part 2): the trial selects on the
     buffer (permutation-null correction + split-half confirm, failure
     modes 1-3 carried), but the spawn settles ONLY on members that
     arrive AFTER it exists (D9): the carve must reproduce on ≥
     FRESH_MIN virgin members of the proposing lineage, else PATIENCE
     prunes.

  PROBE FINDING (the statistic): the s031 probe scored candidates by
  null-corrected gain in raw R units. R saturates near 1, and ring
  marginals are themselves clumpy (cos²φ piles at its extremes), so on
  ring buffers the null carve sits at 0.93 and a 0.995 relation shows
  "gain" 0.06 — invisible. The honest unit is RESIDUAL INCOHERENCE:
  ratio = (1 − null) / (1 − carve) > NULL_RATIO.
  Measured: dendrite on E+F 13.1×, tanh on A+B 2.5-2.9×, noise pairs
  1.0-1.2×, mixed/world buffers below threshold (trials correctly
  defer until division isolates the relational residue).

  PROBE FINDING (family trials — the ring blocker and its repair):
  per-buffer trials NEVER discover the ring relation: division
  legitimately fragments continuous manifolds into raw-separable arcs
  (the s031 testbed premise "no raw dim separates E|F" is false for
  FRAGMENTS), so no E/F-confounded buffer of trial size ever exists,
  and mixed parents fail the null (clumpy marginals). The repair:
  trial over UNIONS OF LINEAGE SIBLINGS — pools of current buffers
  descending from a retired division ancestor, deepest (smallest)
  pools first. Measured: reunified-E pool ratio 21.4, half-ring 22.5,
  E+F 13.9, vs control unions (C+D blobs 1.47, A+E 1.16). The composer
  is thereby the ANTI-FRAGMENTATION mechanism: a family one
  composition explains better than its pieces — which is also the
  probe-level answer to the deferred under-division/sibling-merge
  carry (s031 NEXT 4).
  4. CAUSAL QUANTIZATION: composed dims carry FIXED static frames
     known at spawn (the forms are bounded by construction) — the
     probe's non-causal batch min-max is gone.

Run: python3 -m experiments.progenitor.run_composer_genome
"""
from __future__ import annotations

import math
import os
from collections import Counter

import torch

DEBUG = bool(int(os.environ.get("COMPOSER_DEBUG", "0")))

from .run_composer_growth import make_data
from .run_mixed_division import circ_2means

N_QUANTA = 1000
WINDOW = 600
S_TRAIN = 800
S_TEST = 200
SEEDS = 3
BUF = 400
MIN_MEMBERS = 60
MIN_CHILD = 25
GAIN_D = 0.08
NULL_SPLIT = 0.72
TRIGGER_R = 0.55
SPAWN_R = 0.80
NULL_RATIO = 2.0       # residual-incoherence ratio (1-null)/(1-carve):
                       # noise pairs measure 1.0-1.2, tanh relations
                       # 2.5-2.9, dendrite on ring pools 13-22
PATIENCE = 3           # sittings for a spawn to gather its virgin verdict
FRESH_MIN = 60         # virgin members needed before the verdict
MAX_SPAWNED = 6
DIV_TRIES = 4
IMPORTANCE_K = 3.0     # genesis lock-in margin (reported ranking seed)
IMPORTANCE_R = 0.9     # any-class buffer coherence a dim needs to be wired
NAMES = ["A:diag", "B:anti", "C:blob1", "D:blob2", "E:ring.40", "F:ring.22"]

GENES = ("linear", "tanh", "dendrite")


def compose(X: torch.Tensor, spec) -> torch.Tensor:
    """The spec's value EXACTLY as the substrate phenotype computes it,
    in pocket space c = q/1000 − ½ (X here is raw ∈ [0,1] ⇒ c = X − ½)."""
    gene, i, j, w = spec
    ci, cj = X[:, i] - 0.5, X[:, j] - 0.5
    if gene == "dendrite":
        return ci * ci + cj * cj
    z = w[0] * ci + w[1] * cj
    return torch.tanh(2.5 * z) if gene == "tanh" else z


def frame_of(spec) -> tuple:
    return (0.0, 0.5) if spec[0] == "dendrite" else (-1.0, 1.0)


def candidates(i: int, j: int):
    for gene in ("linear", "tanh"):
        for w in ((1.0, 1.0), (1.0, -1.0)):
            yield (gene, i, j, w)
    yield ("dendrite", i, j, None)


def composed_phase(X: torch.Tensor, spec) -> torch.Tensor:
    """Causal quantization: the composed value through the spec's FIXED
    static frame → pocket → phase. No batch statistics anywhere."""
    lo, hi = frame_of(spec)
    q = (N_QUANTA * (compose(X, spec) - lo) / (hi - lo)).clamp(1, N_QUANTA - 1)
    return 2 * math.pi * q / N_QUANTA


def phasors(X: torch.Tensor, dims: list) -> torch.Tensor:
    cols = [2 * math.pi * X.t().clamp(1 / N_QUANTA, 1 - 1 / N_QUANTA)]
    cols += [composed_phase(X, s).unsqueeze(0) for s in dims]
    return torch.exp(1j * torch.cat(cols).t())


def genesis_importance(Xw: torch.Tensor) -> list:
    """Per-dim lock-in margin over the genesis window — the
    Germline.perception_importance semantics (first-sitting evidence)."""
    Z = torch.exp(1j * 2 * math.pi * Xw)
    return [float(Z[:, d].sum().abs() / math.sqrt(len(Xw)))
            for d in range(Xw.shape[1])]


def carve_r(X: torch.Tensor, spec) -> float:
    phi = composed_phase(X, spec)
    side = circ_2means(phi)
    if int((~side).sum()) < MIN_CHILD or int(side.sum()) < MIN_CHILD:
        return 0.0
    z = torch.exp(1j * phi)
    return max(float(z[~side].mean().abs()), float(z[side].mean().abs()))


def null_r(X: torch.Tensor, spec, g: torch.Generator) -> float:
    """Permutation null ×3: shuffle col j against col i — marginals and
    the static frame survive, the RELATION dies."""
    _, _, j, _ = spec
    total = 0.0
    for _ in range(3):
        shuf = X.clone()
        shuf[:, j] = X[torch.randperm(len(X), generator=g), j]
        total += carve_r(shuf, spec) / 3
    return total


def trial_ratio(X: torch.Tensor, spec, g: torch.Generator) -> tuple:
    """(carve, residual-incoherence ratio) — the trial statistic."""
    r = carve_r(X, spec)
    if r < SPAWN_R or r >= 1.0:
        return r, 0.0
    return r, (1.0 - null_r(X, spec, g)) / (1.0 - r)


def run_seed(seed: int, composers: bool):
    Xs, ys = make_data(S_TRAIN, seed)
    Xt, yt = make_data(S_TEST, 100 + seed)
    dims: list = []
    book = {p: 4.0 for p in GENES}
    raw_bufs: list = []
    spawn_log, prune_log, settle_log = [], [], []
    pending: list = []     # [spec, gene, age, fresh-member store, buf idxs]
    importance: list = []
    wired: list = []
    # lineage forest over buffers: node_of[i] = stable node id of buffer i;
    # parent[node] = the retired division ancestor. Family trials pool the
    # current descendants of an internal node (deepest pools first).
    node_of: list = []
    parent: dict = {}
    depth: dict = {}
    next_node = [0]

    def new_node(par=None):
        nid = next_node[0]
        next_node[0] += 1
        parent[nid] = par
        depth[nid] = 0 if par is None else depth[par] + 1
        return nid

    def bet(gene: str, success: bool, amount: float = 1.0):
        """step4 transfer between the gene's group and the others."""
        others = [p for p in book if p != gene]
        src, dst = (others, [gene]) if success else ([gene], others)
        share = amount / len(src)
        moved = 0.0
        for p in src:
            take = min(share, max(0.0, book[p] - 1.0))
            book[p] -= take
            moved += take
        for p in dst:
            book[p] += moved / len(dst)

    for w0 in range(0, len(Xs), WINDOW):
        Xw = Xs[w0:w0 + WINDOW]
        if not raw_bufs:
            raw_bufs = [Xw[-BUF:]]
            node_of = [new_node()]
            importance = genesis_importance(Xw)
            continue
        Z = phasors(Xw, dims)
        T = torch.stack([phasors(b, dims).mean(0) for b in raw_bufs])
        member = (Z.unsqueeze(1) * T.conj().unsqueeze(0)).real.sum(-1).argmax(1)
        for k in range(len(raw_bufs)):
            xk = Xw[member == k]
            if len(xk):
                raw_bufs[k] = torch.cat([raw_bufs[k], xk])[-BUF:]
                for p in pending:            # future deposits: virgin members
                    if k in p[4]:
                        p[3] = torch.cat([p[3], xk])[-2 * FRESH_MIN:]

        # importance-gated wiring: a dim some SETTLED class coheres on
        # (raw dims; judgable-size buffers only — tiny selected fragments
        # confabulate coherence on noise, failure mode 4)
        wired = [d for d in range(4) if any(
            len(b) >= MIN_MEMBERS and
            float(phasors(b, [])[:, d].mean().abs()) > IMPORTANCE_R
            for b in raw_bufs)]

        new_bufs, new_nodes, buf_map = [], [], {}   # old buffer idx → new
        for bi, b in enumerate(raw_bufs):    # the sitting: divide XOR compose
            if len(b) < MIN_MEMBERS:
                buf_map[bi] = [len(new_bufs)]
                new_bufs.append(b)
                new_nodes.append(node_of[bi])
                continue
            Zb = phasors(b, dims)
            R = Zb.mean(0).abs()

            div_r, div_split = 0.0, None     # remedy 1: division, multi-try
            order = torch.argsort(R)[:DIV_TRIES]
            for d in order.tolist():
                side = circ_2means(torch.angle(Zb[:, d]))
                if int((~side).sum()) < MIN_CHILD or int(side.sum()) < MIN_CHILD:
                    continue
                r = max(float(Zb[~side, d].mean().abs()),
                        float(Zb[side, d].mean().abs()))
                if r > max(float(R[d]) + GAIN_D, NULL_SPLIT) and r > div_r:
                    div_r, div_split = r, (side, d)

            # remedy 2: composer trial. Division and composition COMPETE
            # (the s031 probe's validated loop): the spawn wins only when
            # its carve evidence beats the best division's — this is the
            # ladder's "division can't clear" read locally (a strict
            # division-first ladder starves the composer: relational
            # buffers fragment forever on legitimate micro-modes and the
            # trial never fires — first-redo finding, this probe).
            comp_spec, comp_ratio, comp_r = None, NULL_RATIO, 0.0
            if (composers and float(R.mean()) < TRIGGER_R
                    and len(b) >= 4 * MIN_CHILD   # split-half must seat 2×2 children
                    and len(dims) + len(pending) < MAX_SPAWNED):
                g = torch.Generator().manual_seed(len(spawn_log))
                half = len(b) // 2
                b1, b2 = b[:half], b[half:]
                for i in wired:                       # importance-gated pairs
                    for j in wired:
                        if j <= i:
                            continue
                        for spec in candidates(i, j):
                            if any(s == spec for s in dims) or \
                                    any(p[0] == spec for p in pending):
                                continue
                            r1, ratio1 = trial_ratio(b1, spec, g)
                            if ratio1 <= comp_ratio:
                                continue
                            r2, ratio2 = trial_ratio(b2, spec, g)
                            if ratio2 > NULL_RATIO:   # split-half confirm
                                comp_ratio, comp_spec, comp_r = \
                                    ratio1, spec, r1

            if DEBUG and composers and len(b) >= MIN_MEMBERS:
                print(f"    w{w0//WINDOW} buf{bi} n={len(b)} "
                      f"meanR={float(R.mean()):.2f} div_r={div_r:.2f} "
                      f"comp={comp_spec} comp_r={comp_r:.2f} "
                      f"ratio={comp_ratio:.2f}")

            if comp_spec is not None and comp_r > div_r:
                dims.append(comp_spec)
                spawn_log.append(comp_spec)
                pending.append([comp_spec, comp_spec[0], 0,
                                b[:0].clone(), {bi}])
                buf_map[bi] = [len(new_bufs)]
                new_bufs.append(b)           # divide on it once it settles
                new_nodes.append(node_of[bi])
            elif div_split is not None:
                side, _ = div_split
                buf_map[bi] = [len(new_bufs), len(new_bufs) + 1]
                new_bufs += [b[~side], b[side]]
                new_nodes += [new_node(node_of[bi]), new_node(node_of[bi])]
            else:
                buf_map[bi] = [len(new_bufs)]
                new_bufs.append(b)
                new_nodes.append(node_of[bi])
        raw_bufs, node_of = new_bufs, new_nodes
        for p in pending:                    # lineage follows divisions
            p[4] = {nk for ok in p[4] for nk in buf_map.get(ok, [])}

        # FAMILY TRIAL: division fragments continuous manifolds into
        # raw-separable arcs before any relational buffer reaches trial
        # size — so trial the UNIONS of lineage siblings: pools of current
        # buffers under a retired ancestor, deepest (smallest) first. One
        # spawn per sitting keeps overproduction bounded.
        if composers and len(dims) + len(pending) < MAX_SPAWNED:
            pools: dict = {}
            for i, nid in enumerate(node_of):
                a = parent.get(nid)
                while a is not None:
                    pools.setdefault(a, set()).add(i)
                    a = parent.get(a)
            found = None
            for a in sorted(pools, key=lambda n: -depth[n]):
                idxs = pools[a]
                if len(idxs) < 2:
                    continue
                pool = torch.cat([raw_bufs[i] for i in idxs])[-BUF:]
                if len(pool) < 4 * MIN_CHILD:
                    continue
                Rp = phasors(pool, dims).mean(0).abs()
                if float(Rp.mean()) >= TRIGGER_R:
                    continue
                g = torch.Generator().manual_seed(7000 + len(spawn_log))
                half = len(pool) // 2
                p1, p2 = pool[:half], pool[half:]
                best, best_ratio = None, NULL_RATIO
                for i in wired:
                    for j in wired:
                        if j <= i:
                            continue
                        for spec in candidates(i, j):
                            if any(s == spec for s in dims) or \
                                    any(p[0] == spec for p in pending):
                                continue
                            _, ratio1 = trial_ratio(p1, spec, g)
                            if ratio1 <= best_ratio:
                                continue
                            _, ratio2 = trial_ratio(p2, spec, g)
                            if ratio2 > NULL_RATIO:
                                best, best_ratio = spec, ratio1
                if best is not None:
                    found = (best, idxs)
                    break
            if found is not None:
                spec, idxs = found
                dims.append(spec)
                spawn_log.append(spec)
                pending.append([spec, spec[0], 0,
                                raw_bufs[0][:0].clone(), set(idxs)])
                if DEBUG:
                    print(f"    w{w0//WINDOW} FAMILY spawn {spec} "
                          f"pool of {len(idxs)} buffers")

        if DEBUG:
            for p in pending:
                print(f"    w{w0//WINDOW} pending {p[0]} age={p[2]} "
                      f"fresh={len(p[3])} bufs={p[4]}")
        survivors = []                       # future-deposit settlement [D9]
        for p in pending:
            spec, gene, age, fresh, _ = p
            p[2] = age + 1
            # verdict when the virgin store is full, OR at PATIENCE on
            # whatever virgin data can seat children — a dim with a
            # near-full store deserves its judgment; PATIENCE-prune is
            # for dims nobody deposits toward
            judge = (len(fresh) >= FRESH_MIN or
                     (p[2] > PATIENCE and len(fresh) >= 2 * MIN_CHILD))
            if judge:
                g = torch.Generator().manual_seed(1000 + len(settle_log))
                _, ratio = trial_ratio(fresh, spec, g)
                ok = ratio > NULL_RATIO
                bet(gene, ok)
                settle_log.append((spec, ok, len(fresh)))
                if not ok:
                    prune_log.append(spec)
                    dims = [s for s in dims if s != spec]
                continue
            if p[2] > PATIENCE:              # never enough virgin data
                bet(gene, False)
                settle_log.append((spec, False, len(fresh)))
                prune_log.append(spec)
                dims = [s for s in dims if s != spec]
                continue
            survivors.append(p)
        pending = survivors

    # ── readout ──
    T = torch.stack([phasors(b, dims).mean(0) for b in raw_bufs])
    Z_all = phasors(Xs, dims)
    member = (Z_all.unsqueeze(1) * T.conj().unsqueeze(0)).real.sum(-1).argmax(1)
    mapping = []
    for k in range(len(T)):
        t = ys[member == k]
        mapping.append(Counter(t.tolist()).most_common(1)[0][0] if len(t) else -1)
    Zt = phasors(Xt, dims)
    pred = (Zt.unsqueeze(1) * T.conj().unsqueeze(0)).real.sum(-1).argmax(1)
    truth = torch.tensor(mapping)[pred]
    per_class = [float((truth[yt == c] == c).float().mean()) for c in range(6)]
    return (per_class, len(T), dims, book, spawn_log, prune_log,
            importance, wired)


def main() -> None:
    for arm, composers in (("division-only", False), ("genome-composer", True)):
        print(f"\n══ {arm} ══")
        acc = torch.zeros(6)
        noise_spawns = 0
        for seed in range(SEEDS):
            (per_class, n_cls, dims, book, spawns, prunes,
             importance, wired) = run_seed(seed, composers)
            acc += torch.tensor(per_class) / SEEDS
            extra = ""
            if composers:
                noise_spawns += sum(1 for s in spawns
                                    if s[1] >= 2 or s[2] >= 2)
                d = [f"{s[0]}({s[1]},{s[2]})" for s in dims]
                imp = " ".join(f"f{d_}={m:.1f}" for d_, m in
                               enumerate(importance))
                extra = (f"; spawned {len(spawns)}, pruned {len(prunes)}, "
                         f"kept [{', '.join(d)}]; importance [{imp}] "
                         f"wired {wired}")
            print(f"  seed={seed}: {n_cls} classes, "
                  f"mean {sum(per_class)/6:.3f}{extra}")
            if composers:
                ranked = sorted(book.items(), key=lambda t: -t[1])
                print("    vote book: " +
                      ", ".join(f"{p} {v:.2f}" for p, v in ranked))
        for c in range(6):
            print(f"  {NAMES[c]:<9s} {acc[c]:.3f}")
        print(f"  mean      {acc.mean():.3f}")
        if composers:
            print(f"  noise-pair spawns across seeds: {noise_spawns}")


if __name__ == "__main__":
    main()
