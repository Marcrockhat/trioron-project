"""De-risk: composer-phenotype growth from FRUSTRATED stress (NEXT 6) on the
mixed-stream division regime — grow, adapt, prune.

Task built so division alone is structurally stuck: 6 classes in 4 dims
(f2/f3 noise). C/D are axis Gaussians (division suffices). A (f1=f0) and
B (f1=1-f0) share identical uniform marginals — only the linear RELATION
differs. E/F are concentric rings (radius 0.40 / 0.22) — only the
QUADRATIC relation differs. No raw dim separates A|B or E|F.

Mechanism: at a sitting, a class whose division was REJECTED while its
buffer stays incoherent (meanR below trigger) holds a composer TRIAL:
every phenotype gene proposes its best candidate dim (a fixed composition
over the two least-coherent informative dims; sum and diff wirings) —
overproduction. The candidate that makes the buffer cohere best is
SPAWNED as a real dim for the whole organism; the rest are never built.
Settlement: if a later division splits ON the spawned dim, the winning
phenotype group takes one vote from the others (step4_grow transfer,
floors, sum conserved); a spawned dim that no class coheres on within
PATIENCE sittings is PRUNED and the bet is paid back.

Run: python3 -m experiments.progenitor.run_composer_growth
"""
from __future__ import annotations

import math
from collections import Counter

import torch

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
NULL_SPLIT = 0.72      # null statistic: halving a uniform dim yields 2/pi=0.64
TRIGGER_R = 0.55       # buffer meanR below this + division rejected -> trial
SPAWN_R = 0.80         # best candidate's carved child must cohere this much
PATIENCE = 3           # sittings for a spawned dim to earn its keep
MAX_SPAWNED = 6
DIV_TRIES = 4          # dims attempted per sitting, ascending coherence
NAMES = ["A:diag", "B:anti", "C:blob1", "D:blob2", "E:ring.40", "F:ring.22"]

PHENOTYPES = {
    "linear": lambda z: z,
    "quad":   lambda z: z + z * z,
    "tanh":   lambda z: torch.tanh(2.5 * z),
    "gcos":   lambda z: torch.cos(math.pi * z),
    "sinc":   lambda z: torch.sinc(2 * z),
    "tan":    lambda z: torch.tan(1.2 * z).clamp(-3, 3),
}
WIRINGS = {"sum": (1.0, 1.0), "diff": (1.0, -1.0)}
SPAWN_GAIN = 0.15      # carve R must beat the permutation null by this


def candidates(i: int, j: int):
    """All composer candidates over a dim pair: each phenotype on the
    sum/diff pre-activation, plus the dendritic quad pair (two branches,
    one input each, quadratic sigma — the Phase-6 ring form)."""
    for pname in PHENOTYPES:
        for wts in WIRINGS.values():
            yield (pname, i, j, wts)
    yield ("quad", i, j, "dendritic")


def make_data(n: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    X, y = [], []
    for c in range(6):
        t = torch.rand(n, generator=g)
        eps = torch.randn(n, 2, generator=g) * 0.02
        if c == 0:   f0, f1 = t, t + eps[:, 0]                       # diagonal
        elif c == 1: f0, f1 = t, 1 - t + eps[:, 0]                   # anti-diag
        elif c == 2: f0, f1 = 0.30 + eps[:, 0], 0.70 + eps[:, 1]     # blob
        elif c == 3: f0, f1 = 0.78 + eps[:, 0], 0.25 + eps[:, 1]     # blob
        else:
            r = 0.40 if c == 4 else 0.22
            phi = 2 * math.pi * torch.rand(n, generator=g)
            f0 = 0.5 + r * torch.cos(phi) + eps[:, 0]
            f1 = 0.5 + r * torch.sin(phi) + eps[:, 1]
        noise = torch.rand(n, 2, generator=g)
        X.append(torch.stack([f0, f1, noise[:, 0], noise[:, 1]], 1))
        y.append(torch.full((n,), c))
    X, y = torch.cat(X).clamp(0, 1), torch.cat(y)
    o = torch.randperm(len(X), generator=g)
    return X[o], y[o]


def apply_dim(X: torch.Tensor, spec) -> torch.Tensor:
    name, i, j, w = spec
    ci, cj = X[:, i] - 0.5, X[:, j] - 0.5
    if w == "dendritic":
        v = ci * ci + cj * cj
    else:
        v = PHENOTYPES[name](w[0] * ci + w[1] * cj)
    return (v - v.min()) / (v.max() - v.min() + 1e-9)


def phasors(X: torch.Tensor, dims: list) -> torch.Tensor:
    cols = [X.t()] + [apply_dim(X, s).unsqueeze(0) for s in dims]
    q = (torch.cat(cols).t() * (N_QUANTA - 2) + 1).clamp(1, N_QUANTA - 1)
    return torch.exp(1j * 2 * math.pi * q / N_QUANTA)


def run_seed(seed: int, composers: bool):
    Xs, ys = make_data(S_TRAIN, seed)
    Xt, yt = make_data(S_TEST, 100 + seed)
    dims: list = []                                # spawned composer dims
    dim_age: list = []                             # sittings since spawn
    book = {p: 4.0 for p in PHENOTYPES}            # the phenotype vote book
    raw_bufs: list = []                            # per-class raw samples
    spawn_log, prune_log = [], []

    def bet(winner: str, amount: float = 1.0):
        losers = [p for p in book if p != winner]
        for p in losers:
            take = min(amount / len(losers), max(0.0, book[p] - 1.0))
            book[p] -= take
            book[winner] += take

    for w0 in range(0, len(Xs), WINDOW):
        Xw = Xs[w0:w0 + WINDOW]
        if not raw_bufs:
            raw_bufs = [Xw[-BUF:]]
            continue
        Z = phasors(Xw, dims)
        T = torch.stack([phasors(b, dims).mean(0) for b in raw_bufs])
        member = (Z.unsqueeze(1) * T.conj().unsqueeze(0)).real.sum(-1).argmax(1)
        for k in range(len(raw_bufs)):
            xk = Xw[member == k]
            if len(xk):
                raw_bufs[k] = torch.cat([raw_bufs[k], xk])[-BUF:]

        dim_age = [a + 1 for a in dim_age]
        new_bufs, used_dims = [], set()
        for b in raw_bufs:               # the sitting: divide XOR compose
            if len(b) < MIN_MEMBERS:
                new_bufs.append(b)
                continue
            Zb = phasors(b, dims)
            R = Zb.mean(0).abs()

            div_r, div_split = 0.0, None         # remedy 1: best division
            for d in range(Zb.shape[1]):
                side = circ_2means(torch.angle(Zb[:, d]))
                if int((~side).sum()) < MIN_CHILD or int(side.sum()) < MIN_CHILD:
                    continue
                r = max(float(Zb[~side, d].mean().abs()),
                        float(Zb[side, d].mean().abs()))
                # carve-out gate: the best child must beat the NULL-split
                # statistic AND the parent — slicing noise can't pass
                if r > max(float(R[d]) + GAIN_D, NULL_SPLIT) and r > div_r:
                    div_r, div_split = r, (side, d)

            comp_r, comp_spec = 0.0, None        # remedy 2: best composer
            if (composers and float(R.mean()) < TRIGGER_R
                    and len(dims) < MAX_SPAWNED):
                g = torch.Generator().manual_seed(len(spawn_log))

                def carve_r(X, spec):
                    phi = 2 * math.pi * apply_dim(X, spec)
                    side = circ_2means(phi)
                    if int((~side).sum()) < MIN_CHILD or \
                            int(side.sum()) < MIN_CHILD:
                        return 0.0
                    z = torch.exp(1j * phi)
                    return max(float(z[~side].mean().abs()),
                               float(z[side].mean().abs()))

                half = len(b) // 2
                b1, b2 = b[:half], b[half:]
                best_gain = SPAWN_GAIN
                for i in range(4):
                    for j in range(i + 1, 4):
                        for spec in candidates(i, j):
                            if spec in dims:               # no duplicates
                                continue
                            r = carve_r(b1, spec)
                            if r < SPAWN_R:
                                continue
                            # permutation null x3: shuffle col j against col
                            # i — marginals and compression survive, the
                            # RELATION dies; selection is by null-corrected
                            # gain, never by raw coherence
                            null = 0.0
                            for _ in range(3):
                                shuf = b1.clone()
                                shuf[:, j] = b1[torch.randperm(
                                    len(b1), generator=g), j]
                                null += carve_r(shuf, spec) / 3
                            if r - null <= best_gain:
                                continue
                            # split-half confirm: the candidate was SELECTED
                            # on b1; it must reproduce on the held-out b2,
                            # or 84-way selection cherry-picks null noise
                            shuf2 = b2.clone()
                            shuf2[:, j] = b2[torch.randperm(
                                len(b2), generator=g), j]
                            r2 = carve_r(b2, spec)
                            if r2 > SPAWN_R and \
                                    r2 - carve_r(shuf2, spec) > SPAWN_GAIN:
                                best_gain, comp_r, comp_spec = r - null, r, spec

            if comp_spec is not None and comp_r > div_r:   # the better bet
                dims.append(comp_spec)
                dim_age.append(0)
                spawn_log.append((comp_spec[0], comp_spec[1], comp_spec[2],
                                  round(comp_r, 2)))
                new_bufs.append(b)               # divide on it next sitting
            elif div_split is not None:
                side, d = div_split
                new_bufs += [b[~side], b[side]]
                if d >= 4:
                    used_dims.add(d - 4)
            else:
                new_bufs.append(b)
        raw_bufs = new_bufs

        keep = [i for i in range(len(dims))
                if dim_age[i] <= PATIENCE or i in used_dims]
        for i in range(len(dims)):                 # settle the bets
            if i in used_dims and dim_age[i] <= PATIENCE + 1:
                bet(dims[i][0])
        if len(keep) < len(dims):                  # prune the failures
            for i in range(len(dims)):
                if i not in keep:
                    prune_log.append(dims[i][0])
            dims = [dims[i] for i in keep]
            dim_age = [dim_age[i] for i in keep]

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
    return per_class, len(T), dims, book, spawn_log, prune_log


def main() -> None:
    for arm, composers in (("division-only", False), ("composer-growth", True)):
        print(f"\n══ {arm} ══")
        acc = torch.zeros(6)
        for seed in range(SEEDS):
            per_class, n_cls, dims, book, spawns, prunes = run_seed(seed, composers)
            acc += torch.tensor(per_class) / SEEDS
            extra = ""
            if composers:
                d = [f"{s[0]}({s[1]},{s[2]})" for s in dims]
                extra = (f"; spawned {len(spawns)}, pruned {len(prunes)}, "
                         f"kept [{', '.join(d)}]")
            print(f"  seed={seed}: {n_cls} classes, "
                  f"mean {sum(per_class)/6:.3f}{extra}")
            if composers:
                ranked = sorted(book.items(), key=lambda t: -t[1])
                print("    vote book: " +
                      ", ".join(f"{p} {v:.2f}" for p, v in ranked))
        for c in range(6):
            print(f"  {NAMES[c]:<9s} {acc[c]:.3f}")
        print(f"  mean      {acc.mean():.3f}")


if __name__ == "__main__":
    main()
