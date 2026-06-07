"""Dream-phase boundary consolidation with a dendrite — Rocky's two-mechanism design.

Separates the two memory systems we untangled:
  - manifold (photographic memory): per-class (μ,Σ) Gaussian — what each species looks like.
    (In this input-space setting the native ManifoldArchive reduces to exactly this, so we
     use a clean per-class Gaussian; it is faithful to what the astrocyte stores here.)
  - dendrite (boundary judgment): a quad nonlinearity σ(z)=z+z² with capacity to carve the
    overlap boundary. Holds no data — its WEIGHTS encode the boundary.

Loop (the architectural point):
  WAKE  : learn the new species only (no replay) + store its photographic memory.
  DREAM : replay ALL species together from the manifold, offline, and train — the dendrite
          consolidates the boundaries between overlapping species (dog↔goat, duck↔chicken),
          which can only form once both are co-present, i.e. at dream time.

Win condition (from the joint diagnostic): hit the Bayes operating point in CL —
dog ≈ 0.68 AND goat ≈ 0.86 together (not overshoot), and recover duck ≈ 0.76.

Run: python3 -m experiments.growth_exercise.cl_dream_dendrite
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.growth_exercise.taxonomy import make_taxonomy, ALL

WAKE_STEPS = 150
DREAM_STEPS = 200
REPLAY_PER_CLASS = 64
LR = 0.03


class Net(nn.Module):
    """Linear heads, optionally fed through a quad-dendrite hidden layer (capacity)."""
    def __init__(self, dendrite: bool, h: int = 24):
        super().__init__()
        self.dendrite = dendrite
        if dendrite:
            self.l1 = nn.Linear(7, h)
            self.head = nn.Linear(h, 10)
        else:
            self.head = nn.Linear(7, 10)

    def forward(self, x):
        if self.dendrite:
            z = self.l1(x)
            return self.head(z + z * z)      # σ(z)=z+z² — the dendrite nonlinearity
        return self.head(x)


class Manifold:
    """Photographic memory. k=0: one full-covariance Gaussian per class (single sketch).
    k>0: native StreamingMixture — K prototypes per class, so a Uniform-shaped class (dog)
    is represented by its actual spread instead of one blurry blob."""
    def __init__(self, k: int = 0):
        self.k = k
        self.mu: dict[int, torch.Tensor] = {}
        self.L: dict[int, torch.Tensor] = {}
        self.mix: dict[int, object] = {}

    def store(self, c: int, X: torch.Tensor) -> None:
        if self.k > 0:
            from trioron.learning.manifold import StreamingMixture
            mx = StreamingMixture(X.shape[1], self.k, X.device)
            mx.update(X)
            self.mix[c] = mx
        else:
            mu = X.mean(0)
            Xc = X - mu
            cov = Xc.T @ Xc / len(X) + 1e-3 * torch.eye(X.shape[1])
            self.mu[c] = mu
            self.L[c] = torch.linalg.cholesky(cov)

    def sample(self, c: int, n: int) -> torch.Tensor:
        if self.k > 0:
            return self.mix[c].sample(n)
        return self.mu[c] + torch.randn(n, self.mu[c].shape[0]) @ self.L[c].T

    @property
    def classes(self):
        return list(self.mix) if self.k > 0 else list(self.mu)


def run(train, test, *, dendrite: bool = True, dream: bool = True, mix_k: int = 6):
    # DEFAULT = the validated current-best config (s023): dendrite + dream + mixture(6)
    # → 0.860 overall (best CL result; goat calibrated to its Bayes ceiling). The OPEN
    # improvement target for next session is dog (~0.51 vs 0.682 ceiling): zero-sum overlap
    # + cross-entropy operating point. Levers to try: cost-sensitive weighting of dog's
    # overlap, or the emergent dog|goat label. Storage/capacity/consolidation are exhausted.
    net = Net(dendrite)
    man = Manifold(k=mix_k)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    for t in range(len(ALL)):
        Xc = train.X[train.y == t]; yc = train.y[train.y == t]
        # WAKE — learn the new species only, no replay
        for _ in range(WAKE_STEPS):
            opt.zero_grad(); F.cross_entropy(net(Xc), yc).backward(); opt.step()
        man.store(t, Xc)
        # DREAM — offline co-replay of all species so far; consolidate the boundaries
        if dream:
            for _ in range(DREAM_STEPS):
                xs, ys = [], []
                for c in man.classes:
                    xs.append(man.sample(c, REPLAY_PER_CLASS))
                    ys.append(torch.full((REPLAY_PER_CLASS,), c))
                X = torch.cat(xs); y = torch.cat(ys)
                opt.zero_grad(); F.cross_entropy(net(X), y).backward(); opt.step()
    with torch.no_grad():
        pred = net(test.X).argmax(1)
    pc = [float((pred[test.y == c] == c).float().mean()) for c in range(len(ALL))]
    return sum(pc) / len(pc), pc


def main() -> None:
    train = make_taxonomy(ALL, n_per_class=128, seed=0)
    test = make_taxonomy(ALL, n_per_class=512, seed=1)
    di, gi, du = ALL.index("dog"), ALL.index("goat"), ALL.index("duck")

    print("dream-phase consolidation, with and without the dendrite\n")
    print(f"{'config':28} {'overall':>8} {'dog':>6} {'goat':>6} {'duck':>6}")
    print(f"{'Bayes ceiling':28} {0.909:>8.3f} {0.682:>6.3f} {0.867:>6.3f} {0.830:>6.3f}")
    print(f"{'CL inline-replay (prior)':28} {0.847:>8.3f} {0.539:>6.3f} {0.828:>6.3f} {0.633:>6.3f}")
    for nm, kw in [("linear + dream", dict(dendrite=False, dream=True)),
                   ("dendrite + dream", dict(dendrite=True, dream=True)),
                   ("dendrite + dream + mix(6)", dict(dendrite=True, dream=True, mix_k=6)),
                   ("dendrite, NO dream", dict(dendrite=True, dream=False))]:
        ov, pc = run(train=train, test=test, **kw)
        print(f"{nm:28} {ov:>8.3f} {pc[di]:>6.3f} {pc[gi]:>6.3f} {pc[du]:>6.3f}")


if __name__ == "__main__":
    main()
