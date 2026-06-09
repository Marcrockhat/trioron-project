"""Capacity-hard procedural taxonomy — multimodal class mixtures + wide disruptors.

The 10-species set is EASY because its features are independent given the class:
independent features → naive Bayes is optimal → summing per-feature log-likelihoods
is LINEAR in the encoding, so a linear readout already reaches Bayes and the council
never needs capacity. Scaling that structure up stays linear-easy.

To give growth real work we need a CAPACITY-hard set: high Bayes (recoverable signal)
but a Bayes-optimal boundary too intricate for a fixed ~20-cell council, so plain-CE
growth measurably closes the gap. The lever here is **multimodality**: each species is
a mixture of M scattered Gaussian blobs in trait space (think morphs / sexual
dimorphism / life stages), heavily overlapping its neighbours, with a fraction of
**disruptor** species spread wide (large σ, dog-style) so they overlap everyone. The
optimal partition is then a multi-piece patchwork — linear cells can't carve it and even
K=2 quads need many of them — while the exact Bayes ceiling stays computable from the
known generative mixture (logsumexp over modes, equal priors).

CRITICAL: the generative SPEC (means/σ — the fixed "species definitions") is generated
ONCE; train and test only differ in their SAMPLE draws and share the spec (and the
standardization stats). Re-randomizing the means per split makes train/test different
distributions → even a linear model scores chance while Bayes reads 1.0. Mirror
``data_taxonomy``'s hardcoded SPECIES dict: fixed params, varying samples.

Dials (hardness ↑): n_class ↑, modes ↑, std ↑ (overlap), disruptor_frac ↑, n_feat ↓.

Run: python3 -m experiments.progenitor.data_hard
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

_LOG2PI = math.log(2 * math.pi)


@dataclass
class HardSpec:
    """The fixed generative model — the 'species definitions'. Generated once."""
    means: torch.Tensor      # (K, M, D) mode centres (raw space)
    log_w: torch.Tensor      # (K, M) log mixture weights (equal → -log M)
    std: torch.Tensor        # (K,) per-class isotropic σ (large for disruptors)
    names: list[str]


@dataclass
class HardTaxon:
    x: torch.Tensor          # (N, D) standardized NN features
    x_raw: torch.Tensor      # (N, D) raw (where the mixture is defined → Bayes)
    y: torch.Tensor          # (N,) class index
    spec: HardSpec
    names: list[str]

    def __len__(self) -> int:
        return self.x.shape[0]


# Canonical "hard taxonomy" defaults — validated capacity-recoverable: Bayes 0.937,
# standing council 0.775 (gap 0.163), and plain-CE growth closes it with DENDRITE
# decisively beating LINEAR (+0.090 vs +0.006 for 40 cells). The dataset the (A) +
# self-correcting growth loop should run on.
def make_spec(n_class: int = 32, n_feat: int = 12, modes: int = 3, std: float = 0.14,
              disruptor_frac: float = 0.15, disruptor_std: float = 0.5,
              seed: int = 0) -> HardSpec:
    """Generate the fixed species definitions (means/σ). Call ONCE; share across splits."""
    g = torch.Generator().manual_seed(seed)
    K, D, M = n_class, n_feat, modes
    means = torch.rand(K, M, D, generator=g)                 # mode centres in [0,1]^D
    stds = torch.full((K,), float(std))
    n_dis = int(round(K * disruptor_frac))
    if n_dis:
        dis = torch.randperm(K, generator=g)[:n_dis]
        stds[dis] = float(disruptor_std)                     # wide-spread disruptor species
    log_w = torch.full((K, M), -math.log(M))
    names = [f"sp{c:03d}" for c in range(K)]
    return HardSpec(means, log_w, stds, names)


def sample(spec: HardSpec, n_per_class: int = 128, seed: int = 0,
           norm: tuple | None = None) -> tuple[HardTaxon, tuple]:
    """Draw a sample from the FIXED spec. Pass the train split's *norm* to the test
    split so both standardize identically. Returns (taxon, norm)."""
    g = torch.Generator().manual_seed(seed)
    K, M, D = spec.means.shape
    xs, ys = [], []
    for c in range(K):
        mode_idx = torch.randint(0, M, (n_per_class,), generator=g)
        mu = spec.means[c, mode_idx]                         # (n, D)
        x = mu + torch.randn(n_per_class, D, generator=g) * float(spec.std[c])
        xs.append(x); ys.append(torch.full((n_per_class,), c, dtype=torch.long))
    X = torch.cat(xs); y = torch.cat(ys)
    perm = torch.randperm(y.numel(), generator=g)
    X, y = X[perm], y[perm]
    if norm is None:
        norm = (X.mean(0), X.std(0).clamp_min(1e-6))
    Xs = (X - norm[0]) / norm[1]
    return HardTaxon(Xs, X, y, spec, spec.names), norm


def make_split(n_per_train: int = 128, n_per_test: int = 256, **spec_kw):
    """Convenience: one spec, matched train/test splits sharing it + the norm."""
    spec = make_spec(**spec_kw)
    tr, norm = sample(spec, n_per_train, seed=0)
    te, _ = sample(spec, n_per_test, seed=1, norm=norm)
    return tr, te


def _log_p_x_given_c(d: HardTaxon) -> torch.Tensor:
    """(N, K) log p(x | c) under the true generative mixture (raw space)."""
    X = d.x_raw
    K, M, D = d.spec.means.shape
    out = torch.empty(len(d), K)
    for c in range(K):
        mu = d.spec.means[c]                                 # (M, D)
        s = float(d.spec.std[c])
        z = (X[:, None, :] - mu[None, :, :]) / s             # (N, M, D)
        logN = (-0.5 * (z * z) - math.log(s) - 0.5 * _LOG2PI).sum(-1)
        out[:, c] = torch.logsumexp(d.spec.log_w[c][None, :] + logN, dim=1)
    return out


def bayes_accuracy(d: HardTaxon) -> float:
    return float((_log_p_x_given_c(d).argmax(1) == d.y).float().mean())


def bayes_per_class(d: HardTaxon) -> list[float]:
    pred = _log_p_x_given_c(d).argmax(1)
    return [float((pred[d.y == c] == c).float().mean()) for c in range(len(d.names))]


def main() -> None:
    tr, te = make_split()                       # canonical hard config
    print(f"hard taxonomy (canonical): {len(te.names)} species, {te.x.shape[1]}-d, "
          f"{len(te)} test samples")
    print(f"  Bayes {bayes_accuracy(te):.3f} | standing council ≈0.775 (gap ≈0.163, "
          f"capacity-recoverable; dendrite ≫ linear under plain CE)")


if __name__ == "__main__":
    main()
