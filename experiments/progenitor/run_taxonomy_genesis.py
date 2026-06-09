"""Genesis → council, end-to-end on the 10-species taxonomy (design-faithful).

The perception layer is GROWN, not given: a wide aperture of candidate sensors is
culled by apoptosis (variance) then saliency (|w·g|) down to the real features, and
the council is attached to exactly those survivors. This wires the genesis stage in
front of the council-decision loop (memory ``progenitor_council_connected_flow``).

Run: python3 -m experiments.progenitor.run_taxonomy_genesis
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .data_taxonomy import make_taxonomy, bayes_accuracy, bayes_per_class
from .genesis import build_aperture, discover_input_layer
from .step3c_council_decides import run_decision

APERTURE_N = 64      # candidate sensors proposed for a 7-feature problem (mostly empty)
N_NOISE = 5          # variance-bearing but uninformative columns planted in the aperture
GENESIS_SEED = 1


@dataclass
class Bundle:
    x: torch.Tensor
    y: torch.Tensor
    names: list[str]


def main() -> None:
    tr = make_taxonomy(seed=0, n_per_class=256)
    te = make_taxonomy(seed=1, n_per_class=512)
    n_out = len(te.names)

    # ── Aperture: plant the 7 real feature columns + noise among 64 candidates ──
    X_tr, real_cols, noise_cols = build_aperture(tr.x, APERTURE_N, N_NOISE, seed=GENESIS_SEED)
    X_te, _, _ = build_aperture(te.x, APERTURE_N, N_NOISE, seed=GENESIS_SEED)  # same placement

    # ── Genesis: apoptosis → saliency → the grown input layer ──
    gen = discover_input_layer(X_tr, tr.y, n_out, real_cols, noise_cols, seed=0)

    print("GENESIS — growing the input layer in front of the council\n")
    print(f"  aperture proposed : {gen.aperture_n} candidate sensors "
          f"({len(real_cols)} real features + {len(noise_cols)} noise + "
          f"{gen.aperture_n - len(real_cols) - len(noise_cols)} empty)")
    print(f"  Phase 1 apoptosis : {gen.aperture_n - len(gen.survived_variance)} empty "
          f"(zero-variance) cells recycled in ONE pass (no training) → "
          f"{len(gen.survived_variance)} survive")
    from .genesis import SAL_FRAC
    keptset, realset, noiseset = set(gen.kept), set(real_cols), set(noise_cols)
    print(f"  Phase 2 saliency  : u=|w·g| on survivors (keep ≥ {SAL_FRAC:.0%} of max):")
    for k, c in enumerate(gen.survived_variance):
        role = "REAL " if c in realset else ("noise" if c in noiseset else "empty?")
        verdict = "KEEP" if c in keptset else "recycle"
        print(f"      col {c:>2} ({role}) u={gen.sal[k]:.4f}  {verdict}")
    recovered = keptset == realset
    print(f"\n  → input layer GROWN to {len(gen.kept)} cells: {gen.kept}")
    print(f"    recovered the {len(real_cols)} real features exactly: {recovered}"
          + ("" if recovered else f"  (missed {sorted(realset-keptset)}, "
             f"extra {sorted(keptset-realset)})"))
    print("\n" + "=" * 60 + "\n")

    # ── Council decision on the grown input layer (the SAME validated loop) ──
    train_d = Bundle(X_tr[:, gen.kept], tr.y, te.names)
    test_d = Bundle(X_te[:, gen.kept], te.y, te.names)
    run_decision(train_d, test_d, bayes_per_class(te),
                 n_in=len(gen.kept), n_out=n_out, bayes_overall=bayes_accuracy(te))


if __name__ == "__main__":
    main()
