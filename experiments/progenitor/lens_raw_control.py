"""The missing baseline: raw pixels vs the lens on the TOY (s042 falsification).

The phasor lens's keystone number (toy identity ~0.98) was NEVER compared to a
raw-pixel baseline — it was a confirming result that skipped scrutiny. Add the
baseline: nearest-centroid on raw 7x5 pixels vs the 2x2 lens descriptor, same
registered clean variants. If raw matches or beats the lens, the lens added no
value and only LOOKED good because the toy is trivially easy.

Pairs with mnist_lens.py (real images): there raw pixels beat the lens with both
back-ends (0.815/0.885 vs 0.710/0.648). This script closes the loop on the toy.

Run:  python3 experiments/progenitor/lens_raw_control.py
"""
from __future__ import annotations

import numpy as np
import torch

from experiments.progenitor import digit_bench_2d as DB


def main():
    digits = [str(i) for i in range(10)]
    gtr = torch.Generator().manual_seed(3)
    gte = torch.Generator().manual_seed(99)

    raw_c = {ch: np.mean([DB.variant(ch, gtr).reshape(-1).numpy() for _ in range(60)], 0)
             for ch in digits}
    lens_c = {ch: np.mean([DB.descriptor(DB.variant(ch, gtr)) for _ in range(60)], 0)
              for ch in digits}

    def acc(cent, feat):
        ok = 0
        for k in range(600):
            ch = digits[k % 10]
            f = feat(DB.variant(ch, gte))
            pred = min(digits, key=lambda c: np.linalg.norm(f - cent[c]))
            ok += int(pred == ch)
        return ok / 600

    raw = acc(raw_c, lambda v: v.reshape(-1).numpy())
    lens = acc(lens_c, DB.descriptor)
    print("TOY 7x5 registered digits, nearest centroid:")
    print(f"  raw pixels (35-d) = {raw:.3f}")
    print(f"  lens 2x2  (48-d)  = {lens:.3f}")
    print(f"\n  -> raw {'>=' if raw >= lens else '<'} lens: the lens adds "
          f"{'NO value (lossy pooling on an easy task)' if raw >= lens else 'value'}.")


if __name__ == "__main__":
    main()
