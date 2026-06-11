"""Per-class accuracy breakdown for the I5 architecture test (one-shot +
dead-3), reusing the gate's discover/one_shot_eval unchanged — predictions
are per-sample, so evaluating one class's test subset at a time yields the
same predictions as the full batch.

Run: python3 -m experiments.progenitor.run_i5_perclass
"""
from __future__ import annotations

import torch

from .data_hard import make_spec, sample
from .run_i5_architecture import (K_DEAD, N_TEST, S_PERIOD, SEEDS,
                                  discover, one_shot_eval)


def main() -> None:
    per_class: dict[int, list[float]] = {}
    per_class_dead: dict[int, list[float]] = {}
    names = None

    for seed in range(SEEDS):
        spec = make_spec()
        tr, norm = sample(spec, n_per_class=S_PERIOD, seed=seed)
        te, _ = sample(spec, n_per_class=N_TEST, seed=100 + seed, norm=norm)
        n_class = len(tr.names)
        names = tr.names

        sub, ctrl, mapping = discover(tr.x, tr.y, n_class)

        clean = [c for c in range(n_class) if float(spec.std[c]) <= 0.3]
        g = torch.Generator().manual_seed(seed)
        dead = torch.randperm(12, generator=g)[:K_DEAD].tolist()

        print(f"\nseed={seed}  ({len(clean)} clean classes; dead sensors {dead})")
        accs, accs_d = [], []
        for c in clean:
            keep = te.y == c
            a = one_shot_eval(sub, ctrl, mapping, te.x[keep], te.y[keep])
            ad = one_shot_eval(sub, ctrl, mapping, te.x[keep], te.y[keep],
                               dead=dead)
            per_class.setdefault(c, []).append(a)
            per_class_dead.setdefault(c, []).append(ad)
            accs.append(a)
            accs_d.append(ad)
            print(f"  class {c:2d} {tr.names[c]:<8s} one-shot {a:.2f}  "
                  f"dead-{K_DEAD} {ad:.2f}")
        mean = sum(accs) / len(accs)
        mean_d = sum(accs_d) / len(accs_d)
        print(f"  mean one-shot {mean:.3f}  dead-{K_DEAD} {mean_d:.3f}  "
              f"(gate reports the same population)")

    print(f"\n── per-class means over {SEEDS} seeds ──")
    for c in sorted(per_class):
        xs, xds = per_class[c], per_class_dead[c]
        print(f"  class {c:2d} {names[c]:<8s} one-shot "
              f"{sum(xs)/len(xs):.3f}  dead-{K_DEAD} {sum(xds)/len(xds):.3f}")


if __name__ == "__main__":
    main()
