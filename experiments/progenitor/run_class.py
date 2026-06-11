"""Run class-driven growth (s028) on the capacity-hard taxonomy.

Growth follows per-class frustration: the worst class recruits capacity wired to its
own evidence — WIDEN (new linear column) while linear width helps it, then DEEPEN (GCU)
on the column most responsible for it. Tests whether depth DISTRIBUTES across columns
(vs piling into one global sink) and whether class-targeted growth beats the phase-split
loop's 0.484.

Run: python3 -m experiments.progenitor.run_class
"""
from __future__ import annotations

import torch

from .data_hard import make_split, bayes_accuracy
from .step5_branch import build_base, grow_class_driven


def main() -> None:
    tr, te = make_split()
    print("Step 5 — class-driven growth on the capacity-hard taxonomy\n")
    torch.manual_seed(0)
    sub = build_base(n_in=tr.x.shape[1], n_out=len(te.names))
    grow_class_driven(sub, tr, te, te.names, bayes_accuracy(te), max_events=80)


if __name__ == "__main__":
    main()
