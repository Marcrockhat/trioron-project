"""Run the self-correcting growth loop (philosophy A) on the capacity-hard taxonomy.

This is where the growth mechanism finally has real work: the standing council
underfits (≈0.775 vs Bayes 0.937), the gap is capacity-recoverable, and DENDRITE
decisively beats LINEAR under plain CE — so the cheap vote can pick correctly and the
loop should climb the gap by spawning dendrites.

Run: python3 -m experiments.progenitor.run_hard
"""
from __future__ import annotations

import torch

from .data_hard import make_split, bayes_accuracy
from .step3_council import build_council
from .step4_grow import grow_council_frustration


def main() -> None:
    tr, te = make_split()                       # canonical hard config (K=32, D=12, M=3)
    print("Step 4 — multi-phenotype frustration loop on the capacity-hard taxonomy\n")
    torch.manual_seed(0)
    sub = build_council(n_in=tr.x.shape[1], n_out=len(te.names))
    grow_council_frustration(sub, tr, te, te.names, bayes_accuracy(te))   # escalates phenotype on stall


if __name__ == "__main__":
    main()
