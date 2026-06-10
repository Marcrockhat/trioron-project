"""Run branch-id thin-chain growth (s027 design) on the capacity-hard taxonomy.

First experiment for the locality-indexed architecture: thin columns addressed by hard
branch_id, lateral=width / axial=depth, sparse top-reading readout, per-branch
frustration. Tests three things at once:
  • does width saturate on its own (Phase 1 stops)?
  • does DEPTH FIRE (Phase 2 deepens columns)? — the recurring fear
  • does depth add anything beyond width?

Run: python3 -m experiments.progenitor.run_branch
"""
from __future__ import annotations

import torch

from .data_hard import make_split, bayes_accuracy
from .step5_branch import build_base, grow_branches


def main() -> None:
    tr, te = make_split()
    print("Step 5 — branch-id thin-chain growth on the capacity-hard taxonomy\n")
    torch.manual_seed(0)
    sub = build_base(n_in=tr.x.shape[1], n_out=len(te.names))
    grow_branches(sub, tr, te, te.names, bayes_accuracy(te))


if __name__ == "__main__":
    main()
