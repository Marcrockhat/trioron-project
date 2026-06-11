"""Capacity-sufficiency check (s028): did growth grow ENOUGH cells?

Question (Rocky): the readout is 32 outputs, genesis gives the input layer, and growth
added hidden cells — but is the grown hidden capacity *enough* to solve the task if we
simply TRAIN LONGER? Grow the organism, FREEZE its structure (no more growth), then train
the frozen net for many more steps and watch where accuracy asymptotes relative to Bayes.

  • asymptote ≈ Bayes  → capacity is sufficient; the growth-loop's lower number was just
    under-training (the gap is OPTIMISATION, and frustration/Numa-Mima is about efficiency).
  • asymptote ≪ Bayes  → not enough cells / wrong structure; capacity is the bottleneck and
    must be fixed before any frustration-signal redesign.

Run: python3 -m experiments.progenitor.run_capacity_check
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .data_hard import make_split, bayes_accuracy
from .step5_branch import build_base, grow_class_driven
from .step3c_council_decides import measure, LR


def main() -> None:
    tr, te = make_split()
    bayes = bayes_accuracy(te)
    C = len(te.names)
    print("Capacity check — grow, freeze, then train long\n")

    torch.manual_seed(0)
    sub = build_base(n_in=tr.x.shape[1], n_out=C)
    grow_class_driven(sub, tr, te, te.names, bayes, max_events=80)

    # ── freeze structure; train the frozen net much longer ──
    n_cells = int(sub.arena.alive[:sub.arena.cursor].sum())
    print(f"\n  frozen structure: {n_cells} cells, {sub.arena.edge_cursor} edges")
    print(f"  training the FROZEN net for 6000 more steps (no growth)\n")
    sub.prepare_training()
    opt = torch.optim.Adam(sub.trainable_tensors(), lr=LR)

    def acc():
        return measure(sub, te, te.names)[2]

    start = acc()
    print(f"    step    0 → overall {start:.3f}")
    for k in range(1, 6001):
        opt.zero_grad()
        F.cross_entropy(sub(tr.x), tr.y).backward()
        opt.step()
        if k % 500 == 0:
            print(f"    step {k:>4} → overall {acc():.3f}")

    final = acc()
    # per-class recall to see WHICH classes the frozen net can/can't reach
    recall = measure(sub, te, te.names)[0]
    solved = sum(1 for r in recall if r >= 0.8)
    dead = sum(1 for r in recall if r < 0.2)
    print(f"\n  ── verdict ──")
    print(f"    grown→frozen long-train asymptote: {final:.3f}   (Bayes {bayes:.3f}, "
          f"gap {bayes - final:+.3f})")
    print(f"    per-class recall ≥0.8: {solved}/{C}   <0.2: {dead}/{C}")
    print(f"    → {'CAPACITY SUFFICIENT (gap is optimisation)' if bayes - final < 0.03 else 'CAPACITY-LIMITED (more/better cells needed)'}")


if __name__ == "__main__":
    main()
