"""Step 4 — single-process growth, philosophy (A): plain CE, no up-weight.

Supersedes step3c's aim-ramp decision. Rocky's calls (s026):

  (A) **No up-weight.** Capacity earns its gains under plain cross-entropy and can
      never force an accuracy drop (verified: adding cells with plain CE is neutral;
      the drop we'd seen was the aim deliberately trading one class for another).
  **ONE process, no per-cell trials.** Because extra cells are harmless under (A),
      there is no gate to enforce and nothing to roll back — so growth is NOT a
      stop-start "commit → retrain → measure → keep/rollback" search. It is a single
      continuous training run that spawns cells into the LIVE network as it goes. The
      optimizer needs no rebuild: new cells occupy the same fixed-capacity weight
      tensors it already holds, so training simply flows into them.

Flow: warm up the council → spawn the trial cohort (one per phenotype) into the live
net and let it compete inside the SAME training → read the winner from readout USAGE
(which phenotype the data actually used, not a side experiment) → keep spawning the
winner into the ongoing training until overall plateaus. COUNT and phenotype are
outputs; nothing is preset.

This only has work to do on a CAPACITY-hard problem (data_hard): the easy 10-species
set sits at its plain-CE ceiling so growth correctly does nothing there.

Run: python3 -m experiments.progenitor.run_hard
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .step3_council import build_council, PHENOTYPES, _PHENO_NAME
from .step3c_council_decides import (
    snapshot, restore, commit_soma, measure, train, train_full, _soma_masks,
    TRAIN_STEPS, FINAL_STEPS, LR,
)

WARMUP = 400          # steps to settle the standing council before the first spawn
PROBE = 200           # steps the trial cohort competes in the LIVE training before we read the winner
GROW_BATCH = 4        # winner cells added per growth event
GROW_TRAIN = 120      # steps of continued training after each growth event
MAX_SOMA = 32         # safety cap on the count
PLATEAU_EPS = 0.004   # stop growing when a growth event no longer raises overall by this

# ── frustration-loop dials (s027) ──
GAP_TOL = 0.010       # frustration is "cleared" when overall is within this of Bayes
REWARD_GAIN = 2.0     # reward a phenotype's score per unit of accuracy it earns
PENALTY = 0.050       # penalty subtracted from a phenotype's score when its batch stalls


def _cohort_saliency(sub, train_d, cohort):
    """Per-cohort-cell saliency u=Σ|w·g| over its edges — the substrate's native
    utility (contribution to LOSS REDUCTION, not weight size). One backward pass on
    the live net. The phenotype the data actually uses to reduce loss scores highest;
    raw |w| misleads (a cell can carry big cancelling weights that don't help). This
    IS the decision, read from the one training — no side experiment."""
    a = sub.arena
    a.edge_weight.grad = None
    F.cross_entropy(sub(train_d.x), train_d.y).backward()
    g = a.edge_weight.grad
    w = a.edge_weight.detach()
    ec = a.edge_cursor
    src, dst = a.edge_src[:ec], a.edge_dst[:ec]
    out = {}
    for ph, cid in cohort.items():
        if cid is None:
            continue
        m = (src == cid) | (dst == cid)
        out[ph] = float((w[:ec][m] * g[:ec][m]).abs().sum())
    return out


def grow_council(sub, train_d, test_d, names, bayes_overall):
    """ONE continuous training process that grows. No per-cell trials, no gate, no
    rollback (philosophy A: extra cells are harmless, so none of that is needed). The
    council's cohort competes inside the single training; the winner is read from
    usage; more winners are spawned into the same ongoing optimization until plateau."""
    sub.prepare_training()
    opt = torch.optim.Adam(sub.trainable_tensors(), lr=LR)

    def step():
        opt.zero_grad()
        F.cross_entropy(sub(train_d.x), train_d.y).backward()
        opt.step()

    def acc():
        return measure(sub, test_d, names)[2]

    # ── warm up the standing council ──
    for _ in range(WARMUP):
        step()
    base = acc()
    print(f"  standing council: overall {base:.3f}  "
          f"(Bayes {bayes_overall:.3f}, gap {bayes_overall - base:.3f})\n")

    # ── spawn the trial cohort INTO the live net; let it compete in the SAME training ──
    cohort = {ph: commit_soma(sub, ph) for ph in PHENOTYPES}
    sub.compile()                              # new topology; opt keeps going (same tensors)
    for _ in range(PROBE):
        step()
    usage = _cohort_saliency(sub, train_d, cohort)
    winner = max(usage, key=usage.get)
    ustr = " ".join(f"{_PHENO_NAME[p][:4]}={usage[p]:.3f}" for p in
                    sorted(usage, key=lambda p: -usage[p]))
    print(f"  cohort competed in-training; saliency |w·g| [{ustr}]")
    print(f"  → the data uses {_PHENO_NAME[winner].upper()}; growing it\n")

    # ── keep spawning the winner into the same ongoing training until plateau ──
    n_soma = len(cohort)
    best = acc()
    while n_soma < MAX_SOMA:
        for _ in range(GROW_BATCH):
            commit_soma(sub, winner); n_soma += 1
        sub.compile()
        for _ in range(GROW_TRAIN):
            step()
        m = acc()
        print(f"  +{GROW_BATCH} {_PHENO_NAME[winner]} (total soma {n_soma}) → overall {m:.3f}")
        if m <= best + PLATEAU_EPS:
            break
        best = m

    final = acc()
    print(f"\n  grown: {n_soma} soma (cohort 5 + {n_soma - 5} {_PHENO_NAME[winner]}), "
          f"ONE training process")
    print(f"  overall {base:.3f} → {final:.3f}  (Bayes {bayes_overall:.3f}; "
          f"closed {final - base:.3f} of {bayes_overall - base:.3f} gap)")
    return n_soma, winner, final


def grow_council_frustration(sub, train_d, test_d, names, bayes_overall):
    """Multi-phenotype frustration loop (Rocky, s027).

    ``grow_council`` commits to ONE phenotype from a single vote and stops when that
    phenotype plateaus — it cannot recover from a wrong pick (the GCU run exposed this:
    the thin-margin vote flipped to ATTENTION, attention stalled, and the loop gave up
    with the gap wide open). This version treats a stall as *unrelieved frustration*:

      • each phenotype carries a REWARD SCORE, seeded from cohort saliency;
      • spawn a batch of the best-scoring phenotype into the live net;
      • if overall rises → REWARD that phenotype (keep investing in what works);
      • if it stalls → PENALISE that phenotype, EXHAUST it, and re-select the next-best
        type (spawn a DIFFERENT cell when the current one stops earning);
      • stop only when the gap to Bayes clears, every phenotype is exhausted, or the
        soma cap is hit — not when the *first* phenotype plateaus.
    """
    sub.prepare_training()
    opt = torch.optim.Adam(sub.trainable_tensors(), lr=LR)

    def step():
        opt.zero_grad()
        F.cross_entropy(sub(train_d.x), train_d.y).backward()
        opt.step()

    def acc():
        return measure(sub, test_d, names)[2]

    # ── warm up the standing council ──
    for _ in range(WARMUP):
        step()
    base = acc()
    print(f"  standing council: overall {base:.3f}  "
          f"(Bayes {bayes_overall:.3f}, gap {bayes_overall - base:.3f})\n")

    # ── spawn the trial cohort; probe; seed per-phenotype reward from saliency ──
    cohort = {ph: commit_soma(sub, ph) for ph in PHENOTYPES}
    sub.compile()
    for _ in range(PROBE):
        step()
    reward = _cohort_saliency(sub, train_d, cohort)        # initial preference = |w·g| usage
    ustr = " ".join(f"{_PHENO_NAME[p][:4]}={reward[p]:.3f}" for p in
                    sorted(reward, key=lambda p: -reward[p]))
    print(f"  cohort competed in-training; initial reward |w·g| [{ustr}]\n")

    n_soma = len(cohort)
    best = acc()
    exhausted: set[int] = set()
    grown = {ph: 0 for ph in PHENOTYPES}

    # ── frustration loop: spawn until the gap clears or every type is exhausted ──
    while (n_soma < MAX_SOMA and (bayes_overall - best) > GAP_TOL
           and len(exhausted) < len(reward)):
        live = [p for p in reward if p not in exhausted]
        winner = max(live, key=lambda p: reward[p])

        for _ in range(GROW_BATCH):
            commit_soma(sub, winner); n_soma += 1
        grown[winner] += GROW_BATCH
        sub.compile()
        for _ in range(GROW_TRAIN):
            step()
        m = acc()
        delta = m - best

        if delta > PLATEAU_EPS:
            reward[winner] += REWARD_GAIN * delta
            best = m
            tag = f"reward +{REWARD_GAIN * delta:.3f}; keep growing it"
        else:
            reward[winner] -= PENALTY
            exhausted.add(winner)
            tag = f"penalty -{PENALTY:.3f} → exhaust; escalate phenotype"

        print(f"  +{GROW_BATCH} {_PHENO_NAME[winner]:>9} (soma {n_soma}) "
              f"→ overall {m:.3f}  Δ{delta:+.3f}  [{tag}]")

    final = acc()
    why = ("gap cleared" if (bayes_overall - best) <= GAP_TOL else
           "soma cap" if n_soma >= MAX_SOMA else "all phenotypes exhausted")
    mix = ", ".join(f"{grown[p]} {_PHENO_NAME[p]}" for p in PHENOTYPES if grown[p]) or "none"
    print(f"\n  grown mix: cohort 5 + [{mix}]   (stopped: {why})")
    print(f"  overall {base:.3f} → {final:.3f}  (Bayes {bayes_overall:.3f}; "
          f"closed {final - base:.3f} of {bayes_overall - base:.3f} gap)")
    return n_soma, grown, final
