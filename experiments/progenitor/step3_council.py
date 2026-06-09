"""Step 3 — the standing 5×4 council, wired with all five GENUINE phenotypes.

Step 1 established the floor (1 perception → 6 linear outputs): overall ~0.807,
but **dog collapses to ~0.324** because a linear readout cannot carve dog's
disjoint band (dog = Uniform(2,85) overlaps cat below and goat above; on the
log-weight axis the Bayes-optimal dog region is a *middle interval*, not a
half-line). Closing that gap is the council's whole job (design §3.3–3.4).

The council is the **standing decision organ**: 5 phenotypes × 4 cells = 20, even
multiplicity so a split vote is a legitimate "undecided" the data must topple, and
never pruned (germline — §3.1/§3.3). The phenotypes are the five expression genes,
and as of this build all five dispatch a GENUINE forward op:

  LINEAR     y = b + Σ w·a
  ATTENTION  softmax over fan-in tokens         → 1 token  ⇒ linear  (reduction)
  CONV       kernel shared via lineage_root      → own root ⇒ linear  (reduction)
  RECURRENT  self/lateral back-edge unroll       → no loop  ⇒ linear  (reduction)
  DENDRITE   per-branch quad σ(z)=z+z²           → K≥2      ⇒ nonlinear

On a single static scalar feature the four non-dendrite phenotypes are linear by
the structure of the input (proven in tests/test_{recurrent,attention,conv}_
phenotype.py), so the trial-vote should topple toward DENDRITE for dog — the one
op that adds the curvature a middle-band decision needs. The other phenotypes
become live differentiators only in the multi-data-type phase, when there is
temporal / multi-token / spatial structure for them to win.

DENDRITE on a 1-D input: a genuine 2-branch quad needs two edges from the
perception cell assigned to *different* branches. ``grow_branch`` partitions
fan-in by SOURCE cell, so it cannot split two same-source edges — we assign the
branch per-edge by hand here (a noted substrate limitation: grow_branch has no
single-source split).

This file is the **connected scaffold only** (3a): the standing organism
``perception → 20 council cells → outputs``, every node and edge visible before
training. The council's *decision* (which phenotype, how many cells) must run IN
PLACE on this organism — the trial-vote and spawn happen here via the live
substrate (`grow_branch`/`divide`), driven by local frustration, looping until the
frustration clears. That decision loop is the next thing to build; earlier offline
per-phenotype simulations were the wrong approach (memory
``feedback_improve_not_rebuild``) and are not carried here.

Run: python3 -m experiments.progenitor.step3_council
"""
from __future__ import annotations

import torch

from trioron.core import Envelope, construct
from trioron.core.epigenome import (
    LINEAR, ATTENTION, CONV, RECURRENT, DENDRITE,
    PERCEPTION, OUTPUT, set_gene, clear_gene,
)
from trioron.phenotype import default_dispatch_table

from .data import make_data, bayes_accuracy, bayes_per_class
from .positions import sensor_positions

# ── Council layout (design §3.3) ──────────────────────────────────
PHENOTYPES: list[int] = [LINEAR, ATTENTION, CONV, RECURRENT, DENDRITE]
CELLS_PER_PHENOTYPE: int = 4          # even multiplicity → split vote = "undecided"
_PHENO_NAME = {LINEAR: "linear", ATTENTION: "attention", CONV: "conv",
               RECURRENT: "recurrent", DENDRITE: "dendrite"}
SEED = 0xC0FFEE


def build_council(n_in: int = 1, n_out: int = 6):
    """Hand-wire the standing council:  perception → 20 council cells → outputs.

    Every council cell is a *trial daughter* of one phenotype, fed by the
    perception cell and feeding every output. DENDRITE cells get a second
    perception edge on branch 1 (K=2) so the quad engages; the other four keep a
    single linear-equivalent fan-in (K=1).

    Council cells are **germline**: CREDIT_ELIGIBLE is left unset so downstream
    consolidation / credit-locking never freezes the decision organ (§3.1).
    """
    def base(sub):
        a = sub.arena
        # 1. perception cell(s) at rank 0
        perc = a.alloc(n_in)
        ppos = sensor_positions(n_in, seed=SEED, dim=1)
        for k, p in enumerate(perc.tolist()):
            a.epigenome[p] = set_gene(int(a.epigenome[p].item()), PERCEPTION)
            a.rank[p] = 0
            a.position[p] = ppos[k]

        # 2. the 5×4 council at rank 1 — one trial daughter per phenotype
        council: dict[int, list[int]] = {ph: [] for ph in PHENOTYPES}
        for ph in PHENOTYPES:
            ids = a.alloc(CELLS_PER_PHENOTYPE)
            for cid in ids.tolist():
                epi = clear_gene(int(a.epigenome[cid].item()), LINEAR)
                a.epigenome[cid] = set_gene(epi, ph)
                a.rank[cid] = 1
                council[ph].append(cid)

        # 3. output cells at rank 2
        outs = a.alloc(n_out)
        for o in outs.tolist():
            a.epigenome[o] = set_gene(int(a.epigenome[o].item()), OUTPUT)
            a.rank[o] = 2
        a.refresh_all_phenotypes()

        # 4. wiring — perception → every council cell → every output
        all_council = [c for ids in council.values() for c in ids]
        src = perc.repeat(len(all_council))
        dst = torch.tensor(all_council, dtype=torch.int32).repeat_interleave(n_in)
        a.add_edges(src, dst)                                   # 1 perc edge per council cell (branch 0)
        cc = torch.tensor(all_council, dtype=torch.int32)
        a.add_edges(cc.repeat(n_out), outs.repeat_interleave(len(all_council)))  # council → outputs

        # 5. DENDRITE cells → genuine K=2: add a 2nd perception edge on branch 1.
        #    grow_branch splits by source and can't separate two same-source edges,
        #    so assign the branch per-edge by hand (noted limitation).
        for cid in council[DENDRITE]:
            e0 = a.edge_cursor
            a.add_edges(perc, torch.tensor([cid], dtype=torch.int32).repeat(n_in))
            a.edge_branch[e0:a.edge_cursor] = 1                 # the new edge(s) → branch 1
            a.n_branches[cid] = 2
            with torch.no_grad():
                a.branch_alpha[cid, 1] = 1.0
            a.refresh_phenotype(cid)

        base.perc = perc.tolist()
        base.council = council
        base.outs = outs.tolist()

    sub = construct(base=base, envelope=Envelope(),
                    dispatch_table=default_dispatch_table(), capacity=64)
    sub.perc = base.perc
    sub.council = base.council
    sub.outs = base.outs
    return sub


def print_structure(sub) -> None:
    a = sub.arena
    print("  council nodes (5 phenotypes × 4 cells = 20, germline / never-freeze):")
    for ph in PHENOTYPES:
        cells = sub.council[ph]
        ks = [int(a.n_branches[c].item()) for c in cells]
        fan = [int(a.fan_in(c)) for c in cells]
        note = ("quad (K=2)" if ph == DENDRITE else "linear-equiv on 1-D")
        print(f"    {_PHENO_NAME[ph]:>9}  cells {cells}  fan_in={fan}  K={ks}  → {note}")
    print(f"  perception: {sub.perc} (rank 0)    outputs: {sub.outs} (rank 2)")
    print(f"  edges: {a.edge_cursor} total "
          f"= perc→council ({len([c for v in sub.council.values() for c in v]) + len(sub.council[DENDRITE])}) "
          f"+ council→out ({20 * len(sub.outs)})")
    for cid in sub.council[DENDRITE]:
        m = (a.edge_dst[:a.edge_cursor] == cid)
        br = a.edge_branch[:a.edge_cursor][m].tolist()
        print(f"    dendrite cell {cid}: incoming-edge branches = {br}  "
              f"(quad engaged = {int(a.n_branches[cid].item()) >= 2})")


def main() -> None:
    test_d = make_data(seed=1, n_per_class=512)
    sub = build_council()

    print("Step 3a — council scaffold wired with all 5 genuine phenotypes "
          "(structure only, no training)\n")
    print_structure(sub)

    # Sanity: with no training, every council cell except dendrite is linear, so a
    # fresh forward should not error and outputs should be finite.
    with torch.no_grad():
        y = sub(test_d.x)
    print(f"\n  forward OK: logits {tuple(y.shape)}, finite={bool(torch.isfinite(y).all())}")

    print("\n  Bayes ceiling (per class):")
    bpc = bayes_per_class(test_d)
    for ci, name in enumerate(test_d.names):
        flag = "  <- the disjoint band the council must carve" if name == "dog" else ""
        print(f"    {name:>8}  bayes {bpc[ci]:.3f}{flag}")
    print(f"    {'OVERALL':>8}  bayes {bayes_accuracy(test_d):.3f}")
    print("\n  (next: build the council's IN-PLACE decision on THIS organism — "
          "detect local\n   frustration, run the trial-vote + spawn via the live "
          "substrate until it clears.)")


if __name__ == "__main__":
    main()
