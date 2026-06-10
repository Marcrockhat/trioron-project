# Trioron Handoff

**Session date:** 2026-06-10
**Session number:** 027
**Session title:** **From council/divide to a branch-id locality architecture: GCU
dendrite, a multi-phenotype frustration loop, then a deep diagnosis (no utility, dup
edges, no depth, a confirmed tangle) that drove a full redesign — thin columns indexed
by hard branch_id, split frustration `[F_lateral, F_depth]`, sparse top-reading readout.
First experiment: width self-saturates and DEPTH FINALLY FIRES (+0.068).**

## The arc (what changed, in order)

1. **Repeated the s026 single-process run** (`run_hard.py`): reproduced 0.777→0.842,
   picks DENDRITE. Clarified for Rocky that `step4_grow` has **no frustration gate** —
   philosophy-A removed it; growth = saliency vote + plateau loop, no rollback.
2. **GCU swap** (`dendrite.py`): replaced `σ(z)=z+z²` with the Growing Cosine Unit
   `σ(z)=z·cos z` (K=1 still suppressed to linear). On the hard taxonomy GCU **lost**:
   baseline 0.777→0.752, vote flipped to ATTENTION (thin margin), final 0.776. Robust
   signal = the averaged baseline drop; the vote flip is partly the known knife-edge.
3. **Built the multi-phenotype frustration loop** (`grow_council_frustration` in
   `step4_grow.py`): per-phenotype reward score, penalty+exhaust on stall, escalate to
   next-best type. It **escalated correctly** (attention→dendrite→recurrent→linear→conv)
   but went **net-negative** (0.752→0.773): philosophy-A's "extra cells are harmless" is
   **falsified** — wrong-phenotype batches measurably *drag* (linear −0.012, conv −0.010)
   and there's no rollback. The penalty acts on the *score*, not the *cells* → needs the
   prune arm.
4. **Diagnosis trail** (`inspect_spawned.py`, `audit_edges.py`, `trace_tangle.py`):
   - **`arena.utility` is all-zero** — `accumulate_saliency` is never called in the
     growth path, so the per-cell `u` prune signal does not exist yet.
   - **Duplicate-edge bug**: `divide()` inherits a random subset of parent inputs THEN
     adds new edges from the same tiny rank-0 pool → ~81 true dup edges (fan_in 14≠12).
   - **No depth**: `same_rank_edges=False` (divide default) → permanently bipartite
     `12→hidden→32`, all-to-all readout (53×32=1696 edges) → dilution. Flipped it on.
   - **Depth formed as a TANGLE**: ranks 2–12 from 29 cells, **112 later-sibling
     back-wires**, longest path to rank-12 = 11 somata in series. Root cause: shared
     parent + `project_to_consumers` wires later siblings into earlier ones. The
     symmetric/axial `divide()` flag is **cosmetic** (position only). Confirmed.
5. **Co-designed the replacement with Rocky** (locked, in design doc §7): **thin
   columns** indexed by **hard `branch_id`** + `layer`; **separate axes** (orientation ×
   phenotype); frustration split **`[F_lateral, F_depth]` per branch**; **readout reads
   tops only**. Locality is the fix for "depth never fires" — depth triggers per-branch,
   so global width can't starve it.
6. **Built + ran it** (`step5_branch.py` / `run_branch.py`): width **self-saturates at 13
   branches** (0.077→0.409 ≈ linear ceiling), **DEPTH FIRES** (+0.068 → 0.478), **no
   tangle** (max depth 5, all composition branch-local). Mechanisms validated.

## State of the build

- **Branch `progenitor-council`.** This session's work is committed + pushed.
- **Live testbed = `step5_branch.py` / `run_branch.py`** (the new architecture). The
  council path (`step4_grow.grow_council` + `grow_council_frustration`) is kept as the
  documented predecessor but is **superseded** (it tangles).
- **`dendrite.py` is GCU `z·cos z` GLOBALLY now** (Rocky's call). Tests asserting `z+z²`
  WILL FAIL — intentional for the branch experiments; future cleanup = per-cell/config
  selection, not a global swap.
- **DO-NOT-COMMIT carries (excluded again, verified):** `trioron/bases/developmental.py`,
  `trioron/lifecycle/developmental.py`, `trioron/viz/export.py`. `.claude/`, `runs/`
  untracked.

## Open questions / next-up (priority order)

1. **Spread depth across columns.** Phase 2 piled ALL depth into branch 0 — post-
   convergence `|w·g|` is a weak depth-target (every converged top ≈0, selector
   tie-breaks to branch 0, then its fresh GCU cell dominates). Replace with a
   **residual / which-class-still-wrong** depth-target so depth distributes. This is the
   single biggest lever on the 0.478 (vs council ~0.78).
2. **Separate-axes grid + phenotype at axial.** Currently lateral=linear, axial=GCU
   hard-coded. Make orientation and phenotype independent choices (attention helped most
   in the council runs — an axial attention column may beat GCU).
3. **Prune arm + wire utility.** Call `accumulate_saliency` in the loop so `arena.utility`
   is live; then prune on `(u low AND λ low)` in **log-domain** (not linear fractions —
   they lose the tail at scale). Output-reference the utility (`|a·g_a|`, not edge `|w·g|`)
   so linear/GCU cells compare fairly.
4. **Normalized epigenetic lock.** Per-weight self-ratio `λ=progress/(progress+velocity)`
   (digit-safe, anchor-free), soft lock, fraction threshold ~0.5 + progress floor (GCU
   flat-spots can false-settle). Still unbuilt; only matters once a 2nd task is chained.
5. **Promote** the validated branch mechanism into `trioron/progenitor/` once depth-spread
   + readout are solid; retire `divide()`'s tangle for this path.

## Pointers

- **Run it:** `python3 -m experiments.progenitor.run_branch` (new architecture, ~2 min).
  `python3 -m experiments.progenitor.run_hard` (old council frustration loop, GCU).
- **Inspectors (read-only, deterministic seed 0):** `inspect_spawned.py` (per-cell
  rank/fan/utility), `audit_edges.py` (edge totals + dup check), `trace_tangle.py`
  (back-wire metric + longest path).
- **Design:** `docs/design/progenitor_council.md` **§7** (the branch-id architecture —
  read this first next session). Spec §3.2–3.6, §5.1.
- **Key code:** `step5_branch.py` (`build_base`, `add_branch`=lateral,
  `deepen_branch`=axial+re-point, `_branch_utility`, `grow_branches`=2-phase loop).

## Environment notes

- `/home/marcrockhat/trioron-project/`, branch `progenitor-council`. Python 3.10.12,
  torch 2.11.0, WSL2, 12 cores, `python3`, `OMP_NUM_THREADS=8`. **`cd` in a compound
  Bash command persists — use absolute paths.**
- Long blocking runs auto-background; rely on task-completion notifications.
- Long, dense session (heavy diagnosis + a redesign + a build). All work committed +
  pushed; safe to break here. Design §7 is the source of truth for the new architecture.
