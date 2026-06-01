# Trioron Handoff

**Session date:** 2026-05-31 → 2026-06-01
**Session number:** 011
**Session title:** Temporal axis gate PASS → the comparison wall → quad-dendrite
cracks it (linear 0/3 → quad 3/3); the sense→logic→symbol curriculum

> This file is **rewritten in full** every session. The previous
> session's handoff is preserved in git history
> (`git log docs/handoff/HANDOFF.md`). Read this file first when
> starting any new session; rewrite it before ending one. See
> `docs/handoff/README.md` for the convention.

## Summary

This session resolved the central question of the depth/reasoning arc: **what
operation is trioron missing for logic, and is it in the architecture?**

Three results, in sequence:

1. **Temporal axis (Axis 7) falsification gate — PASS.** Delayed grounded
   recall: a memoryless substrate is pinned at chance; a substrate carrying its
   own leaky interior trace (BPTT through the sequence) recovers the cue —
   0.995→0.726 over 1→4 step hold, Δ +0.53 to +0.80, n=3. First mechanism in
   the whole arc that *lifts* (depth never did).

2. **The comparison wall.** A standardized structural-metrics suite + a fine
   temporal task (delayed match-to-sample, DMS) revealed: the substrate
   **recalls** perfectly (1.000) but **cannot learn a relation/comparison**
   (frozen at chance, zero gradient). Same wall as depth — *composition/
   relation*, not memory, is the bottleneck for logic.

3. **Quad-dendrite cracks it.** Comparison is an inner product (Σ sampleₖ·testₖ)
   — a sum of input *products*. Linear+ReLU cells can't form products (the
   parity wall); the dendrite quad σ(z)=z+z² expands to Σ wᵢwⱼ·aᵢaⱼ — all
   pairwise products in one unit. Result: DMS goes from **linear 0/3 (chance)**
   to **quad-dendrite 3/3 → 1.000**. The relational primitive that logic needs
   is *already in the spec* (dendrite gene, §3.6) — it was just never
   implemented (all shipped phenotypes are linear stubs).

Conceptual throughline established with Rocky: reasoning ≠ eyesight (depth is
for *fine* comparison, not coarse perception); emotion = *learned* predictive
valence (not innate; racism = same mechanism miscalibrated); the curriculum is
**sense → logic → symbol → language**, taught bottom-up; and it has NOT been
taught as a staged curriculum yet (each rung tested standalone).

## Headline numbers

**Axis 7 gate (delayed recall, n=3), `bench_temporal_gate.py`:**
| cue held | temporal-off | temporal-on | Δ |
|---|---|---|---|
| 1 step | 0.200 | 0.995±0.003 | +0.80 |
| 2 steps | 0.200 | 0.957±0.011 | +0.76 |
| 4 steps | 0.200 | 0.726±0.010 | +0.53 |
PASS at every delay. off = exactly chance (memoryless can't recall).

**Comparison wall + quad fix (DMS K=4, n=3), `bench_quad_dendrite_dms.py`:**
| phenotype | result |
|---|---|
| linear+ReLU | 0/3 — 0.445±0.13 (chance, flat gradient) |
| quad-dendrite (z+z², RMS-norm trace) | **3/3 — 1.000±0.000** |

**Structural metrics (coarse grounded task, n=5), `analyze_self_arrange_structure.py`:**
self-arrange grows real depth (eff_depth 5.6 vs 1.0, 274 i↔i edges, axon
fan-out 14.6 vs 32, module silhouette 0.27 vs 0.10) but **no wiring modularity**
(Newman r≈0.018≈chance — corrected an earlier crude n=1 guess) and **no
abstraction gradient** (−0.18±0.19). On a coarse task that doesn't demand
composition, the cortex hallmarks don't appear (expected).

## What was done

- **`experiments/bench_temporal_gate.py`** (new) — Axis 7 gate. Substrate carries
  its own leaky interior trace `m=λm+(1-λ)h` (λ=0.6), fed back as input, BPTT.
  `live_activations` is differentiable → real internal recurrence, not an echo.
- **`experiments/structural_metrics.py`** (new) — standardized Substrate
  Organization Metrics: population sparseness (Treves-Rolls), Newman categorical
  assortativity, abstraction gradient (rank vs eta² Spearman), module silhouette
  + morphometric census (cells/dendrites/lineage-clusters/axon fan-out/params).
  All established measures; novelty is application to a grown substrate.
- **`experiments/analyze_self_arrange_structure.py`** (new) — n=5 structural
  report, bipartite vs self-arrange.
- **`experiments/bench_fine_temporal.py`** (new) — DMS fine-comparison task;
  revealed the comparison wall (recall 1.000 via same harness, compare frozen).
- **`experiments/bench_quad_dendrite_dms.py`** (new) — implements the quad
  forward σ(z)=z+z² via a custom dispatch table, sets dendrite phenotype on
  interior cells, re-runs DMS. linear 0/3 → quad 3/3. Stability: gradient
  clipping + smooth RMS-norm of the trace (hard clamp saturates → strands seeds).

## Key findings

1. **Memory works, comparison doesn't (on linear cells).** Recall 1.000;
   relation frozen at chance. The logic bottleneck is the *operation*
   (composition/comparison), not memory or depth-structure.
2. **Comparison is a product; ReLU can't make products; z² can.** Inner
   product / coincidence detection = pairwise input products. Single ReLU unit:
   none. Single quad unit: all of them. Hence learnable with quad, flat-gradient
   with linear. Same family as parity (XOR = a+b−2ab).
3. **Quad-dendrite robustly solves comparison (3/3 → 1.000)** with smooth RMS
   normalization of the recurrent trace. A *hard* clamp saturates to zero
   gradient and strands seeds — the smooth norm is load-bearing.
4. **ALL shipped phenotypes are linear stubs.**
   `trioron/phenotype/__init__.py` registers attention/conv/recurrent/dendrite
   → `linear.forward_batch`; `dendrite.py` is an empty docstring. The quad here
   is the first non-linear phenotype exercised. (To make a cell dendrite: clear
   LINEAR bit — bit 0, always set, and `primary_phenotype` returns the *lowest*
   set gene — and set DENDRITE, then `refresh_phenotype`.)
5. **Self-arrange depth is real but conv-distinct and (on coarse tasks) not
   cortex-like.** Topological rank-layering, no weight-tying/gene (conv =
   weight-tied lineage + spatial receptors). No abstraction gradient where the
   task doesn't demand composition.
6. **Right primitive per operation.** ReLU to sense, quad to compare. z² is not
   "better" — it matches a multiplicative operation and hurts (instability)
   where multiplication isn't needed. The substrate should *grow* quad cells on
   relational rungs (frustration→grow-dendrite, like frustration→divide).

## State of the build

- **Branch:** `v2.0-scaffold`. All work pushed.
- **Commits this session:** `03d6d5a` (Axis 7 gate PASS), `0107a39` (structural
  metrics + DMS comparison wall), `6a362f6` (quad-dendrite cracks it).
  Session 010's were `5547ab2` (self-arrange + grounded world + Axis 7 design).
- **Pre-existing uncommitted, NOT touched (from session 005):**
  `trioron/bases/developmental.py`, `trioron/lifecycle/developmental.py`,
  `trioron/viz/export.py`. Left as-is, as in 009/010.
- **Untracked:** `.claude/`, `runs/` (gitignored).
- **Pre-existing test failure (NOT mine, flagged in 010):**
  `test_lifecycle.py::TestGrowth::test_growth_trigger_logic`.

## Decisions made

- **Quad/relational primitive is the answer to "what's missing for logic."**
  Confirmed empirically. The dendrite gene exists for it; it needs a real
  (non-stub) non-linear forward in core — but only after spec review.
- **Quad-dendrite code stays in `experiments/` for now** (custom dispatch),
  pending a spec §3.6 review before a real `dendrite.py` forward lands in core.
- **Temporal axis ships live at the gate level**; per-cell `mnemonic` gene in
  core is still a follow-on.
- **Walked back an over-claim mid-session** (smoke said quad=1.000; n=3 said
  chance; diagnosed instability; RMS-norm fixed it to 3/3). Lesson re-logged:
  do not headline from a single smoke seed.

## Open questions

1. **Make quad a real phenotype in core** (spec §3.6 `dendrite.py` forward +
   the smooth-normalization story) — but the normalization currently lives in
   the recurrent-trace wrapper, not the cell. Where does it belong?
2. **Self-organized quad growth:** can frustration on a relational rung trigger
   `promote_dendrite` (grow quad cells), the way frustration triggers `divide`?
   This is the "teach the curriculum by growing the primitive" step.
3. **Does the abstraction gradient appear on a task the substrate can now
   learn?** DMS structural measurement was moot while DMS was unlearnable; with
   quad it learns — re-measure SOM on the quad-DMS substrate (do deep quad cells
   encode match/non-match?).
4. **Staged curriculum:** sense → consolidate → relation, with quad growth on
   the relational rung — the first genuinely *taught* curriculum.
5. **Other stub phenotypes:** attention (Q·K is also native comparison) and
   recurrent are linear stubs too. Attention may be an alternative relational
   primitive worth comparing to quad.

## Next-up tasks (priority order)

1. **Re-measure structure (SOM) on the quad-DMS substrate** — now that it
   learns the relation, does the cortex abstraction gradient finally emerge?
   This closes Rocky's structural question on a task that demands composition.
2. **Self-organized quad growth** — `promote_dendrite` under relational
   frustration; test that the substrate grows quad cells when (and only when) a
   rung needs comparison.
3. **Staged mini-curriculum** — sense → consolidate → relation, the first
   actually-taught sense→logic curriculum, with primitive growth per rung.
4. **Spec review for §3.6** — fold the quad forward + normalization into the
   spec, then implement a real non-stub `dendrite.py` in core.

## Pointers

- **`experiments/bench_quad_dendrite_dms.py`** — quad forward `make_quad_forward`
  (σ=z+z²), `build` (custom dispatch + set dendrite phenotype), RMS-norm trace +
  grad clip in `run`. linear 0/3 → quad 3/3.
- **`experiments/bench_temporal_gate.py`** — Axis 7 gate (recall).
- **`experiments/bench_fine_temporal.py`** — DMS; the comparison wall.
- **`experiments/structural_metrics.py`** — SOM suite + morphometrics; accepts
  precomputed `acts` for recurrent/sequence tasks.
- **`experiments/analyze_self_arrange_structure.py`** — n=5 structural report.
- **`docs/design/temporal_axis_v7.md`** — Axis 7 design (gate §7 marked PASS).
- **`trioron/phenotype/__init__.py`** — the linear-stub registrations (all
  phenotypes → linear). `trioron/core/epigenome.py:primary_phenotype` — returns
  the *lowest* set gene (LINEAR bit must be cleared to express another).
- **spec §3.5** recurrent (within-pass), **§3.6** dendrite/quad, **§2.2**
  epigenome bits.

## The conceptual through-line (read to understand the arc)

Reasoning isn't eyesight: depth/composition is for *fine* comparison (good-vs-bad
human, wild-vs-domestic cat), not coarse perception — which is why depth never
lifted coarse classification. The real cognitive stack is **sense → logic →
symbol → language**, bottom-up. trioron is the grounded *interface/inner voice*
(not an LLM-scale reasoner; Gemma does language). This session located the exact
gap: it can **sense** and **remember** and **recall**, but the **relational
operation** (compare/compose) — the substance of logic — was missing, because
interior cells were linear and comparison is multiplicative. The quad-dendrite
primitive (already in the spec, never implemented) supplies it. Emotion is
*learned* predictive valence over innate drives. The curriculum has not been
*taught* (each rung standalone); teaching = staged rungs + the substrate growing
the right primitive per rung (depth, memory, quad).

## Memories saved this session (cross-PC persistent)

- `temporal-gate-pass` — Axis 7 PASS; recall works (substrate's own trace+BPTT).
- `quad-dendrite-comparison-result` — comparison: linear 0/3, quad 3/3; z²=products;
  all phenotypes are linear stubs; needs smooth RMS-norm.
- (010) `self-arrange-depth-result`, `grounded-sense-valence-curriculum`,
  `temporal-axis-is-axis-7`.

## Environment notes

- Working dir `/home/marcrockhat/trioron-project/`, branch `v2.0-scaffold`.
- Python `/usr/bin/python3` 3.10.12, torch 2.11.0+cu130. WSL2, 12 cores,
  7.4 GiB RAM. `python3`, `OMP_NUM_THREADS=8`. sklearn 1.7.2 / scipy 1.15.3 /
  numpy 2.2.6 available (used by structural_metrics).
- Bench timings: temporal gate / DMS ~30s per arm-seed; structural n=5 ~2 min.
