# Trioron Handoff

**Session date:** 2026-05-31 → 2026-06-01
**Session number:** 011
**Session title:** Temporal gate PASS → comparison wall → quad cracks it →
ROOT FINDING: substrate was purely LINEAR → quad nonlinearity into core +
satellites memory cell type

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
   — a sum of input *products*. The dendrite quad σ(z)=z+z² expands to
   Σ wᵢwⱼ·aᵢaⱼ — all pairwise products in one unit. Result: DMS goes from
   **0/3 (chance)** to **quad 3/3 → 1.000** (with grad clipping + RMS-norm of
   the recurrent trace).

4. **ROOT FINDING: the substrate was PURELY LINEAR.** `scheduler.forward`
   applies NO activation; the linear phenotype is `y=b+Σw·a`; all phenotypes
   were linear stubs. A stack of linear cells = one linear map — so
   self-arrange depth was *mathematically vacuous*, comparison was
   *unrepresentable*, and z² was the *first nonlinearity the substrate ever
   had*. Even chained-15's 0.83 came from the external MLP router, not the
   linear core. (See `substrate-was-purely-linear` memory.)

5. **Fix landed in core (commit a56137a).** Real quad phenotype
   (`trioron/phenotype/dendrite.py`, was a stub), registered, and
   `seeded(nonlinear=)` puts the quad on interior triorons (output stays
   linear). Quad preserves perception (0.992–0.995 vs linear 0.991), stable
   with grad clip. Per Rocky's directives: "implement improvements in the
   triorons themselves" (→ quad in core) and "set satellite if the change is
   too massive" (→ memory to satellites). **Decision rule established: small
   change → triorons core; massive change → satellites.**

6. **Satellites v1** (`experiments/satellites_v1.py`) — a new cell type
   alongside triorons (satellite-glia analogue, recurrent slot): intrinsic
   MEMORY (leaky trace + BPTT, no Python harness), RESOURCE SENSING
   (arena.pressure), resource-gated DIVISION. Delayed recall OFF 0.200 (chance)
   → ON 0.996, memory living in graph cells.

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
  metrics + DMS comparison wall), `6a362f6` (quad-dendrite cracks it), `32a7a0a`
  (handoff), `a56137a` (**core: real quad nonlinearity in triorons + satellites
  v1**). Session 010's were `5547ab2`.
- **CORE CHANGED this session:** `trioron/phenotype/dendrite.py` (real quad),
  `phenotype/__init__.py` (registered), `bases/seeded.py` (`nonlinear=` option,
  default False). First substrate change of the arc. `experiments/` benches +
  chained-15 read `TRIORON_NONLINEAR=1` env to build quad triorons (grad clip
  added to chained-15, gated on the flag so linear baseline is unchanged).
- **RUNNING at handoff:** the full `nonlinear=True` suite (grounded + chained-15
  smoke, linear vs quad) — `/tmp/run_nonlinear_suite.log`. Decides whether quad
  becomes the substrate default.
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

1. **DONE — suite verdict: DO NOT flip `nonlinear=True` default.** Quad is
   neutral on perception (grounded SYS 0.894→0.895) and essential for
   comparison (DMS 0/3→3/3) but **REGRESSES chained-15 CL** (router-class
   0.809→0.750, task-aware 0.960→0.849, −11pp — a forgetting signal; the CL
   machinery (credit-freeze/manifold/router) is tuned for linear stats). So
   quad stays **opt-in**. Implication: the substrate must be **heterogeneous** —
   mostly linear, with quad cells GROWN only where relational computation is
   needed → makes `promote_dendrite`-under-relational-frustration (task 2)
   empirically required, not just elegant. (chained-15 here was smoke; a
   full-epoch run would confirm magnitude.)
2. **Promote satellites into core** as a real stateful phenotype (the
   recurrent/mnemonic slot, not a stub) with a clean stateful-forward + reset
   boundary; then self-organized satellite growth under temporal/memory demand.
3. **Re-measure structure (SOM) on the quad substrate** — now that it can learn
   relations, does the cortex abstraction gradient emerge? Closes Rocky's
   structural question on a composition-demanding task.
4. **Satellites v2** — modulation (gate trioron plasticity/gain), phase
   (wake/dream) timescale.
5. **Staged mini-curriculum** — sense → consolidate → relation, the first
   actually-taught sense→logic curriculum, growing the right primitive per rung.
6. **Spec review** — fold the quad phenotype (§3.6) + satellites (new §) +
   "substrate needs a nonlinearity" into the spec; CI's 4 pre-existing failures.

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
- **`trioron/phenotype/dendrite.py`** — the real quad σ(z)=z+z² (NEW; was a
  stub). **`trioron/phenotype/__init__.py`** — DENDRITE now → quad; attention/
  conv/recurrent still linear stubs. **`trioron/bases/seeded.py`** —
  `nonlinear=` option (interior cells → quad). `epigenome.py:primary_phenotype`
  returns the *lowest* set gene (clear LINEAR to express another).
- **`experiments/satellites_v1.py`** — `SatelliteOp` (stateful memory + resource
  sensing), `add_satellite`/`divide_satellite`, recurrent-slot dispatch.
- **`/tmp/run_nonlinear_suite.log`** — the quad-vs-linear suite (running at
  handoff). Re-run: `TRIORON_NONLINEAR=1 python3 experiments/<bench>.py ...`.
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
- `substrate-was-purely-linear` — ROOT FINDING: no nonlinearity at all; deep
  linear = one linear map; fixed by real quad phenotype + seeded(nonlinear=).
- `satellites` — new cell type: intrinsic memory + resource sensing + division;
  decision rule (small→triorons, massive→satellites).
- (010) `self-arrange-depth-result`, `grounded-sense-valence-curriculum`,
  `temporal-axis-is-axis-7`.

## Environment notes

- Working dir `/home/marcrockhat/trioron-project/`, branch `v2.0-scaffold`.
- Python `/usr/bin/python3` 3.10.12, torch 2.11.0+cu130. WSL2, 12 cores,
  7.4 GiB RAM. `python3`, `OMP_NUM_THREADS=8`. sklearn 1.7.2 / scipy 1.15.3 /
  numpy 2.2.6 available (used by structural_metrics).
- Bench timings: temporal gate / DMS ~30s per arm-seed; structural n=5 ~2 min.
