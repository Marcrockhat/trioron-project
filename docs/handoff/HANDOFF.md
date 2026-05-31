# Trioron Handoff

**Session date:** 2026-05-31
**Session number:** 010
**Session title:** Self-arrange depth shipped (emerges, doesn't lift) →
pivot to grounded sense→valence world + Axis 7 (temporal) design

> This file is **rewritten in full** every session. The previous
> session's handoff is preserved in git history
> (`git log docs/handoff/HANDOFF.md`). Read this file first when
> starting any new session; rewrite it before ending one. See
> `docs/handoff/README.md` for the convention.

## Summary

Implemented the **self-arranging substrate** (session 009's next-up #1):
relaxed `divide()`'s rank policy so a new cell may draw edges from
same-rank cells (`same_rank_edges`), and Kahn's BFS then promotes it to
rank+1 — depth self-organizes through growth, no hand-crafting. Wired to a
`--self-arrange` bench flag. **Depth provably emerges** (rank 5, 40 deep
cells in probes) and is **cycle-safe by construction** (a freshly-divided
cell is a sink, so any incoming edge is acyclic — no cycle check needed).

**But depth does not lift accuracy on any learnable task tested (4 task
families).** chained-15 smoke gave a marginal +1pp (router-class 0.8182 vs
0.8080), but synthetic parity (unlearnable), teacher-student (width
suffices), staircase (unlearnable), and a new grounded-world bench all
show self-arrange ≈ flat width. The depth-favoring regime *coincides* with
the SGD-unlearnable regime on this substrate — reproducing the project's
prior `bench_2_0_single_task` "depth didn't win" finding three more times.

Session then **pivoted, at Rocky's direction**, from "does depth help" to
the actual roadmap question: **can the substrate learn things that need
composition — logic, then language.** Long design conversation produced:
(1) the **sense → logic → symbol → language** curriculum made concrete
("logic before language" / discover-fire framing); (2) **emotion = learned
predictive valence** (not innate — racism is the same mechanism
miscalibrated by biased experience); (3) a **grounded sense→valence world**
bench; (4) a full **Axis 7 (Temporal / Working Memory) design proposal**.

Built `experiments/bench_grounded_world.py`: ~16 sensory primitives →
innate drive-consequence labels (the only hardwired valence), substrate
learns to predict the consequence (= emergent "emotion"). **The loop works
(iid 0.996)** but **compositional generalization is weak** (nourish-recall
0.99 i.i.d. → 0.39 on a held-out cue) and **depth doesn't fix it**. Static
depth is not the missing piece; *state over time* is the untested one —
which is why the temporal axis is the next real lever.

## Headline numbers

**chained-15 smoke (seed=42, 2ep), clean machinery, MLP class-router:**
| Config | router-class full | task-aware |
|---|---|---|
| baseline (no self-arrange) | 0.8080 | 0.9603 |
| `--self-arrange` | **0.8182** | 0.9599 |

+1pp at half training budget — directional only, NOT confirmed at full n=3
(that run was deferred when the pivot happened).

**grounded-world bench, n=3 (60ep, h_init=8, grow 40@20, noise=0.1):**
| Arm | iid | SYS | nourish-recall(SYS) | depth |
|---|---|---|---|---|
| no-growth | 0.996 | 0.894 | 0.388 ± 0.032 | rank 1 |
| bipartite | 0.996 | 0.894 | 0.395 ± 0.040 | rank 1 |
| self-arrange | 0.996 | 0.895 | 0.391 ± 0.037 | rank 5 |

SYS = systematic (umami-cued) test; nourish-recall(SYS) is the real
compositional metric (train cues nourishing with *sweet*, test with
*umami*). All arms identical → depth irrelevant; substrate memorizes the
surface cue.

## What was done

### Self-arrange depth (the mechanism — SHIPPED, opt-in)
- `trioron/lifecycle/grow.py`: added `GrowthConfig.same_rank_edges` (default
  **False** — keeps CI/v1 behavior). When True, the edge-source mask uses
  `a.rank <= child_rank` (vs `<`), and excludes the child id. Sink-safety
  argument documented inline (no cycle check needed). 60-division probe:
  baseline stays max_rank=1 (bipartite); self-arrange reaches rank 5 with
  36 interior cells at rank>1.
- `experiments/bench_chained_15_v2.py`: `--self-arrange` flag → threads a
  `GrowthConfig` through `train_one_task` → `divide`. Added to the flags
  log line.

### Synthetic depth probes (all non-discriminating)
- `experiments/bench_arena_hierarchical.py` (new): hierarchical parity /
  teacher-student / staircase tasks, A/B no-growth / bipartite / self-arrange
  with deterministic matched growth. **Mechanism confirmed** (depth emerges
  under real training) but **no task discriminates** depth from width.
  Fixed a missing `sub.prepare_training()` (substrate wasn't training).

### Grounded sense→valence world (the pivot, NEW direction)
- `experiments/bench_grounded_world.py` (new): the sense→logic curriculum
  rung. Sensory schema (vision/touch/taste/smell, ~16 dims) → innate drive
  consequence (nourishing/hydrating/painful/toxic/neutral) via conjunctive
  grounded rules → substrate learns predictive valence ("emotion"). Reports
  i.i.d. + **systematic compositional split** + per-class nourish-recall.

### Axis 7 — Temporal / Working Memory (design proposal, NEW)
- `docs/design/temporal_axis_v7.md` (new): full design. Two timescales —
  within-pass recurrence (already specced §3.5, `recurrent` gene, never
  exercised) + across-pass working memory (NEW gene bit 10 `mnemonic`,
  leaky trace). Phase signal (wake/dream) as the clock. Temporal frustration
  as the growth trigger (promote_recurrent / promote_mnemonic, analogue of
  frustration→divide). Falsification gate (delayed grounded association).
  Lifetime budget (cheapest axis by storage). **Corrects the
  `temporal-cognition-gap` mis-numbering: temporal is Axis 7, NOT 6 — Axis 6
  is `axis6_spawn`.**

## Key findings

1. **Self-arrange depth emerges and is cycle-safe, but doesn't lift
   learnable tasks.** A divided cell is a sink → relaxing the rank policy
   can't create cycles. Depth grows (rank 5). But across 4 task families,
   self-arrange ≈ flat width. *Why:* the depth-favoring regime is the
   SGD-unlearnable regime (parity); everything learnable is width-sufficient
   (teacher-student, grounded-world). This is intrinsic on this substrate,
   reproducing `bench_2_0_single_task`'s "depth didn't win."
2. **The two-substrate split.** chained-15 uses the **Arena** substrate
   (`divide()`, ranks) where self-arrange lives. CIFAR / hierarchical /
   absorption use **TrioronNetwork** (fixed layer-list MLP, `axis6_spawn`
   for *width* growth). My depth change *cannot run* on the latter without a
   port — Rocky correctly stopped that direction. The CIFAR "self-arrange"
   arm is a *different* (width/spawn) mechanism.
3. **Emotion = learned predictive valence, not innate.** Only the drives
   (hunger/thirst/pain/temperature) are hardwired valence; emotion *emerges*
   as the substrate learns percept→drive-consequence. Biased experience →
   biased emotion (racism is the same mechanism). This makes the temporal
   axis load-bearing: prediction is temporal credit assignment.
4. **The grounded loop works but the substrate memorizes surface cues.**
   iid 0.996, but nourish-recall collapses to 0.39 on a held-out cue. Static
   depth doesn't help. The compositional gap is real.
5. **My systematic split is arguably too hard** (holds out a *primitive*
   cue entirely — "umami is edible" never seen in any context). The 0.39 is
   partly an impossible-without-prior inference, not pure weak-composition.
   Refine toward a SCAN-style split (all primitives seen, novel
   *combinations* held out) next.

## State of the build

- **Branch:** `v2.0-scaffold`
- **Committed this session** (selectively — see Decisions): `grow.py`,
  `bench_chained_15_v2.py`, `bench_arena_hierarchical.py`,
  `bench_grounded_world.py`, `docs/design/temporal_axis_v7.md`, this handoff.
- **Pre-existing uncommitted, NOT touched (carried from session 005):**
  `trioron/bases/developmental.py`, `trioron/lifecycle/developmental.py`,
  `trioron/viz/export.py`. Left as-is, as in session 009.
- **Untracked:** `.claude/`, `runs/` (gitignored).
- **Pre-existing test failure (NOT mine):**
  `tests/test_v2/test_lifecycle.py::TestGrowth::test_growth_trigger_logic`
  — fails on clean HEAD too (threshold mismatch: `check_growth_trigger(1.5,
  50)` returns True, test expects False). Flag for a future cleanup.

## Decisions made

- **Ship self-arrange as opt-in (`same_rank_edges=False` default).** Keeps
  CI/v1 tests intact; bench opts in via `--self-arrange`. Do NOT flip the
  default until depth demonstrably helps something.
- **Stop chasing synthetic depth discriminators.** 4 families failed; it's
  intrinsic. Self-arrange stays available as a mechanism, not a headline.
- **Pivot to the grounded sense→logic→symbol→language curriculum** as the
  real track (Rocky's call). Depth and temporal are composition-in-space vs
  composition-in-time — the two halves of reasoning.
- **Emotion is learned, not hardcoded.** Schema hardcodes senses + drives +
  survival disposition; emotion emerges.
- **Committed the session's code + handoff and pushed**, per the project's
  unconditional handoff rule (CLAUDE.md), surgically excluding the
  session-005 carried files.

## Open questions

1. **Does a SCAN-style compositional split (novel combinations, all
   primitives seen) discriminate depth — or even there does width suffice?**
   The current split holds out a primitive entirely; refine it.
2. **chained-15 full n=3 with `--self-arrange`** — still un-run. Worth the
   3.5h to confirm/deny the +1pp smoke signal? (Low priority given depth's
   track record, but it's the one real-data signal.)
3. **Temporal axis falsification gate** — build the delayed grounded
   association task; does mnemonic/recurrence beat the feed-forward baseline?
   This is the decisive next experiment for the new direction.
4. **Survival disposition (run/safe/normal) + active/embodied loop** —
   deferred; the passive associative version is what's built. Agency pulls
   temporal in immediately.

## Next-up tasks (priority order)

1. **Build the temporal axis falsification gate** (`docs/design/
   temporal_axis_v7.md` §7): delayed grounded-association task on the Arena
   substrate, mnemonic-on vs mnemonic-off. This is the decisive test for the
   whole new direction and the natural continuation. Implement the
   `mnemonic` gene (bit 10) + leaky trace + phase-aware forward first.
2. **Refine the grounded-world compositional split** to SCAN-style
   (all primitives seen in *some* context, novel combinations held out) so
   "weak composition" isn't confounded with "unfair held-out primitive."
3. **Review the Axis 7 design** (`docs/design/temporal_axis_v7.md`) with
   Rocky; fold accepted parts into spec §3.7 + §9 partition before any
   `trioron/` temporal code lands (spec-is-source-of-truth discipline).
4. **(Optional) chained-15 full n=3 `--self-arrange`** if we want the depth
   verdict on real data closed out.

## Pointers

- **`trioron/lifecycle/grow.py:93-117`** — the `same_rank_edges` policy +
  sink-safety comment. The one-line lever.
- **`experiments/bench_chained_15_v2.py`** — `--self-arrange` flag;
  `growth_cfg` threaded through `train_one_task` (line ~390) → `divide`
  (line ~493).
- **`experiments/bench_grounded_world.py`** — the grounded world: `SENSES`,
  `innate_consequence`, `make_world` (sweet→umami cue swap), `train_arm`,
  `class_recall`. Run `--smoke` (~1 min) or default n=3 (~3 min).
- **`experiments/bench_arena_hierarchical.py`** — synthetic depth probes
  (parity/teacher-student/staircase); mechanism proof that depth emerges.
- **`docs/design/temporal_axis_v7.md`** — Axis 7 full design.
- **spec §3.5** (`recurrent` gene, line 873) — the within-pass temporal
  primitive that already exists. **spec §2.2** — epigenome bits (bit 3
  recurrent, bits 10-15 free; `mnemonic` proposed at bit 10). **spec line
  3127** — the axis write-function list (confirms Axis 6 = spawn).
- **Logs (uncommitted, /tmp):** `run_selfarrange_smoke_s42.log`,
  `run_baseline_smoke_s42.log`, `run_grounded_world_n3.log`.

## The conceptual through-line (read this to understand the pivot)

Rocky reframed the whole session: depth-for-its-own-sake is the wrong
target. The real question is whether trioron can do **logic and language** —
and the honest answer is it shouldn't try to do them like an LLM (that's
billions-of-params *coverage*, and it's Gemma's job). Trioron is the
**grounded interface / inner voice** that represents the user's bounded,
evolving intent — and *that* needs composition (depth in space) + memory
(depth in time), at honest small scale. The curriculum is **sense → logic
→ symbol → language**, bottom-up. We're at the **sense→logic** rung:
grounded percepts → learned valence → compositional generalization. The
baby-babble test ("understand a 'baaa' you've never heard") = compositional
generalization = the grounded-world bench's systematic split. Pictographs
(日+月=明) are the symbol rung above it. Language (frozen Gemma) is last.

## Memories saved this session (cross-PC persistent)

- `self-arrange-depth-result` — depth emerges via `same_rank_edges`, is
  cycle-safe (sink argument), but doesn't lift 4 learnable task families;
  the depth-favoring⟺unlearnable tradeoff is intrinsic.
- `grounded-sense-valence-curriculum` — the sense→logic→symbol→language
  design; emotion=learned predictive valence; the grounded-world bench +
  weak-composition result.
- `temporal-axis-is-axis-7` — corrects `temporal-cognition-gap`: temporal =
  Axis 7 (Axis 6 = spawn); design at `docs/design/temporal_axis_v7.md`.

## Environment notes

- Working dir: `/home/marcrockhat/trioron-project/`. Branch `v2.0-scaffold`.
- Python `/usr/bin/python3` (3.10.12), torch 2.11.0+cu130. Linux WSL2, 12
  cores, 7.4 GiB RAM. Use `python3`, `OMP_NUM_THREADS=8` solo.
- grounded-world: ~3 min n=3. chained-15 smoke: ~8.5 min. Full-epoch n=3
  chained-15: ~3.5h serial (RAM-bound, OOMs on 2 concurrent full runs).
