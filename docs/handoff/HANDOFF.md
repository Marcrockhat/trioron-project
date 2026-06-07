# Trioron Handoff

**Session date:** 2026-06-07
**Session number:** 023
**Session title:** **Taxonomy continual-learning bench built & validated — manifold
replay is the workhorse, the dog/goat overlap shown irreducible; memory
mechanisms (manifold vs dendrite) untangled; λ-plasticity gate on `divide()`
shipped (default off).**

Started from s022's progenitor–council design but Rocky steered the session into a
deep, experiment-driven exploration of trioron's **memory / continual-learning
machinery**, using the weight-only animals probe as a sandbox and growing it into a
proper taxonomy CL bench. The council build (s022 step b) was **not** started; step
(a) — the λ-gate — was. Net output: a validated CL bench + several real mechanism
findings, all captured in memory and below.

## Summary (the arc)

1. **λ-plasticity gate on `divide()`** (step a from s022). `grow.py` gained
   `GrowthConfig.divide_lambda_max`: a parent whose epigenetic-lock λ exceeds it has
   *matured* and won't divide (division → stem/germline-only). **Default `None`
   (off)** — the council path turns it on; old growth paths unchanged. Test
   `test_plasticity_gate_blocks_mature_cell` passes (the 2 carried failures remain).

2. **Manual-grounded clarification of the dog disruptor.** Native retrieval probe
   (`retrieval_probe.py`): `StreamingMixture` K=4 prototypes hits the weight-only
   6-class ceiling (0.843); a single Gaussian underfits dog. Linear readout caps at
   0.78 because dog needs **disjoint** bands (it's flanked by cat below, goat above).

3. **Emergent-label experiment (B)** (`emergent_labels.py`). Naming the overlap
   (`dog|goat`) is output-side growth; output- and input-growth are complementary.
   Linear-on-emergent 0.92, +nonlinearity 0.985. Honest "100%" = honest coverage,
   not beating the Bayes overlap.

4. **10-species taxonomy dataset** (`taxonomy.py`), lean-4 features (weight, height,
   body_cover, lays_eggs). Naive-Bayes ceiling exact & cheap. Cumulative ceiling:
   weight 0.736 → +height 0.873 → +cover 0.909. (Bayes-compute scaling: closed-form
   to ~thousands of features; the wall is leaving the synthetic regime, not the CPU.)

5. **Class-incremental CL bench** (`cl_incremental.py`, 1 species/task). Native
   machinery turns naive **0.20** → **0.85** (97% of joint-achievable). **Mechanism
   ablation (tested, after Rocky caught an overclaim):** *manifold replay does ALL
   the work* (replay-only 0.854); *credit-lock is idle* (bipartite net → only heads
   lockable, no interior to protect); *λ-EWC HURTS* (over-anchors). Three distinct
   mechanisms: replay=manifold, credit-lock=`CreditTracker`, λ=`epigenetic_lock`.

6. **Generative beats discriminative memory.** Hard-exemplar / boundary replay is
   *harmful* (dog → 0): rehearse the **prototype**, not the contested zone. Validates
   the manifold's generative design.

7. **Two-mechanism architecture** (Rocky's correction): **manifold/astrocyte** =
   generative photographic memory (unwired memory shelf); **dendrite** = discriminative
   boundary judgment, consolidated **during dreaming** (the only moment overlapping
   classes co-exist via replay). `cl_dream_dendrite.py`: dream is load-bearing
   (no-dream collapses to 0.10); **dendrite+dream+mix(6) = 0.860 = best CL result**.

8. **dog gap shown irreducible.** Across every lever (storage form, exemplars,
   dendrite capacity, dream) dog stays ~0.52 vs its 0.682 Bayes recall. The overlap
   is **zero-sum** + CE picks a goat-favoring operating point. Only real levers: more
   features or the emergent `dog|goat` label. On the simpler 6-species/weight-only
   problem the machinery hits the ceiling **exactly** (0.837), so this is a property
   of the harder regime, not the machinery.

## Headline numbers

| bench | naive | machinery | ceiling | notes |
|---|---:|---:|---:|---|
| CL 6 species, 1 feat (weight) | 0.332 | **0.837** | 0.837 | **100% of ceiling** |
| CL 10 species, 4 feat | 0.200 | 0.847 | 0.909 | 93%; replay-only |
| CL 10 species + dream+dendrite+mix(6) | — | **0.860** | 0.909 | best CL; goat calibrated |
| dog per-class (any method) | — | ~0.52 | 0.682 | irreducible overlap |

## State of the build

- Branch `v2.0-scaffold`. **Committed this session** (new files + handoff):
  `experiments/growth_exercise/{retrieval_probe,viz_demo,taxonomy,cl_incremental,emergent_labels,cl_dream_dendrite,cl_weight_only}.py`,
  `trioron/lifecycle/grow.py` (λ-gate), `tests/test_v2/test_lifecycle.py` (gate test).
- **Fixes set as defaults** (per Rocky, to be improved next session):
  `cl_incremental.run` → `replay=True, EWC_STRENGTH=0` (validated workhorse);
  `cl_dream_dendrite.run` → `dendrite+dream+mix(6)` (best CL).
- **DO NOT COMMIT / leave alone** (carried since s005): `trioron/bases/developmental.py`,
  `trioron/lifecycle/developmental.py`, `trioron/viz/export.py` (your in-flight viz/stem
  edits). `runs/`, `.claude/` untracked.
- **Council build (s022 step b) NOT started** — still pending.

## Decisions made (the why)

- **EWC off by default** — tested: it monotonically hurts on the small shared net;
  hard credit-lock + replay are the levers (matches the manual).
- **Generative > discriminative memory** — tested: boundary exemplars erase the class
  anchor; the manifold should store prototypes.
- **dog is not a memory/capacity problem** — it's the Bayes overlap + CE operating
  point; stop chasing it with storage tricks.
- **Lean-4 features, class-incremental 1-at-a-time** — Rocky's picks for the first
  small-batch CL test.
- **torch proxy for the dream/dendrite test** — faithful because the input-space
  manifold = a per-class Gaussian; native-substrate port deferred.

## Open questions / next-up

1. **Pick the thread.** Two live paths: **(α)** continue s022's **council build**
   (step b: trial-vote differentiation on disruptor-dog, λ-gate now exists); or
   **(β)** push the **taxonomy CL bench** — close the 0.847→0.909 gap (duck
   consolidation), and the dog operating-point via **cost-sensitive weighting** or the
   **emergent `dog|goat` label**. Ask Rocky which.
2. **Port dream+dendrite to the native substrate** — use the real `dream_cycle`
   (offline) instead of inline rehearsal, with interior dendrite cells (so credit-lock
   has something to protect too). Validate 0.860 holds through the canonical path.
3. **Viz: render the astrocyte memory shelf** — diamonds beside the outputs, unwired,
   dashed-linked to their head; and fix the layered layout into the real
   `trioron/viz/export.py` (currently force-directed/random — hides depth).
4. **Multi-seed** everything (current CL/dream numbers are single-seed).
5. **Carried:** 2 pre-existing `test_lifecycle` failures; s020 growth audit.

## Pointers

- **Datasets:** `taxonomy.py` (10-species, 4-feat, `bayes_accuracy`), `chicken_goat.py`
  (weight-only `SPECIES`).
- **CL benches:** `cl_incremental.py` (ablation harness), `cl_weight_only.py`
  (6/1 comparison), `cl_dream_dendrite.py` (dream + dendrite).
- **Probes:** `retrieval_probe.py` (StreamingMixture on dog), `emergent_labels.py`
  (output-growth), `viz_demo.py` (layered render → `runs/viz_demo.html`).
- **λ-gate:** `trioron/lifecycle/grow.py:divide` (`divide_lambda_max`).
- **Machinery:** `learning/{manifold,credit,epigenetic_lock,dream}.py`. Astrocytes =
  `forward_inclusion=False` (spec §2.11). Manual §4–5.
- **Memory (s023):** [[cl_replay_workhorse]], [[generative_beats_discriminative_memory]],
  [[manifold_astrocyte_dendrite_split]], [[dream_dendrite_validated]],
  [[dog_gap_irreducible]], [[taxonomy_cl_dataset]].
- **Still open from s022:** `docs/design/progenitor_council.md` (council build).

## Environment notes

- `/home/marcrockhat/trioron-project/`, branch `v2.0-scaffold`. Python 3.10.12,
  torch 2.11.0+cu130, WSL2, 12 cores. `python3`, `OMP_NUM_THREADS=8`.
- Exercise/CL runs are fast (seconds–~2 min). All s023 CL numbers are **single-seed**.
- Heavy session (long, many experiments) — flagged per the warn-on-context-pressure rule.
