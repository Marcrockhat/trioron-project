# Trioron Handoff

**Session date:** 2026-06-08
**Session number:** 024
**Session title:** **Hard pivot to a from-scratch, supervised progenitor–council
rebuild. Package made self-contained + experiments archived; new `progenitor-council`
branch; clean-room input-layer *genesis* built and validated (Steps 1–2b).**

## Summary (the arc)

The session opened on a dendrite build-conflict and resolved it (spec §3.6 branch
dendrite, commit `efe1f29`), but Rocky then **reset the whole approach**. His
verdict: the v2 experimentation produced too much slop/drift under-supervised, and
he wants to **rebuild the progenitor–council architecture from scratch, one step at
a time, fully supervised — watching the actual nodes and edges grow, patching the
structure one fix at a time.** The animal-taxonomy weight problem is the sandbox.
The rest of the session executed that: a packaging cleanup to get a clean base, a
new branch, and the first clean-room steps (the input layer). **Three commits this
session, all on the new branch's lineage.**

## State of the build

- **Branch `progenitor-council`** (off `v2.0-scaffold` @ `d46a99f`, NOT off `main`
  — `main` is the flat v1.1 layout with **no v2 substrate at all**: no Arena,
  epigenome, divide, core/. The progenitor–council design needs those, so it must
  branch off scaffold).
- **Commits (this session):**
  1. `efe1f29` — v2 dendrite phenotype → spec §3.6 (branch partition + per-branch
     quad + learned α; K=1≡linear). On `v2.0-scaffold`. *Largely superseded by the
     pivot, but it's a correct substrate fix and the tests pass.*
  2. `d46a99f` — **packaging cleanup**: version single-sourced from metadata
     (`__init__` was stale `1.1.0`, now reads pyproject `0.2.2`); the 4 donor/bench
     modules the legacy API+CLI need relocated `experiments/` → `trioron/legacy/donorkit/`;
     **all of `experiments/` archived → `archive/experiments/`** (not shipped); donor
     path **un-broken** (it was dead in this layout — stale flat imports + a sys.path
     hack shadowing stdlib `profile`).
  3. `b2e90c8` — **clean-room input-layer genesis** in `progenitor/` (Steps 1–2b).
- **The published API is the v1.1 *legacy* surface** (`trioron/__init__.py` re-exports
  only `trioron.legacy.*`; CLI = `trioron.legacy.cli:main`). The v2 substrate and the
  clean room are NOT in the public API → the rebuild is automatically API-safe.
- **DO NOT COMMIT / carried since s005:** `trioron/bases/developmental.py`,
  `trioron/lifecycle/developmental.py`, `trioron/viz/export.py` (Rocky's in-flight
  stem/viz edits — the design doc's §4 germline primitives live here). `runs/`,
  `.claude/` untracked.

## THE DESIGN — progenitor input-layer genesis (capture this; it's not in any doc)

Emerged this session in conversation; extends `docs/design/progenitor_council.md`
§3.2/§3.6. The progenitor's job is **input-layer genesis**, and it works like this:

1. **Universal oversized intake.** The progenitor accepts a fixed, oversized
   aperture (the 1.5 Mi = 1,572,864 ceiling, §3.2) regardless of the data's true
   dimensionality. Every problem enters through the same wide aperture.
2. **Position-tagged chunks.** It chunks the aperture into candidate **sensor cells**,
   each given a **position** from a **seeded-Gaussian field = the *spatial S***. Same
   seed → identical positions across organisms → graft/absorption-aligned by
   construction. This is the position-layer analog of the **R·S weight handshake**
   (`W=R·S`, shared subspace S + per-organism rotation R, lossless cross-seed graft —
   see memory `l0_subspace_factor_trump_card`); positions **compose with** R·S, they
   don't replace it. Gaussian → center-dense/foveated (retinal). A 1-D sensor is just
   **one input node** (lowest resolution); finer resolutions are added later by CL.
3. **APOPTOSIS FIRST (the key ordering, Rocky's correction).** A signal diluted
   1/1.5 M IS noise — you can't detect it by pooling. So **variance apoptosis fires
   first**: it's a *statistic* (no training), so it mass-recycles every empty
   (zero-variance) cell immediately and cheaply, removing the diluting "solvent." Only
   the small variance-bearing survivor set goes to the council.
4. **Saliency second.** The council judges survivors by `u = |w·g|` (the **third
   triparametric node param**): a chunk with variance but no *learning* (doesn't
   reduce loss) is recycled; high-`u` chunks feed back to the progenitor and are
   **cloned into a branch** routed to the council.
5. **Converge + disconnect.** For 1-D weight this collapses to **one node** = the
   input layer. The progenitor then **disconnects** (steps back, stays plastic —
   germline, never frozen). **CL:** upgraded data spawns a *new* input layer, **linked**
   to the prior ones to provide multi-resolution.

**Cost / depth:** collapsing 1.5 Mi to 1 by recursive division is `ceil(log_b N)` —
**21 binary, 5 via the council's 20-way (5 phenotypes × 4)**; total ≈ depth × patience.
But apoptosis-first means the cheap variance pass does the bulk cull, so the expensive
(training) saliency runs on only a handful of cells.

**Open design knob — patience / judge-timing.** Don't judge saliency on step 1 (a
slow-to-learn feature would be wrongly recycled). The council owns "keep judging until
the balance topples to a stable verdict" (§3.6 stop signal). In the apoptosis-first
toy this matters less (variance is instant; saliency on a tiny set converges fast).

## Decisions made (the why)

- **Branch off scaffold, not main** — main has no v2 substrate; the design needs it.
- **Relocate donor/bench into the package** (Option B) over leaving them in `experiments/` —
  makes the wheel self-contained; surfaced + fixed pre-existing donor-path breakage.
- **Apoptosis-first ordering** — kills the 1/N dilution problem; the naive
  hierarchical "re-train a pooler per partition and read `|w·g|`" FAILED (kept 51/64
  cols: diluted signal vanishes at coarse levels, `u` not comparable across re-trained
  levels). Deleted; replaced by variance-first.
- **Seeded-Gaussian positions = spatial S**, composing with R·S (not replacing it).

## Clean room — `progenitor/` (Steps 1–2b, all validated)

- `data.py` — self-contained 6-animal weight testbed (dog = `Uniform(2,85)` disruptor;
  weight-only Bayes ≈ 0.84/0.85). No dependency on the archived experiments.
- `positions.py` — seeded-Gaussian sensor positions (the spatial-S handshake).
- `step1_baseline.py` — the floor: 1 perception → 6 linear outputs, every node/edge
  printed. Overall 0.807; **dog collapses to 0.324** (vs 0.482 Bayes) — the gap the
  council must fix.
- `step2_genesis.py` — flat genesis loop: oversized aperture → variance gate (empties)
  + saliency `|w·g|` gate (no-learning noise) → 1 node.
- `step2b_apoptosis.py` — **the canonical version**: variance apoptosis FIRST (mass,
  no training) → saliency on the concentrated few. 256 → 250 culled → 6 → 1. Converges.

## Open questions / next-up (priority order)

1. **Input-layer tail (small):** amplify the surviving sensor → a branch; the
   progenitor **disconnect**. Both near-trivial at 1 node; can fold into Step 3.
2. **Step 3 — the standing 5×4 council + trial-vote differentiation** (§3.3–3.4): the
   council that grows a **DENDRITE for dog** (and stays linear elsewhere), which is
   what closes the 0.324→0.482 dog gap. This is the real differentiation mechanism.
3. **Position-handshake under growth (deferred this session):** same seed aligns
   *birth* positions, but independent growth/recycle diverges the draw order. Needs a
   basis-level invariant (the R·S analog) not per-index seed equality.
4. **CL multi-resolution linking (deferred):** upgraded data → new linked input layer.
5. **Carried:** 2–4 pre-existing `test_v2` failures (handoff s023 said 2; it's 4 —
   verified pre-existing by stashing); the s020 growth audit.

## Decisions/results from the (now-superseded) dendrite thread

- `efe1f29` made the v2 dendrite spec-§3.6-correct (branch-partitioned quad). Validated
  on the rings task + `selective_quad_growth` (1.000 relational, 0 quad on linear).
- BUT the native **selective-dendrite CL bench was a wash** (n=3): linear floor already
  ~0.877 (96% of Bayes), quad trades duck for goat, dog gap stays ~0.58. The
  `cl_selective_dendrite` bench got archived with the rest. Lesson that drove the pivot:
  the taxonomy CL had no headroom; the real test regime is the weight-only disjoint band.

## Pointers

- **Design (the only reference for the rebuild):** `docs/design/progenitor_council.md`
  (+ the genesis spec above). Deliberately ignoring `MEMORY.md` / `paper/v3/spec.md`
  during the rebuild to avoid the doc-overload drift that derailed earlier.
- **Clean room:** `progenitor/` (run `python3 -m progenitor.step2b_apoptosis`).
- **Substrate primitives the design uses:** `trioron/core/{arena,epigenome}.py`,
  `trioron/lifecycle/{grow,developmental}.py`, `trioron/bases/developmental.py`.
- **Handshake:** memory `l0_subspace_factor_trump_card` (R·S), `foreign_donor_pool_vs_seed`.
- **Archived experiments:** `archive/experiments/` (not shipped; not maintained).

## Environment notes

- `/home/marcrockhat/trioron-project/`, branch `progenitor-council`. Python 3.10.12,
  torch 2.11.0, torchvision 0.26.0, WSL2, 12 cores. `python3`, `OMP_NUM_THREADS=8`.
- Version is single-sourced from `pyproject.toml` (`0.2.2`) via metadata; bump there +
  the fallback literal in `__init__.py` on release.
- **NOT pushed yet** — push `progenitor-council` (and `v2.0-scaffold` if wanted) at the
  start/end of next session. Heavy/long session (flagged per warn-on-context-pressure).
