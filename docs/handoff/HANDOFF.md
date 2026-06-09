# Trioron Handoff

**Session date:** 2026-06-09
**Session number:** 025
**Session title:** **Made the council's five phenotypes GENUINE (recurrent/attention/
conv were linear-aliased stubs); separated experiment from architecture for the
coming API. Then two hard course-corrections from Rocky: the dendrite must spawn
AUTONOMOUSLY (no hand-wiring), and we must IMPROVE the existing organism, not build
parallel scaffolds. The connected council-decides organism is the next build.**

## Summary (the arc)

Continued the from-scratch progenitor–council rebuild. Three real wins this
session — and one important reset:

1. **Genuine phenotypes.** `recurrent`, `attention`, `conv` were empty stub files
   *aliased to `linear.forward_batch`* in the dispatch table — so 4 of the 5
   council phenotypes computed the identical function. Implemented all three as
   genuine ops (spec §3.3–3.5), each proven to **reduce exactly to linear on
   trivial fan-in** (a 1-D scalar). On a 1-D input only the dendrite (quad) is
   nonlinear *by the math* — the other four can only differentiate once the
   multi-data phase gives them structure (multiple tokens / spatial / temporal).
2. **Experiment/architecture separation** (Rocky's request, for the coming API):
   `trioron/` ships (architecture); the clean room moved to `experiments/progenitor/`
   (does not ship). New `trioron/progenitor/` subpackage is the API home;
   mechanism migrates in **as each piece validates**.
3. **The council mechanism — and the reset.** I built a council scaffold and a
   local-frustration signal, but then ran the trial-vote/spawn as **offline,
   isolated, hand-wired simulations** (a fresh substrate per phenotype with a
   daughter I pre-attached to dog). Rocky corrected this twice: (a) **there is no
   hand-wired "dog-dedicated dendrite"** — the structure must spawn it autonomously
   where the data demands; if it doesn't, our code failed to reveal the spawn
   signal; (b) **improve OUR organism, don't build new parallel architecture.** The
   offline harness is the wrong approach. The next build is the real connected
   organism deciding in place.

## State of the build

- **Branch `progenitor-council`.** New commit this session: **`d107d65`**
  (genuine 5-phenotype council + experiment/architecture split — 20 files).
- **Committed (architecture, ships):**
  - `trioron/phenotype/{recurrent,attention,conv}.py` — genuine ops, registered.
  - `trioron/core/arena.py` — per-cell `k_unroll` (recurrent depth, cap 8).
  - `trioron/progenitor/__init__.py` — API skeleton (empty; migration policy inside).
  - `pyproject.toml` — registers `trioron.progenitor`.
  - `tests/test_{recurrent,attention,conv}_phenotype.py` — 6 tests, all pass.
- **Committed (experiment, does not ship):**
  - `experiments/progenitor/` (moved from top-level `progenitor/`): `data.py`,
    `positions.py`, `step1_baseline.py`, `step2_genesis.py`, `step2b_apoptosis.py`,
    `step3_council.py` (connected scaffold 3a ONLY), `step3b_frustration.py`
    (local frustration = the spawn signal).
- **NOT committed — wrong approach, left on disk untracked:**
  - `experiments/progenitor/step3c_spawn.py`, `experiments/progenitor/bounded_dendrite.py`
    — the offline per-phenotype spawn *simulation* + bounded-σ experiment. Superseded
    by the connected organism. Contains the comprehension-gate logic + bounded-dendrite
    idea if useful for reference; otherwise delete.
- **DO NOT COMMIT / carried since s005:** `trioron/bases/developmental.py`,
  `trioron/lifecycle/developmental.py`, `trioron/viz/export.py`. `runs/`, `.claude/`
  untracked. (All verified excluded from `d107d65`.)
- **Published API is still the v1.1 legacy surface** — the rebuild remains API-safe.

## THE NEXT BUILD — the council decides IN PLACE (capture this)

The correct flow, as ONE connected organism (memory `progenitor_council_connected_flow`):

1. Feed the table + progenitor → **input layer** (genesis: aperture → variance
   apoptosis → saliency). Exists: `step2b_apoptosis`.
2. The input layer comes back **already attached to the council** — one substrate:
   `input → 5×4 council → outputs`. Exists: `build_council` in `step3_council.py`.
3. **The council decides, from the data flowing through it, in place:**
   - **what type** — trial-vote among the council's phenotype cells; whichever
     relieves the local (per-output-cell) frustration wins;
   - **how many** — keep committing soma cells until frustration clears (§3.6 stop
     signal = comprehension). **The count is an OUTPUT of the loop, never preset.**
   Use the LIVE substrate spawn (`grow_branch`/`divide`) on the running organism —
   not a rebuilt offline substrate.
4. Committed soma cells = what grew; the council stays standing (germline).
5. **Comprehension gate** (validated this session): a commit must relieve the
   frustrated cell AND regress no other class — else veto.

## Findings this session (honest)

- **Genesis** still converges (step2b: 256 aperture → 250 apoptosed → 1 input node).
- **Local frustration localizes** correctly: on the 6-animal weight testbed, per-
  output-cell residual CE puts **dog alone** above the `mean+1·std` spawn threshold
  (dog residual 0.995). The spawn signal IS revealable — 3b's earlier null was our
  *global-CE* trial, not the data.
- **Offline dendrite OVER-claims.** A dendrite fed to dog-only, scored on dog's
  residual, lifts dog 0.324→0.611 (overshooting Bayes 0.482) by stealing chicken+
  goat; overall regresses 0.807→0.772. The comprehension gate correctly REJECTS it.
- **Bounding σ did NOT help.** `σ=z+z·tanh(z)` ≈ same over-claim (dog 0.622, overall
  0.768). Hypothesis "unbounded z² spills at the tails" **falsified**. Per Rocky's
  principle (change the mandate only if the experiment wins) → **keep shipped
  σ=z+z²**. Root cause is SCOPE (dendrite fed dog only), not the activation.
- These offline numbers are diagnostic only — they were NOT produced by the real
  in-place council. Re-derive them with the connected organism.

## Open questions / next-up (priority order)

1. **Build the connected council decision** (the NEXT BUILD above) on
   `build_council`: detect local frustration → council runs the trial-vote +
   `grow_branch`/`divide` spawn IN PLACE → comprehension gate → loop until clear.
   This replaces `step3c_spawn.py`. Re-derive type AND count from the organism.
2. **Spawn scope** (the open design fork): the dendrite fed dog-only over-claims;
   the gate's regression list (chicken, goat) names dog's true competitors. The
   faithful scope may be "dendrite feature read by dog + its regressed competitors",
   shaped jointly — but let the *organism/gate* decide scope, don't hand-set it.
3. **`grow_branch` single-source limit** (noted): it partitions fan-in by SOURCE
   cell, so it can't split two same-source edges into two branches — the 2-branch
   quad on a 1-D input needed per-edge branch assignment by hand. A `grow_branch`
   variant that splits a single source's edges is likely needed for genuine spawn.
4. **Multi-data-type phase (deferred, Rocky's sequencing):** cement 1-D + CL first,
   THEN mix data types so attention/conv/recurrent become live differentiators.
5. **Carried:** 4 pre-existing `test_v2` failures (credit-lock ×2, growth-trigger,
   dream-lock) — unchanged, verified pre-existing.

## Pointers

- **Design reference:** `docs/design/progenitor_council.md` (+ genesis spec in the
  s024 handoff history). Spec §3.3–3.6 for the phenotype definitions.
- **Run the organism:** `python3 -m experiments.progenitor.step3_council` (3a
  scaffold), `...step3b_frustration` (spawn signal), `...step2b_apoptosis` (genesis).
- **Tests:** `python3 -m pytest tests/test_{recurrent,attention,conv}_phenotype.py`.
- **Substrate primitives:** `trioron/core/arena.py` (`grow_branch`, `k_unroll`,
  `n_branches`, `edge_branch`), `trioron/phenotype/*`, `trioron/lifecycle/{grow,
  developmental}.py` (divide, spawn_stem, MorphogenField).
- **Memories written this session:** `feedback_improve_not_rebuild`,
  `progenitor_council_connected_flow` (both indexed in MEMORY.md).

## Environment notes

- `/home/marcrockhat/trioron-project/`, branch `progenitor-council`. Python 3.10.12,
  torch 2.11.0, WSL2, 12 cores. `python3`, `OMP_NUM_THREADS=8`. **`cd` in a Bash
  compound command persists the working dir — use absolute paths or `cd` back.**
- Version single-sourced from `pyproject.toml` (`0.2.2`) via metadata.
- **Push status:** `d107d65` committed locally; **push pending** (push this + the
  handoff commit before closing). Long session (flagged per warn-on-context-pressure).
