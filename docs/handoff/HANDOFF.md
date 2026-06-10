# Trioron Handoff

**Session date:** 2026-06-10
**Session number:** 026 (long — spanned 06-09→06-10)
**Session title:** **The council grows for real: genesis wired in front, then a major
philosophy reset (no up-weight; ONE training process, not per-cell trials), a
capacity-hard procedural dataset built to give growth real work, and a validated
single-process growth loop that autonomously picks phenotype (dendrite) + count and
climbs the capacity gap.**

## The arc (what changed, in order)

1. **In-place council decision** (commits `cb09468`, `ae421b8`): the council decides
   type+count on ONE living organism; validated on 6-animal and the 10-species/4-feature
   taxonomy. Finding: dendrite differentiates from linear as soon as input is >1-D
   (multi-FEATURE, not the full multi-data-TYPE phase).
2. **Genesis wired in front** (`e8473bb`): perception layer GROWN, not allocated —
   aperture → variance apoptosis (free) → saliency `|w·g|` keep → input layer. Recovers
   all real features; `SAL_STEPS` cut 200→**21** (separates by ~10 steps).
3. **Rocky's design resets (the heart of the session):**
   - **Decision ≠ training.** Deciding which phenotype is a cheap SCREEN (~50 steps,
     need not converge); the real fit is ONE pass at the end. 600 steps was overkill
     (council plateaus by ~150).
   - **(A) No up-weight.** The accuracy DROP we saw from "2 dendrites" was the *aim*
     deliberately trading goat for dog — NOT the capacity (verified: plain-CE capacity
     is neutral/harmless). Adopted (A): plain CE, capacity earns gains, never forces a
     drop. dog stays at its CE optimum unless added resolution finds real signal.
   - **ONE process, not per-cell trials.** Under (A) there's no gate to enforce and
     nothing to roll back, so growth is a single continuous training that spawns cells
     into the LIVE net — not a train-measure-rollback search.
4. **Capacity-hard dataset** (`0db0871`): the easy sets are at their plain-CE ceiling
   (independent features → naive Bayes is linear), so growth had nothing to do. Built
   `data_hard.py`: multimodal class mixtures + wide disruptors, exact Bayes, FIXED spec
   shared across train/test. Canonical **K=32 species, D=12 features**, Bayes 0.937,
   standing council 0.775 (gap 0.163), capacity-recoverable; DENDRITE ≫ LINEAR under
   plain CE (+0.090 vs +0.006 for 40 cells — first clean phenotype differentiation).
5. **Single-process growth loop** (`bf0c667`): `step4_grow.py` + `run_hard.py`. Warm up
   council → spawn trial cohort (one/phenotype) into the live net → read winner from
   saliency `|w·g|` → keep spawning the winner into the same ongoing training until
   plateau. Result on hard taxonomy: 0.777 → **0.842** (closed 0.066 of 0.161), grows
   **20 dendrites** (picked by the data), ONE process, **115 s**, 0.9 GB.

## State of the build

- **Branch `progenitor-council`**, all pushed. Latest: `bf0c667`.
- **Current decision path = `step4_grow.py` (single-process, philosophy A).** It
  SUPERSEDES the aim-ramp logic in `step3c_council_decides.py` (kept as the documented
  predecessor; `run_decision` there still uses the old aim path — consolidate later).
- **Runners:** `run_hard.py` (hard taxonomy, the live testbed), `run_taxonomy_genesis.py`
  / `run_taxonomy.py` (10-species, still on the old `run_decision`).
- **Data:** `data_hard.py` (canonical hard config in `make_split()` defaults),
  `data_taxonomy.py` (10-species, easy), `data.py` (6-animal).
- **`build_council(n_in, n_out, capacity=None)`** now scales arena capacity (was 64).
- **DO-NOT-COMMIT carries (verified excluded):** `trioron/bases/developmental.py`,
  `trioron/lifecycle/developmental.py`, `trioron/viz/export.py`. `runs/`, `.claude/`
  untracked. The viz SVG/PNG of the final architectures live in `runs/viz/` (gitignored,
  regenerable; HTML viewer's CDN dep fails offline — use the SVG/PNG).

## Open questions / next-up (priority order)

1. **Winner-read robustness.** The dendrite-vs-attention saliency margin was thin
   (0.099 vs 0.095) — robust this seed, but could flip. Use a longer cohort probe or a
   more discriminative signal (e.g. marginal-accuracy on a held mini-batch), and run
   multi-seed.
2. **Climb further.** Closed 0.066 of 0.161; the recovery curve reaches ~0.86 at 40
   dendrites. `PLATEAU_EPS` / `GROW_TRAIN` / `MAX_SOMA` are dials — tune so it climbs
   more of the gap before plateauing.
3. **Consolidate decision paths.** Fold `step4_grow` into the shipping flow and retire
   step3c's aim-ramp `run_decision` (or make it call step4). Then promote the validated
   mechanism into `trioron/progenitor/`.
4. **Multi-locus / harder still.** Scale the hard dataset (more species, more modes) and
   confirm the single-process loop keeps growing the right phenotype; attention/conv/
   recurrent still need the multi-data-TYPE phase to ever win.

## Pointers

- **Run it:** `python3 -m experiments.progenitor.run_hard` (single-process growth, ~2 min).
  `python3 -m experiments.progenitor.data_hard` (dataset summary).
- **Design:** `docs/design/progenitor_council.md`. Spec §3.2–3.6.
- **Key primitives reused:** `step3c_council_decides.{snapshot,restore,commit_soma,
  measure,train,train_full}`, `step3_council.build_council`, `lifecycle/grow.divide`.
- **Memories this session:** `progenitor_council_connected_flow` (updated with the
  taxonomy + multi-feature correction). Consider adding: philosophy-(A) + single-process
  growth, and the capacity-hard-dataset recipe.

## Environment notes

- `/home/marcrockhat/trioron-project/`, branch `progenitor-council`. Python 3.10.12,
  torch 2.11.0, WSL2, 12 cores, `python3`, `OMP_NUM_THREADS=8`. **`cd` in a compound
  Bash command persists — use absolute paths.**
- Long blocking waits get auto-backgrounded by the harness; rely on task-completion
  notifications.
- **VERY long session — flagged per warn-on-context-pressure.** All work committed +
  pushed at each step; safe to break here.
