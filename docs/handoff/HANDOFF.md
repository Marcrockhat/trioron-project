# Trioron Handoff

**Session date:** 2026-06-09
**Session number:** 026
**Session title:** **Built the connected council decision the s025 handoff specified:
the council now decides phenotype TYPE *and* soma COUNT in place, on ONE living
organism, autonomously. Validated end-to-end on the 6-animal testbed — dog and goat
both move toward Bayes, the comprehension gate stops the count at the Bayes ceiling.
The trial-vote is a genuine 5-way tie on a 1-D feature (expected); the multi-data-type
phase is what makes phenotypes differentiate.**

## Summary (the arc)

This session built THE NEXT BUILD from s025: the in-place council decision, on the
existing connected organism, no offline simulation, no hand-wired answer. It works.

1. **One living organism throughout.** `build_council`'s `perception → 5×4 council →
   outputs` is the only substrate. A full arena `snapshot`/`restore` (capacity is
   fixed, so a clone of every tensor + the two cursors is an exact rollback) lets each
   phenotype *trial in vivo* — spawn a soma daughter on the real arena, train it,
   measure, roll back — instead of rebuilding a substrate per phenotype (the s025
   rejected approach).
2. **The council decides, autonomously, in place:**
   - **locus** — per-output residual CE above `mean+1·std` localizes to **dog**
     (the disjoint Uniform band), chosen by the data.
   - **type** — trial-vote: each phenotype divides one soma daughter aimed at dog,
     picked by best *gated* relief. On a 1-D scalar all five tie (LINEAR wins the
     tiebreak — it re-routes the council's existing curvature through the adapting
     readout). **No phenotype is privileged** (Rocky's correction this session).
   - **count** — commit loop recruits winner soma via real `divide()`; the count
     **emerges** from the §3.6 comprehension stop signal (net accuracy peaks at the
     Bayes ceiling, then a non-improving soma is rolled back). Canonical run: count=1.
3. **Result (deterministic, seed before build):** dog 0.408→0.453, goat 0.854→0.811
   (both toward their Bayes recalls 0.482 / 0.783), overall 0.838→**0.841 ≈ Bayes
   0.840**. The council stays standing (germline, never pruned).

## State of the build

- **Branch `progenitor-council`.** New commits: **`cb09468`** (in-place decision),
  **`ae421b8`** (taxonomy run + dataset-agnostic `run_decision` refactor).
- **Committed (experiment, does not ship):**
  - `experiments/progenitor/step3c_council_decides.py` — the in-place decision. ONE
    substrate; snapshot/restore rollback; trial-vote; ramped-aim commit loop;
    comprehension gate; `run_decision(...)` is dataset-agnostic. Run:
    `python3 -m experiments.progenitor.step3c_council_decides`.
  - `experiments/progenitor/data_taxonomy.py` — clean-room 10-species/4-feature
    taxonomy (incl. turtle). `experiments/progenitor/run_taxonomy.py` — the runner.
- **Deleted (superseded, were untracked):** `step3c_spawn.py`, `bounded_dendrite.py`
  — the offline per-phenotype simulation + bounded-σ experiment. Both replaced.
- **DO-NOT-COMMIT / carried since s005 (verified excluded from `cb09468`):**
  `trioron/bases/developmental.py`, `trioron/lifecycle/developmental.py`,
  `trioron/viz/export.py`. `runs/`, `.claude/` untracked.
- **Published API still the v1.1 legacy surface** — rebuild remains API-safe.
- **Tests:** the 6 phenotype tests still pass (`test_{recurrent,attention,conv}_phenotype`).

## Findings this session (honest, load-bearing — don't re-derive)

1. **Global CE is blind to the rare hard class.** A soma trained on plain CE against
   the frozen council washes into the field mean (orphaned global frustration in pure
   form). Leverage requires the **readout to adapt**; aim requires the frustrated
   class **up-weighted** — the local learning signal (§3.4). Focal/frustration-weighted
   loss self-calibrates but is too gentle here (dog only reaches 0.42).
2. **dog≈0.40 is the genuine softmax-CE optimum.** CE is a log-likelihood, not 0/1;
   dog = Uniform(2,85) has low flat density so it is never high-confidence, and CE
   settles at dog≈0.40 while the **Bayes argmax** gives 0.48. Relaxing any aimed soma
   back to unweighted CE ALWAYS returns dog to 0.40. The aim re-weight is the only
   thing that crosses the gap; the gate bounds how far.
3. **Comprehension gate binds on OVERALL-HOLDS, not per-class no-regression** (a
   deliberate change from the s025 gate). cat (0.898) and goat (0.854) sit *above*
   their Bayes ceilings by holding dog's overlap mass, so a genuine dog fix MUST pull
   them back — "regress no other class" forbids the exact correction. Over-claim is
   still caught: it drops overall (the s025 dog-only land-grab fell 0.838→0.785).
   **Stop signal = net accuracy peaks** (diminishing returns), NOT local frustration
   clearing — dog's residual stays the field's highest even at Bayes, so the relative
   threshold never fires.
4. **No phenotype is privileged** (Rocky). LINEAR winning is a legitimate outcome —
   on 1-D it re-routes the council's existing curvature. **Keep shipped σ=z+z²** (the
   z IS its linear term); the bounded σ=z+z·tanh(z) stays falsified.
5. **Reproducibility:** seed BEFORE `build_council` — the council's edge init (hence
   its trained basin) is otherwise non-deterministic and dog lands anywhere 0.41–0.52.

## Also this session — taxonomy run (turtle dataset), and a corrected claim

Ran the SAME decision loop on the **10-species / 4-feature taxonomy** (s023 dataset,
brought clean-room into `experiments/progenitor/data_taxonomy.py` + `run_taxonomy.py`;
`main` refactored into a dataset-agnostic `run_decision`). Commit **`ae421b8`**.

- Standing council overall 0.896 (Bayes 0.911); frustration flags **chicken+dog**,
  locus = dog (worst, 0.566 vs Bayes 0.682).
- **Trial-vote: DENDRITE WINS outright** (dog relief 0.674 vs ~0.63 for the linear
  family) — the first genuine phenotype differentiation.
- Commits **1 dendrite soma** (aim 1.6) → dog 0.566→**0.695** (≈ Bayes 0.682), overall
  0.896→**0.901**, gave back duck+goat toward Bayes. Final substrate **38 cells / 395
  edges**, count emergent.

**Corrects the s025/s026 claim that the multi-data-TYPE phase is needed to break the
vote tie.** Only multi-FEATURE is needed: on 1-D all 5 phenotypes tie, but on the 7-d
taxonomy the dendrite's quad models feature interactions a linear re-route cannot, so
it separates with >1 input dim. Tokens/spatial/temporal are still required for
attention/conv/recurrent specifically.

## Also this session — genesis wired in front of the council, + convergence check

Rocky: "wire the genesis in front of the council, stick with the design." Done.
`experiments/progenitor/genesis.py` (+ `run_taxonomy_genesis.py`), commit **`e8473bb`**.
The perception layer is now GROWN, not hand-allocated (design §3.2/§3.6):

- Aperture of **64** candidate sensors (7 real features planted + 5 noise + 52 empty).
- **Phase 1 apoptosis** (variance, NO training): 52 empties recycled in one pass → 12 survive.
- **Phase 2 saliency** `u=|w·g|` (200-step probe): keeps the **7 real**, recycles all 5 noise.
- → input layer GROWN to 7 cells; the SAME council decision runs on the survivors and
  reproduces the hand-allocated result **identically** (0.896 → dendrite soma → dog
  0.566→0.695, 38 cells/395 edges).
- **Fix vs step2b:** the 0.10-of-max saliency threshold (calibrated for ONE planted
  feature) recycled the weak categorical one-hots (kept 3/7); lowered to **0.01-of-max**
  (noise sits at ~0 saliency, well below any real feature) → recovers all 7 exactly.

**Convergence (Rocky's 600-step question):** the standing council hits **0.889 by step
50** and plateaus ~0.896 by **step ~150**; TRAIN_STEPS=600 is conservative headroom
(300→600 buys +0.003, noise-level). Safe to cut to ~200 if speed matters; left at 600
to preserve the committed canonical numbers — **knob, Rocky's call.** The real time cost
is the decision's aim-ramp (SOMA_STEPS=800 × ~7 aims × trials), not the council train.

## Open questions / next-up (priority order)

1. **Promote to `trioron/progenitor/` + multi-locus.** The mechanism (genesis + council
   decision) is validated on both testbeds; migrate it into the shipping subpackage. While
   doing so, generalize the single-locus decision to re-pick the most-frustrated cell each
   round (the taxonomy flagged chicken AND dog; the loop only addressed dog) — work down
   the frustrated set until overall stops improving globally.
2. **One-arena genesis.** Genesis currently runs in its own aperture probe and hands the
   surviving columns to `build_council`; faithful to the design's stage sequence but not
   yet one living arena. In-arena apoptosis (recycle aperture cells + rewire to council)
   is the tighter integration if wanted.
3. **attention/conv/recurrent differentiation** still needs the multi-data-TYPE phase
   (tokens / spatial / temporal) — only the dendrite separates on tabular features.
2. **Promote to `trioron/progenitor/`.** The mechanism (snapshot/restore in-vivo
   trial, frustration locus, ramped-aim commit, overall-peak comprehension gate) is
   validated; migrate it into the shipping subpackage once (1) exercises it on a real
   differentiator. Until then it stays in `experiments/`.
3. **`divide` single-source limit** (carried Q3): a DENDRITE soma child from `divide`
   is K=1 (inherits one perception edge on branch 0); the 2nd branch is added by hand
   to engage the quad. A `grow_branch` variant that splits one source's edges per
   branch is still the clean fix.
4. **CL across this decision.** The council decision is a single-task carve so far;
   wire the validated CL machinery (manifold replay / credit-lock) so committed soma
   survive task boundaries — the s023 finding is that **manifold replay is the CL
   workhorse** on this substrate.
5. **Carried:** 4 pre-existing `test_v2` failures (credit-lock ×2, growth-trigger,
   dream-lock) — unchanged, pre-existing.

## Pointers

- **Design reference:** `docs/design/progenitor_council.md` (§3.3–3.6 phenotype /
  council / division law). Spec §3.3–3.6 for the phenotype op definitions.
- **Run the decision:** `python3 -m experiments.progenitor.step3c_council_decides`.
  Scaffolds: `...step3_council` (3a structure), `...step3b_frustration` (spawn signal),
  `...step2b_apoptosis` (genesis).
- **Tests:** `python3 -m pytest tests/test_{recurrent,attention,conv}_phenotype.py`.
- **Substrate primitives:** `trioron/lifecycle/grow.py` (`divide`, project-to-consumers,
  `divide_lambda_max` plasticity gate), `trioron/core/arena.py` (`grow_branch`,
  `add_edges`, `n_branches`, `edge_branch`, `branch_alpha`), `trioron/phenotype/*`.
- **Memories updated this session:** `progenitor_council_connected_flow` (now carries
  the validated result + the 4 findings). `feedback_improve_not_rebuild` unchanged.

## Environment notes

- `/home/marcrockhat/trioron-project/`, branch `progenitor-council`. Python 3.10.12,
  torch 2.11.0, WSL2, 12 cores. `python3`, `OMP_NUM_THREADS=8`. **`cd` in a Bash
  compound command persists the working dir — use absolute paths or `cd` back.**
- The decision run is slow (trial-vote ×5 + aim ramp ×~7, each 800 steps) — ~2–3 min.
- Version single-sourced from `pyproject.toml` (`0.2.2`) via metadata.
- **Push status:** `cb09468` committed locally; **push pending** (push this + the
  handoff commit before closing).
