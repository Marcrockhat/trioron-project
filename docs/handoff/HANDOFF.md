# Trioron Handoff

**Session date:** 2026-08-15
**Session number:** 049
**Session title:** **Phasecyte session — the PCLL learner is NAMED (Phasecyte,
Rocky) and re-framed as a sibling learner on the trioron substrate; NESTING
transfers to it (3 domain leaves + manifold router, paired +0.071 full over the
monolith, 8× faster); and the first working WAKE/DREAM loop between the two
learners ships: dream-distilled trioron leaves beat their phasecyte teachers
9/9, routed full 0.540±0.037 — best phasecyte-line number ever, no stored
data, no wake gradients.**

---

## READ THIS FIRST

1. **Naming (Rocky's decision, this session):** the PCLL learner is
   **Phasecyte** ("Coheron" rejected — Rust-crate collision; "Phaselite" —
   supplement brand; Phasecyte had zero web hits). PCLL stays as the mechanism
   acronym. Framing pinned in spec §10 naming note: **two learners, one body**
   — trioron (gradient, triparametric (w,λ,u)) and phasecyte (gradient-free,
   phase-coherent). Verified: `trioron/pcll/` has ZERO λ/epigenetic-lock
   references — λ was never in the PCLL loop. Scope = naming/spec only; code
   stays in `trioron/pcll/`; package split deferred.
2. **All headline numbers below are n=3, seeds 0–2, gabor sense, centered
   readout, paired per seed.** The s048 gaming arc is untouched and remains
   the product focus — this session was a phasecyte detour Rocky opened
   ("can the phasor optics perform better if we nest them?"). Answer: yes.
3. **s045's structural close STANDS**: fixed phasor encodings still lose to
   learned readouts *as front-ends*. Nesting won and dreaming won because they
   attack the OTHER wall (s039's routing/readout diagnosis), not that one.

## WHAT RAN (all committed; logs in `outputs/*_s049_*.log`)

1. **`pcll_nested.py`** (`77ef15a`) — 3 domain leaves (digits/fashion/
   letters), each a full gradient-free phasecyte organism, enrolled as its
   domain arrives; class budget matched (cap 43×3 vs monolithic 128); routed
   per sample by core `ManifoldRouter` full-cov QDA over the gabor descriptor
   (routing acc 0.937–0.938 every seed). vs monolithic rerun n=3
   (0.403±0.039 full / 0.694±0.050 ta — seed 0's archived 0.446/0.736 was its
   BEST seed):
   - **nest full 0.474±0.031 (paired +0.071±0.009), ta 0.758±0.035 (paired
     +0.064±0.028) — positive every seed.**
   - **Router is load-bearing:** flat union of the same leaves (no router)
     = 0.261±0.028, far below even the monolith. Mirrors s048 gaming (flat
     fails, nest works) in a pure classification stream.
   - **8× wall-clock** (399s vs 3239s) — observe cost scales with per-leaf
     class count. Also kills the s039 raw-arm-stall problem.
   - Deterministic leaf pathology: per-task 0.00 (task 3) and 0.49 (task 5)
     EVERY seed — digits leaf tiles to cap-43 inside task 1, later classes
     unnamed/evicted ≈0.07 ta headroom. Oracle routing +0.036 full more.
2. **`measure_pcll_size.py`** (`77ef15a`) — exact byte audit
   (`outputs/pcll_size_audit_s049.log`): deploy-minimal = **406 KiB
   templates+names** (parity with gradient v0.2.2 substrate 363 KiB,
   `validate_router_real.log:119`) **+ 1,805 KiB router Σ (82% of deploy —
   shrinkable: diag ≈9 KiB, cost unmeasured)**. Phasecyte state = sufficient
   statistics, FLAT in samples: beats raw retention past ~720 samples; hippo
   K=20 costs 4.5× the templates. IGNORE the audit's "641 MB training state"
   line — walker counted preallocated capacity buffers, not content.
3. **`hybrid_nest.py`** (`2f04978`) — **the wake/dream loop.** Wake =
   phasecyte single pass. Dream = per-domain gradient trioron leaf (`Seeded`
   nonlinear quad, H=64, 8ep Adam) trained on pseudo-pockets sampled from the
   phasecyte's OWN per-class manifold sketches — all 10 domain classes
   jointly, so the leaf sees a STATIONARY set: **the nest converts continual
   learning into per-leaf joint learning; the gradient leaf needs no CL
   machinery.** Same router.
   - **Student beats teacher in ALL 9 domain×seed cells** (e.g. seed 0:
     digits 0.509→0.591, fashion 0.554→0.619, letters 0.580→0.662).
   - **Routed full: dreamed 0.540±0.037** vs phasecyte-nest 0.474±0.031
     (paired +0.066±0.021) vs monolith 0.403±0.039 (paired +0.137±0.020).
     Gradient-stack headline 0.601 (n=10, 8ep/task, full CL machinery) is now
     within ~0.06 — with zero stored data and zero wake gradients.
   - Mechanism: dreaming installs the DISCRIMINATIVE readout s039 prescribed,
     without touching the wake path. This closes the s039→s045 loop.
   - "Hybrid" arm currently = oracle per-domain pick (chose gradient 3/3 at
     full data); phasecyte's irreplaceable role is the EARLY stream — the
     honest hybrid claim is the accuracy-vs-samples curve, not the endpoint.
4. **Naming/spec pass** (`77ef15a`): spec §10 re-headed "Phasecyte — PCLL"
   + naming note; §9.15; directory-tree comment; manual §2 PCLL block;
   both design docs. Paper has no PCLL mentions — untouched.
5. **API PROMOTION + v0.3.0** (Rocky's call, end of session): the nest +
   wake/dream machinery promoted to **`trioron/pcll/nest.py`**
   (`PhasecyteLeaf`, `PhasecyteNest`, `dream_distill`, `dreamed_predict`;
   exported from `trioron.pcll`); both experiments refactored to thin
   drivers over the package. **Parity validated BIT-IDENTICAL** at smoke
   scale for both paths (nested 0.403/0.432/0.322/0.591/0.936; hybrid
   0.403/0.386 + matching dream losses). Version bumped **0.2.2 → 0.3.0**
   (pyproject + `__init__` fallback; editable install refreshed). Spec
   §9.15 partition row added BEFORE code landed, per the house rule.
   Tests: 131+6 pass; only the 4 known pre-existing failures.
   **`conscience-core` fast-forward merged into `main`** (main was 227
   behind, 0 ahead) and both branches pushed. **Released to PyPI:**
   https://pypi.org/project/trioron/0.3.0/ (twine, wheel + sdist,
   both checks PASSED) — the first release since 0.2.2.

6. **WORLD SHOWCASE (Rocky's ask, end of session):**
   `archive/experiments/world/world_phasecyte.py` — 5 phasecyte leaves
   absorb the skill masters (banded pairs, ACTION-BLOCKED stream — the
   chained class-sequential analog; window 50, without which division
   forms 1-2 classes named the majority action), manifold router, all
   through the released v0.3.0 API. **n=3 over 40 maps: wake 47.4±4.6 —
   beats the recorded DQN (37.9) every seed, one gradient-free pass vs
   300 TD episodes. In-world dream lift is a WASH (+2.0±3.8; one seed
   negative)** — unlike chained-15's 9/9; the wake matched filter is a
   weak action readout (train-acc 0.545 vs majority 0.423 on WARM) and
   dream inherits class impurity. Public page: **`tour/phasecyte.html`**
   (linked from tour index; GIF `tour/assets/phasecyte_vs_dqn_map987654.gif`,
   nest t=67 vs DQN t=13; wash stated plainly). Commit `7b59835`,
   merged to main → live on GH Pages.

7. **OPTIMIZED-ROW CONTROLS (overnight, Rocky's 2×2):**
   - **TD trioron router n=3: 148.5±12.9** (163.2/143.2/139.2,
     `runs/router_td_s049_seed{0,1,2}.log` → outputs/). **HEADLINE
     CORRECTION: s048's 163.2 was the best seed.** Claim "TD ≥ Gaussian
     nest (148.1)" survives on the mean, barely.
   - **DQN @3000 episodes (10×) n=3: 49.0±14.3** (62.3/50.9/33.7,
     `outputs/dqn3000_s049.log`) — 10× budget lifts DQN only to the
     wake-nest's one-pass level; optimized nest 3× above.
   - 2×2 published on `tour/phasecyte.html` with the
     demonstrations-vs-tabula-rasa asymmetry stated (BC-warm-started
     DQN = unrun symmetric control). Per-map DQN distribution measured:
     σ=15, modal death t=13, 10/40 maps ≤20 ticks.
   - Map rendering upgraded to tile GLYPHS (shape+color, legend,
     night shading) in `watch_duel._panel` — all future duel GIFs
     inherit it.

8. **STRUCTURAL DREAMING VALIDATED n=3** (Rocky: "self-reflect, break
   from its limit") — `world_dream_newleaf.py`: baseline TD nest →
   organism reads its OWN cause-of-death table (diagnosis maps disjoint
   from eval) → new trioron leaf TD-trained on the implicated drive's
   delta only (NO master) → 6-leaf cold-start router → eval. **Paired
   +1.7/+23.6/+5.8 (mean +10.4; 148.5→158.9); diagnosed cause shrank
   3/3 seeds; seed 2 self-diagnosed integrity (not cold) and birthed a
   different leaf — per-individual adaptation.** Extended nests near
   the arbiter 167.6. Follow-ups: incremental enrollment (warm-start),
   iterate the loop. Logs `outputs/dream_newleaf_s049_seed{0,1,2}.log`.
9. **CHECKPOINT BUG (found via showcase render):** `watch_duel`
   router ckpt saved bias+edge_weight only — quad-dendrite
   `branch_alpha` (trainable!) dropped; reload = lobotomized router
   (89.7 vs 163.2). FIXED + round-trip verified (163.25=163.25;
   map-987654 replay = s048's t=144 exactly). **Rule: substrate ckpts
   must save every tensor in `trainable_tensors()`.** Old
   `runs/duel/router_td.pt` was replaced (retrained, reproduces).
   Showcase page: colored in-GIF+HTML legends, speed controls
   (½/1/2×, fps-variant GIFs), optimized duel added (TD nest t=144 vs
   DQN@3000 t=30, `render_optimized_duel.py`; dqn3000.pt ckpt saved).

10. **NEST-AS-TEACHER (kickstarted DQN) n=3** — `world_kickstart.py`
   (Rocky: "can the DQN be trained by our model?"). Same student/budget
   (300 eps), three knowledge paths:
   - tabula rasa 37.9±2.7 (matches control; CURVE DEGRADES — peaks
     ep50≈49 then declines as it learns fire-camping and cooks);
   - BC warm-start + TD 40.9±1.1 (clone acc 0.926 — the nest is far
     more clonable than the perfect master's 0.73 — but deploy +3
     only: the missed 7% are the critical fire decisions). SETTLES the
     page's symmetric-control caveat: demonstrations alone don't
     rescue the flat learner;
   - **teacher-guided exploration 57.2/33.7/139.4 — BIMODAL: 1/3
     seeds locks onto the teacher's strategy at near-teacher level
     (139 vs 148.5).** KEY CORRECTION: the flat QNet CAN represent
     near-teacher play — the barrier is exploration/optimization, not
     capacity. Next lever if pursued: target net / annealed teacher
     mixing to raise the hit rate.
   Logs `outputs/kickstart_s049_seed{0,1,2}.log`; page section added.

## GOTCHAS (hard-won this session)

- Substrate gradient training needs **`sub.prepare_training()`** — plain
  `compile()` + `trainable_tensors()` leaves requires_grad off → loss has no
  grad_fn.
- Long benches at 2×4 threads can die SILENTLY (mono seed 1 died at task
  5/15, log just stops; the s048 pgrep-self-match trap struck again during
  diagnosis — check log mtime, not process greps).
- Rocky Q&A settled on the record: N_QUANTA is **1000** (not 100); quanta
  count sets angular resolution, NOT bank size (banks are C×F complex sums)
  — coarse quanta is an int8-compression lever, not a simplification;
  logits→quanta works via softmax (non-negative → receptor-native).

## NEXT (priority)

0. **Close the in-world dream gap** — the wash traces to wake class
   impurity (dream trains pseudo-samples under the class's MAJORITY
   action). Candidates: purer class formation (smaller windows /
   per-action enrollment), or dream targets weighted by tap
   composition instead of majority naming. Chained-15 says the
   mechanism works when classes are clean.
1. **Learned arbitration for the hybrid** (replace the oracle pick): route
   phasecyte-vs-gradient per domain by sketch-space confidence, or the
   consequence-taught substrate router (reward = correctness) — would also
   answer "100% trioron router" for the phasecyte nest.
2. **Accuracy-vs-samples curve** — the honest hybrid headline: phasecyte
   answers from sample 1, dreamed leaf overtakes, curve dominates both pure
   nests at every point.
3. **Linear-probe-on-pockets control** — how much of the +0.066 dream lift is
   readout-generic vs dream-specific. Cheap; run before quoting mechanism.
4. **Fix the cap-43 tiling pathology** (tasks 3/5 die every seed) — ≈0.07 ta
   free; likely naming/eviction under cap pressure, not learning.
5. **Router Σ diet** (diag/low-rank; 1.8 MiB → KiB) + **int8 phasor
   templates** (unit-modulus → dream-archive demotion should be ~lossless).
6. **s048 gaming NEXT items unchanged** (dream_loop wiring, incremental
   enrollment, nonlinear leaf retrain, n≥3, NAV leaf) — the game arc is still
   the product; this session's wake/dream loop is directly reusable for its
   day/night mechanic.

## OPEN / unresolved

- 4 pre-existing test failures (test_learning TestCredit ×2, test_lifecycle
  ×2) — predate s047, untouched.
- s047 parked items unchanged (novelty-alarm probe, conscience API bundle,
  router n≥3) — behind the game arc.
- N_QUANTA ablation ({1000,100,10}) discussed, not run — N_QUANTA is a core
  constant, needs a parameterization to sweep.
- Package split for phasecyte deferred (naming/spec only this session);
  arena lock-in rows are the awkward coupling if/when it happens.

## State of the build / Pointers

- **Commits (branch `conscience-core`):** `77ef15a` (nested + size audit +
  naming pass + memory), `2f04978` (hybrid nest + mono seed 2 log),
  session-final commit adds mono seed 1 log + this handoff.
- **Scripts:** `experiments/progenitor/{pcll_nested,measure_pcll_size,hybrid_nest}.py`
  (pcll_nested.run_seed takes `return_state=True` — hybrid + audit reuse it).
- **Logs (committed):** `outputs/pcll_nested_s049_gabor{,_seed1,_seed2}.log`,
  `outputs/pcll_size_audit_s049.log`, `outputs/hybrid_nest_s049_{full,seed1,seed2}.log`,
  `outputs/pcll_chained15_s049_mono_seed{1,2}.log`.
- Monolithic per-seed (centered): seed0 0.736/0.446 (archived s039), seed1
  0.709/0.393, seed2 0.638/0.369.
- s048 gaming state unchanged (duel GIF, checkpoints, router 163.2) — see git
  history of this file.

## DO-NOT-COMMIT carries (unchanged since s034, LEAVE THEM)

`trioron/bases/developmental.py`, `trioron/lifecycle/developmental.py`,
`trioron/viz/export.py`; `.claude/`, `runs/`, `trioron/legacy/outputs/`,
`notebooks/` checkpoints, output PNGs/uncommitted logs untracked.

## Environment notes

- `/home/marcrockhat/trioron-project/`, branch `conscience-core`, Python
  3.10.12, torch 2.11.0, WSL2, `python3` (NOT `python`), `OMP_NUM_THREADS=8`
  (4 per process when two run in parallel — but see GOTCHAS: two 4-thread
  benches + a third job can silently kill one).
- Phasecyte benches run from repo root:
  `PCLL15_SENSE=gabor python3 -m experiments.progenitor.pcll_nested`
  (~7 min/seed); hybrid: `... -m experiments.progenitor.hybrid_nest`
  (~20 min/seed, env HYB_SEED/HYB_PSEUDO/HYB_EPOCHS/HYB_HIDDEN).
- Bench logs buffer — 0 bytes until exit is normal; check log mtime before
  assuming a hang.
