# Trioron Handoff

**Session date:** 2026-06-04
**Session number:** 015
**Session title:** Finish the λ restoration — Fisher calibration → **Rocky's catch: the
native λ driver is `|w·g|` saliency, NOT Fisher** → world n=5 verdict (soft λ complements
the hard freeze) → arena survival (world-ceiling, not lock-discriminating) → **fixed the
MEMORY.md overflow that was silently dropping memories** (the actual anti-drift fix) +
rediscovered the forgotten selective-quad capability + manual/spec/API updates →
**conclusive imitation-ceiling diagnosis: substrate & percept CLEARED; the survival cap is
holistic imitation of partly-random masters → use the Pong Mode-E primitive-vocabulary +
dream-loop recipe**

> Rewritten in full every session; prior handoffs in git history. Session 014
> (`e3ed911`) restored λ to v2 core. This session calibrates it, **corrects its driver**,
> validates it on the world, and — the meta-result — repairs the memory index that was
> dropping the very features we keep forgetting.

## Summary

1. **λ Fisher calibration (open item 0a).** The restored λ pinned at `LAMBDA_FLOOR`
   (dead/uniform) because **empirical Fisher (`g²`) vanishes at convergence** (g→0). Fix:
   `fisher_loss` — sample the backward target from the model's softmax (the canonical
   *model-distribution* Fisher). Verified live in `lambda_ewc_smoke.py`: λ off the floor,
   real stability-plasticity trade (retention 0.229→0.488 at s=100).
2. **Rocky's catch (the key turn): the native λ driver is `|w·g|`, not Fisher.** Trioron's
   own importance signal everywhere — **KIBRA edge-tagging** (`dream.py`), the **utility
   `u`**, the **pruner** — is `|weight·gradient|`. Fisher (`g²`) is the imported
   academic-EWC outlier. `|w·g|` keeps the weight-magnitude factor, so it differentiates λ
   at convergence **without** the washout, using plain labels. Probe: `|w·g|` gives 46/2048
   nodes above floor (vs Fisher-true-label 0/2048) — as good as the model-dist band-aid,
   simpler. Added `accumulate_saliency`; swapped it in as the world λ driver.
3. **World validation (open item 0c), n=5 `consolidate_base.py`.** Verdict below: **hard
   FULL-LOCK still wins raw retention + water acquisition; soft λ-WG uniquely *improves*
   the old fire skill while learning water (the freeze holds it static).** Driver ranking
   confirmed **`|w·g|` > Fisher > reward**. λ is a **complement** to credit-locking, not a
   replacement — matching v2's design doctrine. **Arc CLOSED (Rocky's call).**
4. **Arena survival (Rocky's question).** `wg_survival.py`: the WG organism survives
   ~47/300 steps — **indistinguishable from FULL-LOCK (44) and PLAIN (50)**; survival is
   the world's lethality ceiling, not the lock. FULL-LOCK has the tightest variance.
5. **THE META-FIX (Rocky's real concern — "previous-you keeps forgetting features").**
   Root-caused it: `MEMORY.md` was **38.4 KB vs a 24.4 KB limit**, so the index was
   silently truncated and dropped entries — including `selective-quad-growth`. The
   knowledge was saved correctly; the *index* lost it. **Compressed MEMORY.md to 24.2 KB**
   (all 184 links verified resolving, 6 superseded handoff pointers dropped, 1 broken link
   fixed). The dropped entries now load every session.
6. **Rediscovered the forgotten capability.** `selective_quad_growth.py` (session 011)
   already demonstrates **problem-driven phenotype selection**: under relational
   frustration a dividing cell's child takes the **quad/dendrite** phenotype — ~8 quad on
   the relational task (1.000), **ZERO** on a linear task (1.000). The substrate *does*
   adapt architecture to the problem. Pinned in manual §2.5 so it stops being forgotten.
7. **Manual / spec / API updated** to match reality (λ restored, `|w·g|` driver, §2.5
   phenotype selection, spec §8.6 corrected, `epigenetic_lock` exported in the public API).
8. **Conclusive imitation-ceiling diagnosis (Rocky: "test all of it then decide").** The
   organism survives ~47/300 ≈ random while the masters survive 75-95. Built
   `imitation_ceiling.py` (4-arm imitation-accuracy probe) + `det_teacher_confirm.py`
   (toggles the masters' `torch.randint` fallback deterministic). Verdict, three
   hypotheses: **(a) substrate too weak — FALSE** (trioron imitates at the MLP ceiling,
   0.71-0.75 on the clean skill); **(b) percept hides the info — FALSE** (zero observability
   gap, MLP-percept = MLP-full-state); **(c) teacher stochasticity + holistic imitation —
   the cap** (deterministic teacher lifts trioron imitation fire 0.59→0.71, water
   0.37→0.75, organism survival 44→51). The twist: the masters' randomness is *load-bearing*
   (predator-evasion — deterministic masters survive far worse, 85→69 / 77→34). So pure
   (Mode-B) imitation is capped at the teacher. **Decision: the Pong recipe** — clean
   primitive donors (WARM/HYDRATE/FORAGE/**EVADE** = the missing primitive masquerading as
   the masters' randomness) → absorb into a vocabulary organism → frustration→dream
   self-improvement to exceed the teachers. **Substrate & perception are settled — stop
   re-litigating them.** (memory `imitation-ceiling-diagnosis`.)

## Headline numbers

**λ-EWC smoke (`lambda_ewc_smoke.py`, orthogonal-random = EWC's worst case).** Model-dist
Fisher rescues the washout:

| driver | task-A after B | λmax | nodes>floor |
|---|---|---|---|
| empirical Fisher (true label) | 0.346 | 1e-3 = floor (dead) | 0/2048 |
| model-dist Fisher (`fisher_loss`) | 0.488 (s=100) | 1.3e-2 | 40/2048 |
| **`|w·g|` saliency (plain labels)** | — | **1.1e-2** | **46/2048** |

**World consolidation (`consolidate_base.py`, n=5, deterministic cold-battery agreement).**
ONLY the lock differs across arms:

| arm | fire-retain | fire-comp f→w | water-acq |
|---|---|---|---|
| PLAIN | 0.141 ± 0.080 | 0.203→0.139 ▼ | 0.226→0.229 |
| MIRROR-LOCK | 0.187 ± 0.146 | 0.203→0.249 | 0.226→0.220 |
| **FULL-LOCK** | **0.376 ± 0.181** | 0.203→0.213 | 0.226→**0.344** |
| **LAMBDA-WG** | 0.271 ± 0.207 | 0.203→**0.264** ▲ | 0.226→0.239 |
| LAMBDA-REWARD | 0.226 ± 0.180 | 0.203→0.101 ▼ | 0.226→0.260 |

FULL-LOCK wins retention + acquisition. LAMBDA-WG: best competence-through-water of any arm
(0.264 — the freeze only holds at 0.213) and best soft arm. **σ huge (degenerate seeds) —
FULL-vs-WG retention gap is within ~1σ, NOT σ-confident.** Leak ordering FULL>MIRROR>PLAIN
reproduced. (An earlier n=5 with the Fisher/reward drivers — `consolidate_lambda_n5.log` —
had Fisher 0.284 / reward 0.264; |w·g| beats reward on competence decisively.)

**Arena survival (`wg_survival.py`, n=3 × 40 eps, out of 300 steps).**

| arm | survival | per-seed |
|---|---|---|
| LAMBDA-WG | 46.9 ± 9.2 | 48, 58, 35 |
| FULL-LOCK | 44.2 ± **2.4** | 42, 48, 43 |
| PLAIN | 49.7 ± 12.4 | 63, 52, 33 |

All within noise — survival = world lethality ceiling (deaths split across cold/overheat/
thirst/integrity), not the consolidation lock. FULL-LOCK = tightest variance.

## What was done (files)

Committed `6b20113` (mid-session): λ `|w·g|` driver + API export + manual.
- **trioron/learning/epigenetic_lock.py** — `accumulate_saliency` (|w·g| EMA, the native
  driver) + `fisher_loss` (model-dist target). Shares `edge_fisher`/`bias_fisher` buffers;
  rolls up via `refresh_lambda` (row-sum + floor).
- **trioron/learning/__init__.py** — export the epigenetic_lock API (λ was the only
  learning module missing from the public surface) — the "API needs updating" Rocky flagged.
- **experiments/world/consolidate_base.py** — `setup_lambda_wg` / `setup_lambda_fisher` /
  `setup_lambda_reward`; LAMBDA-WG + LAMBDA-REWARD arms; soft `ewc_penalty` in the
  non-mirror-gated TD backward; `--ewc-strength`; `run_arm(keep_sub=)`.
- **docs/TRIORON_MANUAL.md** — λ RESTORED + `|w·g|` driver (§1); §2.5 problem-driven
  phenotype selection (NEW); §5/§8/§9 fixes.

Uncommitted at handoff time (commit with this handoff):
- **paper/v3/spec.md §8.6** — corrected the stale "v2 drops per-weight Fisher" claim to
  record the λ restoration + `|w·g|` driver + world verdict.
- **experiments/world/wg_survival.py** — NEW arena-survival eval for the consolidated
  organism (reuses `run_arm(keep_sub=True)` + `fire_taming.evaluate`).
- **~/.claude/.../memory/MEMORY.md** — compressed 38.4 → 24.2 KB (NOT in repo; lives in
  ~/.claude, does not sync — that's why this handoff exists).

Imitation-ceiling diagnosis (commit with this handoff):
- **experiments/world/imitation_ceiling.py** — NEW. 4-arm imitation-accuracy probe
  (chance / MLP-percept / MLP-full-state / trioron-substrate) → substrate & percept cleared.
- **experiments/world/det_teacher_confirm.py** — NEW. Toggles the masters' random fallback
  deterministic and re-measures imitation + survival → confirms teacher stochasticity is the cap.
- **experiments/world/fire_taming.py / quest.py** — added `EXPLORE_DETERMINISTIC` flag
  (default OFF; one shared global) + deterministic `_explore` fallback. consolidate_base unaffected.
- `runs/` (LOCAL): `imitation_ceiling.log`, `det_teacher_confirm.log`.

**Pre-existing, STILL DO NOT TOUCH**: `trioron/bases/developmental.py`,
`trioron/lifecycle/developmental.py`, `trioron/viz/export.py` (carried session-005).

## Key findings

1. **λ's native driver is `|w·g|`, not Fisher.** Fisher washes out at convergence; `|w·g|`
   (the trioron-wide saliency dialect) does not. Don't reach for Fisher EWC on the
   substrate — use saliency.
2. **Soft λ complements, doesn't replace, hard credit-locking.** Freeze wins retention +
   acquisition; soft λ's edge is *improving* a consolidated skill during new learning. Both
   are legitimate; pick by whether you want a frozen skill or a still-improving one.
3. **Survival is the world's ceiling, not the lock.** Consolidation benefit shows in skill
   retention (battery), not whole-episode survival — many death modes the skill doesn't fix.
4. **The forgetting mechanism was the MEMORY.md index overflow.** Memories were saved fine;
   the index that surfaces them was truncated. Keep MEMORY.md under 24.4 KB — one short
   line per entry, detail in the topic files.
5. **Trioron already does problem-driven phenotype selection** (selective quad growth). Not
   aspirational — validated n=3. The unbuilt part is generalizing the selector to pick
   conv/attention/recurrent (and conv-by-emergence is separately *closed* on the flat
   substrate — use a cortex upstream).

## Decisions made

- **λ driver = `|w·g|` saliency** (default), Fisher available but deprecated-for-substrate.
- **λ arc CLOSED** on the honest result (Rocky); spec §8.6 updated rather than chasing a
  strength sweep on a noisy probe.
- **MEMORY.md hard cap discipline**: ≤ 24.4 KB, terse one-liners, prune superseded handoffs.

## Open questions / next-up

1. **THE next arc (now well-motivated by the imitation diagnosis): the Pong Mode-E recipe
   for survival.** Stop holistic imitation. Build clean primitive donors —
   WARM / HYDRATE / FORAGE / **EVADE** (the missing predator-evasion primitive) — each
   individually learnable (proven ~0.75 fidelity); absorb into a vocabulary organism with a
   router (`pong_vocabulary_organism` pattern); close the loop with frustration→dream
   self-improvement to *exceed* the teachers (Pong −20→+1, Breakout above oracle). This is
   the learn-to-*use* path, not learn-from.
2. **Architecture vision (Rocky): generalize problem-driven phenotype selection** beyond
   quad — a divide-time selector picking conv/attention/recurrent from the frustration
   signature. Its own arc; flat substrate geometry blocks conv (`conv_by_emergence_null`).
2. **(Deferred) λ sharpening** — competence-floor-before-consolidate + n=10 to make the
   FULL-vs-WG gap σ-confident; a LAMBDA-WG strength sweep. Lower priority (arc closed).
3. **Routing to core** (carried from s014 open #2): `trioron.learning.route()` — the
   dual-manifold H-routing is still bench-local (`bench_chained_15_v2.py`).
4. **Spec sweep** for other stale post-restoration claims (§2997, §3032 reference v2
   dropping Fisher — §8.6 fixed, these two lines not yet).

## Pointers

- `experiments/world/` — `consolidate_base.py` (5-arm lock comparison + λ drivers),
  `wg_survival.py` (arena survival), `lambda_ewc_smoke.py` (driver smoke). Each has
  `--quick`/`--smoke`; full = `--seeds 5`. `runs/` (LOCAL): `consolidate_lambda_n5.log`
  (Fisher/reward), `consolidate_lambda_wg_n5.log` (the WG verdict), `wg_survival_n3.log`.
- λ module: `trioron/learning/epigenetic_lock.py` (now in `trioron.learning` public API).
- `selective_quad_growth.py` + memory `selective-quad-growth` — problem-driven phenotype.
- Manual §1 (λ + driver), §2.5 (phenotype selection). Spec §8.6 (corrected).

## Environment notes

- `/home/marcrockhat/trioron-project/`, branch `v2.0-scaffold`. Python 3.10.12, torch
  2.11.0+cu130, WSL2, 12 cores, 7.4 GiB. `python3`, `OMP_NUM_THREADS=8`.
- Benches are CPU/core-bound (~0.7 GB RAM each), not memory-bound; safe to run alongside
  light host (Windows) tasks. n=5 consolidate ≈ 45 min; n=3 survival ≈ 15 min.
- Survival is the noisy whole-episode metric (±9–12 across seeds); the deterministic
  cold/thirsty battery is the low-noise instrument. Seed `torch.manual_seed` before any
  battery/eval (masters' explore uses the global RNG).
