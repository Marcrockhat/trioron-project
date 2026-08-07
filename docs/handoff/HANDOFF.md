# Trioron Handoff

**Session date:** 2026-08-07
**Session number:** 047
**Session title:** **Conscience-core pivot — council line PARKED; H-space routing
PROMOTED to `trioron/learning/router.py` and validated BIT-EXACT against the
archived chained-15 bench (4 arms + smoke, all MATCH); the 0.76 headline shown to
be ERA-BOUND (doesn't reproduce on current core — cite 0.7111). New deployment
thesis: trioron = task-aware conscience layer for Aidos, adaptable across LLMs.**

---

## READ THIS FIRST

1. **Branch `conscience-core`** (new, off `progenitor-council`). The council /
   progenitor line is **PARKED, not deleted**: its gate (typed growth must
   rediscover the dendrite advantage, s045 item 5) was never met, and Rocky
   confirmed the councils were trioron covering functions it shouldn't. History
   stays on `progenitor-council`. DO-NOT-COMMIT carries continue (bottom).
2. **The pivot (Rocky, this session):** enhance what trioron demonstrably excels
   at — task awareness — and deploy it to Aidos as the conscience layer. Aidos
   context: "project tefilin" moved to a Flip 7 device; Aidos focus returns to
   the heart = trioron's adaptability to different LLMs. Chosen shape: trioron
   reads **model-agnostic** inputs (its own features, not host hidden states —
   learn-to-use-not-from), emits **text-first** context (portable to any host),
   per-host soft-prompt projections later. Trioron's job = infer which
   context/persona/task is live + retrieve/protect the right competence.
3. **Work order agreed:** (1) park council ✓; (2) promote H-routing to core ✓
   (this session); (3) streaming context-shift detection ← NEXT (the §10.2.1
   surprise machinery survives the council pivot as exactly this detector).

## WHAT SHIPPED (all committed on `conscience-core`)

1. **`trioron/learning/router.py`** (commit `b6d9290`) — `ManifoldRouter` over a
   `ManifoldArchive` of interior codes:
   - `route_class` — pure QDA argmax over per-class log-likelihoods (head
     bypassed entirely; the forgetting-prone edges unused).
   - `route_group` / `route_prediction` — manifold picks the task group, head
     logits pick the class within it.
   - `build_h_archive_from_data` (oracle path) and
     `build_h_archive_from_manifold` (storage-free: sample the perception
     manifold, forward through the CURRENT substrate, collect H codes — fixes
     stale statistics). Exported from `trioron.learning`. 6 new tests in
     `tests/test_v2/test_router.py` (all pass; suite 135 pass / 4 PRE-EXISTING
     failures in test_learning/test_lifecycle — they fail with s047 changes
     stashed too, likely the s034 DO-NOT-COMMIT carries; untouched).
2. **`experiments/validate_router_promotion.py`** (commit `21c70f8`) — shim
   runner: aliases `experiments.datasets` → legacy donorkit, loads
   `archive/experiments/bench_chained_15_v2.py`, monkeypatches
   `evaluate_all_tasks` so the final pure-H post-refresh eval ALSO routes with
   the promoted `ManifoldRouter` and compares. **All 5 runs MATCH bit-exact.**
3. **Manual updated** (§ header, §5.5, §7, §9) with promoted status + corrected
   numbers.

## THE NUMBERS (seed 42, full-cov, current core — logs in outputs/validate_router_*.log)

| config | storage-free (manifold refresh) | oracle (real refresh) |
|---|---|---|
| task-mode, 4ep | 0.6833 (task-aware 0.9504) | 0.6962 (0.9507) |
| class-mode QDA, 4ep | **0.6989** (0.9491) | **0.7111** (0.9505) |
| class-mode QDA, 2ep | — | 0.6776 (0.9564) |

**Claim correction (important):** the manual/paper-adjacent "0.76" (commit
`7e561e4`) is ERA-BOUND — it does not reproduce on current core even at its own
2-epoch config (0.6776 today). The archived bench imports LIVE trioron modules;
the core evolved since (dendrite/tanh/growth/credit/soft-apoptosis...). Not a
router bug — bit-exact MATCH rules that out; the same-run bench and promoted
router always agree to the last digit. **Cite 0.7111 class-oracle / 0.6989
storage-free.** Note 4ep > 2ep on current core (old relationship reversed).
Single-seed; n≥3 before any paper use.

## NEXT (priority)

1. **Streaming context-shift detector** — deployment has no task boundaries;
   the router needs a "context changed" signal. Reuse spec §10.2.1's design
   (per-feature surprise `|x−μ|/σ`, Kalman-derived EWMA α, optional
   phase-conditioned comb) — the design survives the council pivot; its
   original niche (arcsin_u magnitude) does not. STILL BLOCKED on
   `P_min..P_max` from Rocky IF the periodic comb is wanted; the aperiodic
   EWMA fallback needs no input and is the deployment-relevant first cut.
2. **Deployment loop: replay + router wired together** (manual §7 open item) —
   a `conscience` API shape for Aidos: enroll context → route → retrieve →
   extend. Then the Aidos bundle: substrate + manifold + router + shift
   detector, text-first interface.
3. **n≥3 seeds on the router numbers** before anything is quoted outside.
4. **Storage note:** full-cov Σ per class is code_dim² (~363 KB at 30 classes
   vs 6.6 KB diag — `7e561e4` note). Selective per-class full-cov upgrade is
   the optimization if the Aidos budget cares.
5. **(Carried, parked)** council-gate items from s045/s046 remain on
   `progenitor-council`: §10.2.1 implementation in receptor.py, dendrite soma
   n=5, raw→dendrite taxonomy swap. Reopen only with a reason.

## OPEN / unresolved

- 4 pre-existing test failures (test_learning TestCredit ×2, test_lifecycle
  ×2) — predate s047; probably the s034 uncommitted carries; diagnose someday.
- The 0.76→0.71 era-drift means OTHER archived-bench claims may also be
  era-bound; re-run before citing any of them.
- EMNIST downloaded fresh this session (562 MB) to the legacy donorkit data
  root — first bench run on a new PC will re-download.

## State of the build / Pointers

- **Core:** `trioron/learning/router.py` (NEW), `manifold.py` (unchanged;
  scoring primitives already lived there), exports in `learning/__init__.py`.
- **Validation:** `experiments/validate_router_promotion.py`; bench source
  `archive/experiments/bench_chained_15_v2.py` (untouched); logs
  `outputs/validate_router_{smoke,real,manifold,real_class,manifold_class,real_class_2ep}.log`
  (untracked, PNG/log convention as before).
- **Docs:** `docs/TRIORON_MANUAL.md` updated (s047 header, §5.5 caveat, §7, §9).
- s045 headline results still stand (manifold replay best CL mechanism;
  bottleneck = acquisition not forgetting; dendrite soma > tanh at same params).

## DO-NOT-COMMIT carries (unchanged since s034, LEAVE THEM)

`trioron/bases/developmental.py`, `trioron/lifecycle/developmental.py`,
`trioron/viz/export.py`; `.claude/`, `runs/`, `trioron/legacy/outputs/`,
`notebooks/` checkpoints, output PNGs/logs untracked.

## Environment notes

- `/home/marcrockhat/trioron-project/`, branch `conscience-core`, Python
  3.10.12, torch 2.11.0, WSL2, `python3` (NOT `python`), `OMP_NUM_THREADS=8`
  (use 4 per process when running two arms in parallel).
- Bench logs buffer: `python3 ... > log` shows 0 bytes until exit; the run is
  fine — check `ps` before assuming a hang.
- Computational framing in code/docs (Gaussian/Mahalanobis/QDA/Kalman);
  optical metaphors stay in prose only.
