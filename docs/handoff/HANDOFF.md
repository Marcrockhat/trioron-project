# Trioron Handoff

**Session date:** 2026-06-06
**Session number:** 020
**Session title:** **Built D2 (one-organism synthesis) — negative result; two core bugs fixed (graft non-leaf, growth sinks).**
S019 built the reflex-vs-wisdom arc A→D and asked D2 next: collapse the 5-donor reflex
into ONE substrate, lock+λ-anchor it, grow curiosity on top, dream. This session built
the whole D2 stack (`experiments/world/arc_d2.py`) — then the investigation that Rocky's
questions drove turned up the real story: **the manifold router's argmax arbitration does
NOT distill into one feedforward policy** (caps ~55% of the teacher at any capacity), AND
**two genuine core defects** that any caller could hit. **D2 is a negative result; D1's
separated composite stands as the arc's answer. The two core fixes are the keepers.**

> Rewritten in full every session; prior handoffs in git history. Session 019
> (`77bd93e`) built arms A→D and asked for D2 as the full-CL-stack headline.

## Summary

1. **Built `experiments/world/arc_d2.py`** — the full D2 stack, reflex-source-agnostic:
   two reflex routes as an ablation (**graft** = absorb donors' frozen features + settle a
   thin head; **distill** = soft-KL behavioural clone), then the shared CL machinery on the
   live substrate (credit-consolidate + **λ-anchor |w·g|** + **manifold-replay** + **dream**
   + frustration **growth**), the **frozen-reflex urgency gate** (danger→frozen reflex,
   slack→live net), and the **retention probe** (live-net survival, machinery ON vs OFF).
   All v2.0 core (`lifecycle.graft/grow`, `learning.{credit,epigenetic_lock,manifold,dream}`)
   — NOT legacy.

2. **CORE FIX 1 — `lifecycle/graft.py` non-leaf bias.** Grafting a *trained* donor copied
   its `requires_grad` bias via in-place index-assign, making the recipient bias a non-leaf
   tensor → optimizer refused it ("can't optimize a non-leaf Tensor"). Fixed at source: the
   field-copy loop runs under `no_grad` + detach (a transplant is never differentiable).

3. **CORE FIX 2 — `lifecycle/grow.py` divided children were SINKS.** `divide()` wired only
   INCOMING edges; the child had **fan-out 0** and so **never reached the fixed output head**
   — grown capacity was structurally unreachable and **growth was inert**. Verified
   empirically (grown cells fan-out 0; uncapped growth to 672 cells gave zero gain). Fixed:
   the child now projects FORWARD to the parent's consumers (output head included),
   `n_forward` edges, small fresh weights (born connected-but-quiet, SGD grows it),
   strictly-higher-rank → cycle-safe. New `GrowthConfig.project_to_consumers=True`,
   `GrowthEvent.n_forward_edges`. **This aligns with the "head = #classes" principle: the
   head doesn't grow in width, its fan-in does.** After the fix growth is productive
   (train-acc 0.64→0.75, survival 78→84, matching hand-set capacity).

4. **Also fixed in `arc_d2.py`:** the curiosity-growth divide-candidate filter required the
   `LINEAR` gene, but nonlinear substrates carry `DENDRITE` — so the candidate set was empty
   and **growth silently never fired** on exactly the nonlinear nets the relational skills
   need. Now filters any active non-output/non-perception cell. (And the students were built
   LINEAR by default — wrong: donor skills are relational/nonlinear born-quad.)

5. **D2 verdict — NEGATIVE, and airtight.** Collapsing the teacher (5 nonlinear donors +
   full-cov manifold router, 151.7) into ONE substrate plateaus at **~84 survival (~55%)**
   across **every** lever tested — linear/nonlinear, capacity 32→672, hand-set vs grown.
   The binding constraint is the **accuracy→survival gap**: even at 0.82 action-match,
   survival sticks ~80 because the missed ~18% are the survival-critical WARM/FLEE urgency
   switches the single net blurs (dies of BOTH cold and overheat — the s018 overheat ceiling
   re-emerging the instant the router is collapsed). **The router's argmax arbitration is
   load-bearing and non-distillable.** D1's separated composite is the right architecture.

6. **The deeper trap (why D2's premise is self-defeating for THIS reflex):** a *lossless*
   reflex (frozen composite) has no shared trainable params → nothing drifts → the
   retention probe is vacuous; a *shared-param* reflex (distilled) makes forgetting testable
   but costs ~45% of survival. You cannot get both from this composite. The native lock/λ/
   dream machinery only pays off under *forced* parameter sharing (capacity pressure), which
   this world isn't.

## Headline numbers

**Reflex fidelity vs the teacher (standalone survival, n=40), full router:**

| student into one net | cells | BC train-acc | survival |
|---|---|---|---|
| linear | 32 | ~0.60 | 58 |
| nonlinear (quad) | 32 | ~0.63 | 58–61 |
| nonlinear hand-set | 160 | 0.766 | 81.7 |
| nonlinear hand-set | 320 | 0.817 | 78.7 |
| growth, SINKS (pre-fix) | 672 | 0.638 | 77.8 |
| **growth, forward-wired (post-fix)** | 784 | **0.754** | **84.2** |
| graft (absorb + settle head) | 235 donor cells | — | 54.7 |
| **TEACHER (D1 composite = arm A)** | ~160 + router | — | **151.7** |

Ceiling ≈ 84 ≈ 55% of teacher, robust across all variants → routing-collapse, not capacity.

## What was done (files)

- **NEW `experiments/world/arc_d2.py`** — D2 stack (both reflex routes, gate, retention probe,
  curiosity driver with machinery toggle). `--reflex graft|distill|both`, `--machinery on|off|both`,
  `--smoke`. NB: the full ON/OFF run was started then KILLED once the fidelity gate failed
  (graft reflex 54.7 << teacher 151.7) — D2's gated survival can't exceed its reflex.
- **`trioron/lifecycle/graft.py`** — non-leaf bias fix (CORE FIX 1).
- **`trioron/lifecycle/grow.py`** — forward-projection fix (CORE FIX 2).
- Commits: this session (graft + grow + arc_d2 + handoff).
- **`runs/` (LOCAL, NOT committed):** `arc_d2_n200.log` (partial — killed at the fidelity gate).

**Pre-existing, STILL DO NOT TOUCH / DO NOT COMMIT** (carried since s005, uncommitted):
`trioron/bases/developmental.py`, `trioron/lifecycle/developmental.py`, `trioron/viz/export.py`.

**Pre-existing test failures (NOT introduced this session — verified on clean grow.py):**
`tests/test_v2/test_lifecycle.py::TestGrowth::test_growth_trigger_logic` and
`::TestDreamCycle::test_dream_locks_eligible_cells` (2 fail, 10 pass). Likely config drift;
out of scope this session.

## Key findings

1. **The manifold router is non-distillable** into one feedforward policy — collapsing
   loses ~45% of survival at any capacity (linear/nonlinear/hand-set/grown all cap ~84).
   The crisp argmax urgency arbitration is the load-bearing part; a single net averages
   WARM/FLEE and oscillates (dies of cold AND overheat).
2. **`divide()` produced dead-end sinks** — grown cells never reached the output head, so
   growth was structurally inert project-wide for bipartite policy/classifier topologies.
   Now fixed (forward-projection). **This is bigger than D2.**
3. **Two layered growth bugs masked each other:** the LINEAR-gene candidate filter stopped
   divides firing on nonlinear nets at all; even when they fired, sinks made them useless.
4. **D1's separation > D2's one-organism collapse** for this reflex. The CL machinery's value
   is under forced parameter sharing, absent here.

## Decisions made

- **Distill route = an ABLATION, not either/or** (Rocky): graft (his pick) + soft-KL distill
  both tested. Both lose to the teacher; graft worst (54.7) — foreign seed-mismatched donor
  absorption pitfall biting.
- **Fix the API, not the experiment** (Rocky): the graft non-leaf bug fixed in core, not
  worked around in arc_d2.
- **Fix the missing re-wiring mechanism** (Rocky): grown cells must project to consumers;
  fixed in core `divide()`.
- **Record D2 as a negative result; D1 stands as the arc's answer** (Rocky).

## NEXT — project-wide growth audit (the s020 fix's implication)

**The `grow.py` sink fix means every prior "grow under frustration" result needs a sanity
check: did grown cells ever connect to the output head, or were they inert dead-ends?**
- Re-examine the growth-dependent claims: `selective_quad_growth` (relational 0.5→1.0),
  `manifold-grown` chained-15, `self_arrange_depth_result`, CIFAR continual growth.
- For each: did survival/accuracy gains come from the NEW cells, or from recompile/rank
  side-effects? Re-run a key one with `project_to_consumers=False` (old behaviour) vs `True`
  to measure how much the fix changes the result. If the fix lifts them, prior numbers were
  a floor; if unchanged, those callers had another path to the head (check their wiring).
- Decide whether the two pre-existing `test_lifecycle` failures need fixing.

**Deferred:** (a) a genuine shared-parameter forgetting demo on a SINGLE distillable skill
(where one net CAN hold the reflex and drift is testable) — the salvage for D2's vacuous-
retention trap; (b) the s019 deferred D1 multi-seed confirm / θ-anneal.

## Open questions / next-up (carried)

1. **Predator/EVADE** still the dominant death in the deployable reflex (integrity 19/40 @
   s018). EVADE donor weak (~44–49). Donor problem or needs anticipation. (s018 carry-over.)
2. **FORAGE routing recall 0.18** (manifold overlaps HYDRATE). Cheap routing fix.
3. **Promote the router to core** (`trioron.learning.route`) — open since s014. Reinforced by
   s020: the router is the load-bearing arbitration; it deserves first-class status.

## Pointers

- D2: `python3 -m experiments.world.arc_d2 --reflex both --machinery both --episodes 200`
  (the full run; expect the graft reflex fidelity gate to fail ~55 << 152). `--smoke` fast.
- Teacher / reflex: `experiments/world/vocabulary.py` (`build_vocabulary`, `danger`,
  `VocabularyOrganism`, full-cov manifold router). Arc harness: `experiments/world/arc.py`
  (`build_observer`, `run_arm`, `FrozenPolicy`, `TDAgent`, density axis).
- Core fixes: `trioron/lifecycle/graft.py` (no_grad+detach field copy),
  `trioron/lifecycle/grow.py` (`project_to_consumers`, `n_forward_edges`).
- CL machinery: `trioron/learning/{credit,epigenetic_lock,manifold,dream}.py` (Manual §5;
  none auto-runs — the driver wires it). NB credit `consolidate()` locks 0 at a single
  boundary (utility<g_min gate); λ-anchor + manifold-replay carry the soft protection.
- Memory: [[reflex_wisdom_arc_build]], [[reflex_vs_wisdom_arc]], [[overheat_ceiling_routing]]
  (the s018 router split this session re-validated as non-distillable),
  [[self_arrange_depth_result]] / [[selective_quad_growth]] (the growth claims to re-audit).

## Environment notes

- `/home/marcrockhat/trioron-project/`, branch `v2.0-scaffold`. Python 3.10.12,
  torch 2.11.0+cu130, WSL2, 12 cores. `python3`, `OMP_NUM_THREADS=8`.
- Full router build (`router_seeds=120, per_prim_cap=1500`) ≈ 1–2 min; teacher survival 151.7.
- Distill/graft fidelity probes ≈ 2–5 min each (n=40 eval). Growth distiller with recompiles
  on large arenas (700+ cells) ≈ several min.
- The grow.py fix changes growth behaviour for ALL callers (default `project_to_consumers=True`);
  set False to reproduce old (sink) behaviour for the audit.
