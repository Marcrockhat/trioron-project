# Trioron Handoff

**Session date:** 2026-06-06
**Session number:** 019
**Session title:** **Built the whole reflex-vs-wisdom arc (A→D) — D wins both axes.**
S018 designed the arc (decisions locked) but wrote no code. This session built it end
to end: a common, policy-agnostic **Numa harness** (`experiments/world/arc.py`) plus
all four arms — **A Reflex** (imitation), **B Wisdom** (curiosity/learning-progress),
**C Reward** (TD on world reward), **D Synthesis** (reflex→wisdom). Headline (n=200,
density axis): **D is the only arm that is A-class on survival AND ≥B-class on learning
at the same time** — the reflex→wisdom thesis confirmed, with honest single-seed
caveats. Two mid-session fixes (Rocky-approved) made the comparison fair.

> Rewritten in full every session; prior handoffs in git history. Session 018
> (`c3ac808`) locked the arc design and asked for A→D. This session built and ran all
> four arms and resolved the comparison.

## Summary

1. **Common Numa harness (`arc.py`).** One **external observer** — a small trioron
   substrate with ONLY the 5-d contrastive-pair head — rides along an arm's life and
   learns to predict next-step pairs from the states the arm's policy visits.
   `NumaLedger` scores held-out (dream) loss-drops as **Numa**, train-only drops as
   **Mima**. **Identical across all arms** (the fairness contract); only the POLICY
   DRIVER differs. `run_arm(agent, …)` agent protocol: `.act` required; `.learn` /
   `.start_episode` / `.diagnostics` optional. Survival uses fire_taming's `ep*7000+7`
   seed set → directly comparable to s018's 151.7.

2. **Arm A (Reflex)** = the frozen s018 vocabulary organism (5 donors + manifold
   router) as policy (`FrozenPolicy`). Reproduces s018 exactly (survival 151.7 @ n=40).

3. **Arms B & C = a shared `TDAgent`**, differing ONLY in reward source (clean
   ablation). One driver substrate: value head (acts ε-greedy) + its OWN pair
   world-model (separate from the neutral observer → Numa stays orthogonal to the
   driver). **B** reward = normalized **learning-progress** `r_int = relu(err_before −
   err_after)` (Rocky's pick over raw-novelty/ensemble — noisy-TV immune); **C** reward
   = world reward `r` (organism_v2 scaffold), pair head a trained passenger.

4. **Two fixes that made the comparison fair (Rocky-approved):**
   - **Normalize `r_int`.** Raw Δerror is ~1e-3 — too weak to move the TD value head, so
     arm B was acting ≈ randomly (survival = random floor). Normalizing by running scale
     → O(1) (`r_int mean ≈ 1.0`) makes B a *real* curiosity agent.
   - **Density axis.** Raw *total* net Numa is survival-confounded (a longer life feeds
     the observer more experience → more Numa for free). Canonical axis is now **net
     Numa DENSITY = net per 1k env steps**; survival stays the separate gate.

5. **Arm D (Synthesis) = D1 urgency-gated** (Rocky: "D1 now, D2 after").
   `GatedReflexWisdom`: frozen reflex acts when `max danger ≥ θ` (=0.5) to SURVIVE;
   curiosity `TDAgent` acts in the safe slack to LEARN, and learns **off-policy from
   every transition** so its world-model stays complete. The reflex is a safety net.

6. **Verdict — D wins the arc.** D is the only arm without a glaring weakness:
   end-state survival (last30 **158.6**) ~A, while density (**+0.177**) edges the best
   single arm. **The deeper finding:** naive curiosity (B) does NOT visit more-learnable
   states per experience — it only consolidates **cleaner** (lower Mima). Pure extrinsic
   reward (C) is the worst arm. No single arm but D wins both axes.

## Headline numbers

**ABCD, n=200, density axis, obs_seed=0** (single seed):

| arm | survival (mean) | last30 | **net-Numa/1k** | Mima % | note |
|---|---|---|---|---|---|
| A Reflex | 168.8 | 170.6 | +0.171 | 48% | owns survival; learns no faster than passive |
| B Wisdom | 45.7 | 38.7 | +0.172 | **42%** | cleanest learning, but starves (overheat 86/cold 80) |
| C Reward | 58.0 | 55.6 | +0.103 | 64% | worst both; r_world near-flat −0.03 (signal-starved) |
| **D Synth** | 132.9 | **158.6** | **+0.177** | 44% | **only arm A-class survival + best density** (gate 44/56) |

**Caveats (do not over-claim):** density margin is slim and single-seed → robust claim
is *"A-class survival AND B-class density at once"*, NOT *"D out-learns all"*. D's
*mean* survival (132.9 < A 168.8) pays an early-wander death tax (untrained curiosity);
last30 is the fair end-state.

## What was done (files)

- **NEW `experiments/world/arc.py`** — the whole arc. `build_observer`, `run_arm`
  (agent protocol, density reporting), `FrozenPolicy` (A), `TDAgent`
  (reward_mode=curiosity|world → B/C; normalized `_norm_lp`), `GatedReflexWisdom` (D),
  `arm_{a,b,c,d}_agent`, `--arm A|B|C|D|AB|ABC|ABCD`, `--theta`, `--smoke`.
- Commits: `c27a27e` (harness + A), `0176c17` (B+C + density + r_int norm),
  `82d65a1` (D).
- **`runs/` (LOCAL, NOT committed):** `arc_armA_n40.log`, `arc_AB_n200.log` (pre-fix,
  confounded), `arc_ABC_n200.log` (the A/B/C verdict), `arc_D_n200.log` (the D verdict).

**Pre-existing, STILL DO NOT TOUCH** (carried since s005, uncommitted):
`trioron/bases/developmental.py`, `trioron/lifecycle/developmental.py`,
`trioron/viz/export.py`.

## Key findings

1. **Reflex→wisdom synthesis wins** the arc's both-axes test: bootstrap survival with an
   innate reflex, grow wisdom in the slack it protects. D = A-class survival + best density.
2. **Naive curiosity does NOT out-learn a reflex per experience** (B ties A on density).
   Its only edge is *cleaner* consolidation (lower Mima). Tempers the curiosity thesis.
3. **Pure extrinsic reward is the weakest learner here** (C: worst density, most Mima) —
   partly signal-starvation (world reward near-flat).
4. **Measurement hygiene mattered twice:** un-normalized `r_int` made B ≈ random;
   raw-total Numa is survival-confounded. Both fixes flipped the apparent result.

## Decisions made

- **Arm B driver = normalized learning-progress** `relu(err_before−err_after)` (Rocky:
  over raw-novelty / ensemble-disagreement).
- **Canonical axis = net Numa DENSITY** (per 1k steps), survival = separate gate (Rocky).
- **Normalize `r_int`** to O(1) so curiosity actually steers (Rocky, a deviation OK'd).
- **Arm D = D1 urgency-gated now; D2 full-stack next** (Rocky).
- **NEXT session leads with D2** (Rocky, over multi-seed-D1 / warm-start-D1).

## NEXT — build D2 (the full-CL-stack headline)

D1 proved the thesis cheaply with a *composite* reflex + a *separate* curiosity net
gated by urgency. **D2 makes it one organism and exercises the native machinery:**

1. **Distill the 5-donor reflex into ONE trioron substrate** (a single policy net that
   reproduces the vocabulary organism's actions — behavioural cloning of `org.act` over
   visited states, or absorb the donors). This is the "nowhere to grow on top" wrinkle
   from s018 resolved by collapsing the composite first.
2. **Credit-lock + λ-anchor it** (`learning/credit.py` consolidate at a boundary;
   `learning/epigenetic_lock.py` anchor + `|w·g|` saliency λ) → the reflex is *innate*,
   drift-protected. (Manual §5; memory [[triparametric_node_lambda]].)
3. **Grow curiosity capacity on top** (divide/grow under frustration; the new cells take
   the learning-progress driver) WITHOUT catastrophically forgetting the locked reflex.
4. **Manifold-replay / dream** (`learning/manifold.py`, `learning/dream.py`) so wisdom
   consolidates without overwriting the reflex. Measure on the SAME density axis vs D1.
   Win = D2 ≥ D1 on both axes while the reflex stays intact (survival never craters).

**Deferred (offer to Rocky):** (a) **multi-seed D1 confirm** (n=3–5 × {A,B,D} @200ep) to
turn the slim density margin into a σ-claim; (b) **warm-start / θ-anneal D1** to kill the
early-wander death tax (mean survival → ~A).

## Open questions / next-up (secondary — the moved s018 ceiling)

1. **Predator/EVADE** is still the dominant death in the deployable reflex (integrity 19/40
   @ s018; 84/200 here). The EVADE donor/master is weak (~44–49). Probe whether it's a
   donor problem or needs anticipation. (s018 carry-over.)
2. **FORAGE routing recall 0.18** (manifold overlaps HYDRATE). Cheap routing fix.
3. **Linear WARM/FLEE** — the s018 split predicts linear suffices (no sign-flip).
4. **Promote the router to core** (`trioron.learning.route`) — open since s014.

## Pointers

- Arc: `python3 -m experiments.world.arc --arm ABCD --episodes 200` (full table);
  `--arm D --theta 0.5`; `--smoke` for a fast sanity pass. Density axis = `net-Numa/1k`.
- Numa: `experiments/world/numa.py` (`contrast_targets`, `NumaLedger`); driver scaffold
  `experiments/world/organism_v2.py` (value⊕pair heads, TD + Numa). Reflex policy:
  `experiments/world/vocabulary.py` (`build_vocabulary`, `danger`, `VocabularyOrganism`).
- CL machinery for D2: `trioron/learning/{credit,epigenetic_lock,manifold,dream}.py`
  (Manual §5; none auto-runs — the driver wires it).
- World physics: importing `vocabulary`/`fire_taming` sets tamed temp on `TileWorld`
  (`WARM_RATE=0.04`, `TEMP_LOW=0.02`, `TEMP_HIGH=0.99`). Death causes: `fire_taming._cause`.
- Memory: [[reflex_wisdom_arc_build]] (this arc), [[reflex_vs_wisdom_arc]] (s018 design),
  [[numa_mima_implemented]], [[overheat_ceiling_routing]] (s018 reflex), [[mode_e_recipe_result]].

## Environment notes

- `/home/marcrockhat/trioron-project/`, branch `v2.0-scaffold`. Python 3.10.12,
  torch 2.11.0+cu130, WSL2, 12 cores. `python3`, `OMP_NUM_THREADS=8`.
- Arc n=200: arm A ≈ 3–4 min; learning arms (B/C/D) ≈ 4–6 min each; full ABCD ≈ 15 min.
- Single-seed survival σ ≈ ±9–12 (s018). The ABCD density margins (0.171–0.177) are
  WITHIN single-seed noise — treat D's density edge as suggestive, not σ-confirmed.
- World benches CPU-bound; born-quad donors (the reflex) exceed the Phase-1 50K-param
  contract (exploration only).
