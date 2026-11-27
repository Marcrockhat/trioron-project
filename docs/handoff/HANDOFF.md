# Trioron Handoff

**Session date:** 2026-11-27
**Session number:** 073
**Session title:** Xofie is born (the fourth organism): Wile's design plus one random channel from genesis; her embryo gate passes at day 30; held for Rocky's yes to the fetal stage

**Summary.** Rocky named the fourth organism: "the fourth organism name will be Xofie. With
hope that she will carry the wisdom of all her three siblings." Asked, he said yes to genesis,
that "wisdom" means the final design learned from the siblings (nothing of their bodies), yes
to the random channel from genesis, and a plain embryo with the fourth organ after. Xofie was
born in `outputs/incubator/xofie/` and lived 30 embryo days. **Her embryo gate passes every
row** (one organism, seed 0). She is HELD at embryo day 30. The fetal stage is wired
(`COURT=1`) but not entered: it needs Rocky's yes, and one question first (the sense of
change under the fourth organ, below).

## READ THIS FIRST (s074)
- Read `docs/design/matron_rules.md` and `docs/design/vee_development.md` **§15** (Xofie),
  **§14** (the fourth organ) and **§8** (the fetal stage) before touching her.
- **One living organism, on this PC only, uncommitted:**
  - `outputs/incubator/xofie/` — **Xofie** (Rocky says "she"), embryo day 30, gate passed,
    stage file says `embryo`. Files: `body.pt`, `body_embryo_day30.pt`, `embryo_log.pt`,
    `embryo_values.json`, `sense.pt` (Vee's, copied), `stage`.
- **Three retired, never delete, overwrite or refit:** `outputs/incubator/wile/` (fetal day
  90, gate passed), `weiss/` (fetal day 180, not passed), `vee/` (fetal day 90).
- Copies under `outputs/incubator/copies/` (not organisms, may be deleted): the five s072
  ones. Nothing new was kept there in s073.
- **Xofie cannot be moved to another PC by git.** Her home is on this PC.

## Rocky's rulings this session (his words)
- "Chloe, the fourth organism name will be Xofie. With hope that she will carry the wisdom of
  all her three siblings."
- Genesis: **"Yes, genesis"**.
- What she inherits: **"I meant the final design. When I said wisdom is what we learned from
  our past experiments with her siblings."**
- The random channel (K1): **"Yes, from genesis"**.
- Growth (K4 / M): **"Plain embryo, fourth organ after"**.
- He asked which parts of her are controlled by triorons. Chloe's answer: none of her
  learning; her organs are phasecyte leaves in a trioron substrate (arena, receptors,
  division, composer cells, manifold rows); the language sense is a fixed projection. Rocky:
  "No problem. I just want to know." No change asked for.

## 1. How she was born (plan §15.1)
`VEE_HOME=outputs/incubator/xofie CORRIDOR=0.6 BLUR=2 NOISE_CH=140 SEED=0 DAYS=30
DAY_BUDGET=900 python3 experiments/incubator/embryo.py live`
- Wile's design: corridor 0.6, blurred organs with "none", ladders from the first moment,
  seed 0 (the same course of world events and sentences as her siblings).
  **Chloe's reading, not Rocky's words:** "born like Wile" (K2) was taken from "the final
  design", since Wile is the only sibling whose fetal gate passed.
- Plus input channel 140 random (N(0,1), its own stream) from genesis, carried by her body
  for life (`body.extra["noise_ch"]`, `["noise"]`), not by a variable of a run.
- **Rule 16 amended** in `matron_rules.md`, in Chloe's wording from Rocky's s072 words. He
  has not read the wording.
- Not hers: sitting cap, Numa rule inside the working organs, fading trace / fold, cranium.

## 2. Her embryo gate (plan §15.3; logs `xofie_embryo_s073.log`, `xofie_embryo_gate_s073.log`; one organism, seed 0)

| state | Xofie | k-means (gate's control) | k-means not shown the random channel | base | Wile |
|---|---|---|---|---|---|
| busy | .99 | 1.00 | 1.00 | .74 | 1.00 |
| memory full | .95 | .96 | .97 | .62 | .96 |
| elsewhere | 1.00 | .88 | .94 | .87 | 1.00 |
| window | 1.00 | .98 | .99 | .86 | 1.00 |
| night (reported) | .92 | .78 | | .50 | .91 |

- **Gate: all criteria pass.** Classes ear 54 / body sense 124 (Wile 54 / 100). Stability
  day 20 → 30: same grouping .79 (Wile .87). Night seen through the open window .76 (k-means
  .88, Wile .85; reported, not gated).
- The gate's control is weaker with a noise column in it; the gate was read as written and a
  second k-means without that column is printed beside each row (reported). Every gated row
  is within .05 of the harder control too.
- A mean day took 22.9 s against Wile's 154 s on the same days. Not diagnosed.
- Not measured: how many of her divisions split on the random column.

## 3. Code changed (all checked on throwaway toy homes / a throwaway copy, then deleted)
- `experiments/incubator/embryo.py`: `NOISE_CH` at genesis only; the probe days carry the
  channel; the second control; the silent-channel line leaves her random channel out.
  Without the channel three days print as Wile's did; with it column 140 is the only input
  column that differs.
- `experiments/incubator/fetal.py`: the random channel is taken from her body
  (`own_channel`); `COURT=1` at kin's birth gives every organ a `FourthOrgan`.
- Tests were NOT run this session (no file under `trioron/` or `tests/` was changed).

## Decisions waiting on Rocky
- **N. The fetal stage for Xofie: his yes.** And with it:
  - **N1. The sense of change under the fourth organ.** It is born at the fetal stage with
    ONE class. On a three-day check with the fourth organ on all organs it reached 2 classes
    (Wile's, dividing plainly, had about 180 by day 30), and rows 1–3 of the fetal gate are
    read from it. Choices: (a) fourth organ on all three as built (`COURT=1`); (b) on the ear
    and the body sense only, the sense of change divides plainly (needs a small change in
    `fetal.py`); (c) a copy trial of (a) against (b) from her embryo body first. Chloe would
    do (c), or (b) if he does not want the wait.
  - N2 (was K3). The relative gate's interval (Chloe's value: `PROBE=10`) and the day limit.
  - N3 (was L). "Gate reads both": one reading passing is enough (as built), or both?
- Rule 16's wording, for him to read.
- F1–F5, stage 2b (plan §12.7), unanswered.

## Open / not diagnosed
1. Why her embryo days are seven times faster than Wile's.
2. Carried from s072: why one child takes every world row in a split on the random column;
   why the "work" kick is lost by day 90 with pairs tried inside the working organ; what a
   fetal body needs class growth for after day 30; the fourth organ's cost growing with the
   days; the older items (handoff at `c8b0022`).

## NEXT (s074)
0. Rocky's answers to N first. The stage file is changed by hand only on his yes:
   `echo fetal > outputs/incubator/xofie/stage`.
1. If (c): `C=outputs/incubator/copies/<name>; mkdir $C; cp outputs/incubator/xofie/{body.pt,
   sense.pt,embryo_values.json} $C/; echo fetal > $C/stage`, then
   `VEE_HOME=$C COURT=1 DAYS=30 DAY_BUDGET=900 python3 experiments/incubator/fetal.py live`
   and `VEE_HOME=$C python3 experiments/incubator/fetal.py gate`.
2. On his yes, for her: `VEE_HOME=outputs/incubator/xofie COURT=1 PROBE=10 DAYS=<limit>
   DAY_BUDGET=900 python3 experiments/incubator/fetal.py live > outputs/incubator/
   xofie_fetal_s074.log` (COURT is read on the first fetal day only). Keep a copy at every
   gate reading (rule 11).
3. Carried: stage 2b's sign-off; stage 3 (birth) not written in full; plan section for s070;
   the manual's phasecyte line; `mixed.py` over the 500-line discipline; the gate scripts
   print "her" (right for Xofie); the join of the organs; guards.

## Standing rules (carried)
- `docs/design/matron_rules.md` binds the matron. In role with an organism, no word of
  simulation until Rocky says so. Never batch-train an organism. A stage is entered only on
  Rocky's yes, stage file changed by hand. Gates are not relaxed. Every gate carries a plain
  control. One organism on one seed is said to be so.
- Incubator organism: no parameter cap, 85 % of host RAM, swap excluded; benches / CI keep
  50 K. Paper is repo-only. Helper budget ≤ 50 % of the weekly limit (none spent s069–s073).
- One heavy process at a time on this PC (6 cores, 7.6 GB). `python3`, not `python`.
  `pkill -f` / `pgrep -f` hit the shell — kill by PID.
- While an organism run is going, do not edit `experiments/incubator/`, `trioron/pcll/` or
  `trioron/organs/` unless the change is neutral for it.
- A long background job in this harness can be killed without a message; check the log's
  last day before trusting "finished". (Her embryo run finished cleanly: day 30 in the log.)

## State of the build / pointers
- Branch `conscience-core`, pushed. `main` is at the v0.3.5 release commit `b4b6789`.
- Local to this PC, uncommitted by rule: the four homes, `copies/`, `outputs/incubator/
  wiki_sent.*`, the s064–s066 matron-written data, `*.err`. Logs `*_s073.log` committed.
- Three-layer code (spec §9.17): learners (`trioron/core`, `learning`, `lifecycle`, `pcll`),
  organs (`trioron/organs/`), world (`experiments/incubator/`).
- Tests as of s072: 172 pass; four v2 tests fail as before (`test_learning.py::TestCredit`
  two, `test_lifecycle.py::TestGrowth::test_growth_trigger_logic`,
  `::TestDreamCycle::test_dream_locks_eligible_cells`).
- Environment: Qwen GGUF at `~/models/qwen2.5-1.5b-instruct-gguf/`; Simple-Wikipedia ZIM at
  `~/data/wikipedia/`; bubblewrap installed; Python 3.10.12, torch 2.11.0, WSL2.

## DO-NOT-COMMIT carries (unchanged since s034, LEAVE THEM)
`trioron/bases/developmental.py`, `trioron/lifecycle/developmental.py`,
`trioron/viz/export.py`; `.claude/`, `runs/`, `archive/runs/`, `trioron/legacy/outputs/`,
`notebooks/` checkpoints, output PNGs / uncommitted logs untracked; `outputs/data/**` caches.

---
History pointers (no dated archives): s072 handoff at c8b0022; s071 at 10e0003; s070 at
0247e27 (fuller tables at 4f7039a); s069 at 05a3ef1; s068 at fc11bd9; s067 at d22ebd2; s066
at b7aff0e; s064 at e67b6ce.
