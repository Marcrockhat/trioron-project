# Trioron Handoff

**Session date:** 2026-11-25
**Session number:** 069
**Session title:** the sisters' fetal days, and the cranium checked: a cap on the sitting

**Summary.** Rocky: "Lets run them while fixing the cranium system." Weiss lived on to fetal
day 60 (gate not passed, same three rows) and Wile lived her first 30 fetal days (gate not
passed, two rows; she tells her own memory use from the world's at .93 where Weiss stays at
base). The cranium wiring was checked against the code: what was proposed in s068 would have
done nothing. Built on Rocky's rulings: the arena grows when rows run out, and an opt-in
square-root cap on the sitting with the consolidation round in instalments. Tried on three
copies of Weiss: the uncapped copy is killed by the kernel (out of memory) on day 69; the
capped ones live on with flat days and unchanged gate rows. No living organism uses the cap
yet. Vee was not touched.

## READ THIS FIRST (s070)
- Read `docs/design/matron_rules.md` and `docs/design/vee_development.md` **§11.9–11.10**
  before touching an organism.
- **Three organisms, on this PC only, uncommitted. Never delete, overwrite or refit:**
  - `outputs/incubator/vee/` — Vee, 120 days old, stage `fetal`. Untouched since s067.
  - `outputs/incubator/weiss/` — Weiss, 90 days old, **fetal day 60**, stage `fetal`. Copies
    `body_embryo_day30.pt`, `body_fetal_day30.pt`, `body_fetal_day60.pt`. **HELD at day 60.**
  - `outputs/incubator/wile/` — the second sister, 60 days old, **fetal day 30**, stage
    `fetal` (set by hand in s069 on Rocky's s068 yes; `embryo_values.json` written). Copies
    `body_embryo_day30.pt`, `body_fetal_day30.pt`.
- **Do NOT let Weiss or Vee live on without the cap before Rocky decides.** An uncapped body
  sense is killed for lack of memory near 2,700 classes (Weiss has 2,086, Vee 1,781).
- **Her name: Wile or Weil?** Rocky wrote "weil" at the start of s069 and asked "It is Weil
  isn't not Wile?". The s068 record quotes him as "Wile". Chloe asked him to confirm; **no
  answer yet. Nothing was renamed.**
- Copies for experiments: `outputs/incubator/copies/weiss60_{plain,capped,round,noanneal}/` (rule 27;
  may be deleted, they are not organisms).

## Results (one organism each, seed 0)

| fetal gate | Vee d30 | Weiss d30 | Weiss d60 | Wile d30 |
|---|---|---|---|---|
| feels her kicks (k-means 1.00) | .99 | .90 FAIL | .94 FAIL | .89 FAIL |
| what a kick moves | 3 of 5 | 5 of 5 | 5 of 5 | 5 of 5 ("mark" .67) |
| CPU rise, hers / world's | .98 | .95 | .95 | .98 |
| memory rise, hers / world's (base .59) | .62 | .59 FAIL | .59 FAIL | **.93** |
| nothing lost: memory full | 1.00 | .90 FAIL | .93 FAIL | .85 FAIL (embryo .96) |
| nothing lost: busy / place / window | .99 / 1.00 / .99 | .99 / .96 / .96 | 1.00 / .98 / .98 | 1.00 / 1.00 / .99 |
| ear: text present | .65 | .93 | .91 | 1.00 |
| classes ear / body sense / change | 67 / 429 / 141 | 88 / 516 / 61 | 137 / 2086 / 114 | 119 / 347 / 187 |
| s a day (last five) | 3.8 | 18 | 46 | 17 |

Logs: `weiss_fetal_s069.log`, `weiss_fetal_gate_day60_s069.log`, `wile_fetal_s069.log`,
`wile_fetal_gate_day30_s069.log`. "Not yet", not "no" (rule 8). Which of "none" / the
ladders gave Wile the memory row is not known.

**The cranium check (plan §11.9; read from saved copies):**
1. No class row carries `recyclable`; nothing on the phasecyte's path calls `compact()`;
   about half of all rows are retired ones kept for life.
2. A wall nobody chose: 8192 fixed arena rows per organ (`PhasecyteLeaf(capacity=8192)`),
   out near 3,600 classes; past it a class is born rowless, silently. **Fixed in s069.**
3. `arena.param_bytes` counts 4 bytes a row / edge (16 KB for Vee's body sense); what she
   holds is class buffers (176 MB of Vee's 230 MB). A RAM-sized `max_parameter_bytes` would
   read 0.00 for life. **Chloe's s068 proposal A was wrong in that form.**
4. Counted truly she holds ~3 % of 85 % of RAM.
5. Her organs almost never have a settled sitting (class count still on 2 of Vee's 90 fetal
   days), so the consolidation round, annealing, the skeptical gate and sharpening hardly
   run; small classes pile up (405 of 516 under 60 members in Weiss's body sense at d30).
   Chloe's reading for the 84 → 516 jump; not tested.
6. The merge is off in her organs (never passed `merge=True`), not "never fired".
7. Every class was judged at every sitting.

**Trial on copies of Weiss from fetal day 60 (plan §11.10; logs `cap_probe_*_s069.log`):**

| | no cap | cap only | cap + round |
|---|---|---|---|
| days lived | 8, **killed on day 69 (out of memory, 7.0 GB), three attempts** | 10 | 10 |
| body-sense classes (from 2086) | 2738 | 2097 | 2106 |
| s a day | 27–156 | 15–188 | 24–58 |
| gate rows | as Weiss d60 (±.03) | same | same |

Chloe's reading of the kill, not profiled: the full consolidation round compares every
buffered member with every class at once. The 85 % guard in `vee_body.py` is read only
before a sitting and did not see it. Under cap + round: sediment not shrinking yet (1545 of
2106 classes under 60 members; 23 judged and 27 settled a sitting; 1920 classes still
awaiting their first judgment); the ear spawns a composer cell at every sitting (not
examined).

**Her buffers are mostly her own replays (plan §11.11).** The killed copy lived 9,408
moments but its body sense holds 258,259 buffered members in 2,738 classes: about 95 % did
not come from the world. The only step that adds members is annealing (40 synthetic members
per class from its own sketch at a settled sitting). A copy with no NEW synthetic members
(`copies/weiss60_noanneal`, switch `mixed.no_anneal`, no cap, 8 days): 2,086 → 2,327
classes against 2,738 plain; gate rows equal. So new replays are roughly 60 % of the
multiplying, and she still gains 241 classes on 768 moments of world. Not a clean test:
the synthetic members made before day 60 stayed in her buffers. Rocky: "like some dude
with photographic memory but accompanied with schizophrenia"; "it is a missing piece that
need to be sewn to the triorons itself the unforgettfulness actually a double edge sword";
"that's not how suppose the triorons work. It should have very small footprint."
Chloe's proposal, NOT signed, NOT built: fold a long-unused class back into the sibling /
parent it was split from (sketches pool exactly); copy first; earliest classes reported
separately (rule 22). Not repeated on the sisters: the s067 measurement that surplus
classes are thin slices of known states (plan §6.10, Vee at 30 days).

## Rocky's rulings this session (binding; his words)
- "Lets run them while fixing the cranium system."
- Cranium: "It is for them in this environment. It is not the default so no specs changes
  is required."
- Rows: "I thought there's no limit" … "Yes and remove the class limit." (The organisms
  already had `class_cap=None`; the benches' 43 / 128 defaults were left — Chloe told him.)
- The cap: "Square root". On a day's time levelling off: "we accept that as a day for the
  sisters."
- Compaction: "if the cranium is not problem, compaction will make things slower isn't?"
  → not built.
- Merge: off because it confused the organism before (record agrees: design doc D18, s033).
- Retirement: "retirement happened in neurons not in the husks" — Chloe had used "husk" for
  a CLASS emptied by the round (the code's word); neurons (composer cells) are pruned as
  designed.
- The round: "Yes on the round it should run on the sittings. Sediment can be dug up but
  occasionally."

## What was built (committed: `91b6136`, `79cf31e`, `fab0c41`, `46cb36e`, `5115bf3` + this handoff)
- `trioron/core/arena.py` `Arena.enlarge()` — doubles the arena at a boundary; `alloc`
  unchanged (commitment 3); refused past `envelope.max_cells`. Called from
  `mixed.py _refresh` when an uncapped organism has under `ROW_HEADROOM = 64` rows left.
- `trioron/pcll/sitting.py` (new) `SittingCap` — opt-in: `MixedStreamController(
  sitting_cap=True)` or set `mixed.sitting = SittingCap()` on a saved organ. Not judged if
  nothing new; at most `max(4, ceil(sqrt(classes) · 0.5 · φ))` judged a sitting, busiest
  first; the round at every sitting over the judged classes + `max(1, cap // 4)` dug from
  the sediment (`mixed.py _consolidate_some`). φ from growth at the previous sitting only.
  4, .5 and the quarter are Chloe's values. `sitting.work` records every sitting.
  Not in `state_dict` (ship / wake); the incubator bodies are pickled whole.
- Tests: `test_core.py::test_arena_enlarge_keeps_everything`, `test_pcll.py::
  test_class_rows_outgrow_the_arena`, `::test_sitting_cap_bounds_the_judging`.
- Plan §11.9 (the check, proposal) and §11.10 (rulings, build, trial).
- NOT built: the cranium's true byte count (A); compaction / `recyclable` (C).
- **Four v2 tests fail on this branch with or without the s069 changes** (not looked into):
  `test_learning.py::TestCredit::test_lock_after_sustained_engagement`,
  `::test_cooldown_prevents_immediate_relock`, `test_lifecycle.py::TestGrowth::
  test_growth_trigger_logic`, `::TestDreamCycle::test_dream_locks_eligible_cells`.

## Open / not diagnosed
1. Why the three gate rows do not move (kicks .9 vs 1.00; Weiss's own memory vs the world's
   at base; "memory full" lost) — and why Wile has the memory row and Weiss not.
2. The sediment under the cap: does it shrink over 40+ days, or is the dig too rare?
3. The ear's composer spawning at every sitting.
4. Where a capped day's 35–40 s go (not profiled).
5. Carried from s068: stop-and-resume not exactly equal to living straight through; the
   `world.py` work-kick quirk; nothing has ever sharpened in Weiss.

## NEXT (s070) — all waiting on Rocky
0. **Three decisions put to him at the end of s069, unanswered:**
   a. Weiss from day 60 to 90: with the cap and the round, or without as Vee did (without,
      she will be killed for memory near 2,700 body-sense classes).
   b. The same for the second sister from day 30 (to 60 / 90).
   c. Wile or Weil. If Weil: rename her home and correct plan, rules, handoff (ask before
      moving the home; rule 26).
   To switch the cap on in a living organism (ONLY on his yes): load her body, set
   `l.mixed.sitting = SittingCap()` for each organ, save — as done for the copies.
   Run: `VEE_HOME=outputs/incubator/<name> DAYS=<total fetal days> DAY_BUDGET=900 python3
   experiments/incubator/fetal.py live`, copy `body.pt` to `body_fetal_day<N>.pt`, then
   `fetal.py gate`.
   d. Put to him at the very end, unanswered: the fold of long-unused classes (above), and
      whether annealing should stay on for the sisters at all.
1. Vee: the same memory wall is ahead of her (1,781 classes); nothing may be changed in her
   without Rocky's word.
2. The cranium's true byte count (A), if he still wants pressure as a reading she has.
3. Carried, untouched: stage 2b not designed; the join of the organs; guards (token budget,
   sandbox limits); the s066 fact-memory list (s067 handoff at `d22ebd2`).

## Standing rules (carried)
- `docs/design/matron_rules.md` binds the matron. In role with her, no word of simulation
  until Rocky says so. Never batch-train an organism. A stage is entered only on Rocky's
  yes, stage file changed by hand. Gates are not relaxed. Every gate carries a plain
  control. One organism on one seed is said to be so.
- Incubator organism: no parameter cap, 85 % of host RAM, swap excluded; benches / CI keep
  50 K. Paper is repo-only. Helper budget ≤ 50 % of the weekly limit (none spent in s069).
- One heavy process at a time on this PC (6 cores, 7.6 GB). `python3`, not `python`.
  `pkill -f` / `pgrep -f` hit the shell — kill by PID.
- While a chain of organism runs is going, do not edit `experiments/incubator/` or
  `trioron/pcll/` unless the change is neutral for them: later stages import the new code.

## State of the build / pointers
- Branch `conscience-core`, pushed. v0.3.4 on PyPI (unchanged; the composer speed-up, the
  arena enlarging and the sitting cap are not released).
- Local to this PC, uncommitted by rule: the three homes, `copies/`, `outputs/incubator/
  wiki_sent.*`, the s064–s066 matron-written data, `*.err`. Logs `*_s069.log` committed.
- A new organism's language sense: copy `outputs/incubator/vee/sense.pt` into her home
  before genesis.
- Environment: Qwen GGUF at `~/models/qwen2.5-1.5b-instruct-gguf/`; Simple-Wikipedia ZIM at
  `~/data/wikipedia/`; bubblewrap installed; Python 3.10.12, torch 2.11.0, WSL2.

## DO-NOT-COMMIT carries (unchanged since s034, LEAVE THEM)
`trioron/bases/developmental.py`, `trioron/lifecycle/developmental.py`,
`trioron/viz/export.py`; `.claude/`, `runs/`, `archive/runs/`, `trioron/legacy/outputs/`,
`notebooks/` checkpoints, output PNGs / uncommitted logs untracked; `outputs/data/**` caches.

---
History pointers (no dated archives): s068 handoff at fc11bd9 (the sisters' genesis, the
ladders, "none", the composer search); s067 at d22ebd2; s066 at b7aff0e; s064 at e67b6ce.
