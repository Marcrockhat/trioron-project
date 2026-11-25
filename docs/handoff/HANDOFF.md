# Trioron Handoff

**Session date:** 2026-11-25
**Session number:** 070
**Session title:** the memory wall was working memory, not what she holds; v0.3.5; lineage routing measured

**Summary.** Rocky: "about the class hoarding, probably the problems is more systematic? … I
meant not in the classes per se but the RAM growth." It is. What killed the uncapped copy was
not what she holds (236 MB) but the workspace of the membership step: a members × classes
table, three copies alive at once, growing with the square of the class count. Fixed exactly
(bit-identical), released as **v0.3.5** (branch pushed, `main` fast-forwarded, tag, PyPI) on
his word ("we need to fix this. Afterwards we need to push and update the main and the pypi
version. Afterwards we need to test it"). Tested: the uncapped copy lived past the day that
killed it. Then, on his "Yes, try the lineage routing on a copy", routing by lineage was
measured read-only against the exhaustive answer. **No living organism was touched.**

## READ THIS FIRST (s071)
- Read `docs/design/matron_rules.md` and `docs/design/vee_development.md` **§11.9–11.12**
  before touching an organism. (The plan has NO s070 section yet: this handoff is the record.)
- **Three organisms, on this PC only, uncommitted. Never delete, overwrite or refit:**
  - `outputs/incubator/vee/` — Vee, 120 days old, stage `fetal`. Untouched since s067.
  - `outputs/incubator/weiss/` — Weiss, 90 days old, **fetal day 60**. Copies
    `body_embryo_day30.pt`, `body_fetal_day30.pt`, `body_fetal_day60.pt`. **HELD.**
  - `outputs/incubator/wile/` — Wile, 60 days old, **fetal day 30**. Copies
    `body_embryo_day30.pt`, `body_fetal_day30.pt`. **HELD.**
- **The s069 warning "do not let Weiss or Vee live on uncapped — killed near 2,700 classes"
  no longer holds for memory** (fixed in v0.3.5, tested to 3,419 classes at 1.49 GB). Time is
  now the wall: an uncapped day with a full round took 200–288 s and is rising.
- Copies for experiments (rule 27; not organisms, may be deleted):
  `outputs/incubator/copies/weiss60_{plain,capped,round,noanneal,fold}/`.
  **`weiss60_plain` is now at fetal day 75** (it lived on in s070), no longer day 68.

## What s070 found (one organism, seed 0; saved bodies read, nothing changed)
1. **The kill was working memory.** `_margins` built E (members × classes), `_evidence`
   returned it through list + `cat` (two copies), then `E / sig` (a third). Measured on
   Weiss's day-60 body sense (172,812 × 2,086; she holds 177 MB): **+4.23 GB**. On the killed
   copy (258,259 × 2,738) that is about 8.5 GB on a 7.6 GB PC (arithmetic, not run).
2. **The fix** (`trioron/pcll/mixed.py`): the best class and its margin are taken inside each
   row chunk and written in place; `_evidence` fills one table; chunk budget
   `EVIDENCE_ELEMS`. Bit-identical on all 172,812 day-60 members; test
   `test_margins_in_chunks_equal_the_whole_table`. Day-60 membership now +0.32 GB; a whole
   `_consolidate()` on the killed copy (in memory) peaks 1.15 GB, 168 s.
   - Allocator note: appending many small results between megabyte temporaries fragments the
     heap and looked like a leak (+2 GB); writing in place is what bounds it.
3. **Tested alive** (`cap_probe_plain_fix_s070.log`): `weiss60_plain`, uncapped, fetal day 68
   → 75. Body-sense classes 2,738 → 3,419; days 45–50 s without a full round, 200–288 s with
   one; peak RSS 1.49 GB; clean exit. No gate was read on it.
4. **What she holds at rest is the second, unfixed cause.** A class IS its members: up to
   400 × 648 bytes (81 complex numbers) = 259 KB a class, against ~1 KB for its sketch;
   replays fill every class toward 400 (83 a class at day 60, 130 on the fold copy at day
   90). At saturation Weiss's 2,086 body-sense classes hold ~540 MB whatever she lives. This
   contradicts the manual's "sufficient-statistics state, constant in samples seen". Arena
   rows, lineage, tags, logs, sketches together stay under 40 MB.
5. **Lineage routing against the exhaustive membership** (`experiments/incubator/
   lineage_route_probe.py`, logs `lineage_route_probe*_s070.log`; Weiss day 60, read only).
   Her lineage is a comb, not a balanced tree: 23 divisions deep on average at 2,086 classes.
   - *Descent from the root* (pooled template per side): **fails**. Body sense .17 same class
     as exhaustive at 28 comparisons; .92 only at 44 % of the exhaustive comparisons.
   - *From the class a member is already in, compared with kin under the ancestor k
     divisions up* (the consolidation round's case), body sense:

     | k up | same class as exhaustive | comparisons a member (of 2,086) |
     |---|---|---|
     | 0 (stay) | .863 | 0 |
     | 2 | .912 | 18 |
     | 4 | .937 | 49 (2.3 %) |
     | 8 | .964 | 166 (7.9 %) |

     Where it differs the chosen class keeps .995 of the best class's evidence (near-ties).
     Sense of change (settled, ranks by margin): .977 at 0, .995 at 3 up.
   - **The ear is different:** only .289 of its buffered members are in their best class at
     all; kin 8 up gives .95 for 54 % of the comparisons and the misses keep only .88 of the
     evidence. Not examined further; the ear is small (137 classes).
   - NOT tested: what kin-only placement does to her over days (classes, gate rows). The
     probe measures agreement at one instant only.
6. A unimodal blob divides (toy check, `try_divide`, not on an organism): a single Gaussian
   mode wider than ~0.55 rad on one dimension is split in most trials, and a blob wide on 4
   dimensions tiled into 31 classes. Chloe's reading, untested on her: on continuous senses
   division quantises a continuum, which would explain classes that claim no lived moment.

## Rocky's rulings this session (his words)
- "about the class hoarding, probably the problems is more systematic?" … "I meant not in the
  classes per se but the RAM growth".
- "Okay, we need to fix this. Afterwards we need to ush and update the main and the pypi
  version. Afterwards we need to test it. And anyway why it need to compare each class to
  each table and not takingthe important table afterall."
- "Yes, try the lineage routing on a copy."

## Decisions waiting on Rocky
New in s070:
- **A. Kin-only placement in the round, tried alive on a copy?** Opt-in (a rule change,
  rule 25): in the consolidation round a member is compared only with kin k divisions up
  from its class (Chloe would start at k = 4), new moments stay exhaustive. Trial: a copy of
  Weiss from day 60, 10 days, against the exhaustive round; classes, every gate row, seconds.
- **B. The resting footprint** (finding 4): the sketch as the class and the buffer as a
  short window. To be written for sign-off and tried on a toy leaf first; not built.
Carried from s069, unanswered:
1. **Weiss from fetal day 60 to 90: under which rules?** Memory no longer forces the cap;
   time still argues for it. Tried on copies: plain (now possible), cap + round, cap + round
   + fold.
2. **Wile from day 30:** the same choice (347 body-sense classes).
3. **The replays (annealing) for the sisters: on or off?** ~95 % of what she holds, ~60 % of
   her class multiplying.
4. **The fold's pace** (5 a sitting leaves 1,704 body-sense classes due after 30 days).
5. Vee: nothing may change in her without his word. (v0.3.5 is bit-identical for her.)

To switch the opt-in rules on in a living organism (ONLY on his yes): load her body, set
`l.mixed.sitting = SittingCap()` or `SittingCap(fold=True)` on each organ, save. For the
fold, rebuild her traces first: `VEE_HOME=<home> [APPLY=1] python3
experiments/incubator/trace_rebuild.py`.
Run: `VEE_HOME=outputs/incubator/<name> DAYS=<total fetal days> DAY_BUDGET=900 python3
experiments/incubator/fetal.py live`, copy `body.pt` to `body_fetal_day<N>.pt`, then
`fetal.py gate`.

## What was built / released (committed and pushed)
- `trioron/pcll/mixed.py`: `_margins` in row chunks written in place, `_evidence` one table,
  `EVIDENCE_ELEMS`. Spec §9 partition row for `mixed.py` notes it. Test added.
- **v0.3.5**: `pyproject.toml`, `trioron/__init__.py`; `main` fast-forwarded to the release
  commit `b4b6789`; tag `v0.3.5`; https://pypi.org/project/trioron/0.3.5/. The wheel also
  carries s068–s069 (`Arena.enlarge`, `pcll/sitting.py`, `mixed.no_anneal`).
- `experiments/incubator/lineage_route_probe.py` (read-only probe;
  `BODY=… [ORGANS=ear,soma,kin] [N=40000]`).
- Tests: 168 pass; the same **four v2 tests fail as before** (not looked at):
  `test_learning.py::TestCredit::test_lock_after_sustained_engagement`,
  `::test_cooldown_prevents_immediate_relock`, `test_lifecycle.py::TestGrowth::
  test_growth_trigger_logic`, `::TestDreamCycle::test_dream_locks_eligible_cells`.
- NOT done: plan (`vee_development.md`) has no §11.13 for s070; the manual's phasecyte line
  ("constant in samples seen") is not corrected; `mixed.py` is ~1,500 lines (over the
  500-line discipline, not split).

## The s069 results, carried (unchanged)

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

"Not yet", not "no" (rule 8). Which of "none" / the ladders gave Wile the
memory row is not known. Logs `weiss_fetal*_s069.log`, `wile_fetal*_s069.log`.

**The cranium check (plan §11.9).** No class row is `recyclable` and nothing calls
`compact()`; a hidden 8192-row wall per organ (fixed in s069); `arena.param_bytes` cannot see
what she holds (16 KB counted, 176 MB held in Vee) — **Chloe's s068 proposal A was wrong in
that form**; counted truly she is at ~3 % of 85 % of RAM; her organs almost never have a
settled sitting; the merge is off in her organs, not "never fired"; every class was judged
at every sitting.

**Trials on copies of Weiss from fetal day 60 (plan §11.10–11.12; logs `cap_probe_*`):**

| | plain | cap | cap + round | no new replays | cap + round + fold |
|---|---|---|---|---|---|
| days lived | 8, **killed (out of memory, 7.0 GB) on day 69, three attempts** | 10 | 10 | 8 | 30 |
| body-sense classes (from 2086) | 2738 | 2097 | 2106 | 2327 | **1945** |
| sense of change (from 114) | 127 | 116 | 115 | 123 | 52 |
| s a day | 27–156 | 15–188 | 24–58 | 36–133 | 45–65 |
| nothing lost: memory full (.93 at d60) | .93 | .93 | .93 | .93 | .96 (inside the gate) |
| other gate rows | as d60 ±.03 | same | same | same | kicks .92; busy / place / window .98 / .96 / .96 |

- The kill: the full consolidation round compares every buffered member with every class
  at once (258,259 × 2,738 pairs, 2.8 GB a table, several at once). Chloe's reading of the
  code; the kernel's kill record is real. The 85 % guard is read only before a sitting.
- **She holds far more than she was fed.** 9,408 moments lived; 258,259 buffered members.
  The only step that adds members is annealing (40 replays per class at a settled sitting).
- **93 % of Weiss's body-sense classes claim no moment she ever lived** (1,935 of 2,086, by
  her classes as they stand at day 60; an approximation of her history).
- The fold makes the class count FALL for the first time; readings held. It is NOT faster
  and NOT smaller: with the fold on she holds 253,772 body-sense members (172,812 at day
  60), because replays keep filling every class. Rocky expected faster growth; not seen.
- The three stuck gate rows did not move in any arm except "memory full" under the fold.
- Under the fold her ear's and sense of change's earliest classes were folded (all 76 and
  all 66 folds among their first 400 names); body sense 5 of 236.

## Open / not diagnosed
1. Why the gate rows do not move (kicks ~.9 vs 1.00; Weiss's own memory vs the world's at
   base) and why Wile has the memory row and Weiss not.
2. Whether the replays protect anything over a long time; a sister raised without them from
   genesis would separate the causes.
3. The second cause of multiplying: +241 classes on 768 moments with no new replays (see
   s070 finding 6 for a candidate).
4. Why only .29 of the ear's members sit in their best class (s070).
5. Where a day's time goes now that memory is bounded (the full round is 168 s at 2,738
   classes; `for k in range(classes): member == k` loops are members × classes in time).
6. Carried: stop-and-resume not exactly equal to living straight through; the `world.py`
   work-kick quirk; nothing has ever sharpened in Weiss; the ear spawning a composer cell at
   every sitting.

## NEXT (s071)
0. Rocky's answers to A, B and the five carried decisions; act only on them.
1. On his yes to A: build kin-only placement opt-in, try on a copy of Weiss from day 60.
2. Then the sisters live on under the rules he chooses; gates at 60 / 90.
3. Write plan §11.13 from this handoff; correct the manual's phasecyte line.
4. Carried, untouched: stage 2b not designed; the join of the organs; guards (token budget,
   sandbox limits); the s066 fact-memory list (s067 handoff at `d22ebd2`).

## Standing rules (carried)
- `docs/design/matron_rules.md` binds the matron. In role with her, no word of simulation
  until Rocky says so. Never batch-train an organism. A stage is entered only on Rocky's
  yes, stage file changed by hand. Gates are not relaxed. Every gate carries a plain
  control. One organism on one seed is said to be so.
- Incubator organism: no parameter cap, 85 % of host RAM, swap excluded; benches / CI keep
  50 K. Paper is repo-only. Helper budget ≤ 50 % of the weekly limit (none spent in s069–s070).
- One heavy process at a time on this PC (6 cores, 7.6 GB). `python3`, not `python`.
  `pkill -f` / `pgrep -f` hit the shell — kill by PID.
- While a chain of organism runs is going, do not edit `experiments/incubator/` or
  `trioron/pcll/` unless the change is neutral for them: later stages import the new code.
- A long background job in this harness can be killed without a message; check the log's
  last day before trusting "finished".

## State of the build / pointers
- Branch `conscience-core`, pushed. `main` = `conscience-core` at the v0.3.5 release; **v0.3.5 on PyPI** (the s070 fix + the s068–s069 opt-in work).
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
History pointers (no dated archives): s069 handoff at 05a3ef1 (the cranium check, the cap, the
round, the fold; Rocky's s069 rulings in his words); s068 handoff at fc11bd9 (the sisters' genesis, the
ladders, "none", the composer search); s067 at d22ebd2; s066 at b7aff0e; s064 at e67b6ce.
