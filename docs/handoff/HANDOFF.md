# Trioron Handoff

**Session date:** 2026-08-18
**Session number:** 051
**Session title:** **Nest-of-nests vs absorption — combining three seed-different
drive organisms into one. Answer: BOTH beat every single organism and the
master-built nest; ABSORB (head-merged graft, exact Q-sum of sibling
leaves, leaves untouched, band router retrained) 190.6±22.3 ≥ nest-of-nests
180.0±7.6 ≥ zero-training majority vote 179.0 ≫ best single 159.0 (s050
band nests 128.2±26.7; s049 master nest 148.5). Post-absorb TD "settle" of
the fat leaves DESTROYS it (98.2±9.6). Package: `lifecycle/graft.py` gained
`merge_output=True` (v2 analog of v1.1 pool-matched absorb) and now carries
dendrite state — the shipped graft silently LINEARISED quad-dendrite donors
(n_branches/branch_alpha/edge_branch were never copied).**

---

## READ THIS FIRST

1. **Rocky's question:** "we have different triorons from different seeds.
   As an organism it should be able to absorb. Is a nest of nests
   comparable to absorb?" Scoped by Rocky: world drive nests (s050 band-
   masked, seeds 0/1/2), same task (committee), absorb = the existing
   pool-matched absorb primitive.
2. **v1 `api.absorb`/`pool_matched_absorb` cannot consume v2 substrates**
   (they take v1 `TrioronNetwork`). The v2 counterpart is
   `lifecycle/graft.py`; as shipped it (a) copied donor OUTPUT cells (→
   12-wide head for two 6-action leaves) and (b) dropped n_branches /
   branch_alpha / edge_branch / λ / k_unroll (→ quad donor becomes
   linear). Both fixed this session (Rocky approved "head-merged v2 graft
   into the package"): `graft(rec, donor, merge_output=True,
   wiring="none", freeze=False)` → `merged(x) == rec(x) + donor(x)` (fp32
   6e-6 measured on real drive leaves and routers). Spec §5.3 paragraph +
   §9.6 row added; `tests/test_v2/test_graft.py` (4 tests, new — there
   were NO graft tests before).
3. **All numbers: fixed 40-map eval (`fire_taming.evaluate`, maps
   ep*7000+7), tamed physics.** Only three organisms exist, so combination
   is n=1 on organisms; **n=3 is on the ROUTER seed** (rseed 0/1/2 → seed
   line 700+) for the learned arms. Single-organism numbers reproduced
   bit-exact from s050 (112.6/113.0/159.0).

## WHAT RAN (all committed `354b9a0`; logs `outputs/non_*_s051*.log`)

Script: `archive/experiments/world/world_nest_of_nests.py --arm
{single,vote,nest,absorb,absorb_settle} --rseed N`.

| arm | what | survival (rseed 0/1/2) | mean±σ | params | 
|---|---|---|---|---|
| single (s050) | each band nest alone | 112.6 / 113.0 / 159.0 | 128.2±26.7 | 13.8K each |
| vote | majority action over the 3 nests, tie→seed 0; NO training | 179.0 (n=1) | — | 41.4K |
| **nest-of-nests** | outer trioron router (77→32 quad→3) picks which seed-nest acts; sub-nests untouched; outer TD 300 eps world reward | 172.3 / 180.3 / 187.5 | **180.0±7.6** | 44.0K |
| **absorb** | per drive: seed-0 leaf ← graft(seed-1, seed-2 leaves, merge_output) = 179-cell fat leaf, exact Q-sum; leaves untouched; band router cold 300 eps | 165.1 / 206.1 / 200.6 | **190.6±22.3** | 35.3K |
| absorb+settle | as absorb + 50 eps TD/leaf on own drive reward (eps .1, lr 1e-3) before router | 87.2 / 103.0 / 104.5 | 98.2±9.6 | 35.3K |

Reference bars: best single 159.0; s049 master-built nest 148.5±12.9;
DQN ~50.

Readings:
- **Combining seed-siblings is a large, robust win** by every method
  that leaves the leaves alone: even zero-training majority vote (+20
  over best single). Sibling organisms are complementary, not redundant
  (each single dies of a different cause profile: seed 2 cold 17/40,
  seeds 0/1 integrity 11-14/40; combined organisms spread deaths ~evenly,
  timeouts 4-11/40 appear).
- **Absorb ≥ nest-of-nests** on mean, at fewer params (35K vs 44K), one
  router instead of four, but higher variance (rseed 0 = 165.1 — router
  seed matters more for a fat-leaf nest). n=3 on router seed; the
  ordering absorb > nest is NOT resolved at σ (Δ+10.6, σ≈22).
- **Q-sum absorption works BECAUSE the leaves are untouched.** The
  50-ep TD settle wrecked it (−92 vs absorb): a 3-fold Q-head sum has
  3× value scale; TD targets pull it back to 1× scale, re-learning the
  leaf from a bad start with 1/6 of the original budget. Settling an
  absorbed substrate needs either a 1/k head rescale first or the native
  machinery (credit-lock the transplanted cells; only new cells plastic)
  — untested. **Do NOT fine-tune absorbed leaves with raw TD.**
- Absorb route_hist: temperature leaf carries 40-45% of routes; integrity
  least (as s050).
- Tick latencies in the logs were measured under 10-way CPU
  oversubscription — do not quote them; re-measure single-process
  (fat leaf alone was 696 µs vs 485 µs live leaf, pre-run).

## API ALIGNMENT + MERGE (second half of session, `f5a6a99`, on `main`)

Rocky: "fix the absorb to match the APIs and merge it to main."
- **`trioron.api.absorb` / `pool_matched_absorb` now dispatch on type:**
  positional v2 `Substrate`s → `graft(merge_output=True, wiring="none",
  freeze=False)` in place (returns `GraftResult`s; `pool_matched_absorb`
  = single-donor form, v1-only kwargs rejected); `donor_paths=/out_path=`
  → unchanged v1 branch-granularity organism. `graft`/`GraftResult`
  exported from `trioron.api`.
- **Found+fixed a v1 breakage in the documented pip flow:** `build_donor`
  (and `absorb` payload / cli) crashed with `'dict' object has no
  attribute 'detach'` — `TrioronLayer.state_dict()` carries a dict
  `_extra_state` (LCN masks) and `legacy/api.py` `.detach()`ed every
  value. `_cpu_state()` helper passes non-tensors through. New
  `tests/test_donor_api_smoke.py` (build_donor→absorb→load_organism).
  This means PyPI 0.3.1's `build_donor` is broken → **0.3.2 release is
  warranted** (not done; Rocky's call).
- Docs: MANUAL §13.7 v2 paragraph + snippet + the settle warning; README
  one-liner; TRIORON_MANUAL §6 graft bullet; spec §9.1 api.py row.
- Tests: 128 pass (`tests` minus the two known-failing modules), + the
  same 4 known failures.
- **`main` fast-forwarded to `f5a6a99` and pushed** (first merge since
  s050 opened the item).

## GOTCHAS

- 10 background procs (OMP=1) on 12 CPUs: learned arms 55-65 min each.
- `absorb_settle` finished BEFORE `absorb` because settled leaves die
  sooner (fewer steps/episode) — not a hang.
- graft `merge_output` requires equal output widths (raises); donor and
  recipient must share input space (perception matched by rank order,
  unchecked beyond count).
- pytest: 119 pass in test_v2 (+4 new graft), the same 4 known failures
  (test_learning TestCredit ×2, test_lifecycle ×2).

## NEXT (priority)

1. **Absorb variance + settle done right:** (a) n=5 router seeds for
   absorb vs nest to resolve the ordering; (b) settle with credit-locked
   transplanted cells / λ-anchored (native machinery, TRIORON_MANUAL §8
   last bullet) or head rescale 1/k — the "organism absorbs then keeps
   learning" path is the real question; (c) absorb into a running s050
   pipeline: does an organism that absorbs its siblings then dream
   (world_drive_dream) still improve, or is the ceiling now
   arbitration again?
2. **Absorb the routers too** (zero-shot: graft the 3 band routers with
   merge_output — same 4-leaf index space) — no retraining at all; if it
   holds ~180+ the whole combine step is training-free.
3. s050 NEXT items still stand: arm 3 integrity-on-threat-distance;
   `Body`/`Organism.live` API shape doc (spec §9 row first); s049 items.
4. ~~Merge `conscience-core` → `main`~~ DONE (`f5a6a99`). **Publish
   0.3.2** (v1 `build_donor` on PyPI 0.3.1 is broken by the
   `_extra_state` crash; the fix is on main) — Rocky's call.

## OPEN / unresolved

- (resolved) `api.absorb`/`pool_matched_absorb` dispatch v1/v2 by type.
- s049/s050 open items unchanged.

## State of the build / Pointers

- **Commits:** `354b9a0` (graft merge_output + dendrite carry + tests +
  spec + bench + logs), `a1d2030` (handoff v1), `f5a6a99` (api dispatch
  + v1 build_donor fix + docs) — all on `conscience-core` AND `main`
  (ff), pushed; + this handoff (pushed to both at close).
- Checkpoints (untracked `runs/nest_of_nests/`): `outer_router_rseed{0,1,2}.pt`,
  `router_absorb_rseed*.pt`, `router_absorb_settle_rseed*.pt`
  (trainable_tensors lists, s049 ckpt rule; fat leaves are rebuilt
  deterministically by `absorb_leaves()`).
- s050 checkpoints (`runs/drive_vocab/`, `runs/drive_band/`) are the
  inputs; do not delete.

## DO-NOT-COMMIT carries (unchanged since s034, LEAVE THEM)

`trioron/bases/developmental.py`, `trioron/lifecycle/developmental.py`,
`trioron/viz/export.py`; `.claude/`, `runs/`, `trioron/legacy/outputs/`,
`notebooks/` checkpoints, output PNGs/uncommitted logs untracked.

## Environment notes

- `/home/marcrockhat/trioron-project/`, branch `conscience-core`, Python
  3.10.12, torch 2.11.0, WSL2, `python3` (NOT `python`), 12 CPUs.
- Bench: `OMP_NUM_THREADS=1 python3 archive/experiments/world/
  world_nest_of_nests.py --arm absorb --rseed 0` (~60 min; `single`/`vote`
  ~4 min).
- Bench logs buffer — check mtime before assuming a hang.
