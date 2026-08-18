# Trioron Handoff

**Session date:** 2026-08-18
**Session number:** 051
**Session title:** **(A) Nest-of-nests vs absorption + (B) the eye / frequency-primitive arc.** (A): Nest-of-nests vs absorption — combining three seed-different
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

## PART B (later in s051): the EYE + the frequency-primitive arc (Rocky AFK at close)

Rocky's chain: "phasecyte first then triorons" → "retina-pooled
phasecyte, human eyes as inspiration" (approved design
`docs/design/retina_phasecyte.md`: continual 20-task, 5 fixations,
opponent colour, fovea 8×8, bar = pure-trioron v2b 0.309/0.606) → "no
convolution in nature; multiplex linears" (§1.1 of the design) → "are we
chasing the wrong primitives? flies see motion, humans match templates"
→ "form = a wave frequency, coordinate-free" → "frequency matching is
fast (speech); we need to know where it starts/ends" → "add simple-shape
classifiers (circle/triangle/square/polkadots/stripes) as primitives —
built with OUR APIs, not hand-coded (generator hand-coded = benchmark
data is fine)" → "how does Shazam find a fragment?" (constellation of
relative landmark pairs). Every step got a probe. Commit `920559f`.

**Built:** `trioron/pcll/eye.py` (`Eye`, `fixations`, `retina_layout`;
tests `tests/test_v2/test_eye.py`), spec §10.11 + §9 row;
`experiments/progenitor/cifar_eye_nest.py` (A0 null / A1 1-fixation /
A2-A3 5-fixation absorb+nest, Phasecyte leaves P+M per fixation → sketch-
dreamed trioron leaf per fixation, band = 20 superclasses); probes
`experiments/progenitor/diag_eye{,2,3,4,5,6}.py`, logs
`outputs/primitive_probe{,2,3,4}_s051.log`, `outputs/cifar_eye_nest_s051_*`.

**Probe protocol (all rows below):** CIFAR-100 first 5 superclasses = 25
fine classes, supervised trioron leaf `Seeded(d, 100, 48, nonlinear)`,
8 epochs Adam, standardized inputs; full = 100-way argmax over the 25
present, task = 5-way superclass-restricted; chance 0.04 / 0.20; "blur"
= eval on 2×avg-pool→bilinear-up test images. Single seed. n=1 — a
ranking, not σ.

| front end (fixed, no learning unless stated) | d | full | task | blur full |
|---|---|---|---|---|
| eye DoG, centre fixation, signed | 412 | 0.202 | 0.383 | 0.202 |
| fly: DoG + ±1px saccade differences | 1236 | 0.176 | 0.366 | 0.203 |
| wave: log-polar power spectra (abs) | 160 | 0.253 | 0.472 | 0.084 |
| collapse: structure fn D(s) | 96 | 0.128 | 0.347 | 0.095 |
| (a) global cepstrum | 100 | 0.258 | 0.466 | 0.104 |
| **(b) cepstral spectrogram, 25 windows, positions kept** | 800 | **0.304** | **0.494** | 0.119 |
| (c) cepstra pooled over positions | 64 | 0.154 | 0.384 | 0.045 |
| (d) onset (max-energy) window only + where | 34 | 0.115 | 0.324 | 0.065 |
| (f) eye DoG + (b) | 1212 | 0.313 | 0.498 | 0.153 |
| (g) primitive vocabulary ×25 windows (detector 0.72 synth) | 150 | 0.144 | 0.342 | 0.084 |
| (h) (g) + (b) | 950 | 0.293 | 0.475 | 0.141 |
| (i) (b) trained WITH blur augmentation | 800 | 0.264 | 0.456 | **0.254** |
| (k) constellation: landmark PAIRS relative only | 1024 | 0.174 | 0.372 | 0.148 |
| (l) same landmarks, absolute 4×4 position | 128 | 0.170 | 0.381 | 0.159 |
| (m) (k) + (b) | 1824 | 0.304 | 0.471 | 0.131 |
| raw pixels (control; 6× the params) | 3072 | 0.356 | 0.533 | 0.359 |

**Readings.** (1) Frequency-SHAPE (cepstra) is the most information-dense
fixed primitive: 100 numbers ≥ 412 DoG numbers; the windowed spectrogram
WITH positions is the best fixed front end (0.304, within 5 pp of raw
pixels at ¼ width). (2) "Where it starts/ends" = keep windows + positions
(pooling halves it; a single onset window is worse). (3) Fly/saccade
differences, the shape vocabulary (detector caps 0.72 on clean 16×16
shapes — resolution), and the Shazam constellation are NULLS on this
probe (constellation: relative == absolute at 8× fewer dims — the
coordinate-free property holds but landmarks are weak at 32×32).
(4) Blur robustness of the frequency features is not free (0.30→0.12)
but is trainable by blur augmentation (0.25 blur / 0.26 sharp) —
"accommodation". The DoG eye is natively blur-robust. (5) Everything
here is a linear/quad readout on fixed features; the leaf is 25–90 K
params.

**Eye + Phasecyte arms (20 superclasses, 100 classes, seed 0):**
A0 (flat luminance null, redundancy retina, 871 receptors): **full 0.028
/ task 0.260**. A1 (1 fixation, Phasecyte P+M → sketch-dreamed trioron):
**full 0.048 / task 0.297** still-eyed (0.043 with microsaccades);
Phasecyte alone 0.028; chance 0.01/0.20. **G1 (eye > null) holds
directionally (+2.0 full / +3.7 task, n=1) but both sit at the floor.**
A2/A3 (5 fixations) died/stalled in the composer-spawn phase after task
5 (log `..._fix5_seed0_killed_task6.log`; 10 Phasecyte leaves is
impractically slow) — not rerun: G3 (0.309/0.606) is not reachable from
0.048/fixation, so its outcome cannot change the conclusion. **Phasecyte-first on CIFAR
is the wrong order**: the template stage's internal classes are so mixed
(no internal class ≥34% one real class) that sketch dreaming keeps ~half
of what the same real pockets support (0.09 vs 0.20 supervised);
microsaccades abolish class discovery outright (templates need a still
eye — verified: jitter 1 → 1 class, jitter 0 → 20 classes by task 2).
Fixes shipped in the bench: signed DoG (ON/OFF zeros were masked as
silence, spec §10.3), sketch-derived standardizer (pockets sit at
0.5±0.07), composition-weighted pseudo sampling (purity³), still eye.

**NEXT (Rocky's call):** the promising line is `(b)`-style windowed
cepstra (± the eye's DoG) as the *sense* of a gradient trioron leaf /
nest — Phasecyte later or not at all for CIFAR; test at 100 classes with
the 5-fixation absorb+nest level-2 (s051 part A) and blur augmentation;
param-match vs raw pixels; then the continual 20-task version. No
background runs left at close.

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
  PyPI 0.3.1's `build_donor` was broken → **v0.3.2 PUBLISHED**
  (https://pypi.org/project/trioron/0.3.2/, `d5c3607`; Rocky ran the
  twine upload). Wheel verified from a clean `pip install --target` of
  the built file BEFORE upload (v1 build_donor→absorb→load + v2 absorb
  exact 5e-10); a post-upload `pip install trioron==0.3.2` check was
  blocked by the sandbox classifier — same file, not re-verified from
  PyPI. README/QUICKSTART install notes say 0.3.2+.
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
4. ~~Merge `conscience-core` → `main`~~ DONE. ~~Publish 0.3.2~~ DONE.
   Cheap follow-up: `pip install trioron==0.3.2` from PyPI on a clean
   venv + run `tests/test_donor_api_smoke.py` against it.

## OPEN / unresolved

- (resolved) `api.absorb`/`pool_matched_absorb` dispatch v1/v2 by type.
- s049/s050 open items unchanged.

## State of the build / Pointers

- **Commits:** `354b9a0` (graft merge_output + dendrite carry + tests +
  spec + bench + logs), `a1d2030` (handoff v1), `f5a6a99` (api dispatch
  + v1 build_donor fix + docs), `d7f2311` (handoff v2), `d5c3607`
  (v0.3.2 bump) — all on `conscience-core` AND `main` (ff), pushed; +
  this handoff (pushed to both at close).
- `dist/trioron-0.3.2-py3-none-any.whl` + sdist = what is on PyPI.
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
