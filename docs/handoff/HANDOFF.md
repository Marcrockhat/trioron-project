# Trioron Handoff

**Session date:** 2026-08-21
**Session number:** 056
**Session title:** **Motion arc steps 1–4 (Rocky: "start the motion arc"): moving-shape
world over CIFAR scenery built; the spectral machinery re-aimed at TIME — 2-D
cross-spectrum PHASE per 8×8 patch + temporal-difference adaptation + population
velocity read-out — decodes velocity to 0.55–0.60 px (cos .83–.86) on flat AND
photo backgrounds with zero learning, while the s052 magnitude spectra are motion-
blind (.056 = chance); wagon-wheel reversal 17 % on alias-tagged textures; common-
fate grouping v3 beats colour/Otsu on photos (IoU .48 vs .34); a 50 K motion leaf
reads the 17-way velocity class at .65–.66 (control .12); shape-from-motion-
grouping on photos .50 vs colour-grouped .39 (chance .33).**

---

## READ THIS FIRST

**s056 addendum (2026-08-21):** Rocky reviewed the sheet and asked for radius
6–14 (was 4–13); world REBUILT (same seeds, `motion.py:76`), feature caches
dropped, diag + leaves rerun (`outputs/motion_diag_smoke_s056.log`,
`motion_leaf_s056.log`). Size was NOT the lever: velocity leaf .658/.660
(unchanged); colour-silhouette shape +5 pp (photo .41 / flat .88); motion-
silhouette shape only +2 pp (photo .52 / flat .64); motion grouping IoU .47/.47
with mask area .20 vs true .29 — bigger bodies have more interior and more
motion-parallel edge that temporal difference can't see. **Grouping v3's
interior fill is the bottleneck (mechanism, not data).** Both-silhouettes
concat now adds slightly on photos (.53). Tables below are the s055 (4–13)
numbers; the s056 log has the current ones.


1. **Rocky's asks:** "start the motion arc" (s054 NEXT-0, scope pinned there and
   still binding: readers PURE trioron/Phasecyte; CNNs = reference bars only;
   NOT tabula rasa — the s053 nest / s054 pre-feeder are carried in as frozen
   organs / absorbed leaves, not equal voters). Mid-session he asked for a
   rendered training sheet to check — `outputs/motion_train_sheet.png` (16
   packets × 6 frames) + `motion_train_masks.png`; **he has not yet replied**
   whether the world matches his intent (small radii 4–13 px, static camera,
   speed 0.5–3 px/frame, 15 % static, T=6). Ask before scaling the world.
2. **New files** (all `experiments/progenitor/`, committed):
   - `motion.py` — generator. Packet = T=6 frames uint8 [N,T,3,32,32] + EXACT
     per-frame mask of object 0 [N,T,32,32]. Per object: shape/fill/colour/pose
     as shapes.py + velocity (dx,dy) px/frame, dθ rad/frame (p .3), dr px/frame
     (loom, p .2). Background: flat colour | CIFAR-100 train photo (static, no
     camera motion). Labels: y_vel 17-way (0 static | 1..8 octant slow <1.75 |
     9..16 octant fast), y_dir, y_speed, y_dx/dy/dth/dr, y_shape/fill/hue,
     y_alias (textured fill displaced > p/2 along its texture normal = wagon-
     wheel regime, ~5 %), y_bgkind. Splits at `outputs/data/motion/*.pt`
     (gitignored; `python3 experiments/progenitor/motion.py build`, ~6 min):
     train 8000 mixed, test 2000 mixed, test_photo 2000, test_flat 2000,
     train_multi 4000 / test_multi 1000 (maxk=2, unused so far).
   - `motion_front.py` — the primitive. `motion_phase` 450 (25 regions × 6
     2-D bins × (cos, sin, coherence)), `motion_energy` 25, `motion_pop` 181
     (per-region decoded v̂, coherence, log E + global 9×9 population response),
     `motion_full` = 656; `velocity_map` / `decode_velocity` (no learning).
   - `motion_diag.py` — parts A (probe + decode), B (wagon-wheel), C (grouping
     v3 vs colour/Otsu); holds `motion_group()` = common-fate grouping v3.
   - `motion_leaf.py` — Seeded leaves on cached streams; log
     `outputs/motion_leaf_s055.log`; feature caches `outputs/data/motion/feat_*`.

## WHAT WAS LEARNED BUILDING THE PRIMITIVE (don't re-derive)
- **Sign:** content moving +dx ⇒ `F_{t+1} = F_t·e^{−2πi k·v/L}` ⇒ cross-
  spectrum phase −2π k·v/L. (I flipped this twice; verified on `torch.roll`
  fields AND rendered shapes: roll +1 → decodes 1.00.)
- **Temporal adaptation must be a DIFFERENCE, not mean-subtraction.** Mean-
  subtract kills the signal when the per-frame phase step is small (large
  apertures). `D_t = F_{t+1} − F_t`, then `conj(D_t)D_{t+1} = |A|²|e^{iω}−1|²e^{iω}`:
  static background cancels EXACTLY, phase is still exactly ω (magnitude is a
  temporal band-pass). Photo backgrounds went from 1.65 px error to ≈ flat.
- **1-D row/column spectra (the literal stereo L/R form) FAIL for 2-D motion:**
  a square sliding along x makes column profiles appear/disappear → coherent
  but meaningless vertical phase (decoded dy≈2 for true 0). The 2-D FFT per
  patch with bins {(1,0),(0,1),(1,1),(1,−1),(2,0),(0,2)} is the right object;
  bin (0,1) of the 2-D FFT = spectrum of row-SUMS, which is a pure shift.
- **Read-out = a population of velocity-tuned units** (MT-like): score(v) =
  Σ_k w_k cos(θ_k + 2π k·v/L) on a 33×33 grid in [−4,4]² (step .25), argmax.
  Aliasing appears as secondary peaks (wagon-wheel for free).
- **Aperture:** L=8 patches under-read magnitude ~20 % (gain 1.25) on small
  bodies; L=16/32 are WORSE (fewer patches see the body, edge effects). Keep L=8.
- Per-pixel temporal-difference energy lights only EDGES; interiors of uniform
  bodies and edges PARALLEL to the motion produce nothing (aperture again).
  Grouping therefore needs the s053 body pipeline (closing + fill) + convex
  hull, a NOISE FLOOR (max(Otsu, 4×median) — without it a static packet's mask
  is Otsu-on-noise covering 60 % of the frame, which also inflated moving IoU),
  and a LOCK-IN baseline: frame spacing chosen from the decoded speed so the
  body moves ~3 px between the compared frames (slow motion integrates longer).
  A velocity-AGREEMENT gate (patch v̂ within 1 px of dominant) HALVED the mask —
  patch decode at edges is too noisy for gating; dropped.

## RESULTS (n=1000 diag smoke `outputs/motion_diag_smoke_s055.log`; leaves n=3 seeds)
### A. Velocity read-out (17-way y_vel, chance .06; direction 8-way chance .125)
| features | probe (linear) y_vel | dir | Seeded leaf y_vel test / photo / flat |
|---|---|---|---|
| s052 static spectra (mid frame, 800) — magnitude only | **.056** | .122 | .115 / .116 / .124 (≈ "always static" prior .15) |
| motion_energy 25 | .213 | .164 | — |
| motion_phase 450 | .346 | .479 | — |
| motion_block 475 | .369 | .497 | — |
| motion_full 656 (50.3 K leaf) | — | — | **.652±.010 / .660±.002 / .662±.003** |
Population decode, no learning: median |v̂−v| 0.55 px flat / 0.60 photo, cos .83/.86,
octant-acc .68/.63; static packets |v̂| .28. Photo ≈ flat = the HP works.
### B. Wagon-wheel (population decode on textured fills)
alias=0 (n=410): err .50 px, cos .84, reversed 4.9 % | **alias=1 (n=46): err .97 px,
cos .67, reversed 17.4 %** | solid: .68 px, reversed 5.6 %. Illusion present, modest
(low bins don't alias; population mixes). A clean demo stimulus (stripe field at
> p/2) is not built yet.
### C. Grouping, IoU vs exact mid-frame mask (n=600)
| bg | colour/Otsu (s053) | common-fate v3 | v3 no-hull |
|---|---|---|---|
| flat, moving | **.844** | .489 | .451 |
| photo, moving | .338 | **.478** | .414 |
| flat/photo static | .82 / .35 | .16 / .11 (mask area .05/.04 — near empty, correct) | |
### D. Shape-from-motion-grouping (3-way y_shape, chance .33; Seeded leaves, "/mov" = moving packets)
| stream | test | test_photo/mov | test_flat/mov |
|---|---|---|---|
| sil_col (colour-grouped silhouette 92) | .616 | .386 | **.833** |
| sil_mot (motion-grouped silhouette 92) | .535 | **.500** | .611 |
| sil_both 184 | **.658** | .491 | .835 |
| static spectra 800 | .403 | .355 | .460 |
| motion_full 656 | .439 | .438 | .452 |
Reading: on photos the colour silhouette is at chance+5 and the motion silhouette
carries the shape (+11 pp); on flat colour wins by 22 pp. The concat does NOT add
on photos (the leaf can't tell which silhouette to trust → a per-packet gate from
motion_energy / bg-kind is the obvious next fix; ties into the router item).
Velocity leaf unchanged by adding sil_mot (.652 → .652).

## NEXT (priority)
0. **Rocky's review of the world** (item 1). Then decide radius range / camera
   pan / T.
1. **Gate, don't concat:** the shape leaf should pick colour vs motion silhouette
   per packet (motion_energy is the fires-only-when-moving signal; s054 NEXT-1
   routed arbitration). Expect photo shape → ~.50 AND flat → ~.83 in one reader.
2. **Absorb the motion leaf into the nest** (the step-4 second half, not done):
   the 50 K velocity leaf as an absorbed leaf / INPUT stream of the s053 nest
   (`pool_matched_absorb` + head settle), on a moving-shape continual stream
   (shape tasks then velocity task). Needs a task schedule — design with Rocky.
3. **Multi-object common fate** (`train_multi`/`test_multi`, maxk=2, built, unused):
   two bodies, different velocities → v3 must split by velocity cluster; this is
   where the dropped agreement gate comes back as a CLUSTERING step, not a gate.
4. **Clean wagon-wheel demo** (stripe field, period p, speeds crossing p/2) +
   render the population response (secondary peak) — "showing the illusion =
   built the mechanism".
5. **Emergence control (s019 doctrine):** feed the raw T-frame packet (or per-
   frame spectra WITHOUT the cross term) to a satellite/recurrent substrate and
   test whether a Reichardt-like correlator emerges under frustration.
6. Grouping v3 ceiling: per-pixel motion compensation (shift by v̂, compare) to
   recover interiors; rotation/looming (dθ, dr) read-out heads (labels exist).

## GOTCHAS (new)
- `motion.py build` must run as a script from repo root (sys.path fixed in-file);
  first run cost one failed launch on `ModuleNotFoundError`.
- `motion_front.HP/L/ST/BINS` are module globals; `velocity_map` returns 4 values
  (v, coh, E, R) — R is [N,13,13,1089] (the population response), big but fine at
  chunk 250–500.
- `motion_diag.py` part A rebuilds train features in-process (~40 s); `motion_leaf`
  caches them. `motion_group` is per-image Python (~15 ms/packet).
- Old s053/s054 gotchas (cifar_continual runs on import, 7 GB RAM ≈ 3 leaf procs,
  ARMS exact-string) still apply.

## State of the build / Pointers
- Commits (`conscience-core`): steps 1–3 commit + this close commit (leaf +
  handoff). `main` NOT advanced. Pushed.
- Logs: `outputs/motion_diag_smoke_s055.log`, `outputs/motion_leaf_s055.log`;
  PNGs `outputs/motion_train_sheet.png`, `motion_train_masks.png`,
  `motion_group_debug.png` (frame | energy | Otsu | closed | truth).
- Package (`trioron/`) untouched.

## DO-NOT-COMMIT carries (unchanged since s034, LEAVE THEM)
`trioron/bases/developmental.py`, `trioron/lifecycle/developmental.py`,
`trioron/viz/export.py`; `.claude/`, `runs/`, `archive/runs/`,
`trioron/legacy/outputs/`, `notebooks/` checkpoints, output PNGs / uncommitted
logs untracked; `outputs/data/**` caches.

## Environment notes
- `/home/marcrockhat/trioron-project/`, branch `conscience-core`, Python
  3.10.12, torch 2.11.0, WSL2, `python3` (NOT `python`), 12 CPUs, 7 GB RAM.
