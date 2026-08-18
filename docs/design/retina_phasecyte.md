# Retina-pooled Phasecyte → trioron nest on CIFAR-100 (design, s051)

**Status:** APPROVED by Rocky (s051): continual 20-task from the start,
5 fixations, opponent colour, fovea 8×8, bar = pure-trioron v2b.
**Owner:** Rocky (direction) / Chloe (engineering).
**Question:** can a nested organism whose *perception* is a
human-eye-inspired retina (pooled Phasecyte receptors) beat the pure-
trioron CIFAR-100 record (blindman v2b: full 0.309 / task 0.606 at 559K),
where every previous nesting lost?

## 0. Why this and not another nesting

Every CIFAR nesting so far (taxonomy v3 0.247, k=20+experts 0.185,
two-trioron stack) shared one perception — a dense L0 over the whole
32×32 sheet — and routed *above* it. Nesting won on chained-15 (Phasecyte
nest) and in the world (drive nest, nest-of-nests, absorb) only when the
leaves were *complementary* and the failure was arbitration. On CIFAR
the failure is perception (no locality: `substrate_no_spatial_memory`).
So the nest has to be **below** perception: leaves that literally see
different things. The human eye is the existence proof that a fixed,
gradient-free front-end with strong locality (fovea), pooling that grows
with eccentricity, opponent colour, centre–surround, and *many
fixations* over one scene is enough sensory structure for a downstream
learner to be sample-efficient.

## 1. Eye → mechanism map (what we copy, what we don't)

| eye | mechanism here | exists? |
|---|---|---|
| photoreceptor mosaic: cones (L/M/S) dense at fovea, rods dominate periphery | 4 channels per pixel: luminance Y (rod-like) + 3 opponent chromatic (L−M "red–green", S−(L+M) "blue–yellow", plus Y for the cone stream); chroma is sampled ONLY inside the fovea+parafovea, periphery is luminance-only | new (`eye.py`) |
| eccentricity-dependent acuity (ganglion RF radius ∝ eccentricity) | **retina pooling**: fovea 8×8 px at full res (64 receptors/channel), parafovea annulus pooled 2×2 (48 regions), periphery pooled 4×4 (≈36 regions), rest of the 32×32 covered by 8×8 blocks (~12). Implemented with `arena.add_pool` region sensors carrying imposed (x,y,scale) positions — the same primitive as `pcll/retina.py`'s RegionSensor, but the pooling is a *body geometry* declared up front, not discovered by redundancy | primitive exists (`add_pool`, RegionSensor); the fixed foveal layout is new |
| centre–surround ganglion cells (ON/OFF) | difference-of-Gaussians on each pooled region against its 8-neighbour surround, ON = +DoG, OFF = −DoG rectified; fixed, no learning; this is the *sense* callable fed to PhasecyteLeaf | new sense; the existing `gabor`/`cs` senses in run_pcll_chained15 are the template |
| **saccades / fixations** | one *fixation* = the retina placed at centre c ∈ F. F = 5 fixations (image centre + 4 at (±6,±6) px) — v1; 9 (3×3 grid) as a knob. Each fixation is a separate Phasecyte leaf family (own genesis, own receptors, own lock-in state). The visual system integrates over fixations; we integrate two ways (see §3) | new; the analog of the blindman positional bank, but the "hand" is a foveated eye, not a blind patch |
| magno vs parvo streams | two leaf types per fixation: **P** (fovea+parafovea, colour + fine DoG) and **M** (periphery + fovea luminance, coarse DoG). Both feed the same downstream trioron leaf | new |
| lateral inhibition / normalisation | per-region contrast normalisation before quantisation (divisive by local RMS) — same role as the standardizers in blindman | trivial |
| NOT copied | temporal dynamics (no motion in CIFAR), pupil, adaptation, binocularity | — |

Everything above the receptors is **unchanged Phasecyte**: RECEPTOR gene
+ phase injection (§10.2), lock-in (§10.3), boundary meeting (§10.4),
signatures/division/composer as configured for the pcll_nested arm.

### 1.1 The no-convolution principle (Rocky, s051)

Convolution (learned weights shared across positions) does not exist in
nature; simple organisms get translation-tolerant perception without
it. Nature's three substitutes are all in this design, and are the ONLY
translation machinery it uses:

1. **Genotypic replication — multiplexed linears.** The same receptive-
   field *type* (DoG ON/OFF) is stamped at every retinal position:
   thousands of independent linear units, identical by gene, not by
   shared parameters. A conv kernel bank without learning and without
   sharing. Everything *learned* downstream is position-specific (LCN,
   not conv) — consistent with blindman (the record) and with the
   conjoined-twin / conv-by-emergence findings (real conv only exists as
   lineage-shared cells; it never emerged).
2. **Time-multiplexing — saccades.** One high-acuity sensor is moved
   over the scene instead of replicating a detector everywhere.
   Fixations give translation *coverage*; **microsaccades** (±1–2 px
   jitter of each fixation centre during wake) give translation
   *tolerance* at zero parameters.
3. **Pooling hierarchy.** Eccentricity pooling + the M stream give
   small-shift tolerance (complex-cell role) — fixed, linear-then-
   rectify.

Knob (not v1): an oriented filter bank (4 orientations, simple-cell
like) stamped at every region — what a CNN's first layer learns. DoG-
only is the purer test of the thesis; the oriented bank is arm A5 if
time allows.

## 2. Architecture (the nest, bottom-up)

```
image 32×32×3
 └─ for each fixation f ∈ F (5):
      eye(f): pooled regions → 4-channel DoG features            [fixed]
       ├─ Phasecyte leaf P_f  (parvo)   ← pockets_P_f            [gradient-free, wake]
       └─ Phasecyte leaf M_f  (magno)   ← pockets_M_f            [gradient-free, wake]
      trioron leaf T_f  (77-ish → 32 quad → 100)  input = [pockets_P_f ; pockets_M_f]/1000
                                                                  [gradient; see §3.2]
 └─ level 2: combine over fixations
      (a) ABSORB      = Σ_f logits_f   (head-merged graft of the T_f, exact; the committee)
      (b) NEST        = trioron router (input: coarse M-stream pockets of the centre
                        fixation → 5) picks ONE fixation's T_f per sample
      (c) SOFT NEST   = router softmax-weighted Σ_f  (the bleed_flip upper bound)
```

Level 0/1 is one organism per fixation. Level 2 is exactly the s051
nest-of-nests-vs-absorb pair, so we reuse `graft(merge_output=True)`
and the router recipe verbatim.

## 3. Protocol

3.1 **Data / metric.** CIFAR-100, `experiments/cifar/datasets.py`
loaders; metrics = full 100-way and task-aware 5-way superclass-
restricted (identical to blindman v2b, `cifar100_fine_to_coarse`).
Report both; v2b's 0.309/0.606 is the bar, blindman v1 uniform 0.239 the
floor. Params counted the same way (bank + combiner).

3.2 **Learning regime — decision.** Two honest options:
  - **Joint (Stage-A style, what blindman did):** Phasecyte leaves wake
    over the whole train set (one pass, gradient-free); trioron leaves
    T_f trained by SGD on pockets, joint 100-way. Cleanest read of
    "does the eye help perception".
  - **Continual (20 superclass tasks × 5 classes, sequential):** the
    Phasecyte leaves *are* continual by construction; T_f are
    dream-distilled from the leaves' per-class sketches at each boundary
    (hybrid_nest recipe, no stored data), CL-honest. Comparable to
    `cifar_continual_v2_vs_v1_n5` (v2.0 beat v1.0 there).
  **Decision (Rocky): continual 20-task from the start.** Stream =
  the 20 CIFAR-100 superclasses in fixed order, 5 fine classes each;
  Phasecyte leaves enroll classes as they stream; at each task boundary
  every T_f is dream-distilled from its leaves' sketches (all classes
  seen so far — stationary joint per leaf, hybrid_nest recipe, no
  stored images). Joint is NOT run.

3.3 **Arms.**
  A0 null: flat Phasecyte on 32×32 luminance pixels (no retina, 1
     fixation) → 1 trioron leaf. Establishes the "Phasecyte-first, no
     eye" baseline the whole design must beat.
  A1 eye, 1 fixation (centre), P+M → T_centre.  Isolates foveation +
     opponent colour + DoG.
  A2 eye, 5 fixations, level-2 ABSORB.
  A3 eye, 5 fixations, level-2 NEST (hard router) and SOFT NEST.
  A4 (knob) 9 fixations, ABSORB — does more saccades keep paying?
  Seeds n=3 on the whole pipeline (Phasecyte genesis + trioron init).

3.4 **Gates (pre-registered).**
  - G1 (eye helps): A1 > A0 by ≥ 3σ on full. If not, the retina is
    not the missing locality and we stop.
  - G2 (fixations help): A2 > A1 by ≥ 3σ.
  - G3 (the record): A2 or A3 ≥ 0.309 full **and** ≥ 0.606 task at
    ≤ 559K params. Either arm clearing both = "beat CIFAR-100 (pure
    trioron)". Clearing full but not task (or vice versa) is reported
    as such, not spun.
  - G4 (nest vs absorb): same read as s051 — expect ABSORB ≥ NEST; a
    NEST win here would be the first per-sample routing win on CIFAR.

3.5 **Budget.** Phase 1: each trioron leaf ≤ 50K params; Phasecyte leaves
capacity 8192 cells (as pcll_nested), class cap 128 → 100 real classes
fits without splitting. 5 fixations × 2 leaves × wake over 50K images:
pcll_nested did 3 leaves × 30K in ~2 min per seed → ~10 min. Trioron
leaves: 5 × joint 100-way on ~200-d pockets, minutes. Whole A2 seed
< 30 min CPU.

## 4. Files (spec §9 partition — row added BEFORE code, house rule)

- `trioron/pcll/eye.py` — `Eye(fixations, fovea, rings, channels)`:
  builds the pooled-region layout for a fixation (returns member pixel
  lists + imposed positions per region per channel), the DoG/opponent
  `sense(x)` callable, and `Eye.attach(substrate)` that spawns the
  region receptors via `add_pool` (RegionSensor). One concept: the eye.
- `experiments/progenitor/cifar_eye_nest.py` — the bench (arms A0–A4).
- spec: §10.11 "The eye — retina-pooled receptor body" (short, points
  here), §9.6/§9.14 rows, §9.11 test `tests/test_v2/test_eye.py`
  (layout covers the sheet exactly once per channel; DoG ON+OFF
  reconstructs; sense is deterministic; attach spawns N regions).

## 5. Decisions (resolved by Rocky, s051)

1. Learning regime for the trioron leaves: **joint first** (my rec) vs
   continual-20-task from the start.
2. Fixation set: 5 (centre + 4 diagonal at ±6 px) — v1; 9 as knob. OK?
3. Colour: opponent Y / L−M / S−(L+M) (eye-faithful) vs raw RGB per
   region. I'd do opponent; raw RGB as a one-line knob.
4. Fovea 8×8 at full res: on 32×32 that is 1/16 of the image at
   full acuity — humans ~1–2° of ~180°, so we are *far* more foveal-
   heavy than the eye; anything smaller than 8×8 starves the P leaf on
   CIFAR objects. Accept 8×8?
5. Metric bar: pure-trioron v2b (0.309/0.606). Cortex/MobileNet
   numbers stay future-work per `feedback_pure_trioron_scope`.
