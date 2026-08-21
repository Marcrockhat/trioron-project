# Trioron Handoff

**Session date:** 2026-08-21
**Session number:** 058
**Session title:** **CNN REFERENCE BAR (NEXT-0 of s057) — compact conv nets on the RAW T=6 packets vs
Kinopsis, n=3. Velocity: a fair CNN at the leaf budget ties (.613±.04 vs .655); an 83K CNN at
60 ep BEATS the leaf (.702). A plain MLP on the leaves' own primitive = the leaf (.644 vs .655):
the primitive, not the trioron substrate, carries velocity. Shape on moving packets: every CNN
≤ .43 vs Kinopsis .653 photo / .870 flat — but the comparison is data-asymmetric (see caveat).**

---

## READ THIS FIRST

Rocky: "let's start the NEXT-0", then AFK 6 h; whole session autonomous. Script
`experiments/progenitor/motion_cnn.py`; logs `outputs/motion_cnn_s058{,_30ep,_strong30ep,_big60ep}.log`.

**Arms** (Adam 1e-3, bs 256, n=3, splits test / test_photo / test_flat; shape reported on
moving-only packets like kinopsis):
- `cnn2d`: 18 stacked channels → 3× conv3x3 (24/48/64) + pool → GAP → head, 43K
- `cnn3d`: [3,T,32,32] → 3× conv3d (16/32/48) → GAP → head, 57K
- `cnn2d_strong`: frames + frame DIFFERENCES (33 ch), BatchNorm, flatten instead of GAP, 23K
- `cnn2d_big`: same, channels 32/64/64, 83K/68K (over the leaf budget, capacity check)
- `mlp656`: the leaves' primitive `motion_full` (656-d, standardised) → plain torch MLP-48, 32K

| arm | ep | vel test | vel photo | vel flat | shape/mov photo | shape/mov flat |
|---|---|---|---|---|---|---|
| cnn2d GAP | 8 | .171 | .180 | .182 | .375 | .371 |
| cnn3d GAP | 8 | .171 | .173 | .184 | .362 | .368 |
| cnn2d GAP | 30 | .384±.113 | .373 | .423 | .403 | .426 |
| cnn3d GAP | 30 | .457±.043 | .445 | .500 | .375 | .392 |
| cnn2d_strong | 30 | .613±.042 | .597 | .641 | .389 | .414 |
| cnn2d_big | 60 | **.702±.005** | .677 | .735 | .395 | .434 |
| mlp656 | 8 | .644±.003 | .645 | .641 | .477 | .486 |
| **Kinopsis s057** | 8 | .655 | — | — | .653 | .870 |

**Readings:**
1. **Velocity: "linear machinery + correct primitive ≈ conv layers" HOLDS at matched budget
   and fails above it.** GAP-CNNs are near chance at 8 ep (GAP discards phase/position — the
   velocity signal); a fair CNN (diffs + BN + flatten, 23K, 30 ep) reaches .613, within 1σ of
   the leaf; 83K/60 ep CNN .702 > .655. The leaf converges in 8 ep / 1 s per seed vs 9–33 min
   per seed for the CNNs — the primitive buys compute, not a ceiling.
2. **mlp656 = leaf.** .644 vs .655 at the same hidden width: on this task the trioron
   substrate adds nothing over an MLP *once the primitive is fixed*. The velocity story is a
   primitive story (motion_front), which is what s054 scope intended (readers pure trioron).
3. **Shape: CNNs ≤ .43 on moving packets vs .653/.870 — NOT a clean win. Caveat:** the
   Kinopsis shape organ was trained on the s053 static shape world (20,000 images) and the msil
   leaf on the 8,000 packets; the CNN saw only the 8,000 packets with 3-way labels. The CNN
   also plateaus at .43 from 23K→83K and 30→60 ep (flat, capacity-insensitive), so data is the
   likely limiter, not architecture. Matched follow-up = NEXT-0b.
4. GAP-CNN velocity σ (.11) is seed-chaotic at 30 ep; report cnn2d_strong / cnn2d_big as the bar.

## NEXT (priority)
0. **Paper**: put the table in as the reviewer's comparison with reading 1–3 verbatim; the
   honest sentence is "at the leaf's budget a CNN on raw frames ties the primitive-fed leaf on
   velocity and a 4× larger CNN beats it; the primitive buys 100–1000× less training compute."
0b. **Matched-data shape bar**: pretrain cnn2d_big on the 20K static shape frames (5-way) then
   fine-tune on the 8K packets (3-way), or give Kinopsis-equivalent data any other way; until
   this runs the shape gap cannot be claimed. Also a CNN on sil_mot (92-d) is meaningless — the
   architecture control for shape is `mlp` on the organ's streams if wanted.
1. Lift the msil voter (photo ceiling .658): grouping v4→v5 per-pixel motion compensation,
   silhouette descriptor (s057 NEXT-0b).
2. Save format: arena-state-only + recompile on load (14.2 MB → ~1 MB; s057 NEXT-1).
3. Cell-level absorption of msil into the organ's shape leaf (s056 graft cost .09 old-world).
4. Multi-object common fate (`train_multi`/`test_multi`, maxk=2, unused).
5. Frozen-CNN + linear head vs frozen-CNN + trioron, continual schedule (the "hardware" claim).
6. Wagon-wheel demo; emergence control (s019 doctrine). Showcase game deferred.

## GOTCHAS
- `motion_cnn.py` runs from repo root, `python3`; cnn3d is slow (35 min/seed at 30 ep);
  cnn2d_big 32 min/seed at 60 ep. Env: SEEDS, EPOCHS, ARMS.
- s057 gotchas unchanged: `Organ._train` leaves grad-bearing `last_activations` (no-grad
  forward before deepcopy); `voters()` without `mshape` = nothing-moves path; motion.py build
  from repo root; 7 GB RAM ≈ 3 leaf procs.

## State of the build / Pointers
- Commit on `conscience-core` this session: motion_cnn.py + 4 logs + this handoff. `main` NOT
  advanced. Pushed.
- Kinopsis (s057) unchanged: `experiments/progenitor/kinopsis.py`, saved organisms
  `outputs/data/motion/motion_organism_s{0,1,2}.pt` (gitignored).
- Package (`trioron/`) untouched.

## DO-NOT-COMMIT carries (unchanged since s034, LEAVE THEM)
`trioron/bases/developmental.py`, `trioron/lifecycle/developmental.py`,
`trioron/viz/export.py`; `.claude/`, `runs/`, `archive/runs/`,
`trioron/legacy/outputs/`, `notebooks/` checkpoints, output PNGs / uncommitted
logs untracked; `outputs/data/**` caches.

## Environment notes
- `/home/marcrockhat/trioron-project/`, branch `conscience-core`, Python
  3.10.12, torch 2.11.0, WSL2, `python3` (NOT `python`), 12 CPUs, 7 GB RAM.
