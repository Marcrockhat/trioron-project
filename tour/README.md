# trioron — a guided tour

A static, single-page Canvas demo that walks readers through trioron's
core mechanisms (genesis, frustration, division, dream, absorption, manifold
replay) using a small population of cells living in valence/arousal space.

Each chapter exposes one knob — the same one you'd pass to the `trioron`
Python API. Thirteen chapters total. Audio at the closing beat is RAVDESS
(drop into `audio/` later).

## Status

Thirteen scenes shipped — 10 narrative chapters plus 3 controlled experiments.

- [x] Scene 1 — Genesis (knob: `l0_dim`)
- [x] Scene 2 — First Task: HAPPY vs SAD (knob: `frustration_threshold`)
- [x] Scene 3 — Population Grows (knob: `branch_width`)
- [x] Scene 4 — Task 2: ANGRY vs CALM (knob: `replay_lambda`)
- [x] Scene 5 — First Dream (knob: `max_downscales_per_layer`)
- [x] Scene 6 — Compatible Donor (knob: `absorption_radius`)
- [x] Scene 7 — Foreign Donor (knob: `pool_match_required`)
- [x] Scene 8 — Cap Reached (knob: `population_cap`)
- [x] Scene 9 — Replay Toggle + closing TTS beat (knob: `manifold_noise_scale`)
- [x] Scene 10 — Expansion: DISGUST + PRIDE (knob: `extend_replay_lambda`)
- [x] Scene 11 — Directed: Convergence (knob: `frustration_threshold`)
- [x] Scene 12 — Directed: Fails to Converge (knob: `population_cap`)
- [x] Scene 13 — Directed: Expansion Forgets (knob: `extend_replay_lambda`)

Scenes 11–13 are *controlled experiments* — they reset the dish on entry and
preload a specific failure/success regime. A small sandbox in `main.js` saves
the world state when entering the directed range and restores it when leaving,
so the continuous world of scenes 1–10 isn't lost.

## On-screen readouts

- **Population / cap / divisions** counter top-left of the dish (updates every tick).
- **Σ frustration** + per-cell average in the directed scenes — color-coded green→red.
- **Decision boundaries** (scenes 3, 4, 11–13): a dashed cross. Vertical = valence
  (HAPPY↔SAD), horizontal = arousal (ANGRY↔CALM). Each line's thickness scales
  with the specialists that built it; either thins independently under forgetting.
- **L0 ghost rings** linger above the dream overlay after apoptosis — the
  architectural truth that L0 substrate is conserved through L1 branch pruning.
- **Metaphor-map panel** below the math-readout: tables every place where the
  petri-dish visual diverges from real-API semantics (`branch_width` ≠ L1
  capacity, inner-ring color = pool tag not seed, apoptosis = L1-only prune,
  fission-spawn distance has no architectural analog, etc.).
- **↺ restart** button in the stepper: wipes the world, all per-scene
  snapshots and remembered slider positions, jumps to scene 1.

## Audio

Scene 9 plays an emotion clip on canvas click. If `audio/<key>.mp3` exists,
that's used. Otherwise falls back to `speechSynthesis` with pitch/rate
flavored by emotion. Drop RAVDESS clips named `happy.mp3`, `sad.mp3`,
`angry.mp3`, `calm.mp3`, `excited.mp3`, `anxious.mp3`, `bored.mp3`,
`tender.mp3` into `audio/` to upgrade.

## Local preview

It's plain static files, so any server works:

    cd tour
    python3 -m http.server 8080

Then open <http://localhost:8080>.

## Hosting on GitHub Pages

1. Commit `tour/` to `main`.
2. Repo → Settings → Pages.
3. Source: "Deploy from a branch", Branch: `main`, Folder: `/tour`.
4. Save. After ~1 minute the page is live at
   `https://marcrockhat.github.io/trioron-project/`.

## Design notes

- Canvas is the (valence, arousal) plane. Each trioron's *position* is its
  specialty — there's no separate latent. This makes the petri dish *also*
  the decision space, so the inset boundary update is just a Voronoi over
  the population.
- The inner ring of every cell is its L0 seed lineage. All descendants of
  the genesis cell share it (same color), which is why later donors will
  be drawn with a *different* inner ring — that visual difference is the
  shared-seed invariant.
- Frustration is per-cell; misclassifications add 0.06, decay 0.8% per
  tick. At threshold the cell divides and frustration resets.
- Division spawns a child sharing seed lineage but at slight position
  offset, which is what differentiates the children's specialty.
- The simulation is **stochastic** — captions describe trends, not
  guaranteed paths. Lower thresholds can produce fewer total divisions
  on binary tasks because the first early division has ~50% odds of
  landing the unmet specialty before the parent drifts.
- `min_live_pop` is a per-world parameter (default 5) — both dream
  apoptosis and cap-driven turnover refuse to drop the population below
  it. Scene 12 lowers it to 2 so a real cap=3 bottleneck can demonstrate
  failure-to-converge without the floor masking it.
