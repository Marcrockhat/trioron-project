# trioron — teaching petri dish

A static, single-page Canvas demo that walks readers through trioron's
core mechanisms (genesis, frustration, fission, dream, absorption, manifold
replay) using a small population of cells living in valence/arousal space.

Each chapter exposes one knob — the same one you'd pass to the `trioron`
Python API. Audio at the closing beat is RAVDESS (drop into `audio/` later).

## Status

All nine scenes are scaffolded. Polish pass pending.

- [x] Scene 1 — Genesis (knob: `l0_dim`)
- [x] Scene 2 — First Task: HAPPY vs SAD (knob: `frustration_threshold`)
- [x] Scene 3 — Population Grows (knob: `branch_width`)
- [x] Scene 4 — Task 2: ANGRY vs CALM (knob: `replay_lambda`)
- [x] Scene 5 — First Dream (knob: `max_downscales_per_layer`)
- [x] Scene 6 — Compatible Donor (knob: `absorption_radius`)
- [x] Scene 7 — Foreign Donor (knob: `seed_match_required`)
- [x] Scene 8 — Cap Reached (knob: `population_cap`)
- [x] Scene 9 — Replay Toggle + closing TTS beat (knob: `manifold_noise_scale`)

## Audio

Scene 9 plays an emotion clip on canvas click. If `audio/<key>.mp3` exists,
that's used. Otherwise falls back to `speechSynthesis` with pitch/rate
flavored by emotion. Drop RAVDESS clips named `happy.mp3`, `sad.mp3`,
`angry.mp3`, `calm.mp3`, `excited.mp3`, `anxious.mp3`, `bored.mp3`,
`tender.mp3` into `audio/` to upgrade.

## Local preview

It's plain static files, so any server works:

    cd teach
    python3 -m http.server 8080

Then open <http://localhost:8080>.

## Hosting on GitHub Pages

1. Commit `teach/` to `main`.
2. Repo → Settings → Pages.
3. Source: "Deploy from a branch", Branch: `main`, Folder: `/teach`.
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
  tick. At threshold the cell fissions and frustration resets.
- Fission spawns a child sharing seed lineage but at slight position
  offset, which is what differentiates the children's specialty.
