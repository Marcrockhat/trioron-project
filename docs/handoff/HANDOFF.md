# Trioron Handoff

**Session date:** 2026-05-24
**Session number:** 001
**Session title:** Spec complete

> This file is **rewritten in full** every session. The previous
> session's handoff is preserved in git history
> (`git log docs/handoff/HANDOFF.md`). Read this file first when
> starting any new session; rewrite it before ending one. See
> `docs/handoff/README.md` for the convention.

## Summary

Drafted the full Trioron v2.0 architecture spec at
`paper/v3/spec.md` (3117 lines, 9 sections). The spec defines a
cell-to-cell graph substrate that adopts every principle of v1.0 /
v1.1 (read in `paper/paper.tex` and `paper/v2/paper.tex`) but
rebuilds them on a lean partition without v1's patch sprawl
(node.py = 2551 lines, api.py = 1446 lines, etc.). No code yet — the
build starts next session per Section 9.14 of the spec.

## State of the build

- **Branch:** `main` (no v2 branch yet — create `v2.0-scaffold` next
  session)
- **Files added this session:**
  - `paper/v3/spec.md` — the spec, authoritative reference
  - `CLAUDE.md` — project-level instructions for future sessions
  - `docs/handoff/README.md` — handoff convention (single-file
    rewrite model, unconditional rule)
  - `docs/handoff/HANDOFF.md` — this file (rewritten every session)
- **Files modified:** none
- **In-flight work:** none — everything reached a natural stopping
  point at the spec

## Decisions made (and why)

| Decision | Why |
|---|---|
| Same `trioron/` package name (not `trioron2/`) | User wants backwards compatibility; one import path |
| Envelope uncapped by default (max_cells/edges/bytes = None) | RAM is the only true bound; caps are optional |
| Two-state lifecycle: active ↔ dormant (rejuvenation-reversible) | "Death" loses lineage and function expensively; dormant preserves both |
| Astrocytes = parameter-holder cells (forward_inclusion=false) | User's framing — biological astrocytes hold information and modulate without firing |
| Per-cell `output_dim` (default 1, cap 256) | Block cells reduce arena slot count for vector-output families |
| Multi-head attention = N independent 4-cell lineages, no sharing | Cellular division costs envelope resources; selection operates on divided cells |
| Conv = sensor receptors + astrocyte kernel | Receptors are zero-weight cells; biologically retinal-photoreceptor-like |
| Mutation OFF by default; three opt-in evolution modes | Substrate must be verified working without mutation first |
| Default graft wiring = `manifold_routed` | Generalizes across donor/recipient pairs; `pool_matched` (v1.1's absorption) demoted to opt-in |
| Lock cap scaled (sqrt of active cells) + dynamic (stability factor) | Fixed caps don't survive substrate-size range from hundreds to millions |
| Saliency = composite (utility + engagement + downstream impact) | TODO inline in spec: calibrate weights $(0.6, 0.3, 0.1)$ |
| Adaptive defrag threshold (loose at low pressure, tight at high) | Defrag is wasted work when arena has slack |
| Recycle by lowest saliency, OLD age as tiebreaker | Old low-saliency cells had time to prove themselves and didn't |
| Phase 1 binding: 50K params, full-fidelity recording, aggressive CI | Earn each scale before claiming it |
| Phase 2 stretch: 1M params, sampled recording | Non-binding for v2.0 |
| EWC fully demoted to `learning/baselines/` | v1.1 established credit-based freezing is strictly stronger |
| Manifold archive entries are astrocytes | Unified arena, visible in snapshots, lifecycle-uniform |
| Visualization: snapshot-record-then-replay, not live | Decouples training from rendering |
| Self-contained HTML viewer (Three.js + InstancedMesh) | Portable; works on any machine without a server |

## Open questions

None blocking the scaffolding step.

Questions to resolve before later steps:

- **Saliency weights** — TODO inline in spec §5.5; calibrate on first
  chained-15 run at Phase 1 scale
- **Phase 2 graduation criterion** — "Phase 1 stable across all
  benchmarks for two release cycles" is vague; tighten when we have
  a release cadence

## Next-up tasks (in priority order)

Per `paper/v3/spec.md` §9.14:

1. **Create branch `v2.0-scaffold`** off `main`.
2. **Scaffold the directory tree** from Section 9 of the spec:
   `trioron/{core,phenotype,bases,learning,lifecycle,viz,evolution,compat,legacy}/`.
   Each directory gets an `__init__.py`. Each spec'd file is created
   as a stub with a docstring linking back to its spec section.
3. **`git mv` v1 modules into `trioron/legacy/`** — pure move, no
   edits. Verify all existing v1 tests still pass after the move.
4. **Implement `core/` first** in dependency order: `Cell`, `Arena`,
   `Envelope`, `CellGraph`, `Scheduler`.
5. **Then `phenotype/linear.py`** for the smallest possible runnable
   forward pass.
6. **Then `learning/credit.py` + `learning/manifold.py`** for the
   first training loop.
7. **Then `lifecycle/grow.py` + `learning/dream.py`** for first
   growth + consolidation experiment.
8. **Add `viz/recorder.py` and `viz/render/`** early — so first
   growth is observable from the start.
9. **Add `bench/run_contract.py`** to CI to gate Phase 1 targets.

Each step gets its own commit and PR.

## Pointers — read these first next session

- **`paper/v3/spec.md` §1** (5 principles) — design north star
- **`paper/v3/spec.md` §9** (partition map) — directory scaffold
- **`paper/v3/spec.md` §9.14** (cross-section index) — fastest lookup
- **`paper/v3/spec.md` §6.3** (8 hot-path commitments) — constraints
  on implementation choices
- **`paper/paper.tex`** — v1.0 paper, for context on what's preserved
- **`paper/v2/paper.tex`** — v1.1 paper, for context on the axis
  system v2 collapses

## Environment notes

- Working directory: `/home/marcrockhat/trioron-project/`
- Branch at session end: `main` (commit `aec18f0` on remote)
- Platform: Linux 6.6 (WSL2)
- Python: from `pyproject.toml` — torch + standard scientific stack
- No new dependencies introduced this session

### Cross-PC setup notes

- **SSH key**: an ed25519 key was generated on this PC at
  `~/.ssh/id_ed25519` and added to the user's GitHub account. Other
  PCs need their own key — generate per machine, add each public
  key to GitHub Settings → SSH and GPG keys, and switch the local
  remote to SSH:
  `git remote set-url origin git@github.com:Marcrockhat/trioron-project.git`
- **GitHub host key** was added to `~/.ssh/known_hosts` on this PC
  via `ssh-keyscan`. Same step needed on each new machine before
  the first push.
- **Git identity** (`user.name`, `user.email`) was set globally on
  this PC. Set the same values on any new machine before committing.
