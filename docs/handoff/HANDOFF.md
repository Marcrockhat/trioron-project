# Trioron Handoff

**Session date:** 2026-05-24
**Session number:** 002
**Session title:** v2 scaffold + core implementation

> This file is **rewritten in full** every session. The previous
> session's handoff is preserved in git history
> (`git log docs/handoff/HANDOFF.md`). Read this file first when
> starting any new session; rewrite it before ending one. See
> `docs/handoff/README.md` for the convention.

## Summary

Completed handoff steps 1–3 and started step 4. Created the
`v2.0-scaffold` branch, scaffolded the full v2 directory tree (69
stub files across 14 directories, each linked to its spec section),
moved all v1 modules into `trioron/legacy/` via `git mv`, and
implemented all 10 `core/` modules (1034 lines).  The core layer is
smoke-tested end-to-end: construct → compile → forward → backward
with gradient flow confirmed.

## State of the build

- **Branch:** `v2.0-scaffold` (ahead of remote by 2 commits —
  **push needed** at start of next session)
- **Commits this session (4):**
  - `234bf3c` — scaffold: v2 directory tree from spec §9
  - `38acf74` — legacy: move v1 modules into `trioron/legacy/`
  - `87f7670` — core: foundation layer (Cell, Arena, Envelope,
    Epigenome, Lineage)
  - `8ca0956` — core: CellGraph, Scheduler, Substrate, construct()
- **Files added (v2 scaffold + core):**
  - `trioron/core/` — 10 modules, all implemented:
    - `state.py` (10 lines) — `CellState` enum: ACTIVE / DORMANT
    - `roles.py` (10 lines) — `CellRole` enum: NEURON / ASTROCYTE
    - `epigenome.py` (40 lines) — gene bit constants + helpers
    - `envelope.py` (73 lines) — resource caps, pressure,
      niche_capacity, stagnation
    - `arena.py` (179 lines) — pre-allocated SoA tensors (cells +
      COO edges), alloc, add_edges
    - `cell.py` (133 lines) — thin read/write facade (not hot-path)
    - `lineage.py` (57 lines) — parent_of, root_of, siblings_of,
      descendants_of
    - `graph.py` (189 lines) — add_edge with 3 invariants, Kahn's
      BFS rank recompute
    - `scheduler.py` (199 lines) — Bucket / DispatchPlan,
      compile(), forward(), dormant grad mask
    - `construct.py` (135 lines) — Substrate class, Base protocol,
      construct() function
  - `trioron/phenotype/`, `trioron/bases/`, `trioron/learning/`,
    `trioron/lifecycle/`, `trioron/viz/`, `trioron/evolution/`,
    `trioron/compat/` — all stub files only (docstring + spec link)
  - `tests/test_v2/`, `bench/` — empty stubs
- **Files modified:**
  - `trioron/__init__.py` — re-exports from `trioron.legacy.*`
  - `pyproject.toml` — new packages list + CLI entry point →
    `trioron.legacy.cli:main`
- **v1 modules relocated:**
  - All v1 `.py` files and sub-packages (`senses/`, `composition/`,
    `bridge/`) moved to `trioron/legacy/` via `git mv`
  - Absolute imports in legacy files fixed: `from trioron.X` →
    `from trioron.legacy.X`
  - Relative imports within legacy unchanged (all resolve correctly)
- **In-flight work:** none — working tree clean

## Decisions made (and why)

| Decision | Why |
|---|---|
| Arena is a plain class, not `nn.Module` | Keeps arena as pure storage. The `Substrate` wraps it and exposes `trainable_tensors()` for the optimizer. Avoids registering large pre-allocated edge buffers as Parameters. |
| Edge storage is COO (3 flat tensors: src, dst, weight) | CSR is hard to append to during growth. COO is append-only. The scheduler compiles to batched gather indices at task boundaries. CSR conversion can be added later as a pure performance optimization. |
| `output_dim = 1` only for Phase 1 | Arena stores `output_dim` per cell but edge weights are scalar. Block-cell support (output_dim > 1) is a documented extension point, not implemented. |
| `phenotype_cache` stores the gene bit position (0–4) | The 5 expression genes (LINEAR through DENDRITE) map 1:1 to phenotype implementations. `primary_phenotype()` returns the lowest set expression gene. Multi-gene cells participate in multiple scheduler passes. |
| Scheduler receives dispatch table via dependency injection | Avoids circular import between `core/` and `phenotype/`. The `construct()` function wires the dispatch table from `phenotype/`. |
| Perception cells excluded from dispatch buckets | Their activation is the raw input, set before dispatch. All non-perception alive neurons are dispatched by (rank, phenotype). |
| Bucket stores `edge_indices` into arena's COO arrays | `arena.edge_weight[edge_indices]` in the forward pass preserves gradient flow back to the arena tensor. No copy of weights. |
| Cycle check is per-edge DFS | Acceptable at Phase 1 scale (~1000 cells). Checks that every cell on a potential cycle has the RECURRENT gene per §2.3. |
| `torch` installed CPU-only via conda pip | System Python lacked torch. Installed `torch-2.12.0+cpu` at `/home/marcrockhat/miniconda3/`. Use `/home/marcrockhat/miniconda3/bin/python` for all runs. |

## Open questions

None blocking the next step.

Questions to resolve before later steps:

- **Autograd integration depth:** Currently `arena.bias` and
  `arena.edge_weight` are plain tensors with `requires_grad_(True)`
  set by `Substrate.prepare_training()`. This works but the
  optimizer sees the full pre-allocated tensor. At Phase 2 scale
  this may need a smarter approach (nn.Module, custom Parameter, or
  masked optimizer).
- **Saliency weights** — TODO inline in spec §5.5 (carried from
  session 001).

## Next-up tasks (in priority order)

Per spec §9.14, continuing from where session 002 left off:

1. **Push branch** — 2 unpushed commits on `v2.0-scaffold`.
2. **Implement `phenotype/linear.py`** (spec §3.2) — the first real
   phenotype. Register it in `phenotype/__init__.py` dispatch table.
   This completes the smallest possible runnable forward pass
   through the substrate.
3. **Implement `phenotype/__init__.py`** dispatch table and
   `register()` function (spec §3.1).
4. **Write first test** in `tests/test_v2/test_core.py` — construct
   a tiny substrate, forward, backward, check gradient flow. Pin
   the smoke test from this session as an automated test.
5. **Then `learning/credit.py` + `learning/manifold.py`** for the
   first training loop (spec §4.1, §4.5).
6. **Then `lifecycle/grow.py` + `learning/dream.py`** for first
   growth + consolidation experiment (spec §5.1, §4.3).
7. **Then `viz/recorder.py`** and `viz/render/` early — so first
   growth is observable from the start (spec §7.1, §7.3).
8. **Add `bench/run_contract.py`** to CI to gate Phase 1 targets
   (spec §6).

Each step gets its own commit.

## Pointers — read these first next session

- **`trioron/core/__init__.py`** — re-exports all 10 modules; good
  starting point for seeing what exists
- **`trioron/core/construct.py`** — Substrate class and
  `construct()` are the user-facing entry points
- **`trioron/core/scheduler.py:forward()`** — the hot-path forward;
  needs a dispatch table entry for LINEAR to produce real output
- **`paper/v3/spec.md` §3.1–§3.2** — phenotype contract + linear
  phenotype spec (next implementation target)
- **`paper/v3/spec.md` §6.3** — 8 hot-path commitments that
  constrain implementation choices
- **`paper/v3/spec.md` §9.14** — cross-section index for fast
  lookup

## Environment notes

- Working directory: `/home/marcrockhat/trioron-project/`
- Branch at session end: `v2.0-scaffold` (commit `8ca0956`,
  **2 commits ahead of remote — push at session start**)
- Platform: Linux 6.6 (WSL2)
- Python: `/home/marcrockhat/miniconda3/bin/python` (3.13) —
  `torch 2.12.0+cpu` installed this session. System python3
  (`/usr/bin/python3`) does NOT have torch.
- numpy is **not** installed (torch warns about this but
  functions fine)

### Cross-PC setup notes

- **Python environment**: the conda python at
  `/home/marcrockhat/miniconda3/bin/python` has `torch` installed.
  On another PC, either install torch in the conda/system python or
  create a venv:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  pip install -e .
  ```
- **SSH key**: per-PC ed25519 key needed for GitHub push. See
  session 001 handoff (in git history) for setup steps.
- **Git identity**: `user.name` and `user.email` must be set on
  each PC before committing.
