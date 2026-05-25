# Trioron v2.0 — Architecture Specification

**Status:** draft, sections 1–2
**Author:** Marcelinus R. Hatorangan (with Chloe)
**Date:** 2026-05-24
**Supersedes:** `trioron_2_0.md`, `trioron_blueprint.md` (kept for v1 reference)

This spec is the source of truth for Trioron v2.0. It defines a cell-to-cell
graph substrate that adopts every principle of the v1.0 / v1.1 architecture
(reported in `paper/paper.tex` and `paper/v2/paper.tex`) but rebuilds them
on a lean partition without the patch sprawl of `node.py` (2551 lines) or
`api.py` (1446 lines).

The defining shift from v1: there is no `TrioronLayer`. There are only
cells, connected by edges in a directed graph, allocated within a bounded
**envelope** of total parameter and edge budget. Layers, in v1's sense,
are not built — they emerge as connectivity strata when the substrate
self-organizes.

---

## 1. Philosophy and Principles

Trioron v2.0 is one substrate that grows into many architectures. The
same cell, configured by its **epigenome**, can express as a linear
combiner, an attention head, a convolutional receptor, a recurrent loop,
or a dendritic compartmentalizer. Transformer-like, convolution-like,
and reasoning-depth behaviors are not built modules — they emerge from
local epigenome configurations under selection pressure.

Five principles govern the architecture. Every design choice in
Sections 2–9 traces back to one of them.

### 1.1 The cell is the only object

There is no layer abstraction. Each cell carries its own incoming edges
as an explicit list, its own per-cell state (weight set, engagement,
utility, position, epigenome, lineage), and its own phenotype tag. The
network is a directed graph of cells. Layers, depth, and width are
emergent properties of connectivity, not declared structural primitives.

The v1.0 paper's contribution was a tri-parametric *node*; v1.1 added
six per-cell *axes*. v2.0 completes that trajectory: the cell is no
longer one entry of an aggregated layer matrix — it is a first-class
graph object with explicit edges, its own state, and its own
expression.

### 1.2 The envelope is a bounded growth medium

Resources are accounted for, but the envelope is uncapped by default.
The substrate is bounded only by the host RAM. Explicit caps on cell
count, edge count, or parameter bytes are *optional* constraining
parameters set by the operator:

```
envelope = Envelope(
    max_cells=None,                 # uncapped by default
    max_edges=None,                 # uncapped by default
    max_parameter_bytes=None,       # uncapped by default; RAM is the floor
)
```

When any cap is set, the envelope enforces it: growth past the cap
requires pairing with a dormancy event (a low-utility cell must
transition `active → dormant` before a new cell can occupy its slot).
When all caps are `None`, the substrate grows freely until host memory
is exhausted, and the operator is responsible for tracking that boundary
through external monitoring.

Whether capped or uncapped, the envelope exposes the same derived
signals (`pressure`, `niche_capacity`, `stagnation` — Section 2.5)
so that lifecycle machinery downstream does not need to know whether
the substrate is bounded by a cap or by RAM.

This is the v1 paper's "byte cap" generalized: the substrate operates
within whatever container the operator provides, but the architecture
does not require a hardware-mismatched arbitrary bound to function. It
is the architecture's commitment to resource-aware continual learning:
the substrate respects whatever container it lives in.

### 1.3 Phenotype emerges from epigenome

Each cell carries a small bitmask, the **epigenome**, where each bit is
a "gene" that expresses a particular operation when set:

- `linear_gene` (default, always on) — standard linear combine over fan-in
- `attention_gene` — cell participates as an attention head (Q/K/V
  triplet identified by lineage tag)
- `conv_gene` — cell expresses as a convolutional receptor (spatial
  fan-in window + weight-tied lineage)
- `recurrent_gene` — cell may carry a self-edge with bounded unroll depth
- `dendrite_gene` — cell partitions its fan-in into compartmentalized
  branches with per-branch nonlinearity

Genes are not mutually exclusive — a cell may express attention + conv
simultaneously if both bits are set. The phenotype dispatcher
(`core/scheduler.py`) reads the epigenome at forward time and routes
the cell to the corresponding batched op.

Genes are written by three sources: inherited from parent (lineage),
mutated during cellular division (sister differentiation), or
explicitly set by the API (developmental priors). The v1.1 axis system
collapses cleanly: Axis 5 (dendrites) is a gene, Axis 1 (long-range
inputs) is the absence of a sequential-source convention, etc.
Section 3 maps each v1.1 axis to its v2.0 expression.

### 1.4 Lineage drives evolution

Each cell records its parent cell's id. Cellular division (Section 5)
inherits the parent's epigenome with optional random bit flips —
sister cells differentiate by mutation. Selection happens at the dream
cycle: utility scores rank cells, low-utility cells transition to
dormant, high-utility cells become eligible to spawn.

Lineage is also the substrate of weight-sharing for convolutional
phenotypes: a lineage group with the `conv_gene` set shares one weight
tensor across positions, with the cells acting as receptors at
different spatial coordinates. The same mechanism, run with
`attention_gene` and Q/K/V tags, produces an attention head.

Evolution under v2.0 is therefore not a separate machinery — it is the
joint operation of lineage tracking, epigenome inheritance with
mutation, and dream-cycle selection.

### 1.5 Every growth event is observable

The substrate is born inside an envelope and grows by cellular division
under selection pressure. Without visualization, claims about what the
substrate produces (transformer-like cortex, conv-like filter banks,
recurrent reasoning loops) cannot be falsified. Visualization is
therefore a first-class architectural concern, not optional telemetry.

The recorder (Section 7) captures snapshots at task boundaries, growth
events, and dream-cycle consolidations. The detector (Section 7) scans
snapshots for **structural fingerprints** — geometric and topological
patterns that match known templates (cortex sheet, column, cluster,
attention head, conv bank, recurrent loop, dormant region). The viewer
(Section 7) renders the result as a self-contained HTML file with
time-scrub, click-to-inspect, and collapse/expand for detected
structures.

Performance contract (Section 6): the recorder writes JSON snapshots in
under 100 ms at 20K cells; the viewer first-paints under 1 s at the
same scale; the detector runs offline after the run, with no impact on
training.

---

## 2. The Substrate

This section defines the cell, the graph, the envelope, the scheduler,
the epigenome, and the lineage tree. These six concepts are mutually
referential — the order below is presentation, not dependency.

### 2.1 The Cell

A cell is the unit of computation, allocation, and selection. Its state
is the following:

| field | type | role |
|---|---|---|
| `id` | int32 | unique within an envelope, assigned on birth, never reused |
| `weights` | float32[fan_in, output_dim] | incoming edge weights |
| `inputs` | int32[fan_in] | source cell ids for each incoming edge |
| `bias` | float32[output_dim] | bias vector (scalar when output_dim=1) |
| `output_dim` | uint16 | per-cell output width; default 1, cap 256 |
| `engagement` | float32 | running mean activation rate; drives credit-based locking (active → dormant) |
| `utility` | float32 | running mean gradient magnitude; drives pruning |
| `position` | float32[3] | continuous coordinate in $[0,1]^3$, the envelope volume |
| `epigenome` | uint16 | gene bitmask (up to 16 genes) |
| `phenotype` | uint8 | cached primary phenotype (linear / attention / conv / recurrent / dendrite) |
| `lineage_root` | int32 | id of the founding ancestor cell in this cell's lineage |
| `parent` | int32 | direct parent cell id (−1 if founding cell) |
| `rank` | int32 | topological rank in the current DAG (cached, recomputed at task boundaries) |
| `state` | uint8 | one of `active`, `dormant` (reversible) |
| `forward_inclusion` | bool | default `true`; `false` for astrocytes (Section 2.11) — cell holds parameters but is not in the scheduler |
| `age` | int32 | number of training steps since birth |
| `phenotype_data` | variant | phenotype-specific extension (Q/K/V tag for attention, branch partition for dendrite, etc.) |

The total per-cell footprint is approximately 100 bytes plus the
`weights` and `inputs` arrays (variable). At 1M parameters distributed
over 20K cells with average fan-in 50 and `output_dim=1`, per-cell
weight + input cost is $50 \cdot (4 + 4) = 400$ bytes; total per-cell
≈ 500 bytes; total cell state ≈ 10 MB. Cells with `output_dim > 1`
("block cells") pay $\text{fan\_in} \cdot \text{output\_dim} \cdot 4$
bytes for weights — a single block cell with `output_dim=64` and
`fan_in=64` is one $64 \times 64$ block matrix, equivalent storage to
64 scalar cells with the same fan-in but one cell entity in the
arena.

The cell is the *user-facing* object — the `Cell` dataclass in
`core/cell.py`. Internally, all per-cell fields are held in parallel
tensors (Structure-of-Arrays) in the arena (Section 2.4) for batched
operations. The `Cell` API returns views into the arena, not copies.

### 2.2 The Epigenome

The epigenome is a 16-bit mask. Reserved gene assignments:

| bit | gene | when set |
|---|---|---|
| 0 | `linear` | default operation; always on for any active cell |
| 1 | `attention` | cell participates as attention head with Q/K/V lineage tags |
| 2 | `conv` | cell is a convolutional receptor (spatial fan-in + lineage-tied weight) |
| 3 | `recurrent` | cell may carry self-edges; unroll depth in `phenotype_data` |
| 4 | `dendrite` | fan-in partitioned into branches with per-branch nonlinearity |
| 5 | `perception` | cell consumes raw input; cannot lock to dormant before task 1 ends |
| 6 | `output` | cell contributes to a head logit; one gene per class slot |
| 7 | `credit_eligible` | engagement-based credit can lock this cell (active → dormant) |
| 8 | `recyclable` | cell's arena slot may be reclaimed under extreme envelope pressure |
| 9 | `weight_tied_lineage` | cell shares its weight tensor with lineage siblings |
| 10–15 | reserved | future genes (modulatory, neuromodulator, etc.) |

The phenotype dispatcher (`core/scheduler.py`) reads the epigenome and
selects the cell's primary phenotype for batched dispatch. When
multiple expression genes are set (e.g., `attention | conv`), the cell
participates in both passes; the scheduler runs the cell once per
expressed gene and combines contributions additively.

Genes are written by:

1. **Inheritance** at cellular division — child epigenome = parent
   epigenome (sister differentiation may flip bits, Section 5).
2. **Developmental priors** — at substrate initialization, founding
   cells are tagged with `perception` or `output` to wire the I/O
   boundary.
3. **Explicit API call** — `api.set_gene(cell_id, gene, value)`. Used
   for benchmarks where we force a phenotype to compare against
   emergent expression.

The epigenome is not learned by gradient descent. It is mutated under
lineage-driven evolution (Section 5) and explicit developmental
control. This is the key distinction from "soft routing" — phenotype
is a structural decision made by the lifecycle, not a continuous
parameter optimized by SGD.

### 2.3 The Graph

The substrate is a directed graph $G = (V, E)$ where $V$ is the
cell set and $E$ is the edge set. Edges are stored CSR-style: for each
destination cell $v$, the list of source ids and corresponding weights.

The graph is a DAG with explicit cycle support for recurrent
phenotypes. A cycle is permitted only if every cell on the cycle has
`recurrent` set in its epigenome; the scheduler unrolls the cycle for
the per-cell `K_unroll` count (default 1; max 8). Cells without
`recurrent` are forbidden from receiving back-edges; the
`api.add_edge` call rejects such attempts.

The graph has three structural invariants enforced at every edit:

1. **No dangling edges** — every src in `inputs[v]` exists in $V$ and
   is not `dormant`.
2. **No phenotype/cycle violation** — back-edges only into `recurrent`
   cells.
3. **No envelope overflow** — `|V| \leq \text{max\_cells}$,
   $|E| \leq \text{max\_edges}$, $\text{bytes}(V, E) \leq
   \text{max\_parameter\_bytes}$.

When any invariant would be violated by an edit, the edit is rejected
or paired with a dormancy event (Section 5) to make room.

**Topological rank.** Each cell carries a cached integer rank. For an
acyclic cell, $\text{rank}(v) = 1 + \max_{u \in \text{inputs}(v)}
\text{rank}(u)$, with rank 0 for cells with no inputs. For a recurrent
cell, the back-edges are ignored when computing rank — they only enter
the forward pass during the unroll. Rank is recomputed at task
boundaries (cheap: one BFS) and not after every individual growth
event, to avoid quadratic-in-edits cost.

### 2.4 The Arena (Storage Layout)

All per-cell state lives in pre-allocated parallel tensors. The arena
owns:

```
arena = {
    "weights":     SparseCSR(max_cells, max_cells),   # block-sparse by rank
    "bias":        Tensor[max_cells],
    "engagement":  Tensor[max_cells],
    "utility":     Tensor[max_cells],
    "position":    Tensor[max_cells, 3],
    "epigenome":   Tensor[max_cells],                 # uint16
    "phenotype":   Tensor[max_cells],                 # uint8 cached
    "lineage_root":Tensor[max_cells],                 # int32
    "parent":      Tensor[max_cells],                 # int32
    "rank":        Tensor[max_cells],                 # int32
    "state":       Tensor[max_cells],                 # uint8
    "age":         Tensor[max_cells],                 # int32
    "cursor":      int,                               # next free slot
    "dead_mask":   BitTensor[max_cells],              # true if slot is dormant + reclaimable
}
```

`grow_node` appends to `cursor`; `dormant` flips the dead_mask bit. A
compaction sweep runs every $N$ dream cycles (default $N = 10$) to
reclaim dead slots and update all cross-references via a single id
remapping pass. This keeps the common-case growth path O(1) and pays
the O(n) compaction cost only periodically.

The `Cell` dataclass is a thin facade that reads/writes the arena
fields by id. User code never touches the arena directly; it operates
on `Cell` views.

### 2.5 The Envelope

The envelope is the bounded growth medium in which cells live and
compete. It is defined by:

```
envelope = {
    "max_cells":           int | None,    # default None (uncapped)
    "max_edges":           int | None,    # default None (uncapped)
    "max_parameter_bytes": int | None,    # default None (uncapped; RAM is the floor)
}
```

Each cap is optional. When all three are `None`, the substrate grows
freely against the host's RAM; when any cap is set, the envelope
enforces it as a hard limit. Operators typically set one cap to
declare an intended deployment envelope (e.g.,
`max_parameter_bytes=4_000_000` for a ~1M-FP32-param target) and leave
the other two as None.

The envelope produces three derived signals used by the lifecycle
machinery (Section 5):

1. **`pressure`** ∈ [0, 1] — the maximum utilization across all set
   caps; defaults to 0.0 when no cap is set. When `pressure > 0.85`,
   growth requires a paired dormancy event.
2. **`niche_capacity`** — the number of additional cells that could be
   added without violating any set cap (`+inf` when uncapped). Used by
   the cellular division trigger to decide whether to attempt growth.
3. **`stagnation`** — number of consecutive tasks during which
   utilization did not increase. Used to opportunistically compact and
   archive when growth has stalled. Computed even when uncapped, on
   absolute cell count.

The envelope is fixed for the duration of a single run. The substrate
can be re-instantiated with a different envelope (smaller for shipping,
larger for extension training); cells, edges, and parameter
populations transfer through the ship-wake-extend loop (Section 5)
unchanged.

### 2.6 The Lineage Tree

The lineage tree records, for each cell, its parent cell id (−1 for
founding cells). The lineage *root* of a cell is the unique founding
ancestor — the cell at the top of the parent chain. Lineage roots
partition the substrate into trees; lineage trees are the unit of
weight-tied sharing for convolutional phenotypes and the unit of
identity for Q/K/V triplets in attention phenotypes.

The lineage tree supports three operations:

- `parent_of(cell_id) -> cell_id` — direct parent lookup, O(1)
- `lineage_root_of(cell_id) -> cell_id` — root lookup, O(1) (cached on
  the cell, updated on division)
- `siblings_of(cell_id) -> list[cell_id]` — direct siblings, O(siblings)

Compaction (Section 2.4) preserves lineage relationships: when a cell
is removed, its children are re-pointed to its parent. This keeps the
tree contiguous and prevents orphan lineages from accumulating in the
arena.

### 2.7 The Scheduler

The scheduler is the load-bearing performance machinery. It converts
the cell-to-cell graph into batched operations grouped by
`(rank, phenotype)`.

For a forward pass on input batch $X$:

1. **Pre-pass**: ensure ranks are current. If `dirty`, run BFS to
   recompute (typically only after a task boundary, not per forward).
2. **Group cells**: bucket cells with `forward_inclusion=true` by
   `(rank, phenotype, output_dim)`. Astrocytes (Section 2.11) are
   skipped at this step — they hold parameters but never enter the
   scheduler's hot loop. Result is a list of buckets in ascending rank
   order.
3. **Per-bucket dispatch**: for each bucket, call the phenotype's
   batched op (`phenotype/linear.py:forward_batch`,
   `phenotype/attention.py:forward_batch`, etc.). The op consumes the
   activations of all input cells (already computed at lower rank) and
   produces activations for the bucket's cells in one call.
4. **Recurrent unroll**: for each rank with at least one recurrent
   cell, unroll up to the per-cell `K_unroll` depth, re-dispatching
   the bucket until convergence or unroll limit.
5. **Output collection**: gather activations for cells with the
   `output` gene set into a logit tensor for the task's classes.

The scheduler maintains a **dispatch table** keyed by phenotype. New
phenotypes register through `phenotype/__init__.py:register`. The
table is read-only after substrate construction.

Performance contract for the scheduler at 1M parameters, 20K cells,
batch 32, on commodity CPU:

| step | target | hard ceiling |
|---|---|---|
| rank recompute (post-edit) | < 5 ms | < 20 ms |
| grouping | < 1 ms | < 5 ms |
| per-bucket dispatch (avg) | < 0.5 ms | < 2 ms |
| total forward | < 30 ms | < 100 ms |

Detailed performance accounting is in Section 6.

### 2.8 Cell State Transitions

Each cell occupies one of two states:

- **active** — plastic, in forward pass, receives gradients, weights mutable
- **dormant** — frozen, in forward pass, zero gradient, weights immutable, **reversible**

Both states participate in the forward pass — dormant cells continue to
contribute their current outputs to downstream cells, preserving the
function they were locked at. The distinction is plasticity: active
cells accept gradient updates; dormant cells do not. Dormancy is
deliberately reversible. Killing a cell loses its learned function and
its lineage role, both of which are expensive to reacquire; locking it
preserves both at zero training cost.

Permitted transitions:

```
active --(credit threshold)--------> dormant      # earned credit, lock
dormant --(rejuvenation signal)----> active       # un-lock, resume plasticity
dormant --(recycle event)----------> [slot reused, new cell id]
```

**Credit-based locking** (Section 4.1): a cell whose engagement scalar
exceeds the threshold for $K$ consecutive tasks earns credit and
transitions `active → dormant`. Its learned function is preserved at
zero further training cost.

**Rejuvenation** (Section 4.4) is the inverse: a dormant cell returns to
`active` when any of the following fires:

1. Explicit API call (`api.rejuvenate(cell_id)`) — for testing and
   operator control.
2. **Frustration-targeted rejuvenation** — when the frustration trigger
   fires (Section 4.2) and gradient pressure on a dormant cell's
   downstream chain is high, the cell is automatically rejuvenated to
   contribute to resolving the plateau.
3. **Absorption-driven rejuvenation** — when a graft (Section 5)
   introduces inputs that significantly increase a dormant cell's
   engagement, the cell rejuvenates to re-fit the new context.

Rejuvenation preserves the cell's id, lineage, position, epigenome, and
weights — only the `state` flag and the gradient mask change. The cell
resumes training from where it was locked.

**Recycling** is the only true loss event, and it is an event, not a
state. Under extreme envelope pressure, a long-dormant cell whose
downstream contribution has been measured negligible (utility below
floor for $M$ consecutive compactions) may have its arena slot
reclaimed for a fresh cell. The fresh cell receives a new id; the
recycled cell ceases to exist. Recycling requires the
`recyclable` gene (bit 8) to be set on the dormant cell — cells
critical to the substrate's identity (perception, output, founding
lineages) can be marked non-recyclable at construction.

Sections 4 and 5 specify the conditions that trigger each transition.

### 2.9 Construction is Modular

The substrate is constructed from a **base** — a reusable construction
recipe that produces an envelope and an initial cell population. A
base is the v2.0 generalization of v1's "frozen L0 random projection":
instead of one hardcoded perception layer, any frozen substrate can
serve as the ground architecture for new growth on top.

The substrate's public construction surface is:

```
substrate = trioron.construct(
    base=trioron.bases.minimal(input_dim=784, initial_classes=10),
    envelope=Envelope(),                  # uncapped by default
)
```

`construct` takes a `Base` (a reusable recipe) and an `Envelope` (the
operator's resource declaration) and produces a runnable substrate.
The base owns the *what* — which cells exist at birth, with what
genes, at what positions, in what state. The envelope owns the *how
much* — the operator's resource cap, if any.

A `Base` is a callable that, given an envelope, returns the initial
cell population:

```python
class Base(Protocol):
    def __call__(self, envelope: Envelope) -> CellPopulation: ...
```

This protocol allows bases to be composed, frozen, and reused as the
foundation of new substrates. The next subsection (Section 2.10)
specifies the base catalogue, the freezing convention, and base
composition.

### 2.10 Bases (Modular Construction Recipes)

A base produces an initial cell population. Three categories of base
ship in `trioron/bases/`:

**1. Founding bases** — start from scratch, born with only perception
and output cells; everything internal is grown by the first task's
gradient signal.

```python
trioron.bases.minimal(input_dim, initial_classes)
# perception strip on the z=0 face, output strip on z=1, no interior
```

**2. Seeded bases** — start with a small random interior to accelerate
first-task convergence. Useful for benchmarks where you want to compare
against v1's "L1 starts at 32 nodes" without re-deriving the substrate.

```python
trioron.bases.seeded(input_dim, initial_classes, interior_cells=32)
# perception + output + 32 active interior cells at random positions
```

**3. Frozen bases** — load a previously trained substrate, lock all its
cells as `dormant`, and expose them as the ground architecture for new
growth on top. This is the v2.0 path for the multi-branch absorption
story: a donor substrate is loaded as a frozen base, and the
recipient grows new active cells that connect into the dormant base.

```python
trioron.bases.frozen(checkpoint_path, freeze=True)
# all loaded cells set to dormant; weights immutable; available as inputs
trioron.bases.frozen(checkpoint_path, freeze=False)
# loaded cells stay active; useful for resuming training mid-run
```

**Base composition.** Bases compose by concatenation:

```python
vision = trioron.bases.frozen("vision_backbone.pt")
language = trioron.bases.frozen("language_backbone.pt")
base = trioron.bases.compose(vision, language,
                              shared_inputs=False)
substrate = trioron.construct(base=base, envelope=Envelope())
```

The composed base merges both cell populations into one substrate,
optionally sharing input dimensions. Each composed sub-base retains its
own lineage roots, so subsequent evolution operations preserve
provenance. This is the cell-granularity replacement for v1's
`MultiBranchOrganism` wrapper: composition happens at construction time,
not via a routing wrapper at inference time.

**Freezing convention.** When a base is loaded with `freeze=True`, every
loaded cell:

1. Has its `state` set to `dormant`.
2. Has its `credit_eligible` gene cleared (dormancy is permanent unless
   explicitly rejuvenated).
3. Has its `recyclable` gene cleared (the base's cells are part of the
   substrate's identity; the operator must opt in to recycling them).
4. Is excluded from cellular-division candidacy (frozen cells don't
   spawn children directly; new growth happens from active cells that
   read the frozen base's outputs).

This produces the v1-style "frozen L0" behavior as a special case of
the general freezing protocol. The frozen base remains in the forward
pass, contributes deterministic outputs to active cells above it, and
participates in manifold replay's stable code space.

**Backwards compatibility.** v1 checkpoints load through
`trioron.bases.from_v1_checkpoint(path)`, which calls into
`compat/load_v1.py` (Section 8) to translate the v1 `TrioronLayer`
representation into a cell population. The translated cells preserve
their v1 weights, positions, and lineage; they are loaded as `dormant`
by default to preserve the v1 donor's function while new v2 growth
occurs above.

**Base catalogue (initial ship list).** The base module ships with:

| base | role |
|---|---|
| `minimal` | empty interior; pure self-organization from task signal |
| `seeded(K)` | empty interior + $K$ random active cells; faster first-task convergence |
| `frozen(path)` | load + freeze any prior substrate as the ground |
| `compose(*bases)` | concatenate multiple bases into one substrate |
| `from_v1_checkpoint(path)` | load a v1.x donor, frozen by default |
| `random_projection(d_in, d_out)` | v1-style frozen Gaussian L0 reproduction |

New bases register through `trioron/bases/__init__.py`. The catalogue
is open — operators can ship their own bases as third-party packages.

### 2.11 Cell Roles: Neurons and Astrocytes

Cells occupy one of two **roles**, orthogonal to the `state` axis
(active/dormant) of Section 2.8:

- **Neuron** (default) — cell is in the forward pass, computes
  per-batch activations, expresses its phenotype genes. `forward_inclusion = true`.
- **Astrocyte** — cell holds parameters and lineage metadata but is
  not in the forward pass. It modulates computation indirectly by
  providing the weight tensor that other cells (typically receptors
  in its lineage) read. `forward_inclusion = false`.

The role is set at cell creation and is rarely changed during a run.
Both roles share the full cell state of Section 2.1: an astrocyte has
weights, bias, position, epigenome, engagement, utility, lineage, age,
and state, exactly like a neuron. The only difference is the
`forward_inclusion` flag — and its consequence, that the scheduler
skips astrocytes when bucketing cells for dispatch.

**Why astrocytes exist.** Several phenotypes need parameter sharing
across many addressable cells. Conv kernels (Section 3.4) are the
first example: many receptors at different positions read one
shared kernel. Storing the kernel on every receptor would be
wasteful; storing it externally to the cell graph would break the
substrate's "everything is a cell" invariant. An astrocyte is the
clean third option — a cell that holds the shared parameter and
exposes it to its lineage descendants. The receptors are zero-weight
sensor cells; the astrocyte is the kernel.

**Plasticity.** An astrocyte's parameters get gradients via the
receptor cells that reference them. Standard PyTorch parameter
sharing: the autograd graph flows backward through every receptor's
forward call into the astrocyte's `weights` tensor, accumulating
gradients there. The astrocyte's `engagement` is the average of
referencing receptors' engagement (so an unused kernel is locked just
like an unused neuron); its `utility` is the sum of gradient
magnitudes received from receptors.

**State and lifecycle.** Astrocytes participate in the same
lifecycle as neurons:

- `active astrocyte` — parameters mutable, gradients flow
- `dormant astrocyte` — credit threshold reached; parameters frozen;
  rejuvenation reactivates them
- Recycling is permitted (with `recyclable` gene set) but rare —
  losing a kernel deletes all its receptor cells' computations

**Genes.** An astrocyte can carry any gene, but expression genes
(`linear`, `attention`, `conv`, `recurrent`, `dendrite`) do not fire,
because the cell is excluded from the scheduler. The genes that
matter on an astrocyte are:

- `weight_tied_lineage` (bit 9) — declares the astrocyte as a shared
  parameter source for its lineage descendants
- `credit_eligible` (bit 7) — allows credit-based locking
- `recyclable` (bit 8) — allows slot reclamation

**Construction.** Astrocytes are created via `api.grow_astrocyte` or
spawned by phenotype helpers (`lifecycle/grow.py:spawn_conv_filter`
spawns one astrocyte + N receptor cells in one call). They cannot
arise from ordinary cellular division — sister differentiation
produces neurons, not astrocytes, because differentiation operates on
expression genes and astrocytes do not express.

**Visualization.** Astrocytes render distinctly from neurons (Section
7): translucent diamond glyphs vs. opaque spheres, positioned at the
centroid of the receptor cluster they serve. Click reveals the
receptors that reference the astrocyte, drawn as faint lines.

**Honest scope.** The astrocyte metaphor is loose. Biological
astrocytes do far more than parameter holding — they participate in
calcium signaling, gliotransmitter release, and active synaptic
modulation. Our astrocyte is a parameter-holder primitive that
borrows the name for its support-cell role in the neuron/glia
dichotomy. No biological-fidelity claim beyond the metaphor.

---

## 3. Phenotypes

A phenotype is a batched operation expressed by a gene. The scheduler
(Section 2.7) groups cells by `(rank, primary_phenotype)` and dispatches
one batched call per group. This section specifies the five phenotypes
that ship in `trioron/phenotype/` and the contract every phenotype must
satisfy.

### 3.1 The Phenotype Contract

Every phenotype module exposes a single batched-forward function and a
small metadata block:

```python
class Phenotype(Protocol):
    name: str                     # "linear", "attention", ...
    gene_bit: int                 # which epigenome bit enables it
    requires: list[str]           # other phenotypes that must run first
    forward_batch(
        cell_ids: Tensor[B],          # which cells in this bucket
        upstream: Tensor[B, fan_in, upstream_dim],  # gathered upstream activations
        arena: Arena,                 # SoA view, read-only
        substrate_state: SubstrateState,  # rank, recurrent unroll counter, etc.
    ) -> Tensor[B, output_dim]        # per-cell output activation (output_dim per-cell)
```

The contract has four hard requirements:

1. **Pure function of its arguments.** The phenotype may not mutate
   the arena or substrate state during forward. Mutations happen in
   the learning/lifecycle passes, never in the dispatcher's hot loop.
2. **Vectorized across cells in the bucket.** No per-cell Python
   iteration. The bucket is the whole batch; the phenotype is one
   torch op.
3. **Shape-stable output.** Output is `[B, output_dim]` where
   `output_dim` is the per-cell width declared on each cell (default
   1). All cells in one batched dispatch share the same `output_dim` —
   the scheduler buckets by `(rank, phenotype, output_dim)` precisely
   so each dispatch call is a single dense op.
4. **Composes additively.** When a cell expresses multiple genes, each
   gene's `forward_batch` is called and contributions are summed
   elementwise. All expressed genes on one cell must agree on the
   cell's `output_dim`; the substrate enforces this at gene-set time.

**Why per-cell vectors.** A cell with `output_dim=1` is a scalar neuron
in the v1 sense. A cell with `output_dim=d` is a "block cell" — one
addressable unit in the arena, internally a $d$-dimensional feature.
Block cells reduce the cell count needed for vector-output families
(attention, conv with multiple channels): one attention OUT cell with
`output_dim=64` replaces 64 scalar OUT cells, saving 63 arena slots
and 63× dispatch overhead without losing the cell-as-addressable-unit
property. The substrate is still "cells, not layers" — block cells are
just denser cells.

A `primary_phenotype` field is cached on each cell to avoid
re-dispatching all genes per forward; the primary is the
highest-priority gene set. Priority order: `attention > conv >
dendrite > recurrent > linear`. The scheduler dispatches the primary
phenotype in the main pass and runs secondary contributions in a
separate accumulation pass per rank, summing elementwise into the
cell's `output_dim`-shaped activation.

### 3.2 Linear (default)

**Gene:** bit 0, `linear`. Always set on any active or dormant cell.

The default phenotype: a standard linear combine of fan-in
activations.

$$y_v = b_v + \sum_{u \in \text{inputs}(v)} w_{vu} \cdot a_u$$

`forward_batch` is one `torch.bmm` or `torch.sparse.mm` over the
bucket. At 1M parameters in mostly-linear cells, this is the dominant
cost of the forward pass.

**Initialization:** new linear cells initialize their incoming weights
via the v1 growth-direction routine (top principal direction of the
residual signal at the rank below — Section 5.1), or to a small random
Gaussian if no residual is available.

### 3.3 Attention

**Gene:** bit 1, `attention`. Cell participates as an attention head.

Attention requires a structured lineage: an attention cell expects its
parent cell to have spawned a **Q/K/V triplet** of cells, each tagged
in `phenotype_data` with one of the roles `{Q, K, V}`. The attention
cell's own role is `OUT` and it combines them via scaled dot-product:

$$\text{attn}(v) = \text{softmax}\!\left(\frac{Q_v K_v^\top}{\sqrt{d_k}}\right) V_v$$

where $Q_v, K_v, V_v$ are the activations of the cell's lineage-tagged
siblings at the same rank, and $d_k$ is the per-head key dimension
(the `output_dim` of the Q/K cells in the triplet).

**Lineage convention:** when a cell spawns an attention head, the
spawn produces four sibling cells in one operation, all carrying
`output_dim = d_head`:

```python
def spawn_attention_head(parent_cell, d_head):
    Q = parent_cell.divide(role="Q", output_dim=d_head,
                            inputs=parent_cell.inputs)
    K = parent_cell.divide(role="K", output_dim=d_head,
                            inputs=parent_cell.inputs)
    V = parent_cell.divide(role="V", output_dim=d_head,
                            inputs=parent_cell.inputs)
    OUT = parent_cell.divide(role="OUT", output_dim=d_head,
                              inputs=[Q, K, V], genes=GENE_ATTENTION)
    return OUT
```

The Q/K/V cells are themselves block cells with `linear` gene; only the
OUT cell carries the `attention` gene. One 4-cell lineage = one
attention head of dimension `d_head`.

**Multi-head attention** is implemented by spawning N independent
attention heads — N separate 4-cell lineages, each with its own Q, K,
V, and OUT. Sharing Q/K/V across heads via fanout is **not** supported:
cellular division costs envelope resources, and the substrate's
selection pressure operates on divided cells. Heads that share
projections are economically equivalent to one head with a wider
output_dim, and the substrate would not maintain them as separate units
under selection. Multi-head attention with N independent heads of
dimension `d_head` costs $4N$ cells.

**Compute cost.** One attention head batches as one scaled-dot-product
call over the bucket. With 8 heads at `d_head=64` and shared rank,
the scheduler dispatches one batched SDPA over 8 (rank, attention)
buckets — cheaper than 8 separate calls due to torch's batched SDPA
kernel.

**Detection (Section 7):** the visualizer detects "attention head"
structures by finding 4-cell lineages with a single OUT (carrying
the gene) and three siblings tagged Q, K, V, all sharing a parent.
This makes emergent attention heads visible without instrumenting the
substrate.

### 3.4 Convolution

**Gene:** bit 2, `conv`. Cell is a convolutional receptor — a **sensor
cell**.

A conv cell is a sensor: its role is *where it samples*, not *what it
weighs*. The cell carries no weight tensor of its own; instead, its
`weights` field is empty, and the kernel weights are stored once at the
lineage root. The receptor's identity is its `position` (where on the
upstream feature manifold it samples) and its `lineage_root` (which
kernel it shares). This is biologically the retinal-photoreceptor
arrangement: photoreceptors carry only a sampling position; the
"filters" that transform their output live downstream.

For an input batch and a conv receptor cell at position $(x, y)$:

$$y_v = b + \sum_{(\delta x, \delta y) \in K} w_{\delta x, \delta y} \cdot a_{x + \delta x, y + \delta y}$$

where $K$ is the kernel footprint stored at the lineage root and
$w_{\delta x, \delta y}$ is read from the shared lineage weight tensor.

**Lineage convention:** a conv "filter" is one **astrocyte** (the
lineage root, carrying the kernel weights, `forward_inclusion=false` —
Section 2.11) and a set of receptor cells at different positions. The
astrocyte holds the kernel; the receptors are zero-weight sensor
cells that sample the upstream manifold and read their weights through
the astrocyte's lineage tie. The astrocyte is not in the forward pass;
gradients flow back to its weight tensor through the receptors that
reference it.

Spawning a new receptor at position $(x', y')$ adds a cell to the
lineage with no new weights — the receptor reuses the shared kernel.
This is how convolution achieves its parameter efficiency under the v2
substrate: receptors are addressable cells but kernels are
lineage-shared parameters. Growing the filter bank's spatial coverage
is cheap (new sensor cells, no new weights); growing the filter bank's
expressive capacity requires spawning a new lineage with its own
kernel.

**Multi-channel output.** A conv cell's `output_dim` corresponds to the
number of output channels at that position. The shared kernel at the
lineage root is then `[in_channels, kernel_h, kernel_w, output_dim]`.
Multi-receptor lineages produce multi-position, multi-channel outputs
in one batched dispatch.

**Batched forward:** the `phenotype/conv.py:forward_batch` op groups
conv cells by lineage root, then dispatches one `torch.nn.functional.
conv2d` per lineage. At 1M parameters with mostly-conv cells (deep
vision model regime), this matches a hand-built CNN's throughput.

**Sensor-cell generalization.** The receptor pattern — a sensor cell
sampling at a position, plus an astrocyte holding the shared
parameters — is reusable beyond conv. Future phenotypes that need
parameter sharing across many addressable cells (embedding tables,
learned positional codes, attention key caches) adopt the same
convention: zero-weight sensor cells in the forward pass, one
astrocyte per shared parameter source. See Section 3.8 for the
recipe.

**Detection:** the visualizer detects "conv filter bank" structures
by finding lineage trees where (a) all cells carry the `conv` gene,
(b) cell positions are spread across a regular grid on the input
manifold, and (c) the lineage shares one weight tensor.

### 3.5 Recurrent

**Gene:** bit 3, `recurrent`. Cell may carry self-edges.

A recurrent cell is permitted to receive back-edges from cells at
equal or higher rank. The `phenotype_data` carries a `K_unroll` count
(default 1, capped at 8 — Section 1, point 4): the scheduler unrolls
the recurrent step up to `K_unroll` times per forward pass.

The unroll convention is fixed-depth, not fixed-point. There is no
convergence criterion — the scheduler runs exactly `K_unroll` steps,
and the final activation is used downstream. This avoids the latency
unpredictability of fixed-point iteration on CPU.

**Forward:** the recurrent cell behaves as a linear cell on its
non-recurrent inputs and accumulates self-edge contributions across
unroll steps:

```
a_v^{(0)} = linear(non_recurrent_inputs)
for t in 1..K_unroll:
    a_v^{(t)} = a_v^{(t-1)} + linear(recurrent_inputs at a^{(t-1)})
```

The unroll depth is per-cell, not per-substrate — different cells can
unroll different depths, and the scheduler batches cells by
`(rank, K_unroll)` within the recurrent bucket.

**Detection:** the visualizer detects "recurrent loop" structures by
finding edges where `rank(src) >= rank(dst)` — back-edges, by
definition only valid on `recurrent`-gene cells.

### 3.6 Dendrite

**Gene:** bit 4, `dendrite`. Fan-in is partitioned into branches with
per-branch nonlinearity.

A dendrite cell partitions its fan-in into $K$ branches (default $K=2$;
capped by the `B_max` profile parameter from v1.1). Each branch
computes a local linear combine over its partition and passes the
result through a per-branch nonlinearity (default `quad`: $\sigma(x) =
x + x^2$ — the NMDA-style supralinear activation from v1.1). Branch
outputs are pooled with learned per-branch weights:

$$y_v = \sum_{k=1}^{K} \alpha_{v,k} \cdot \sigma\!\left(\sum_{u \in B_k(v)} w_{vu} \cdot a_u\right)$$

where $B_k(v)$ is the $k$-th branch's input partition for cell $v$, and
$\alpha_{v,k}$ is the learned branch weight.

When $K = 1$ the dendrite phenotype is byte-identical to linear (the
quad nonlinearity reduces to a per-cell affine if $\alpha$ collapses to
$1$). This preserves the v1.1 backward-compatibility guarantee.

**Lineage convention:** dendrite cells use `phenotype_data` to store
the branch partition (`branch_id: int[fan_in]`) and the per-branch
weights (`alpha: float[K]`). A new branch is added with `grow_branch`,
which extends `alpha` by one entry and re-assigns some fan-in columns
to the new branch.

**Detection:** the visualizer detects dendritic cells through the
`dendrite` gene and `B > 1` and renders each cell's branches as
sub-glyphs (small marks on the cell sphere showing branch count and
relative weights).

### 3.7 Composite Expression

When a cell carries multiple expression genes, every gene's
`forward_batch` runs, and contributions are summed elementwise into
the cell's `output_dim`-shaped activation. All expressed genes on one
cell must share the cell's `output_dim` (enforced at gene-set time).
Composite expression is rare in practice but architecturally permitted:

- `attention | dendrite` — an attention OUT cell whose linear combine
  is partitioned dendritically. Useful for testing whether dendritic
  routing improves attention.
- `conv | recurrent` — a recurrent convolutional receptor. Useful for
  reasoning-depth at convolutional features.
- `attention | conv` — an attention head that reads from a
  weight-tied lineage. Rare; the user must wire it deliberately.

The scheduler dispatches the primary phenotype in the rank's main
pass and runs secondary contributions in a per-rank accumulation
pass. Total cost is the sum of per-phenotype costs; cells with
multiple expressed genes pay the full cost of each.

**Default behavior:** cellular division initializes children with
exactly one expression gene (sister differentiation may add a second
gene; Section 5.2). The substrate does not gratuitously stack
phenotypes; composition is a deliberate operator choice.

### 3.8 Adding a New Phenotype

The phenotype catalogue is open. Adding a phenotype requires:

1. Assign an unused gene bit in `core/epigenome.py`.
2. Implement `forward_batch` in `phenotype/<name>.py`.
3. Register the phenotype in `phenotype/__init__.py`.
4. Provide a visualizer detection template in `viz/detect.py` (so the
   phenotype is observable in the rendered HTML).
5. Provide a spawn helper in `lifecycle/grow.py` if the phenotype
   requires structured lineage (attention's 4-cell Q/K/V/OUT lineage;
   conv's astrocyte-plus-receptors lineage).

If the new phenotype requires parameter sharing across many
addressable cells, follow the **astrocyte recipe** (Section 2.11,
Section 3.4):

- The astrocyte holds the shared parameter tensor; it has
  `forward_inclusion=false` and carries the `weight_tied_lineage`
  gene.
- Each addressable cell (sensor, receptor, position) is a zero-weight
  cell that references the astrocyte through its lineage tie.
- The spawn helper creates the astrocyte first, then the addressable
  cells, then wires the lineage.
- Gradients flow back through the addressable cells into the
  astrocyte's parameters via standard PyTorch autograd; no special
  handling needed.

The scheduler's dispatch table picks up the new phenotype
automatically; no changes to `core/scheduler.py` are needed unless the
phenotype introduces a new dispatch invariant (e.g., a non-additive
composition rule).

### 3.9 Honest Limits

The phenotype set ships with five entries because each maps to a
recognized architectural family (linear → MLP, attention → transformer,
conv → CNN, recurrent → RNN, dendrite → multi-compartment neuron). The
claim is *not* that v2.0 reaches state-of-the-art on any of these
families' home benchmarks. The claim is:

1. The substrate can *express* each family at cell granularity, by
   setting the right gene on the right lineage.
2. Each family can *emerge* under selection pressure when the gene is
   available — Section 5.2 documents the conditions, Section 7 the
   visual evidence.
3. Multiple families can *coexist* in one substrate, with the
   scheduler handling dispatch in one forward pass.

The substrate does not yet match purpose-built architectures on any
single family. Section 4 documents the learning machinery that drives
expression, and Section 5 the selection pressure that decides which
phenotypes survive.

---

## 4. Learning

This section specifies the learning machinery: how the substrate
acquires knowledge from gradient signal, how it protects acquired
knowledge from drift, how it consolidates between tasks, and how it
recovers from premature locking. The primary anchoring mechanism is
**credit-based locking** (the v1.1 finding that strictly beat EWC).
EWC and its variants live in `learning/baselines/` for benchmark
comparison only.

The four mechanisms specified below run on a fixed schedule:

```
for each task t:
    while task t is in progress:
        for each batch:
            forward
            compute task loss (scaled by frustration multiplier — §4.2)
            backward
            apply gradient mask (dormant cells get zero gradient)
            optimizer step
        update engagement and utility EMAs (§4.1)
        check rejuvenation triggers (§4.4)
        check growth triggers (§5.1 — lifecycle)
    end-of-task: dream cycle (§4.3)
        replay
        consolidate (credit threshold → lock)
        archive (compact dormant slots — §5)
```

### 4.1 Credit-Based Locking (Primary Anchor)

The v1.1 paper established that credit-based row freezing is a
strictly stronger anchor than EWC's quadratic penalty: zero gradient
is a stronger constraint than a soft pull toward an anchor weight.
v2.0 commits to this finding as the default anchoring mechanism. EWC's
machinery (Fisher accumulators, anchor snapshots, $\Lewc$ penalty) is
not instantiated on the default path.

**Engagement scalar.** Each cell carries an `engagement` scalar — an
exponential moving average of the cell's activation rate over training
batches:

$$e_v^{(t+1)} = (1 - \rho) \cdot e_v^{(t)} + \rho \cdot \text{rate}(a_v^{(t)})$$

where $\rho = 0.01$ is the EMA decay (one task ≈ 100–1000 batches gives
several effective time constants) and $\text{rate}(a)$ is the fraction
of batch positions for which the cell's output activation exceeds zero
(for ReLU-like activations) or a calibrated threshold (for other
nonlinearities). Engagement is updated at the end of each batch in the
same hook that updates utility.

**Lock condition.** A cell transitions `active → dormant` when:

1. It has the `credit_eligible` gene set (bit 7), AND
2. Its engagement $e_v$ has exceeded threshold $\theta_e$ (default
   $0.3$ from v1.1) for $K$ consecutive task boundaries (default
   $K=2$), AND
3. The most recent gradient magnitude on the cell's weights is below
   a stability floor $g_{\min}$ (default $10^{-4}$).

The conjunction prevents premature locking: a cell must be both
*used* (engagement above threshold), *stably used* (sustained across
tasks), and *settled* (low gradient signal). The condition is checked
once at each task boundary, not per-batch, so locking is a
discrete-time event aligned with task transitions.

**Lock action.** When the condition fires:

1. The cell's `state` flips to `dormant`.
2. The arena's `dead_mask` is left unset — dormant cells are NOT
   recyclable by default; only cells with the `recyclable` gene set
   can be reclaimed.
3. The cell remains in the forward pass (per Section 2.8). Its
   weights are now immutable; the optimizer skips them on every
   subsequent backward pass.
4. The `lock_event` is logged for the recorder (Section 7), with the
   task index and the cell's engagement/utility profile.

**Gradient masking.** Dormant cells receive zero gradient via a
per-cell mask applied between `loss.backward()` and `optimizer.step()`:

```python
def mask_dormant_gradients(arena, opt):
    dormant = arena.state == STATE_DORMANT
    arena.weights.grad[dormant] = 0.0
    arena.bias.grad[dormant] = 0.0
```

The mask is the only mechanism enforcing the lock. There is no
quadratic penalty term, no anchor weight snapshot, no Fisher
accumulator. The optimizer simply receives zero updates on dormant
cells. This is mechanically equivalent to setting
`requires_grad=False` on the cell's parameters, but cheaper at scale
because the mask is one tensor op rather than 20K Python attribute
assignments.

**Why this is strictly stronger than EWC.** EWC pulls a weight toward
its anchor with strength $\lambda$; with a small task loss in the
opposite direction, the weight can still drift, and the drift
accumulates over many tasks. Credit-based locking sets the gradient
to exactly zero — the weight does not drift at all, regardless of the
task loss. The substrate trades EWC's continuous knob ($\lambda$) for
a binary decision (lock or don't), but the decision is made per-cell
based on observed engagement, so the substrate is naturally
selective: only cells doing useful work get locked.

**Why per-cell, not per-weight.** A cell is the unit of identity and
selection in v2.0. Per-weight locking (zeroing individual entries of
$W$) would require per-weight engagement tracking, which collapses to
EWC-style Fisher accumulation. Per-cell is cheaper (one scalar per
cell vs. one scalar per weight) and matches the substrate's
biological metaphor: cells differentiate and consolidate; individual
synapses do not.

### 4.2 The Frustration Trigger

Frustration is the per-task plateau detector inherited from v1.0
(`paper/paper.tex`, §3.2). It produces a multiplier $m$ applied to the
task loss only, scaling the gradient signal when the optimizer stops
making progress. The mechanism is unchanged in v2.0; the integration
point is.

**Detector.** The detector accumulates the task loss in windows of
length $W_f$ (default 50 batches). At each window boundary it compares
the window's mean loss to the prior window's; if improvement is below
$\varepsilon_f$, it increments a stuck counter $s$. The multiplier is:

$$m = \min(M_{\max},\, 1 + g \cdot \max(0, s - \tau + 1))$$

with hinge $\tau = 2$, gain $g = 0.5$, ceiling $M_{\max} = 4$
(defaults from v1). The multiplier is applied only to the task loss,
not to any regularizer:

$$L = m \cdot L_{\text{task}}$$

There is no $\Lewc$ term on the default path (Section 4.1), so the
total loss in v2.0 reduces to the multiplied task loss alone.

**Three downstream consumers.** Frustration is the architecture's
stress proxy. Three downstream mechanisms read its state:

1. **Growth trigger (Section 5.1)** — when frustration is sustained
   above a threshold, the cellular division trigger fires, attempting
   to add capacity at the rank where stress is concentrated.
2. **Frustration-targeted rejuvenation (Section 4.4)** — when
   frustration fires and gradient pressure on a dormant cell's
   downstream chain is high, the dormant cell is rejuvenated to
   contribute to plateau resolution.
3. **Compaction trigger (Section 5)** — sustained frustration with
   no successful growth indicates the substrate cannot accommodate
   the current task; compaction is scheduled to free dormant slots
   for new growth attempts.

**Reset.** The frustration counter $s$ resets to 0 at each task
boundary. It also resets when the task loss drops by more than
$\varepsilon_{\text{reset}}$ (default $0.1$ absolute) in any single
window — a sign that the substrate has resolved the plateau and the
stress signal is no longer informative.

### 4.3 The Dream Cycle

Between tasks the substrate runs an offline **dream cycle** — a
sleep-time pass that consolidates the just-learned task without
exposing the network to new data. Frustration is disabled during the
dream cycle (no environmental stress during sleep); the gradient mask
of Section 4.1 remains active (dormant cells stay dormant).

The dream cycle has four sequential stages:

```
replay  →  consolidate  →  rejuvenate-check  →  compact
```

**Stage 1: Replay** (Section 4.5). The substrate is exposed to
synthetic samples drawn from the per-class manifold archive (Section
4.5), spanning all classes seen so far. Loss is cross-entropy over
the synthetic batch; backward applies the gradient mask. The purpose
is to anchor the head and the active interior cells against drift on
prior classes.

Replay step count is $K_r$ per past class (default $K_r = 20$); a
30-class curriculum at $K_r = 20$ runs 600 replay steps per dream
cycle.

**Stage 2: Consolidate.** The credit-based lock condition
(Section 4.1) is evaluated at each cell. Cells passing the condition
transition `active → dormant`. The lock events are logged for the
recorder.

This is the moment of structural consolidation: cells that have done
useful, stable work earn credit and freeze. Subsequent tasks must
work around them.

**Stage 3: Rejuvenate-Check.** Dormant cells are scored for
rejuvenation (Section 4.4). Cells whose conditions fire transition
`dormant → active`. This is the inverse of consolidation — cells
that were locked but whose function is no longer well-fit to current
tasks regain plasticity.

In practice this stage rarely fires during normal training. Its
primary use cases are after base loading (a frozen base may need
selective rejuvenation when the new task domain partially overlaps
the base's training distribution) and after absorption (Section 5).

**Stage 4: Compact** (Section 5). The arena is swept: long-dormant
cells with the `recyclable` gene set, whose utility has been below
the floor for $M$ consecutive dream cycles (default $M = 5$), are
recycled — their arena slots are reclaimed, their cell entities cease
to exist, and the slot is available for next growth. A single
remapping pass updates all references (lineage parents, edge sources,
attention OUT inputs).

**Caps are scaled and dynamic.** Each dream cycle is bounded, but the
bounds scale with substrate size and modulate with stability signals.
Fixed numerical caps do not survive the range of substrates the
architecture must support (hundreds of cells to a million).

The cap formulation:

$$\text{cap}_x(t) = \max\!\left(\text{floor}_x,\ \left\lceil \sqrt{N_\text{active}(t)} \cdot s_x \cdot \phi(t) \right\rceil\right)$$

where:

- $N_\text{active}(t)$ is the count of active neurons at the start of
  the dream cycle (astrocytes excluded)
- $s_x \in (0, 1]$ is the per-stage base rate (lock: $0.5$,
  rejuvenate: $0.1$, recycle: $0.1$)
- $\phi(t) \in [0.1, 1.0]$ is the stability factor
- $\text{floor}_x$ is the per-stage substrate floor (lock: $1$,
  rejuvenate: $1$, recycle: $1$)

The sqrt-scaling is sublinear by design — at a 20K-active-cell
substrate the lock cap is $\lceil \sqrt{20{,}000} \cdot 0.5 \cdot 1.0
\rceil = 71$ per cycle; at 1M it is $\lceil 1000 \cdot 0.5 \rceil =
500$ per cycle. Linear scaling would lock 1% of a 1M substrate every
cycle (10K cells) — far too aggressive given v1's finding that
"fewer events $=$ better."

The stability factor is the dynamic component:

$$\phi(t) = \text{clip}\!\left(1.0 - \alpha \cdot \text{frustration\_pressure}(t) - \beta \cdot \text{growth\_rate}(t),\ 0.1,\ 1.0\right)$$

with defaults $\alpha = 0.7$, $\beta = 0.3$.
`frustration_pressure(t)` is the mean of the frustration multiplier
$m$ over the last task; `growth_rate(t)` is the count of new active
cells added during the last task, normalized by $\sqrt{N_\text{active}}$.

The intuition: when the substrate is in flux (high frustration or
heavy recent growth), $\phi$ shrinks toward $0.1$, throttling all
structural change to let the gradient signal settle. When the
substrate is stable, $\phi = 1.0$ and the cycle operates at its full
sqrt-scaled cap. The floor of $0.1$ guarantees that *some* structural
change is always possible — the substrate never freezes itself out of
consolidation.

| stage | $s_x$ | floor | example at $N_\text{active}=20{,}000$, $\phi=1.0$ |
|---|---|---|---|
| replay batches | (deterministic, not capped — runs $K_r \cdot |\text{past classes}|$) |
| consolidation locks | $0.5$ | 1 | 71/cycle |
| rejuvenations | $0.1$ | 1 | 15/cycle |
| recyclings | $0.1$ | 1 | 15/cycle |

These are defaults. Profiles (Section 5) may override $s_x$ and
$\text{floor}_x$ per stage; operators can also pass a custom
$\phi$-policy callable to the substrate. The defaults are conservative;
v1's "smaller per-cycle change preserves accuracy" finding motivates
starting with $s_\text{lock} = 0.5$ and tuning up only if substrate
consolidation is the bottleneck.

### 4.4 Rejuvenation

Rejuvenation is the inverse of credit-based locking: a dormant cell
returns to `active` state with its weights and lineage preserved. The
gradient mask is removed; the cell resumes plasticity from where it
was locked.

Three triggers fire rejuvenation:

**Trigger 1: Explicit API.** The operator calls
`api.rejuvenate(cell_id)`. Used for testing, ablation, and external
control. The dormant cell flips to active immediately; no further
condition is checked.

**Trigger 2: Frustration-Targeted.** During a frustration-driven
growth attempt (Section 5.1), the growth trigger scans the
upstream-downstream chain of cells contributing to the high-loss
batches. If a dormant cell appears in the chain with non-trivial
contribution (engagement above $\theta_e / 2$ on the high-loss
sub-batch), it is rejuvenated. The intuition: this cell was locked at
a time when its function was settled, but the current task has shifted
the gradient landscape such that the cell's contribution is no longer
adequate. Rejuvenating it is cheaper than spawning a new cell and
re-learning its function from scratch.

**Trigger 3: Absorption-Driven.** When a graft (Section 5)
introduces inputs that significantly increase a dormant cell's
engagement (engagement EMA crosses $\theta_e$ during the first $W_a$
batches after the graft), the cell is rejuvenated. The intuition: the
graft has brought new context that re-activates a previously dormant
function; the cell should regain plasticity to re-fit the new context.

**Rejuvenation event.** The transition is atomic:

```python
def rejuvenate(arena, cell_id):
    assert arena.state[cell_id] == STATE_DORMANT
    arena.state[cell_id] = STATE_ACTIVE
    # gradient mask is recomputed from state[]; no explicit update needed
    record_event(REJUVENATE, cell_id, current_task)
```

The cell's weights, lineage, position, epigenome, and `output_dim`
are unchanged. Only the `state` flag flips. The optimizer's next
backward pass will include the cell in the gradient flow as usual.

**Cooling.** A just-rejuvenated cell is exempt from the credit-based
lock condition for the next $C_r$ task boundaries (default $C_r = 2$).
This prevents thrashing — a cell that was locked, rejuvenated, and
immediately re-locked would consume rejuvenation budget without
contributing learning.

### 4.5 Manifold Replay

Manifold replay is the v2.0 inheritor of the v1 mechanism: storage-
free pseudo-rehearsal that summarizes each past class as a small
diagonal-Gaussian sketch in a stable code space. The mechanism is
unchanged in spirit; the implementation moves into the cell substrate.

**Archive entries are astrocytes.** Each past class $c$ is
represented in the arena by one **manifold astrocyte** —
`forward_inclusion=false`, `weights` holding the per-dimension mean
and standard deviation of the class's code-space activations:

$$\text{astrocyte}_c.\text{weights} = \begin{bmatrix} \mu_c \\ \sigma_c \end{bmatrix} \in \R^{2 \times d_\text{code}}$$

Per-class storage is $2 \cdot d_\text{code} \cdot 4$ bytes for the
sketch plus ≈100 bytes of cell metadata. The default
$d_\text{code} = 128$ matches v1 (1024 bytes for the sketch, ≈1100
bytes including cell overhead); a 30-class curriculum costs $\approx
33$ KB. The slight overhead vs. v1's external dict (≈3% per class)
buys unified arena layout, visibility in snapshots and the renderer,
and uniform lifecycle handling — manifold astrocytes lock, rejuvenate,
and (rarely) recycle through the same machinery as every other cell.

**Lineage and position.** Each manifold astrocyte's `lineage_root` is
itself; its `parent` is the substrate's class-introduction event
(recorded as a synthetic spawn). Its `position` is on the $z=1$ face
of the unit cube near the output cell for its class, so the
visualizer places it as a small translucent disk adjacent to the
class head slot.

**Default state lifecycle for manifold astrocytes.** A manifold
astrocyte is born during a class's first task in the `active` state
(its $(\mu_c, \sigma_c)$ is being collected). At the dream cycle's
consolidate stage, the astrocyte transitions to `dormant` — the
sketch is frozen, replay continues to sample from it, but no further
collection updates the weights. Rejuvenation is rare and operator-
driven: typically only after a deliberate code-space shift that
invalidates the stored sketch (e.g., unfreezing a previously frozen
base).

**Stable code space.** The replay sketch requires a code space that
does not drift across the curriculum, otherwise statistics collected
at task $t$ are invalid at task $t+k$. v1 enforced this by freezing
the L0 random projection at initialization. v2.0 generalizes the
concept: any base loaded as a frozen base (Section 2.10) provides a
stable code space, and manifold astrocytes collect activations
*after* that frozen base. The architecture's invariant is *not* "L0
is frozen" — it is "the substrate exposes one stable code-space
boundary, downstream of which manifold astrocytes collect."

For the `minimal` base (perception cells only, no frozen interior),
the stable code space is the perception layer itself — manifold
astrocytes store statistics of the perception cells' outputs. This is
the v1 L0-frozen case as a special instance.

For composed bases (`bases.compose(...)`), the stable boundary is
the union of every frozen base's output cells. Manifold astrocytes
collect across the union; the sketch dimension is the concatenated
output width of the composed boundary.

**Replay.** During the dream cycle (Section 4.3, Stage 1), the
substrate iterates over manifold astrocytes for past classes and
draws pseudo-samples:

$$\tilde z_c = \mu_c + \sigma_c \odot \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, I)$$

The sampled $\tilde z_c$ is injected at the stable code-space
boundary — the substrate's downstream cells receive
$\tilde z_c$ in place of the boundary cells' real activations. The
forward continues through the active interior to the head, where
cross-entropy loss is computed against the class label $c$.

The L1-detached pattern from v1.1 carries over: during replay,
gradients on cells with the `credit_eligible` gene set but not yet
locked are zeroed for the head pass. This prevents replay from
disturbing cells that are close to consolidation. Head weights and
unfrozen interior weights still receive gradient.

**Why per-class statistics work.** The v1 paper documented the
falsification: DeepInversion-style logit inversion peaked at $0.27$
full-softmax on chained-15 because logit attractors locate
class-$c$ argmax peaks, not class-$c$ manifold samples. The
$(\mu, \sigma)$ summary lives on the real code-space manifold by
construction. v2.0 inherits this finding; we do not re-derive it.

**Why astrocytes for archive storage.** Three benefits unified
storage delivers beyond the v1 external dict:

1. **Visibility.** Manifold astrocytes appear in snapshots
   (Section 7), so the recorder timeline shows the archive growing
   as new classes are introduced. The viewer renders them at the
   $z=1$ face; collapsing them into a "manifold ring" reveals the
   archive's class-by-class accumulation at a glance.
2. **Lifecycle uniformity.** Locking, rejuvenation, and recycling
   apply to manifold astrocytes through the same code paths as any
   other astrocyte. No special-case bookkeeping in
   `learning/manifold.py`.
3. **Absorption compatibility.** When a donor substrate is loaded as
   a frozen base (Section 2.10), its manifold astrocytes come along
   automatically as part of the cell population. No separate archive
   serialization. Section 5's graft mechanism inherits this for free.

**API.** `learning/manifold.py` exposes:

- `spawn_manifold_astrocyte(substrate, class_id, code_boundary)` —
  creates the astrocyte during a class's first task
- `update_sketch(astrocyte, code_batch)` — accumulates the per-batch
  mean/variance into the astrocyte's weights (online algorithm; only
  while the astrocyte is active)
- `sample(astrocyte, batch_size)` — draws pseudo-samples for replay

The substrate's `dream_cycle()` orchestrates these calls; user code
typically never touches the manifold API directly.

### 4.6 EWC and Baselines

EWC and the other competitor families from v1 (Online EWC, LwF,
PackNet, HAT) live in `trioron/learning/baselines/` and are not
instantiated on the default training path. They are present for
benchmark comparison only.

**What the baselines provide.** Each baseline implements one
mechanism from the v1 competitor sweep (`paper/paper.tex` Table 3),
configured to drop into the v2.0 training loop in place of the
credit-based locking + manifold replay default. The harness for
comparison runs is at `experiments/baseline_sweep.py`; it iterates
over baselines and re-runs the same task curriculum for each.

**Memory savings on the default path.** With EWC removed from the
default, the substrate does not maintain:

- Anchor weight tensors $\anchorW, \anchorB$ (one per active cell:
  ≈ 4 MB at 1M params)
- Fisher accumulator $F_W$ (one per weight: another ≈ 4 MB)
- Online EWC EMA buffer ($\gamma F_{t-1}$, another ≈ 4 MB)

Total ≈ 12 MB freed on the default path. The credit-based locking
mechanism's additional state is one scalar per cell (`engagement`),
totaling 80 KB at 20K cells — three orders of magnitude smaller.

**Honest reconciliation with v1.** v1.1's credit + manifold result at
$n=3$ was $0.563$ full-softmax vs. v1.0's EWC + manifold at $n=10$ at
$0.601$. v2.0 carries credit-freezing as the default but the spec
commits to re-running at $n=10$ before publishing any v2.0
benchmark. If credit + manifold trails EWC + manifold on full-softmax
at matched seeds, the spec's response is a head-side calibration step
(the `settle_head_via_manifold` 200-step pass already used in v1.1
absorption), not a return to EWC.

### 4.7 Honest Limits

- **Lock thresholds are tuned, not derived.** $\theta_e = 0.3$,
  $K = 2$ consecutive tasks, $g_{\min} = 10^{-4}$ are inherited from
  v1.1 tuning. Different tasks may want different thresholds; the
  defaults are starting points, not theoretical optima.
- **Rejuvenation is heuristic.** The "high gradient pressure on a
  dormant cell's downstream chain" criterion (Trigger 2) is
  defined operationally, not derived from first principles. It may
  fire spuriously or fail to fire when needed. Operator-controlled
  rejuvenation (Trigger 1) is the fallback for cases where the
  automatic triggers misjudge.
- **Manifold replay assumes stable code space.** If the operator
  configures a non-frozen base (`bases.frozen(path, freeze=False)`),
  manifold archives collected before unfreezing become stale. The
  archive may be re-collected after a deliberate code-space shift,
  but this is an operator decision, not an automatic mechanism.
- **The dream-cycle caps are intentionally conservative.** The v1
  finding was that smaller per-cycle structural change preserved
  accuracy. v2.0 keeps that conservatism. Operators chasing faster
  convergence may relax the caps; we recommend doing so only after
  reading the v1.0 §3.5 ablation arms.

---

## 5. Lifecycle

This section specifies how the substrate grows, evolves, gets shipped,
and gets extended. The mechanisms compose: a cell is born by
**division**, mutates from its parent through **sister
differentiation**, may be locked or rejuvenated under selection
(Section 4), and the whole substrate may be **shipped** to deployment
and **woken** for further growth on new data. Multi-substrate
composition happens through **grafting** — a v2.0 generalization of
v1.1's pool-matched absorption.

### 5.1 Cellular Division (Growth)

A cell grows by **division**: one parent cell produces one or more
children, each inheriting state from the parent with optional
modifications. Growth is the primary structural plasticity mechanism;
it is gated by three signals.

**Growth trigger conjunction.** A division is attempted only when all
three of the following are satisfied over a window of $W_g$ steps
(default 100 batches):

1. **Frustration above threshold.** The frustration multiplier $m$
   (Section 4.2) has exceeded $\theta_m$ (default $2.0$) for at least
   $W_g / 2$ consecutive steps in the window — the substrate is
   stuck.
2. **Rank saturation at the candidate site.** The effective rank
   $\reff = \exp\!\left(-\sum_k p_k \log p_k\right)$ at some rank's
   active-cell activation matrix has approached the bucket's cell
   count (saturation gap below $\varepsilon_\text{rank}$, default
   $0.1$). This signals that adding capacity *at this rank* would
   actually be used.
3. **Gradient stability.** The median per-cell gradient magnitude at
   the candidate site is bounded $g_\text{min} \leq |\tilde g| \leq
   g_\text{max}$, indicating the substrate is in a region of the loss
   landscape where new capacity can settle.

The conjunction is v1's growth trigger ported to the graph substrate.
The difference: in v1 the candidate site was "L1" (the only plastic
layer); in v2 the candidate site is a `(rank, phenotype, output_dim)`
bucket. Multiple sites may satisfy the conjunction simultaneously;
the substrate orders attempts by frustration concentration and
processes them one per growth event.

**Envelope check.** Before division fires, the envelope's
`niche_capacity` (Section 2.5) is consulted. If capacity is positive,
division proceeds. If capacity is exhausted, division is paired with
a recycling event: a `recyclable`-gene dormant cell at the same or
adjacent rank is recycled to free its slot. If no recyclable cell is
available, the division is **denied** and a `growth_denied` event is
logged; the next dream cycle attempts to compact dormant slots.

**Division mechanics.** The default division is one-to-one (one
parent, one child):

```python
def divide(parent_cell, mutations=None):
    child = arena.allocate()
    child.parent = parent_cell.id
    child.lineage_root = parent_cell.lineage_root
    child.epigenome = parent_cell.epigenome
    child.position = parent_cell.position + small_jitter()
    child.output_dim = parent_cell.output_dim
    child.inputs = []                # wired in next step
    child.weights = init_from_residual(parent_cell, ...)
    child.state = STATE_ACTIVE
    apply_mutations(child, mutations)  # see §5.2
    return child
```

**Initial wiring.** A newly divided cell needs incoming edges. The
default wiring policy:

1. Inherit a fraction $f$ (default $0.5$) of the parent's incoming
   edges, with weights initialized to half the parent's value
   (function-preserving partial copy).
2. Establish $K_\text{new}$ new edges (default $K_\text{new} = 4$) to
   randomly chosen active cells at lower rank, with weights from the
   v1 growth-direction routine — top principal direction of the
   residual signal at the rank below.

Alternative wiring policies are pluggable through
`lifecycle/grow.py:wiring_policy`. The default is the v1-inspired
balance between inheritance and exploration.

**Multi-cell division.** Some phenotypes spawn multiple cells in one
event (Section 3.3's attention 4-cell lineage, Section 3.4's conv
astrocyte + N receptors). These are atomic operations: the helper
either allocates all required cells or fails as a unit. The envelope
check is performed against the total cost; partial spawns are
forbidden.

### 5.2 Evolution Strategies

Evolution in v2.0 is not a single mechanism but a family of
strategies, all opt-in except for selection. The substrate ships with
all mutation-based strategies **off by default** so that the core
learning machinery (Section 4) is exercised cleanly before
evolutionary pressure is layered on top. Once the substrate is
verified working with selection alone, the operator opts in to
additional strategies for specific experiments.

Three evolution modes are specified:

| mode | default | mechanism | when to enable |
|---|---|---|---|
| **Selection** | ON (always) | dream-cycle credit-lock + utility-recycle | inherent to the lifecycle |
| **Sister mutation** | OFF | random gene flips at division time | exploring phenotype diversity within one run |
| **Multi-substrate exploration** | OFF | parallel substrate variants, pick the best | architecture search at substrate granularity |

#### 5.2.1 Selection (default)

Selection is the population dynamics produced by the lifecycle itself
and is always active — it requires no operator configuration. Cells
that accumulate engagement get locked into dormant (their function
preserved). Cells with low engagement and low utility, marked
`recyclable`, are recycled. Cells with high utility but unstable
gradient stay active — they're under selection pressure but haven't
settled.

Selection alone is sufficient for the substrate to organize its
interior in response to task pressure. Sister cells with identical
epigenomes are still differentiated by the random initial wiring of
their incoming edges (Section 5.1) and by the gradient signal each
receives; over time some siblings earn credit and lock, others lose
utility and get recycled, and the substrate's interior reflects what
worked.

The recorder (Section 7) captures every lock and recycle event,
making selection-driven evolution observable on the timeline:
lineages diversify by survival rather than mutation.

#### 5.2.2 Sister Mutation (opt-in)

When the operator wants the substrate to explore phenotype diversity
within a single run, sister mutation is enabled at construction:

```python
substrate = trioron.construct(
    base=...,
    envelope=...,
    mutation_rates={
        "attention": 0.01,
        "conv": 0.01,
        "recurrent": 0.01,
        "dendrite": 0.01,
    },
)
```

When set, each multi-cell division applies per-gene flip
probabilities to non-canonical sisters (cells not spawned with a
specifically-required role like attention OUT or conv receptors):

$$P(\text{flip gene } g \text{ for sister } i) = p_g$$

The defaults when mutation is enabled are deliberately small (1–2%).
This produces gradual phenotype drift over many divisions, selected
on by the credit dynamics of 5.2.1. Mutation rates higher than ~5%
introduce too much noise: most sisters carry mutations that hurt
their engagement and get recycled, wasting envelope cells and dream-
cycle budget.

The default — mutation off — produces a substrate where every cell's
epigenome is inherited unchanged from its parent, and phenotype
diversity comes only from explicit operator gene-setting and from the
phenotype-specific spawn helpers. This is the configuration we
recommend for production deployment: predictable substrate behavior
with no hidden evolutionary drift.

#### 5.2.3 Multi-Substrate Exploration (opt-in)

The most expensive but most expressive evolution mode: spawn $N$
substrate variants from a common ancestor, evolve each independently
for $K$ tasks, score them on a target metric, and continue with the
winner. This is architecture search at substrate granularity.

```python
controller = trioron.evolution.MultiSubstrateController(
    ancestor=substrate,
    n_variants=4,
    variant_policies=[
        {"mutation_rates": {"attention": 0.05}},
        {"mutation_rates": {"conv": 0.05}},
        {"mutation_rates": {"recurrent": 0.05}},
        {"mutation_rates": {}},   # no-mutation control
    ],
    tasks_per_generation=3,
    score_fn=lambda s: s.eval_task_aware(test_loader),
)
winner = controller.run(curriculum)
```

Each generation forks $N$ copies of the current best substrate, runs
each on the next $K$ tasks with its assigned mutation policy, scores
them with `score_fn`, picks the winner, and discards the rest. The
winner becomes the ancestor for the next generation.

Compute cost is $N\times$ baseline training compute. Operators run
this mode when (a) the target task domain is novel enough that the
default phenotype distribution is suspect, (b) compute is available
for parallel runs, and (c) the score metric meaningfully discriminates
substrates.

The controller writes per-variant snapshots through the standard
recorder (Section 7), so the entire exploration is reproducible and
observable. The viewer can load all variants and overlay their
substrate states, making the diversification visible across the
generation.

**Honest scope.** Multi-substrate exploration is an orchestration
pattern, not a substrate mechanism. The substrate provides the
primitives (forking via ship, scoring via inference, swap via
construction); the controller is a thin layer that composes them.
Operators who want different selection rules (Pareto-front pick,
tournament selection, etc.) write their own controller subclass.

#### 5.2.4 Lineage as the unit of structural identity

Across all three evolution modes, lineage is the unit that persists.
Two cells in the same lineage carry the same `lineage_root`.
Lineages persist across compactions: if a child cell is recycled, its
parent's lineage tree loses one branch but the root and other
branches remain. If the root itself is recycled (rare, requires
`recyclable` to be set and an explicit operator decision because
roots are foundational), the entire lineage's descendants are
re-pointed to the next-deepest common ancestor.

The visualizer renders lineage trees as colored chain graphs — pick a
lineage in the side panel, the corresponding cells highlight across
the entire timeline. This is how "evolution" becomes observable in
all three modes: selection shows survivor lineages, sister mutation
shows phenotype-diversified branches, multi-substrate exploration
shows the winning variant's lineage tree against discarded ones.

### 5.3 Grafting (Multi-Substrate Composition)

Grafting is the v2.0 mechanism for composing independently-trained
substrates: one substrate is grown above another, with the lower
substrate optionally frozen as a base (Section 2.10). It generalizes
v1.1's pool-matched absorption from layer-granularity to cell-
granularity.

**Two grafting protocols** ship in `lifecycle/grow.py`:

**Protocol A: Frozen-base graft.** The donor substrate is loaded via
`bases.frozen(donor_path)`, producing an initial cell population
where every donor cell is `dormant` and `forward_inclusion=true` (the
donor stays in the forward pass; its weights are frozen). The
recipient's own perception and output cells are appended through the
base composition mechanism. New growth on the composed substrate
happens in the active interior above the donor's frozen cells.

This is the cleanest composition — no merging math, no parameter
collision, no class-namespace ambiguity. The donor's manifold
astrocytes (Section 4.5) come along automatically and continue to
serve replay for the donor's classes.

**Protocol B: Active-graft.** The donor substrate is loaded with
`freeze=False`. Donor cells start `active` and merge into the
recipient's lifecycle: they receive gradients, accumulate engagement,
may be locked or recycled. The recipient's existing cells and the
donor cells are wired together by a graft helper at construction
time:

```python
substrate = trioron.construct(
    base=trioron.bases.compose(
        existing_substrate,
        trioron.bases.frozen(donor_path, freeze=False),
        graft_edges="manifold_routed",  # default; see below for alternatives
    ),
    envelope=Envelope(),
)
```

The `graft_edges` parameter selects the wiring policy:

- `manifold_routed` (**default**) — for each donor cell, find
  recipient cells whose activation distribution overlaps the donor's
  manifold archive; wire those. The most selective policy; requires
  per-cell manifold histograms (computed once at graft time).
- `dense` — every recipient output-rank cell adds edges from every
  donor output-rank cell. Expensive but exhaustive; useful when the
  donor's manifold archive is absent or untrusted.
- `sparse` — random subset of cross-edges; bounded by a per-cell
  fan-in cap.
- `pool_matched` — cells in matching pools (lineage + position) are
  preferentially wired. The v1.1 absorption mechanism, ported.
  **Use only for strict absorption** — donor and recipient must share
  the same base, the same pool grid, and the same class space.
  Absorption is known to be fragile: v1.1's pool-matched absorption
  required tight donor/recipient compatibility and did not always
  recover task accuracy. Manifold-routed is the more general and more
  robust option for arbitrary cross-substrate composition.

The default is `manifold_routed` because it generalizes across the
broadest range of donor/recipient pairs and relies only on signals
the substrate already maintains (per-cell manifold histograms). When
the donor was produced from the same base as the recipient and the
operator specifically wants pool-matched absorption semantics from
v1.1, opt in with `graft_edges="pool_matched"`.

**Absorption-driven rejuvenation.** Grafting triggers the
absorption rejuvenation pathway (Section 4.4, Trigger 3). During
the $W_a$ batches after a graft (default $W_a = 50$), engagement
EMAs are computed on dormant donor cells. Those whose engagement
crosses $\theta_e$ are rejuvenated, on the theory that the new
recipient context has reactivated their function.

**Soft routing at the head.** When the donor and recipient cover
different class spaces, the composed substrate must reconcile their
heads. The v1.1 mechanism (per-branch log-softmax composition) is
retained; the composed head's output for class $c$ is:

$$\text{logit}_c = \log \sum_{b \in \text{branches covering } c} g_b \cdot e^{\text{branch}_b \text{logit for } c}$$

where $g_b$ is the per-branch gate from the donor's manifold
log-likelihood (Section 4.5). This is unchanged from v1.1 and
imported into v2.0 via `lifecycle/graft.py:compose_heads`.

### 5.4 The Ship-Wake-Extend Loop

Deployment in v2.0 is not a one-shot training run. The substrate is
**shipped** as a checkpoint, **woken** on new data in the field,
**extended** with new cellular growth, and re-shipped. The loop is
the same as v1's ship-wake-extend mechanism but operates on the cell
substrate.

**Ship.** Shipping serializes the entire arena to a checkpoint file:

1. All cells (active, dormant) with their full state (weights,
   epigenome, lineage, position, etc.)
2. All edges in CSR format
3. The envelope configuration (caps, current pressure)
4. The lineage tree (parent pointers, lineage roots)
5. All manifold astrocytes (their $(\mu, \sigma)$ weights are
   serialized as ordinary cell weights)
6. The current task index and frustration counter state

A shipping checkpoint is one file: `substrate_<task>.pt`. The format
is the PyTorch state dict augmented with a sidecar `meta.json`
recording envelope config and the schema version.

Before shipping, an optional **consolidation-dream pass** runs:

1. Full-coverage replay over all past tasks (longer than a normal
   dream cycle's replay)
2. One final consolidate pass — any cells eligible for locking are
   locked
3. Optional int8 quantization of dormant cells' weights (one
   per-cell scale factor; weights stored as int8). This is the v1
   Phase 2 dream archive ported to the cell substrate.

The consolidation-dream pass is opt-in via
`substrate.ship(consolidate=True)`. The default is to ship without
extra consolidation, preserving the substrate's exact training
state.

**Wake.** Waking deserializes a checkpoint into an arena. The
deserialized substrate is immediately runnable for inference; for
further training, the operator typically calls `substrate.wake_for_training()`
which:

1. Restores the optimizer state (Adam moments are stored alongside
   the cell state if shipped from mid-training).
2. Re-marks any cells that should be eligible for rejuvenation (by
   default, none — the operator opts in).
3. Resets the frustration counter and engagement EMAs for active
   cells (so the wake doesn't immediately fire growth from stale
   pre-ship statistics).

**Extend.** Extension is the operation that adds new task capacity to
a woken substrate. The substrate's envelope may be lifted (cap
increased) to accommodate growth; manifold astrocytes for the new
tasks are spawned during their first task; existing dormant cells
contribute to forward but do not anchor new tasks (unless rejuvenated
via the absorption-driven trigger).

Internal state — manifold astrocytes, lineage tree, locked cells —
all persist across the boundary. The substrate "sleeps long" before
shipping (long consolidation-dream pass), then wakes on new data and
extends with fresh active growth on top of the dormant foundation.

**Extension envelope.** When lifting the envelope:

```python
substrate.wake_for_training()
substrate.envelope.max_parameter_bytes = 8_000_000   # was 4 MB; now 8 MB
substrate.envelope.max_cells = None                  # was 20K; now uncapped
# ... train on new tasks; substrate grows freely within the new cap
```

The new cap takes effect at the next growth attempt; in-flight cells
are not affected. The substrate does not retroactively recycle to
shrink to a lower cap — operators must explicitly trigger a
compaction pass if they want to shrink.

### 5.5 Compaction

Compaction is the operation that sweeps the arena, recycles dormant
cells with the `recyclable` gene whose utility has been below floor
for $M$ consecutive dream cycles (default $M = 5$), and re-packs the
arena to remove gaps.

**Trigger.** Compaction runs:

1. **At the end of each dream cycle** (Section 4.3, Stage 4) — the
   normal path; processes a bounded number of recyclings per the
   recyclings cap.
2. **On explicit operator call** — `substrate.compact()` — for
   deployment-time aggressive compaction (e.g., before shipping).
3. **Under envelope pressure** — when `envelope.pressure > 0.9` and a
   growth attempt is pending, an unscheduled compaction runs to free
   slots.

**Saliency.** Recycling priority is driven by per-cell **saliency** —
a composite measure of how much the cell matters to the substrate's
current behavior. In v2.0, saliency is operationalized as:

$$\text{saliency}(c) = w_u \cdot \text{utility}(c) + w_e \cdot \text{engagement}(c) + w_d \cdot \text{downstream\_impact}(c)$$

with starting defaults $w_u = 0.6$, $w_e = 0.3$, $w_d = 0.1$. Utility
(running mean gradient magnitude) carries the most weight because it
directly measures the cell's contribution to learning. Engagement
(activation rate) is a secondary signal — a cell can fire often
without producing gradient if its outputs are already well-fit.
Downstream impact (the sum of saliency of cells that consume this
cell's output, computed once per compaction pass via reverse BFS)
folds in the cell's structural role beyond its immediate signal.

> **TODO (saliency weights).** The defaults
> $(w_u, w_e, w_d) = (0.6, 0.3, 0.1)$ are tuned by intuition, not by
> data. Revisit after the first benchmark runs land — calibrate
> against the recycling-decision outcomes (do recycled cells later
> turn out to have been important? do retained cells justify their
> slot?) and sweep on the chained-15 curriculum at Phase 1 scale.
> The weights are exposed as a substrate-construction parameter so
> operators can override; the defaults will be revised once the
> calibration sweep is complete.

For cells that have been dormant a long time, utility and engagement
decay (they receive no gradient and may not fire on current inputs);
saliency therefore naturally lowers for cells that have outlived
their function. Downstream impact protects cells whose outputs are
still consumed by salient downstream cells — recycling a cell whose
descendants depend on it would break their computation.

**Mechanics.**

```python
def compact(arena, max_recyclings, envelope_pressure):
    candidates = [c for c in arena.cells
                  if c.state == DORMANT
                  and GENE_RECYCLABLE in c.epigenome
                  and c.utility_below_floor_for(M)]
    # Sort: lowest saliency first; among cells tied at low saliency,
    # OLDER cells are recycled first — they had time to prove
    # themselves and didn't. Young cells get more time.
    candidates.sort(key=lambda c: (saliency(c), -c.age))
    recycled = candidates[:max_recyclings]
    for c in recycled:
        arena.mark_recycled(c.id)
        record_event(RECYCLE, c.id, current_task)
    # Defrag is gated by a pressure-adaptive threshold.
    if free_fraction > defrag_threshold(envelope_pressure):
        run_defrag_pass(arena)
```

**Saliency computation cost.** Utility and engagement are EMAs
already maintained per batch (Section 4.1); they cost nothing extra
to read. Downstream impact requires one reverse BFS over the active
cell graph per compaction pass; at 20K cells with 1M edges this is
~3 ms on commodity CPU. The full saliency computation for a
compaction pass costs ~5 ms total.

**Adaptive defrag threshold.** The fragmentation threshold that
triggers a defrag pass is loose when the envelope has room and tight
when it is nearly full:

$$\text{defrag\_threshold}(p) = \max\!\left(0.05,\ 0.5 \cdot (1 - p)\right)$$

where $p = \text{envelope.pressure} \in [0, 1]$.

| envelope pressure | defrag threshold | behavior |
|---|---|---|
| $p = 0.0$ (empty / uncapped) | $0.50$ | defrag only when half the arena is fragmented; almost never fires |
| $p = 0.5$ (half full) | $0.25$ | defrag at moderate fragmentation |
| $p = 0.85$ (high pressure) | $0.075$ | defrag aggressively to free slots for growth |
| $p \geq 0.9$ (saturated) | $0.05$ | defrag at any small free fraction; every cycle |

The intuition: at low pressure, defrag is wasted work — there are
plenty of fresh slots above the cursor for new growth, so reclaiming
gaps mid-arena doesn't help. At high pressure, every free slot
matters, so defrag earns its compute cost. When the envelope is
uncapped, `pressure` defaults to 0 and defrag effectively never
runs (operator can force `substrate.compact()` if needed).

**Cost.** Recycling itself is O(recycled cells). Defrag is
$O(|V| + |E|)$ — one BFS over the cell graph re-points all references.
At 20K cells and 1M edges, defrag is ~10 ms on commodity CPU. Defrag
runs at most once per task boundary; under low envelope pressure it
is substantially less frequent.

### 5.6 Honest Limits

- **Growth trigger is heuristic.** The frustration + rank-saturation
  + gradient-stability conjunction was tuned on v1 benchmarks; it
  carries no guarantee on out-of-distribution curricula. Operators
  may need to tune $\theta_m$, $\varepsilon_\text{rank}$, and the
  window sizes for new task domains.
- **Sister mutation is off by default.** The substrate ships with
  mutation rates all zero. Operators must explicitly enable mutation
  via `mutation_rates={...}` at construction (Section 5.2.2), or use
  multi-substrate exploration (Section 5.2.3) for a more deliberate
  search. The default keeps substrate behavior predictable while the
  operator verifies the core learning machinery works on their task.
- **Graft wiring is policy-driven, not learned.** The `graft_edges`
  parameter picks a wiring policy at construction time; the
  substrate does not learn the wiring through gradient descent. The
  default `manifold_routed` policy relies on the manifold archive
  being well-populated; for donors with sparse archives the operator
  should consider `dense` or `sparse`. `pool_matched` is the v1.1
  absorption protocol and is documented as fragile — use only when
  donor and recipient strictly share base and class space.
- **Compaction defrag is O(|V|+|E|).** At very large substrates (≥1M
  cells), defrag becomes a noticeable cost. The current spec defers
  this to an operator concern — at the targeted 20K–100K cell
  regime, defrag is fast.
- **Ship-wake-extend has been validated only at $n=1$ in v1.** The
  v1 paper noted the extension experiment was single-seed. v2.0
  inherits this caveat; before publishing v2.0 extension results, the
  experiment must be re-run at multi-seed.

### 5.7 Developmental Program (Stem Cells, Morphogens, Redifferentiation)

The v2.0 substrate is a cell-to-cell graph, but the construction
recipes (Section 2.10) build pre-wired layer-like structures. This
section specifies the **developmental program** — the mechanism by
which an undifferentiated cell population self-organizes into a
functional substrate. The developmental program replaces pre-wired
construction with signal-driven differentiation, making the
substrate's architecture an emergent property of its inputs.

#### 5.7.1 Stem Cells

A **stem cell** is a cell with no expression genes set (epigenome
bits 0–4 all zero). It occupies a position in the arena, has edges
(or none), and participates in the forward pass as a pass-through
(identity on its summed inputs). It is uncommitted — it has not
differentiated into a phenotype.

```python
STEM = -1  # sentinel; no expression gene set

def spawn_stem(arena, n, region):
    """Spawn n undifferentiated cells in the given region.

    region: (center, radius) in position space. Cells are
    scattered uniformly within the ball.
    """
```

Stem cells are the substrate's **progenitor pool**. They are spawned:

1. **At construction** — a `developmental` base spawns perception
   cells, output cells, and a pool of stem cells between them.
   Unlike `seeded`, no interior cells are pre-assigned a phenotype;
   no edges are pre-wired between perception and output.
2. **At graft boundaries** — when a donor is grafted onto a
   recipient (Section 5.3), stem cells are spawned in the boundary
   region between the two populations. They differentiate to bridge
   the donor and recipient, forming connective tissue.
3. **Under frustration** — when the frustration detector fires and
   the growth trigger condition is met (Section 5.1), new cells are
   spawned as stems, not as clones of the parent's phenotype.

#### 5.7.2 Morphogen Field (Adaptive)

A **morphogen field** is a smooth, learnable function over position
space that encodes regional identity. The field tells each cell
"where it is" in a developmental sense — not just its (x, y, z)
coordinate, but its role in the substrate's functional hierarchy.

The morphogen field is parameterized as a small learnable
projection from position to regional identity:

$$\phi(c) = \sigma(W_\phi \cdot \text{position}(c) + b_\phi)$$

where $W_\phi \in \mathbb{R}^{R \times 3}$ maps 3D position to
$R$ regional identity channels (default $R = 4$, one per candidate
phenotype), $b_\phi$ is a bias, and $\sigma$ is sigmoid. The
parameters $W_\phi, b_\phi$ are substrate-level parameters trained
alongside cell weights.

**Initialization.** $W_\phi$ is initialized so that the z-component
dominates: the starting morphogen approximates the static z-gradient
from Section 5.7.3, but the substrate can learn to reshape regional
boundaries as training progresses. For example, if a vision task
benefits from more CONV depth than the default z-threshold assigns,
the morphogen field shifts the CONV→LINEAR boundary deeper.

**Adaptation signal.** The morphogen is updated by the same
optimizer that trains cell weights. Its gradient comes from the
loss: if reassigning a cell's phenotype (via the morphogen shift)
would reduce loss, the morphogen shifts. This is automatic — no
explicit morphogen-training phase is needed.

**Cost.** $W_\phi$ has $4 \times 3 + 4 = 16$ parameters. The
morphogen evaluation is a single matrix multiply per cell, run once
per compile (not per batch). Negligible cost.

#### 5.7.3 Positional Differentiation (Phase B)

When a stem cell differentiates, its morphogen value $\phi(c)$
determines the default phenotype. The $R$-channel morphogen output
maps to phenotypes via argmax:

| Channel | Phenotype | Typical region |
|---|---|---|
| 0 | CONV | Near perception; local spatial features |
| 1 | LINEAR | Mid-depth; feature integration |
| 2 | ATTENTION | Deep; relational reasoning |
| 3 | LINEAR | Near output; final projection |

Differentiation is triggered by the cell's first non-zero gradient
signal. When a stem cell receives gradient (from the loss
propagating through its connections), it:

1. Evaluates the morphogen field: $\phi(c) \in \mathbb{R}^R$.
2. Selects phenotype $= \arg\max_r \phi_r(c)$.
3. Sets the corresponding expression gene in its epigenome.
4. Sets the CREDIT_ELIGIBLE gene.
5. If the selected phenotype is CONV, sets the WEIGHT_TIED_LINEAGE
   gene (Section 5.7.4a).
6. Refreshes its phenotype cache.
7. Is now committed — it participates in dispatch as its new
   phenotype on the next compile.

A stem cell that never receives gradient (disconnected, or in a
quiescent region) remains undifferentiated indefinitely. It costs
one bias parameter and zero compute (pass-through in dispatch).

#### 5.7.4 Axon Guidance (Proximity-Based Wiring)

Stem cells do not start with edges. Edges form through **axon
guidance** — a proximity-based wiring rule that runs after each
division or spawn event:

```python
def axon_guidance(arena, cell_id, max_edges=6, radius=0.3):
    """Wire cell_id to nearby cells based on position proximity.

    Edges are directed: cell_id receives input from cells at
    LOWER z (closer to perception) and sends output to cells at
    HIGHER z (closer to output).

    Within the radius, candidates are ranked by distance; the
    closest max_edges cells are wired.
    """
```

Axon guidance replaces the pre-wired fully-connected edges of the
`seeded` base. The resulting connectivity is:

- **Spatially local** — each cell connects to its neighbors, not
  to all cells in the substrate. This naturally produces
  convolutional-like receptive fields near the perception face.
- **Hierarchically directed** — edges flow from low-z to high-z,
  creating a feedforward hierarchy without explicit layer
  boundaries.
- **Sparse** — each cell has at most `max_edges` inputs, bounded
  by the fan-in cap. Total edge count grows as
  $O(n_\text{cells} \cdot \text{max\_edges})$, not
  $O(n^2)$.

Edge budget allocation:
- 80% of `max_edges` to lower-z inputs (feedforward)
- 20% to same-z lateral connections (see §5.7.4b)
- No backward edges (high-z to low-z) without the RECURRENT gene
  (Section 3.4)

#### 5.7.4a Lineage-Based Weight Sharing (CONV cells)

CONV cells from the same **lineage root** share edge weights. When
a CONV cell divides, its child inherits a pointer to the parent's
weight row, not a copy. All cells in the lineage apply the same
learned filter at their respective positions.

This is controlled by the `WEIGHT_TIED_LINEAGE` gene (epigenome
bit 9, already defined in §2.2). The mechanism:

1. When a CONV cell with WEIGHT_TIED_LINEAGE divides, the child's
   edge weights are **aliased** to the parent's weights. The arena
   stores one weight per lineage root; all descendants index into
   the same weight.
2. During backward, gradients from all cells in the lineage
   accumulate into the shared weight. This is mathematically
   equivalent to a convolution: the same filter applied at every
   position, trained from the union of all positions' gradients.
3. Non-CONV cells do NOT share weights even if they share a lineage
   root. Weight sharing is phenotype-gated.

**Why this explains optical illusions.** Shared weights mean the
same edge detector fires identically regardless of spatial position.
A Müller-Lyer arrow triggers the same line-end detector at every
location — the cell cannot distinguish "this is an arrow" from
"this is a corner" because it applies the same filter everywhere.
The illusion is not a bug; it is the architectural cost of
translation invariance. Weight sharing makes perception efficient
but brittle to adversarial spatial patterns.

**Implementation.** The arena gains a `weight_root` int32 tensor
(per-cell). For non-tied cells, `weight_root[c] = c`. For tied
cells, `weight_root[c] = lineage_root[c]`. The scheduler's
forward pass indexes `arena.edge_weight` via `weight_root` instead
of directly. The optimizer sees only the root's weight; tied
descendants contribute gradient but own no parameters.

#### 5.7.4b Lateral Signaling

Cells broadcast a **lateral signal** to neighbors within a
signaling radius $r_s$ (default $r_s = 0.15$). The lateral signal
carries three pieces of information:

1. **Position** — the cell's (x, y, z) coordinate.
2. **Phenotype** — what expression gene the cell has committed to
   (or STEM if uncommitted).
3. **Engagement** — how active the cell is (from the credit
   tracker's EMA).

Lateral signaling serves three functions:

**A. Differentiation coordination.** When a stem cell is about to
differentiate, it checks its lateral neighbors' phenotypes. If
$\geq 50\%$ of neighbors within $r_s$ already express the same
phenotype that the morphogen would assign, the cell may select an
under-represented alternative phenotype instead (the next-highest
morphogen channel). This prevents homogeneous clumps and encourages
phenotypic diversity in dense regions — analogous to lateral
inhibition (Notch signaling) in real neurogenesis.

**B. Shortest-path wiring.** During axon guidance (Section 5.7.4),
a cell knows the positions of all neighbors within $r_s$ via their
lateral broadcast. This allows guidance to prefer the shortest
physical path: among candidate input cells at lower z, pick the
nearest ones first. The result is that wiring efficiency tracks
physical proximity — long-range connections form only when local
connections are saturated.

**C. Density regulation.** The number of lateral signals a cell
receives is a proxy for local cell density. When a cell's neighbor
count exceeds $N_\text{dense}$ (default 12), it inhibits further
division in that region: the frustration-gated growth trigger
(Section 5.1) will not spawn new cells within $r_s$ of a dense
cell. This prevents overcrowding and distributes growth to sparse
regions — analogous to contact inhibition in tissue growth.

**Cost.** Lateral signaling is evaluated at compile time (not per
batch). Each cell scans neighbors within $r_s$ using the position
tensor — an $O(n^2)$ scan in the naive case, or $O(n \log n)$
with spatial indexing. At Phase 1 scale (≤2048 cells), the naive
scan is ~1 ms. At 100K cells, spatial indexing (k-d tree or grid
hash) is needed; this is an implementation concern, not a design
change.

#### 5.7.5 Signal-Driven Redifferentiation (Phase A Override)

Positional differentiation (Phase B) sets the default. **Signal-
driven redifferentiation** (Phase A) overrides it when the cell's
phenotype is a poor fit for the signal it receives.

The mechanism is inspired by plant hormone signaling: a cell that
is frustrated (receiving strong gradient but not reducing loss)
emits a **redifferentiation signal**. The signal is local — it
affects only the cell itself, not its neighbors.

**Trigger.** A committed cell becomes eligible for
redifferentiation when:

1. Its per-cell frustration (loss contribution not decreasing over
   $W_r$ batches, default $W_r = 200$) exceeds the
   redifferentiation threshold $\theta_r$ (default $\theta_r = 3.0$,
   calibrated to fire only under sustained poor fit).
2. It has been committed to its current phenotype for at least
   $T_\text{min}$ tasks (default $T_\text{min} = 2$) — no
   thrashing on first exposure.

**Process.** When triggered:

1. The cell's current expression gene is cleared.
2. The cell enters a **trial period** of $W_\text{trial}$ batches
   (default 100), during which it tries each candidate phenotype
   in sequence (one per $W_\text{trial} / |\text{candidates}|$
   batches).
3. Loss-under-each-phenotype is recorded. The cell commits to the
   phenotype with the lowest mean loss.
4. If no phenotype improves over the original, the cell reverts.

**Analogy.** This is the auxin response: a root cell that finds
itself at a graft junction receives wound signals, dedifferentiates,
then redifferentiates as shoot tissue. The cell doesn't "know" what
to become — it tries candidates and the gradient selects.

#### 5.7.6 The Developmental Base

A new base `developmental` replaces `seeded` for the cell-first
developmental path:

```python
trioron.bases.developmental(input_dim, initial_classes,
                            stem_pool=64)
```

Construction sequence:

1. Allocate `input_dim` perception cells at z=0, scattered along
   the y-axis.
2. Allocate `initial_classes` output cells at z=1.
3. Allocate `stem_pool` stem cells uniformly in the interior
   (z ∈ [0.1, 0.9]).
4. Initialize the adaptive morphogen field $W_\phi, b_\phi$ with
   z-dominant defaults.
5. Run axon guidance on every cell — this produces the initial
   sparse, proximity-based wiring.
6. Do NOT set any expression genes on stem cells. They
   differentiate on first gradient.

The first forward pass propagates input through the stem cells
(pass-through). The first backward pass triggers differentiation.
By the end of the first training batch, the substrate has a
committed architecture — but one that emerged from position and
signal, not from pre-wiring.

#### 5.7.7 Interaction with Existing Mechanisms

| Mechanism | Interaction |
|---|---|
| **Frustration (§4.2)** | Fires on the substrate level as before. For stem cells, frustration triggers spawn (not division of existing phenotyped cells). |
| **Division (§5.1)** | Daughter of a committed cell inherits the parent's phenotype. Daughter of a CONV cell with WEIGHT_TIED_LINEAGE shares the parent's weights. Daughter of a stem cell is also a stem cell. |
| **Credit locking (§4.1)** | Stem cells are not credit-eligible (no CREDIT_ELIGIBLE gene by default). They acquire it upon differentiation. |
| **Manifold replay (§4.5)** | Unchanged. Manifold astrocytes are phenotype-independent; they store code-space statistics regardless of whether the producing cells are CONV, ATTENTION, or LINEAR. |
| **Astrocyte-gated replay** | The manifold log-likelihood evaluation is independent of cell phenotype. Gated replay works identically whether the substrate was built by `seeded` or `developmental`. |
| **Grafting (§5.3)** | Stem cells spawned at the graft boundary differentiate based on the donor's signal profile. The adaptive morphogen field extends continuously across the boundary. |
| **Compaction (§5.5)** | Undifferentiated stem cells that remain uncommitted for $M_\text{stem}$ tasks (default 5) are recyclable — they had the chance to differentiate and didn't. |
| **Redifferentiation (§5.7.5)** | Only committed cells redifferentiate. Stem cells differentiate (first commitment), which is a one-way transition on a different code path. |
| **Lateral signaling (§5.7.4b)** | Coordinates differentiation, guides wiring, regulates density. Evaluated at compile time, not per batch. |

#### 5.7.8 Honest Limits

- **Phenotype implementations required.** This section assumes
  CONV, ATTENTION, RECURRENT, and DENDRITE phenotypes are
  implemented in `phenotype/`. Currently only LINEAR is
  implemented. The developmental program degrades gracefully: if
  only LINEAR exists, all stem cells differentiate to LINEAR
  regardless of morphogen value, and the substrate is equivalent
  to a dynamically-wired dense network.
- **Weight sharing is CONV-only.** LINEAR, ATTENTION, and other
  phenotypes do not share weights across lineage. This is by design:
  weight sharing is the defining property of convolution (translation
  invariance). Non-convolutional cells need position-specific
  weights to learn position-dependent features.
- **Lateral signaling is $O(n^2)$ naive.** At Phase 1 scale
  (≤2048 cells), the naive position scan is ~1 ms. At 100K+
  cells, spatial indexing is required — k-d tree or grid hash.
  This is an implementation optimization, not a design change.
- **Adaptive morphogen can overfit.** With only 16 parameters,
  the morphogen is unlikely to overfit in practice. But if the
  task is very small (few batches), the morphogen may not receive
  enough gradient to adapt meaningfully. The z-dominant
  initialization ensures it works even without adaptation.
- **Redifferentiation is expensive.** The trial period pauses
  useful learning for that cell. The threshold $\theta_r$ must be
  high enough that redifferentiation fires rarely — it is a last
  resort, not a routine operation.
- **No inter-cell chemical signaling.** Lateral signaling
  broadcasts position/phenotype/engagement but does not carry
  arbitrary learned messages. True chemical signaling (where cells
  emit and receive learned signal molecules) would enable richer
  coordination but adds significant complexity. Deferred.

---

## 6. Performance Contract

Performance is a first-class design constraint, not a "later"
concern. The substrate is a cell-to-cell graph; naive
per-cell-Python-loop execution would turn a 1 ms matmul into a 1000 ms
graph traversal. This section declares hard bench targets and the
hot-path commitments that produce them.

The contract applies to commodity CPU (no GPU dependency, no SIMD-
specific tuning) and to FP32 weights by default. GPU and BF16 paths
are optional accelerations; the contract holds without them.

### 6.1 Bench Targets at Scale

**Phase 1 (current, binding): development cap at 50K parameters.**
The substrate's primary bench contract is the development scale.
Architecture work, profiling, and CI gating all run at 50K. The
substrate is not authorized for runs beyond this cap during Phase 1 —
not because larger substrates can't function, but because we
prioritize verifying the small-scale path works cleanly before
scaling up. This protects against the trap of over-engineering for a
scale we haven't earned.

**Phase 2 (stretch, non-binding): deployment target at 1M
parameters.** The deploy column is the aspirational stretch target
for v2.x. It is not a release gate for v2.0; numbers here are
informational, derived from the dev-scale profile extrapolated under
the hot-path commitments of Section 6.3. v2.0 graduates to Phase 2
when the dev-scale path is stable across all benchmarks for at least
two release cycles.

| operation | **dev (50K params)** *binding* | deploy (1M params) *stretch* | hard ceiling (dev) |
|---|---|---|---|
| forward pass, batch 32 | < 3 ms | < 30 ms | < 10 ms |
| backward + opt step | < 8 ms | < 80 ms | < 25 ms |
| growth event (single division) | < 0.1 ms | < 1 ms | < 0.5 ms |
| growth event (atomic 4-cell spawn) | < 0.5 ms | < 2 ms | < 2 ms |
| dream cycle (full replay over 30 classes) | < 0.5 s | < 5 s | < 1.5 s |
| rank recompute (after edits) | < 0.5 ms | < 5 ms | < 2 ms |
| compaction (no defrag) | < 1 ms | < 10 ms | < 5 ms |
| defrag pass (full) | < 1 ms | < 10 ms | < 5 ms |
| snapshot JSON dump | < 10 ms | < 100 ms | < 50 ms |
| HTML viewer first paint | < 0.2 s | < 1 s | < 1 s |

The hard ceiling applies to dev — Phase 1 binding contract. Above
ceiling at dev = a release-blocking regression. The deploy column
has no enforcement until Phase 2 begins.

**Why 50K cap during Phase 1.** A 50K-parameter substrate is large
enough to test every mechanism (growth, dream cycle, manifold
astrocytes, grafting, all five phenotypes) and small enough that
iteration is fast — a full task takes seconds, not minutes, so
benchmark runs land in CI under a minute and developer-facing
profiling is interactive. Locking the upper bound at 50K also
prevents the substrate from being optimized for a scale it cannot yet
defend at multi-seed reproducibility.

### 6.2 Memory Budget

Memory budgets are reported at the Phase 1 binding scale (50K params)
and the Phase 2 stretch scale (1M params). Both assume average fan-in
50 and default `output_dim=1`; block-cell-heavy substrates trade cell
count for per-cell density at the same parameter total.

**Phase 1 (binding) — 50K parameters / ~1K cells:**

| component | size |
|---|---|
| `weights` (FP32, sparse CSR) | 200 KB |
| Adam moments (m, v at FP32) | 400 KB |
| edge indices (int32) | 200 KB |
| per-cell metadata | 100 KB |
| manifold astrocytes (30 classes × 1024 B) | 30 KB |
| optimizer scratch / activation cache | 200–500 KB |
| **total resident** | **~1.5 MB** |

**Phase 2 (stretch) — 1M parameters / ~20K cells:**

| component | size |
|---|---|
| `weights` (FP32, sparse CSR) | 4.0 MB |
| Adam moments (m, v at FP32) | 8.0 MB |
| edge indices (int32) | 4.0 MB |
| per-cell metadata | 2.0 MB |
| manifold astrocytes (100 classes × 1024 B) | 0.1 MB |
| optimizer scratch / activation cache | 2–6 MB |
| **total resident** | **~25 MB** |

EWC on the default path is absent (Section 4.6); the substrate does
not maintain anchor weights, Fisher accumulators, or Online EWC EMA
buffers. The memory freed against an EWC baseline is ~600 KB at
Phase 1 scale and ~12 MB at Phase 2 scale.

For substrates with block cells (`output_dim > 1`), per-cell weight
storage scales as $\text{fan\_in} \cdot \text{output\_dim} \cdot 4$
bytes. Total memory still tracks the parameter count, not the cell
count, because block cells are denser per cell.

### 6.3 Hot-Path Commitments

The bench targets are achievable only because of specific design
commitments that constrain the hot path. These commitments are
load-bearing for the contract; violations require either fixing the
violation or relaxing the contract.

**Commitment 1: Structure-of-Arrays storage.** All per-cell state is
held in parallel tensors in the arena (Section 2.4). Per-cell access
in the forward path is `arena.weights[cell_id]`, not
`cell.weights`. The `Cell` dataclass is a view-only facade for the
user-facing API; the forward path never instantiates a Python `Cell`
object.

**Commitment 2: Rank-batched dispatch.** The scheduler buckets cells
by `(rank, phenotype, output_dim)` (Section 2.7) and dispatches one
batched op per bucket. No bucket contains a Python loop over cells;
every dispatch is one PyTorch op over the entire bucket. The
scheduler is benchmarked against a hand-built reference network at
matched parameters; the substrate must come within 3× of the
reference's wall-clock at 1M parameters.

**Commitment 3: Append-only growth.** Cellular division writes to the
next free arena slot via a cursor (Section 2.4). No realloc, no
tensor reshape during growth. Compaction (Section 5.5) defragments
the arena periodically when fragmentation crosses the adaptive
threshold; in steady state defrag is rare.

**Commitment 4: Deferred rank recomputation.** Topological rank is
recomputed only at task boundaries, not after each cellular division.
Within a task, newly grown cells inherit `rank = parent.rank + 1`
provisionally; the BFS-based recomputation runs once during the dream
cycle to reconcile. This trades exactness for amortized cheapness:
ranks may be approximate within a task but are correct at every
task boundary.

**Commitment 5: Gradient mask, not requires_grad toggling.** Dormant
cells receive zero gradient via one mask tensor op
(Section 4.1), not by setting `requires_grad=False` on per-cell
parameters. PyTorch's per-parameter `requires_grad` involves Python
attribute walks; the mask is one element-wise multiply on the gradient
tensor.

**Commitment 6: No Python in the scheduler's per-batch hot loop.**
The scheduler's `forward` implementation has zero Python-level cell
iteration. Buckets are precomputed at task boundaries; dispatch
within a forward is a series of torch ops with no per-cell branching.
Profile the scheduler — every cycle in the forward must be a torch
op, not a Python statement.

**Commitment 7: CSR edges with block-sparse matmul.** Edges live in
CSR format with row-pointer arrays. The linear-phenotype forward op
is a `torch.sparse.mm` or, when the bucket is dense enough, a gather
followed by a single `torch.bmm`. Operators may swap between sparse
and dense within a bucket based on a density threshold (default 30%
fan-in saturation).

**Commitment 8: Snapshot serialization is incremental.** The
recorder (Section 7) writes per-cycle deltas where possible, not full
arena dumps. A full snapshot is taken once per task; intra-task
snapshots only record cells whose state changed. This keeps snapshot
cost under the bench target even with frequent snapshots enabled.

### 6.4 Profiling and Benchmarking

The substrate ships with a built-in profiler that records per-op
wall-clock and call-count statistics:

```python
with substrate.profile() as p:
    substrate.train_one_task(task_loader)
    substrate.dream_cycle()
print(p.summary())   # per-op wall-clock, call counts, mean/median/p95
```

The profiler is mode-toggled at substrate construction (default off)
and adds < 1% overhead when enabled because it uses
`torch.cuda.synchronize`-free timing on CPU.

Continuous benchmarking is part of CI: `bench/run_contract.py`
exercises the bench-target operations at both dev and deploy scales
and fails the build if any operation exceeds the soft target. The
hard ceiling triggers a release blocker, not just a CI warning.

**Reference benchmarks ship at:**

- `bench/bench_forward_pass.py` — forward at batch 32 across cell
  counts from 100 to 100K
- `bench/bench_growth.py` — growth event latency under varying
  envelope pressure
- `bench/bench_dream_cycle.py` — dream cycle wall-clock as a
  function of class count
- `bench/bench_viz_export.py` — snapshot JSON dump + viewer first
  paint at varying cell counts

Each bench writes a CSV and a comparison HTML plot; the CI gate is
the CSV against the bench-target table of Section 6.1.

### 6.5 Honest Limits

- **Phase 1 cap is a discipline, not a hardware limit.** The 50K-
  parameter ceiling during Phase 1 is enforced at the API surface —
  `substrate.construct(...)` raises if any envelope cap or runtime
  growth would push past it. The substrate is technically capable of
  larger runs; the cap exists so we earn each scale before claiming
  it.
- **Targets assume commodity CPU (x86_64 with AVX2 or equivalent).**
  Numbers will differ on ARM, on older CPUs without AVX2, and in
  WSL-like environments where syscall latency is higher. Phase 1
  reference hardware: a modern x86_64 laptop CPU at single-thread.
- **The 1M-parameter target is a Phase 2 stretch.** Training beyond
  1M parameters is supported by the architecture but is not
  bench-targeted in either phase. Operators training at 10M+
  parameters should expect the forward pass to scale linearly in
  parameter count and super-linearly in cell count.
- **GPU is optional, not contractual.** The substrate runs on GPU
  if `torch.device("cuda")` is set, but the contract is written
  against CPU. GPU-specific optimizations (CUDA graph capture,
  fused kernels) are deferred to v2.x.
- **The 3× hand-built-reference target** is a rough commitment. The
  reference is a dense MLP at matched parameter count; this is a
  conservative comparison because the substrate's sparse and
  heterogeneous structure prevents it from being as cache-friendly as
  a dense matmul. v2.0 aims for 3× at Phase 1 scale; closing the gap
  further with phenotype-specific kernels is Phase 2 work.

---

## 7. Visualization

Visualization is a first-class architectural concern (Section 1.5):
if the substrate produces a transformer-like cortex or a conv-like
filter bank but we cannot see it, the claim is unverifiable. This
section specifies the snapshot-record-then-replay pipeline, the
structural-fingerprint detector, and the HTML viewer.

The pipeline has three stages, each in its own module under
`trioron/viz/`:

```
training       recorder.py    →   snapshots/         (live, cheap)
                                  ├─ t000.json
                                  ├─ t001.json
                                  └─ ...
post-run       detect.py      →   structures.json    (offline)
viz/export     export.py      →   network.html       (self-contained)
browser        viewer.js      ←   user opens HTML
```

The recorder runs during training. The detector runs after the run.
The exporter produces a single self-contained HTML file that opens in
any modern browser without a server.

### 7.1 The Recorder

The recorder captures snapshots of the substrate state at significant
events. It runs in-process during training and writes JSON files to
the run's snapshot directory.

**Snapshot triggers.** Three event types fire snapshots by default:

| trigger | when | snapshot kind |
|---|---|---|
| `on_task_boundary` | start and end of each task | full snapshot |
| `on_dream_consolidate` | after dream cycle's consolidate stage | full snapshot |
| `on_growth_event` | each cellular division (sampled, see below) | delta snapshot |

Triggers are configurable through the recorder constructor:

```python
recorder = trioron.viz.Recorder(
    snapshot_dir="runs/exp001/snapshots/",
    triggers={
        "on_task_boundary": True,
        "on_dream_consolidate": True,
        "on_growth_event": "full",  # Phase 1 default; see below
        "on_recycle_event": False,
    },
    sample_growth_every_n=1,
)
substrate.attach_recorder(recorder)
```

**Growth-event sampling.** Cellular division can fire hundreds of
times per task; the snapshot policy must trade fidelity against
storage cost.

- **Phase 1 default: full fidelity** (`sample_growth_every_n=1`).
  Every growth event writes a delta snapshot. The 50K-parameter cap
  bounds total event count to the low thousands per run; snapshot
  storage stays in the tens of MB. Full fidelity is the right default
  while we are still verifying the substrate works correctly — we
  cannot debug a substrate whose growth history is gappy.
- **Phase 2 recommended: sampled** (`sample_growth_every_n=10` or
  higher). At 1M parameters, growth events run into the high
  thousands per task; full fidelity produces hundreds of MB of
  snapshots per run. Sampling keeps storage tractable at the cost of
  losing per-event resolution between samples.

The operator can override at any time. The default ships at 1
(full fidelity) and will be revisited when Phase 2 begins.

**Full snapshot format.** A full snapshot is a JSON object:

```json
{
  "schema_version": "trioron-v2.0",
  "task_index": 5,
  "timestamp_iso": "2026-05-24T10:30:15Z",
  "envelope": {"max_cells": null, "max_edges": null, ...},
  "cells": [
    {"id": 0, "state": "active", "role": "neuron",
     "position": [0.12, 0.5, 0.0], "epigenome": 33,
     "phenotype": "linear", "output_dim": 1,
     "lineage_root": 0, "parent": -1, "rank": 0,
     "engagement": 0.42, "utility": 0.018, "age": 1532},
    ...
  ],
  "edges": [{"src": 0, "dst": 100, "weight": 0.3}, ...],
  "manifold_astrocytes": [
    {"id": 9000, "class_id": 0, "mu": [...], "sigma": [...]},
    ...
  ],
  "events_since_last_snapshot": [
    {"type": "lock", "cell_id": 42, "task": 5},
    {"type": "growth", "parent_id": 100, "child_id": 105, "task": 5},
    ...
  ]
}
```

**Delta snapshot format.** Same schema, but the `cells` and `edges`
arrays contain only entries that changed since the previous full
snapshot. Removed cells are listed in a separate `removed_cells`
array. Delta snapshots cost ~5–10% of a full snapshot when growth
events are sparse.

**Performance contract.** The recorder's snapshot write cost is bounded
by the targets in Section 6.1: at Phase 1 scale, a full snapshot
dumps in under 10 ms; at Phase 2 scale, under 100 ms. Snapshots are
written asynchronously to a background thread so the substrate's
training loop never blocks on disk I/O. The recorder maintains a
bounded write queue; if the queue fills (slow disk), the recorder
warns and drops the lowest-priority snapshot (oldest growth-event
delta).

### 7.2 The Detector

The detector runs offline after a training run. It scans each
snapshot, identifies cells that form recognized structural
fingerprints, and emits a `structures.json` file consumed by the
viewer's collapse/expand UI.

**Template-based detection.** The detector ships with a fixed set of
templates, each a function that scores a candidate cell group:

| template | detection signal | viewer render |
|---|---|---|
| **cortex sheet** | cells co-positioned on a 2D manifold (PCA eigenvalue₃ ≪ ₁,₂), shared phenotype, shared lineage root | translucent flat ellipsoid + thickness |
| **column / tube** | PCA eigenvalue₁ ≫ ₂,₃, ordered by rank | translucent capsule |
| **cluster / nucleus** | dense isotropic blob, same phenotype + same lineage root | translucent sphere |
| **attention head** | 4-cell lineage with Q/K/V/OUT roles | 4-cell glyph with directed arrows |
| **conv filter bank** | lineage with astrocyte root + receptor cells spread across a regular grid | translucent grid + astrocyte glyph at centroid |
| **recurrent loop** | back-edges from higher to lower rank | curved arrow on the loop |
| **dormant region** | dormant cells co-positioned in space | dimmed, frozen color |
| **manifold ring** | manifold astrocytes for $\geq 4$ classes | translucent ring at $z=1$ face |

Each template emits a structure record:

```json
{
  "id": "cortex-A-task5",
  "type": "cortex_sheet",
  "task_first_seen": 5,
  "cell_ids": [120, 121, 122, ...],
  "hull_geometry": {"type": "ellipsoid", "center": [0.5, 0.3, 0.4],
                    "axes": [[...], [...], [...]], "extents": [0.2, 0.18, 0.02]},
  "label": "cortex sheet (linear, lineage 100, 87 cells)",
  "confidence": 0.93
}
```

**Anomaly mode.** Cells that match no template are placed in an
`ungrouped` bucket. The detector clusters ungrouped cells by spatial
proximity + lineage similarity and emits "anomaly" structures for any
cluster with $\geq 8$ cells. These render as dashed outlines in the
viewer; they are flagged as candidates for new templates. This is
how we discover patterns we didn't anticipate.

**Cross-snapshot tracking.** The detector tracks structures across
snapshots by lineage root: a cortex sheet at task 5 with lineage root
100 is the same structure as a cortex sheet at task 6 with lineage
root 100, even if cell count changes. The viewer uses this to draw
continuous structure trajectories on the timeline.

**Detection cost.** Detector runs in Python with NumPy; per-snapshot
cost is dominated by the PCA and clustering steps. At Phase 1 scale
(~1K cells per snapshot, ~100 snapshots per run), the detector
completes in under 5 seconds. At Phase 2 scale (~20K cells, ~500
snapshots), under 2 minutes. Detection is single-shot per run —
re-running is only needed if templates are updated.

**Honest scope.** The detector finds what its templates look for.
Calling a region a "cortex" is a label about geometry + lineage, not
a functional claim. Novel emergent structures show up as anomalies
and require an operator to look at them and decide whether they merit
a new template.

### 7.3 The Viewer

The viewer is a single self-contained HTML file that opens in any
modern browser. It loads the snapshot directory's JSON files plus the
`structures.json` from the detector and renders the substrate's
state over time.

**Tech stack:**

- Three.js loaded from CDN (no npm, no build step)
- `InstancedMesh` for cell glyphs (one draw call for all spheres /
  diamonds / glyphs)
- `LineSegments` for edges
- `dat.gui` for slider/toggle controls
- Plain HTML for the side panel (cell inspector, structure list,
  timeline)

The exporter (`trioron/viz/export.py`) embeds the JSON snapshots
directly into the HTML file as base64-encoded blobs, so the result
is genuinely self-contained — no asset directory, no server required.
File size at Phase 1 scale: ~500 KB. At Phase 2 scale: ~5–15 MB.

**Render layout.**

- **3D scene** dominates the viewport — the $[0,1]^3$ envelope cube
  with cells as glyphs
- **Right side panel** — selected cell details (id, parent, lineage,
  phenotype, engagement, utility, age, state), structure list (toggle
  visibility per structure)
- **Bottom timeline** — task-aligned scrubber, with markers for
  growth/lock/recycle event clusters
- **Top toolbar** — preset filters (show only attention / show only
  this lineage / collapse all structures / hide dormant)

**Cell glyphs.**

| role | glyph | meaning |
|---|---|---|
| active neuron | opaque sphere | colored by phenotype |
| dormant neuron | dim sphere | grayed but visible |
| active astrocyte | opaque diamond | parameter-holder, no compute |
| dormant astrocyte | dim diamond | locked parameter-holder |
| manifold astrocyte | translucent disk | per-class manifold sketch at $z=1$ |

**Phenotype color key:**

- linear: gray
- attention: blue
- conv: red
- recurrent: green
- dendrite: purple

Cells with composite expression (multiple genes) render with a
pie-segmented sphere showing the gene mix.

**Edges.**

- forward edges (lower rank → higher rank): solid lines, color from
  source cell
- back-edges (recurrent): dashed curved lines, green
- lineage edges (parent → child): faint dotted lines, only shown
  when a lineage is selected
- shared-parameter edges (astrocyte → receptors): orange dotted lines,
  only shown when an astrocyte is selected

**Collapse/expand structures.** Clicking a structure hull in the side
panel toggles between rendering its cells as individual glyphs vs.
rendering the structure as a single hull (translucent shell + label).
Default: all detected structures collapsed; loose cells (ungrouped or
non-matching) rendered individually. Operators can switch to
all-expanded for detailed inspection.

**Timeline scrub.** The bottom timeline shows snapshots as ticks.
Dragging the scrubber loads the corresponding snapshot's state into
the scene. Cells that were locked between two snapshots flash briefly
on transition. The scrubber supports keyboard arrows for
frame-by-frame stepping.

**Filter sliders.**

- **Max cells shown** — caps the rendered cell count at a user-set
  value (default 5,000; presets 1K/5K/20K/all). Above the cap, the
  viewer shows top-N cells by saliency.
- **Min saliency** — hides cells below the saliency threshold
- **Phenotype filter chips** — toggles per phenotype
- **Lineage filter** — show only cells in selected lineage(s)
- **State filter** — toggle active/dormant visibility

The cap is the primary tool for keeping the renderer responsive at
Phase 2 scale. At Phase 1 the substrate is small enough that the cap
is rarely engaged.

### 7.4 Anatomy of an Observability Run

A complete observability run from training to inspection:

```
# 1. Train with recorder attached
substrate = trioron.construct(...)
recorder = trioron.viz.Recorder("runs/exp001/snapshots/")
substrate.attach_recorder(recorder)
substrate.train(curriculum)

# 2. Run detector offline
python -m trioron.viz.detect runs/exp001/snapshots/ \
    --output runs/exp001/structures.json

# 3. Export to self-contained HTML
python -m trioron.viz.export runs/exp001/ \
    --output runs/exp001/network.html

# 4. Open in browser (no server)
xdg-open runs/exp001/network.html
```

The result: a single HTML file the operator can scrub through to see
the substrate's full developmental history, with detected structures
highlighted and click-to-inspect for any cell.

### 7.5 Honest Limits

- **Detector templates are heuristics.** Geometric fingerprints
  (sheets, columns, blobs) are fast and reliable. Functional
  fingerprints (attention head, conv filter bank) require the
  epigenome + lineage to be present and correctly tagged. Calling
  a region a "cortex" is a label about geometry, not a function
  claim.
- **Recorder cost compounds at sub-task granularity.** Snapshot
  every growth event at Phase 2 scale produces ~10–100 snapshots
  per task, ~MB each — easily 100 MB of snapshots per task. The
  default sampling (`sample_growth_every_n=10`) trades fidelity for
  size; operators chasing rare events can tighten the sample.
- **Viewer is read-only.** No edit-the-substrate-from-the-browser
  feature. Substrate changes happen in the training loop; the
  viewer is for inspection.
- **Browser performance ceiling.** Above ~50K rendered cells in one
  frame, even with `InstancedMesh`, frame rates drop below 30 fps
  on mid-range laptops. The cap slider is the answer; if operators
  need to see all cells in a >50K substrate at once, they should
  pre-collapse via the detector or render at a lower zoom level.
- **JSON format is verbose.** A snapshot at Phase 2 scale is several
  MB before gzip. Future versions may switch to a binary format
  (msgpack, Arrow) once the visualization workflow stabilizes.

---

## 8. Backwards Compatibility

v1 checkpoints, v1 API entry points, and v1 test suites all continue
to work under v2.0. The compatibility commitment is:

1. **Donor loading**: any v1.x checkpoint loads via
   `trioron.bases.from_v1_checkpoint(path)` and produces a runnable
   substrate without manual conversion.
2. **API stability**: v1 callers of the top-level `trioron.api`
   module continue to work — the v1 entry points are preserved as
   shims that translate to v2 calls.
3. **Test parity**: every v1 test in `tests/test_*.py` (excluding
   tests that exercise specifically-removed internals) passes
   unchanged.

These are commitments for the v2.0 release. Future major versions may
deprecate the compat path; v2.x will not.

### 8.1 The Compat Layer

The compat layer lives in `trioron/compat/` and is the bridge between
v1's layer-based architecture and v2's cell graph:

```
trioron/compat/
├── __init__.py
├── load_v1.py           # v1 checkpoint → v2 cell population
├── layer_view.py        # synthesize v1-style layer view from cell ranks
├── api_shim.py          # v1 api entry points → v2 calls
└── test_parity.py       # runs the v1 test suite against v2 substrate
```

Each module is single-purpose and has no upward dependencies; the
core, phenotype, learning, lifecycle, and viz packages are unaware
of the compat layer.

### 8.2 v1 Checkpoint Loading

v1 checkpoints were produced by `TrioronLayer` aggregates with L0,
L1, and head submodules. The loader translates this into the v2 cell
substrate via the following correspondence:

| v1 concept | v2 equivalent |
|---|---|
| `L0` frozen Gaussian projection | astrocyte holding the random matrix + perception cells reading it |
| `L1` plastic layer of $N$ nodes | $N$ neuron cells at rank 1, each with the L1 row as `weights` |
| Head per-class column | output cell per class, fan-in from L1 cells |
| Per-node $\lambda$ (Fisher stiffness) | not preserved (v2 uses engagement, not Fisher) |
| Per-node $u$ (utility) | carried over to v2 `utility` field |
| Per-class manifold sketch | one manifold astrocyte per class |
| `dream archive` (locked rows) | dormant cells |
| `multi-branch` wrapper | composed base (Section 2.10) |

**Loader API:**

```python
substrate = trioron.construct(
    base=trioron.bases.from_v1_checkpoint(
        "v1_donor.pt",
        freeze=True,            # default: load as frozen base
        translate_lineages=True, # reconstruct lineage from L1 row indices
    ),
    envelope=Envelope(),
)
```

**`freeze=True` (default)** loads every translated cell as dormant.
The v1 donor's function is preserved exactly; new v2 growth happens
on top. This is the v2.0 path for the v1.1 multi-branch absorption
use case.

**`freeze=False`** loads cells as active, allowing the woken
substrate to continue training the donor's interior. Useful for
resuming a v1 training run mid-curriculum.

**Lineage translation.** v1 had no lineage tree. The loader
synthesizes a flat lineage: every translated cell gets its own
lineage root (itself), except for cells that v1 marked as "spawned
from L1 row $i$" — those inherit row $i$'s id as their lineage root.
Operators who need richer lineage reconstruction can post-process
with `compat.load_v1.assign_lineages_by_phenotype()`.

**Fisher loss.** v1's per-node $\lambda$ (Fisher-derived stiffness)
does not translate to v2's engagement scalar. The loader does not
attempt to convert; cells loaded from v1 start with engagement = 0
and accumulate fresh statistics. If the v1 donor is loaded as a
frozen base, this is irrelevant (locked cells don't update
engagement). If loaded as active, the first few task boundaries
re-establish engagement organically.

### 8.3 v1 API Shim

The v1 top-level API (`trioron.api`) exposed functions like
`build_donor`, `chain_train`, `extend`, `serve_http`, etc. The shim
preserves these signatures and routes them to v2 implementations:

```python
# v1 caller code, unchanged
import trioron.api as api

donor = api.build_donor(
    input_dim=784,
    initial_classes=10,
    cap_bytes=32_000,
)
api.chain_train(donor, curriculum, epochs_per_task=4)
api.extend(donor, new_curriculum, extension_cap=64_000)
```

Under the hood, each shim call constructs a v2 substrate, invokes
the corresponding v2 method, and returns a wrapper object whose
attributes mirror v1's expected return shape. The wrapper is
sufficient for the v1 test suite; new code should call the v2 API
directly through `trioron.construct(...)`, `substrate.train(...)`,
`substrate.ship(...)`.

**Deprecation timeline.** Shim functions emit a `DeprecationWarning`
when called from non-test code. The warning includes a pointer to
the equivalent v2 call. The shim stays in place through v2.x and
may be removed in v3.0.

### 8.4 Layer View

Some v1 callers expected `donor.l0`, `donor.l1`, `donor.head`
attributes — direct access to the layer aggregates. The
`compat/layer_view.py` module synthesizes these on demand from the
v2 cell substrate:

```python
class LayerView:
    @property
    def l0(self):
        # synthesize: cells at rank 0 with the perception gene
        return SynthesizedL0(substrate, [c for c in substrate.cells
                                          if c.rank == 0])

    @property
    def l1(self):
        # synthesize: cells at rank 1 with linear phenotype
        return SynthesizedL1(substrate, [c for c in substrate.cells
                                          if c.rank == 1 and ...])
```

The synthesized objects support read access (`l1.weights`, `l1.lambdas`,
`l1.utility`) by gathering from the arena's parallel tensors. Write
access raises — modifying a synthesized layer would require
reconciling against the cell-level state, and the compat path does
not support write-back. Callers that need to mutate state must use the
v2 API.

### 8.5 v1 Test Suite Parity

The v1 test suite lives in `tests/test_*.py` and exercises v1
behavior. Under v2, the suite runs in two modes:

1. **Compat mode** (default for the parity gate) — tests run against
   the v1 API shim, which routes calls to v2 implementations. Any
   test failure here is a regression in the compat layer.
2. **Native mode** (opt-in via `pytest --native`) — tests that have
   been ported to call v2 directly are run against the v2 API. The
   ported test set grows over time; un-ported tests fall back to
   compat mode automatically.

The parity gate is a CI requirement: every PR runs the full v1 test
suite in compat mode, and any new failure blocks the merge. This
keeps the compat path honest as v2 evolves.

### 8.6 What Is Not Preserved

A small set of v1 internals is explicitly not carried into v2:

- **Per-weight Fisher accumulators.** v2 does not maintain
  per-weight importance estimates. EWC competitor runs (`learning/
  baselines/`) instantiate Fisher on demand for their own use; the
  default path does not.
- **The 6-axis API surface from v1.1.** Axes 1–6 were each their
  own write function in v1.1 (`set_input_source`, `archive_input`,
  `insert_layer`, `set_axonal_gain`, `grow_branch`, `axis6_spawn`).
  These collapse into epigenome bits in v2 (Section 2.2). Operators
  who specifically need the v1.1 axis API call them through the
  shim, which translates each into the corresponding gene-setting
  call.
- **Profile presets `open` / `classification` / `edge` /
  `reasoning`.** v1.1's profile system is replaced by v2's
  construction-time configuration. The shim provides equivalent
  presets through `trioron.api.profile_classification()` etc., which
  return a `Base + Envelope` pair pre-configured to v1.1 semantics.
- **`MultiBranchOrganism` wrapper.** Replaced by composed bases
  (Section 2.10). The shim's `api.build_multibranch(...)` returns a
  substrate constructed via `bases.compose(...)`.

### 8.7 Honest Limits

- **Compat is one-way.** v1 checkpoints load into v2; v2 checkpoints
  do **not** export back to v1. The v2 cell graph carries state
  (epigenome, lineage, position, dormant astrocytes) that has no v1
  representation. Operators who need both v1 and v2 deployment
  artifacts must train under v1 and migrate forward.
- **Fisher loss may degrade v1.0 donors.** v1.0's EWC path relied on
  per-weight Fisher; losing it means a v1.0 donor loaded as
  `freeze=False` may drift more than under v1's own EWC anchoring.
  v1.1 donors (which used credit-based freezing) translate
  losslessly.
- **Test parity is functional, not behavioral.** v1 tests check that
  output shapes, return types, and exception semantics match. They
  do not check exact numerical equivalence between v1 and v2 forward
  passes — v2's scheduler may dispatch in a different order, and
  finite-precision arithmetic is order-sensitive.
- **API shim adds dispatch overhead.** v1 calls through the shim
  pay one extra Python-level indirection. Negligible for most calls;
  measurable for very-hot calls (e.g., per-batch forward). The shim
  is for compat, not performance; new code should use v2 directly.

---

## 9. Partition Map

This section is the final, authoritative directory layout for the v2.0
package. Each file's responsibility is listed alongside its
cross-reference to the spec section that defines its contract. Files
listed here are the v2 ship layout; v1's modules continue to exist
under `trioron/legacy/` (not shown in detail; preserved unchanged).

### 9.1 Top-Level Layout

```
trioron/
├── __init__.py                # public surface (re-exports for v1 callers)
├── api.py                     # v2 public API (replaces v1's 1446-line api.py)
│
├── core/                      # the new substrate — one concept per file
├── phenotype/                 # how genes express into operations
├── bases/                     # modular construction recipes
├── learning/                  # credit, frustration, dream, manifold
├── lifecycle/                 # growth, evolution, ship, graft, compact
├── viz/                       # recorder, detector, viewer, exporter
├── evolution/                 # multi-substrate controller (opt-in)
├── compat/                    # v1 ↔ v2 bridge
└── legacy/                    # v1 modules, untouched
```

### 9.2 `core/` — The Substrate

| file | responsibility | spec section |
|---|---|---|
| `cell.py` | `Cell` dataclass (user-facing view), per-cell field schemas | 2.1 |
| `epigenome.py` | 16-bit gene bitmask, gene constants, expression rules | 2.2 |
| `graph.py` | `CellGraph` — nodes + edges, invariants (no dangling, no phenotype/cycle violation, no envelope overflow) | 2.3 |
| `arena.py` | Structure-of-Arrays storage, append-only cursor, dead_mask, compaction sweep | 2.4 |
| `envelope.py` | Envelope dataclass, pressure / niche_capacity / stagnation signals, cap enforcement | 2.5 |
| `lineage.py` | Lineage tree, parent_of, lineage_root_of, siblings_of, cross-compaction preservation | 2.6 |
| `scheduler.py` | Topological-rank batched dispatch, phenotype dispatch table, forward orchestration | 2.7 |
| `state.py` | Cell state transitions (active ↔ dormant, rejuvenation), gradient mask construction | 2.8 |
| `construct.py` | `trioron.construct(base, envelope)` entry point | 2.9 |
| `roles.py` | Neuron vs. astrocyte role definitions, forward_inclusion handling | 2.11 |

### 9.3 `phenotype/` — Genes to Operations

| file | responsibility | spec section |
|---|---|---|
| `__init__.py` | Phenotype registration, dispatch table | 3.1 |
| `linear.py` | Default linear-combine phenotype | 3.2 |
| `attention.py` | Q/K/V/OUT lineage, scaled-dot-product, multi-head as N lineages | 3.3 |
| `conv.py` | Sensor receptors + astrocyte kernel, batched grouped conv | 3.4 |
| `recurrent.py` | Self-edges, fixed-depth unroll (cap 8) | 3.5 |
| `dendrite.py` | Branch partition, quad nonlinearity, soma pooling | 3.6 |
| `composite.py` | Multi-gene composition rules, additive accumulation pass | 3.7 |

### 9.4 `bases/` — Modular Construction Recipes

| file | responsibility | spec section |
|---|---|---|
| `__init__.py` | Base protocol + catalogue registration | 2.10 |
| `minimal.py` | Empty interior, perception + output only | 2.10 |
| `seeded.py` | Empty interior + N random active interior cells | 2.10 |
| `frozen.py` | Load + freeze any prior substrate as ground architecture | 2.10 |
| `compose.py` | Concatenate multiple bases into one substrate | 2.10 |
| `random_projection.py` | v1-style frozen Gaussian L0 reproduction | 2.10 |
| `from_v1_checkpoint.py` | v1.x donor loader (delegates to compat/load_v1) | 2.10 / 8.2 |

### 9.5 `learning/` — Anchor, Stress, Sleep, Replay

| file | responsibility | spec section |
|---|---|---|
| `__init__.py` | Top-level learning hooks for the training loop | 4 |
| `credit.py` | Engagement tracking, lock condition, active → dormant transition | 4.1 |
| `frustration.py` | Plateau detector, multiplier formula, downstream-consumer signals | 4.2 |
| `dream.py` | Four-stage dream cycle orchestration, scaled/dynamic caps | 4.3 |
| `rejuvenate.py` | Three rejuvenation triggers, dormant → active transition, cooling | 4.4 |
| `manifold.py` | Manifold astrocyte spawn, online sketch update, sample for replay | 4.5 |
| `baselines/` | Subpackage — competitor families for benchmarking only | 4.6 |
| `baselines/ewc.py` | Per-weight Fisher + EWC penalty | 4.6 |
| `baselines/online_ewc.py` | Fisher EMA across tasks | 4.6 |
| `baselines/lwf.py` | Logit distillation against anchored network | 4.6 |
| `baselines/packnet.py` | Iterative magnitude pruning + per-task masks | 4.6 |
| `baselines/hat.py` | Hard binary attention masks per task | 4.6 |

### 9.6 `lifecycle/` — Growth, Evolution, Deployment

| file | responsibility | spec section |
|---|---|---|
| `__init__.py` | Top-level lifecycle hooks | 5 |
| `grow.py` | Cellular division, growth trigger conjunction, wiring policies | 5.1 |
| `evolve.py` | Sister mutation rules (opt-in), gene flip probabilities | 5.2.2 |
| `graft.py` | Multi-substrate composition, four wiring policies (default `manifold_routed`) | 5.3 |
| `ship.py` | Substrate serialization, optional consolidation-dream pass | 5.4 |
| `wake.py` | Substrate deserialization, wake_for_training | 5.4 |
| `extend.py` | Envelope lifting, growth on top of frozen foundation | 5.4 |
| `compact.py` | Recycling pass, saliency-based sort, adaptive defrag threshold | 5.5 |
| `saliency.py` | `saliency(cell)` composite metric + downstream-impact BFS | 5.5 |

### 9.7 `viz/` — Observable Growth

| file | responsibility | spec section |
|---|---|---|
| `__init__.py` | `Recorder`, `Detector`, `Exporter` re-exports | 7 |
| `recorder.py` | Snapshot triggers, full + delta snapshot writers, async queue | 7.1 |
| `snapshot.py` | JSON schema, full/delta serialization | 7.1 |
| `detect.py` | Template-based fingerprint detector, anomaly mode, cross-snapshot tracking | 7.2 |
| `templates/` | One file per template (cortex_sheet, column, cluster, attention_head, conv_bank, recurrent_loop, dormant_region, manifold_ring) | 7.2 |
| `export.py` | CLI: snapshot dir → standalone HTML | 7.3 |
| `render/viewer.html` | Three.js scaffolding, dat.gui controls, side panel | 7.3 |
| `render/viewer.js` | Scene setup, instanced rendering, timeline scrubbing | 7.3 |
| `render/style.css` | Side panel + toolbar styling | 7.3 |

### 9.8 `evolution/` — Multi-Substrate Exploration

| file | responsibility | spec section |
|---|---|---|
| `__init__.py` | Exports `MultiSubstrateController` | 5.2.3 |
| `controller.py` | Fork-N, train-K, score, pick-winner generation loop | 5.2.3 |
| `score.py` | Default score functions (task-aware accuracy, full-softmax) | 5.2.3 |
| `policies.py` | Per-variant mutation/wiring policies | 5.2.3 |

### 9.9 `compat/` — v1 Bridge

| file | responsibility | spec section |
|---|---|---|
| `__init__.py` | Public compat surface | 8 |
| `load_v1.py` | v1 checkpoint → v2 cell population | 8.2 |
| `layer_view.py` | Synthesize L0/L1/head views from cell ranks | 8.4 |
| `api_shim.py` | v1 api entry points → v2 calls | 8.3 |
| `test_parity.py` | Run v1 test suite against v2 substrate | 8.5 |

### 9.10 `legacy/` — v1 Modules

Existing v1 modules move here unchanged via `git mv`:

```
trioron/legacy/
├── node.py · network.py · dreaming.py · api.py
├── manifold.py · multibranch.py · pruner.py · spatial.py
├── classification.py · curriculum.py · profile.py
├── frustration.py · growth_direction.py · triggers.py
├── ceilings.py · incubator.py · cli.py · serve_http.py
├── hat.py · packnet.py
├── senses/ · composition/ · bridge/
```

The legacy package is not modified by v2 development. Compat code
imports from `legacy/` when it needs the original v1 implementation
to drive a test or a fallback. Operators may also import from
`legacy/` directly (deprecated but supported through v2.x) for
unported workflows.

### 9.11 Tests

```
tests/
├── test_v2/                  # new v2 tests, one file per concept
│   ├── test_cell.py
│   ├── test_epigenome.py
│   ├── test_graph.py
│   ├── test_arena.py
│   ├── test_envelope.py
│   ├── test_lineage.py
│   ├── test_scheduler.py
│   ├── test_construct.py
│   ├── test_phenotype_linear.py
│   ├── test_phenotype_attention.py
│   ├── test_phenotype_conv.py
│   ├── test_phenotype_recurrent.py
│   ├── test_phenotype_dendrite.py
│   ├── test_bases.py
│   ├── test_credit.py
│   ├── test_frustration.py
│   ├── test_dream.py
│   ├── test_rejuvenate.py
│   ├── test_manifold.py
│   ├── test_grow.py
│   ├── test_evolve.py
│   ├── test_graft.py
│   ├── test_ship_wake_extend.py
│   ├── test_compact.py
│   ├── test_saliency.py
│   ├── test_recorder.py
│   ├── test_detect.py
│   ├── test_export.py
│   └── test_evolution_controller.py
│
├── test_v1*.py               # existing v1 tests, unchanged
└── conftest.py               # shared fixtures, compat mode toggle
```

The v2 test directory is structured one file per concept, matching
the partition. Each test file's first import is from the
corresponding module under `trioron/<package>/`; a missing
counterpart in the test directory means the spec is missing test
coverage for that concept.

### 9.12 Benches

```
bench/
├── run_contract.py            # CI gate: exercises all bench targets
├── bench_forward_pass.py      # forward at batch 32, cell counts 100 → 100K
├── bench_growth.py            # growth event latency vs. envelope pressure
├── bench_dream_cycle.py       # dream cycle wall-clock vs. class count
├── bench_viz_export.py        # snapshot + viewer paint vs. cell count
└── reference/
    ├── dense_mlp.py           # hand-built MLP at matched params (3× target reference)
    └── README.md              # how to add a new reference network
```

### 9.13 Documentation

```
paper/v3/
├── spec.md                    # this document — authoritative architecture spec
├── paper.tex                  # v2.0 paper (not yet drafted)
└── refs.bib                   # citations (shared with v1 papers)

docs/
├── quickstart.md              # 5-minute hello-substrate tutorial
├── migration_from_v1.md       # for users coming from v1.x
├── observability.md           # using the recorder, detector, viewer
└── phenotype_authoring.md     # how to add a new phenotype
```

### 9.14 Cross-Section Index

For navigating the spec, the following table maps each major concept
to its defining section and primary implementation file:

| concept | spec § | primary file |
|---|---|---|
| Cell | 2.1 | `core/cell.py` |
| Epigenome | 2.2 | `core/epigenome.py` |
| Cell graph | 2.3 | `core/graph.py` |
| Arena (SoA storage) | 2.4 | `core/arena.py` |
| Envelope (uncapped default) | 2.5 | `core/envelope.py` |
| Lineage tree | 2.6 | `core/lineage.py` |
| Scheduler (rank-batched) | 2.7 | `core/scheduler.py` |
| State transitions | 2.8 | `core/state.py` |
| Construction | 2.9 | `core/construct.py` |
| Bases | 2.10 | `bases/` |
| Neurons vs. astrocytes | 2.11 | `core/roles.py` |
| Phenotype contract | 3.1 | `phenotype/__init__.py` |
| Linear | 3.2 | `phenotype/linear.py` |
| Attention | 3.3 | `phenotype/attention.py` |
| Conv | 3.4 | `phenotype/conv.py` |
| Recurrent | 3.5 | `phenotype/recurrent.py` |
| Dendrite | 3.6 | `phenotype/dendrite.py` |
| Credit-based locking | 4.1 | `learning/credit.py` |
| Frustration | 4.2 | `learning/frustration.py` |
| Dream cycle | 4.3 | `learning/dream.py` |
| Rejuvenation | 4.4 | `learning/rejuvenate.py` |
| Manifold replay | 4.5 | `learning/manifold.py` |
| EWC + baselines | 4.6 | `learning/baselines/` |
| Cellular division | 5.1 | `lifecycle/grow.py` |
| Selection (default evolution) | 5.2.1 | `learning/credit.py` + `lifecycle/compact.py` |
| Sister mutation (opt-in) | 5.2.2 | `lifecycle/evolve.py` |
| Multi-substrate exploration (opt-in) | 5.2.3 | `evolution/controller.py` |
| Grafting | 5.3 | `lifecycle/graft.py` |
| Ship-wake-extend | 5.4 | `lifecycle/ship.py` + `wake.py` + `extend.py` |
| Compaction + saliency | 5.5 | `lifecycle/compact.py` + `saliency.py` |
| Performance contract | 6 | `bench/run_contract.py` |
| Recorder | 7.1 | `viz/recorder.py` |
| Detector | 7.2 | `viz/detect.py` |
| Viewer | 7.3 | `viz/render/` |
| v1 compat | 8 | `compat/` |

---

## End of Spec

This document is the authoritative reference for Trioron v2.0
architecture, Phase 1. Changes to any commitment listed here require
a spec amendment with version bump. Implementation work proceeds
against this spec; the partition map (Section 9) is the scaffold
plan.

**Next steps in the v2.0 build:**

1. Scaffold the directory tree with stub modules + docstrings linking
   back to their spec sections.
2. Move v1 files into `legacy/` (pure `git mv`, no edits) and verify
   all v1 tests still pass.
3. Implement `core/` first (`Cell`, `CellGraph`, `Scheduler`,
   `Envelope`, `Arena`), then `phenotype/linear.py` to get the
   smallest possible runnable forward pass.
4. Layer in `learning/credit.py` and `learning/manifold.py` to enable
   the first training loop.
5. Wire in `lifecycle/grow.py` and `learning/dream.py` for the first
   growth + consolidation experiment.
6. Add `viz/recorder.py` and `viz/render/` so the first growth is
   observable from the start.
7. Add `bench/run_contract.py` to the CI and gate the Phase 1
   targets.
8. Subsequent phenotypes, lifecycle features, evolution controller,
   and compat layer in dependency order.

Each step has its own commit and PR; spec changes are pre-requisites,
not afterthoughts.

