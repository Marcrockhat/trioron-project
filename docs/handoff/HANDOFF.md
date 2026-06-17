# Trioron Handoff

**Session date:** 2026-06-17
**Session number:** 041
**Session title:** **Conv DEPTH delivered: the conv→pool→conv stack §8 deferred
is built, and it is the FIRST positive CONV result.** On shifted-MNIST (a 28×28
digit dropped at a random offset on a 36×36 canvas, so the object *moves*), the
two-layer conjoined-twin conv hierarchy reaches **0.842** vs single-layer conv
**0.498** vs raw logreg **0.493** — a **+0.344** depth lift. A fixed
column-shuffle control collapses conv2 to **0.417** (locality gap −0.426) while
permutation-invariant logreg is unchanged: the win is **geometry, not capacity**.
This closes the s040 honest gap ("DEPTH untested") — single-layer conv only
*ties* the tabular learner; depth is what flips it. An experiment session; no
package code touched.

---

## READ THIS FIRST

1. **EXPERIMENT session. No package code changed.** New file:
   `experiments/progenitor/conv_depth_shifted_mnist.py`. Design §8 extended
   with the depth result. The conv→pool→conv hierarchy is VALIDATED as a
   mechanism + a positive, but NOT promoted into `trioron/`.
2. **What the depth stack IS.** Both conv arms use the real
   `conv.forward_batch` weight-tie (sharing by `lineage_root`+tap). L1 = C1
   conjoined-twin cohorts over the input (s040 form). The NEW piece is **L2 = a
   2nd conjoined-twin cohort that tiles over the 2×2-pooled L1 feature map**,
   each L2 channel reading ACROSS all C1 L1-channels in its K2×K2 patch in
   matched tap order (real cross-channel 2nd-layer convolution, one shared
   kernel per channel). Pooling/ReLU between layers is the consumer's job — §8
   predicted "the consumer must pool too," and that held.
3. **Local `spawn_cohort` (in the experiment), NOT the shared
   `spawn_conv_cohort`.** The shared spawner (`conv_proposer.py`) hardcodes the
   root-kernel size to `k²`, which breaks for L2's non-square `C1·K2²` fan-in.
   The experiment's `spawn_cohort` sizes the kernel from `len(patch[0])`, so it
   handles both single-channel K×K (L1) and cross-channel C·K2² (L2). If this
   gets promoted, generalize the shared spawner the same way.
4. **REGIME CAVEAT carries from s040: this is GRADIENT-BASED (Adam on
   kernels).** The chained-15 headline 0.736/0.446 is the GRADIENT-FREE PCLL
   organism — a different pipeline. Gradient-free conv (lock-in / Hebbian
   kernel) is still unsolved and gates any drop-in to the matched-filter
   pipeline.
5. **DO-NOT-COMMIT carries (unchanged since s034, LEAVE THEM):**
   `trioron/bases/developmental.py`, `trioron/lifecycle/developmental.py`,
   `trioron/viz/export.py`; `.claude/`, `runs/` untracked. Two stalled s039 raw
   logs still untracked. This session's new files SHOULD be committed (the
   experiment script + design §8 update + the run log + this handoff).

---

## What was built

**`experiments/progenitor/conv_depth_shifted_mnist.py`** — the conv→pool→conv
depth stack on shifted-MNIST, three arms (raw-logreg / conv1 single-layer /
conv2 depth), with a fixed column-shuffle locality control. Env-tunable
(`CANVAS/OFFSET/C1/C2/K1/S1/K2/S2/EPOCHS/N_TRAIN/N_TEST/SEEDS/SHUFFLE`).
Helpers: `_shift_onto_canvas` (the moving-object task), `tile_patches_mc` (the
multichannel L2 patch tiler in channel-major layout), `spawn_cohort` (the
fan-in-sized conjoined-twin spawner), `build_l1`/`build_l2`/`l1_map`/`l2_feats`
(the two-arena chained forward). Gradients verified to reach BOTH L1 and L2
kernels end-to-end (through the maxpool + cross-channel tie).

**`docs/design/progenitor_council.md` §8** — extended with the "Depth result +
positive (s041)" block: the task, the architecture, the table, the three
findings, and the honest scope.

## The numbers (shifted-MNIST, 10-way, n=2 seeds, Adam)

C1=8 K1=5 s2 / pool 2×2 / C2=12 K2=3 s1, 8 epochs. Log:
`outputs/conv_depth_shifted_mnist_s041_run1.log`.

| arm | REAL | SHUFFLED |
|---|---|---|
| raw-logreg | 0.493 | 0.488 |
| conv1 (1-layer) | 0.498 | 0.261 |
| **conv2 (DEPTH)** | **0.842** | 0.417 |

- **DEPTH lift** conv2 − conv1 = **+0.344**; conv2 − logreg = **+0.350**.
- **LOCALITY** conv2 REAL − SHUF = **+0.426** → the advantage is spatial
  wiring, not parameter count (logreg, permutation-invariant, is flat under
  shuffle: 0.493→0.488).
- Single-layer conv merely *ties* logreg (0.498≈0.493) — neither follows the
  moving object. DEPTH is the lever. This is the positive that single-layer
  could never show on centered chained-15; it confirms DEPTH (not channel
  count) was the untested lever flagged in s040.

## NEXT (priority)

1. **(Optional) close the chained-15 depth gap too.** s040's honest gap was
   "single-layer conv doesn't beat logreg on CENTERED chained-15, depth
   untested." Now that the depth stack exists, run it on centered chained-15
   (the `conv_proposer_chained15.py` data) to settle whether depth helps there
   (likely still loses — centered, perception-saturated — but it would make the
   claim airtight). The shifted-MNIST positive already covers the "conv CAN
   help" requirement, so this is rigor, not blocker.
2. **Gradient-free conv (the real open problem).** All conv numbers are
   Adam-on-kernels. The PCLL organism is gradient-free. Whether a tied conv
   kernel can be learned without backprop (lock-in / Hebbian / matched-filter
   correlation) is unsolved and gates any drop-in to the chained-15 pipeline.
3. **Package home & promotion (only after a regime decision).** The three
   primitives (coordinate field, proposer, conjoined-twin cohort) + the depth
   stack are validated. Promotion target is §7's lateral-growth event (mint a
   conjoined CONV cohort when a spatial trial-vote wins) or the §3 council
   scaffold. Generalize the shared `spawn_conv_cohort` to size the kernel from
   the patch (see READ-THIS-FIRST #3) before promoting.
4. **Vote weighing accuracy-per-param** (s040 open question) — under the 50K
   envelope, conv's accuracy-at-low-params is attractive; the current vote uses
   raw accuracy. Still a pending design decision.

## OPEN / unresolved

- **Gradient-free conv** — see NEXT#2; the gating unknown for the PCLL
  pipeline.
- **Centered chained-15 + depth** — NEXT#1; the last piece of "conv doesn't
  help chained-15."
- s039 carries (untouched this session): plain-gabor discrepancy
  (0.712/0.385 vs s038 0.690/0.304); chained-15 over-segmentation (cap=128);
  the generative mean-template readout (−0.32) — the s039 keystone NEXTs.
- All conv numbers are gradient-based, n=2 seeds.

## State of the build

- Branch `progenitor-council`. Package code UNTOUCHED; gate battery green
  (from s036). New files to commit this session: the experiment script + design
  §8 update + the run log + this handoff. DO-NOT-COMMIT carries left alone.
- Run cost: smoke seconds; full n=2 with shuffle control 154s (CPU,
  OMP_NUM_THREADS=8).

## Pointers

- **Design:** `docs/design/progenitor_council.md` §8 (conjoined-twin rule +
  s040 single-layer validation + s041 depth result); §3.3–3.4 (council /
  trial-vote); §3.6 (positional sensors = the coordinate field); §7 (branch-id
  sheet — the promotion target).
- **Conv primitive:** `trioron/phenotype/conv.py` (weight-tying by
  `lineage_root`+tap; the LINEAR-reduction guarantee, lines 15–20).
- **Experiments:** `experiments/progenitor/conv_depth_shifted_mnist.py` (s041,
  DEPTH); `test_conjoined_conv.py` (s040 mechanism); `conv_proposer.py` (s040
  trial-vote, synthetic); `conv_proposer_chained15.py` (s040 real sensor
  field, single-layer).
- **Data:** plain MNIST via `DatasetBundle(["mnist"]).task_view("mnist",
  range(10), range(10), split=…)` → 784-d [0,1] with labels 0–9. chained-15:
  `chained_15_specs` + `build_task_views`. Root `outputs/data/`.
- **Memory:** `conv_depth_shifted_mnist_result` (this session);
  `confirmation_drift_underpowered_experiment` (s040 caveat, now closed for
  depth).

## Environment notes

- `/home/marcrockhat/trioron-project/`, branch `progenitor-council`,
  Python 3.10.12, torch 2.11.0, WSL2, `python3` (NOT `python`),
  `OMP_NUM_THREADS=8`. Data `outputs/data/`.
- **Regime distinction (DO NOT CONFLATE):** chained-15 headline 0.736/0.446 is
  the GRADIENT-FREE PCLL organism (`run_pcll_chained15.py`). All conv work
  (s040 + s041) is GRADIENT-BASED (Adam on kernels). Different pipelines.
- **Model note:** lead with computational framing (convolution = weight-shared
  local filter; weight-tying = parameter sharing; depth = conv→pool→conv
  hierarchy; the shuffle control isolates locality) — biological metaphors
  ("twins") can trip mid-session safety downgrades.
