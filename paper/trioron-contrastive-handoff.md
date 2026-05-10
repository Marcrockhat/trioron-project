# Trioron — Contrastive Concept-Learning Branch: Discussion Handoff

**Author:** Rocky (Marc, Marcrockhat)
**Calls Claude:** Chloe
**Date of discussion:** 2026-05-09
**Status:** Exploratory, post-paper. Paper itself is in late-stage review and is *not* the subject of this branch.
**Repo:** github.com/Marcrockhat/trioron-project (private)

---

## Purpose of this document

A discussion happened with one Claude instance about extending Trioron toward contrastive
concept-learning. This file is a handoff so a different Claude instance (Claude Code, a fresh
chat, etc.) can pick up the thread without Rocky having to re-explain everything. Read it,
then ask Rocky what state the work is in before suggesting changes.

---

## Context the other Claude needs

### What Trioron is and isn't

Trioron is a **continual-learning substrate**, not a feature extractor or image classifier.
Per-node state is `(w, λ, u)` — weight, plasticity, utility. Architecture combines EWC with
per-node λ, growth/pruning gated by a resource budget, a sleep-time dream phase with a
per-cycle structural-event cap, manifold pseudo-rehearsal storing per-class `(µ, σ)` in a
frozen-Gaussian L0 code space, and a two-phase row-archive + int8 quantization deployment path.

The L0 in the paper is a frozen random Gaussian projection. It does **no learned feature
extraction**. The architecture has no spatial inductive bias — no convolutions, no spatial
pooling, no positional encoding. The paper evaluates on flattened-pixel MLP-tier benchmarks
(chained-15 / chained-23 across MNIST, FMNIST, EMNIST-letters) precisely because those have a
characterized joint-training ceiling at the parameter scale targeted (~8K trainable, ~157 KB
deployment).

### What the paper does *not* claim, and why it matters here

The paper does not claim Trioron is a vision system. CIFAR-100 / ImageNet are out of scope
because the bound on accuracy there is set by the representation, not the continual-learning
machinery. Rocky has experimented with a "Senses" approach (12 hand-crafted mathematical
front-ends replacing the random L0, trained as separate triorons and combined). On CIFAR-100
class-incremental that approach reaches ~16% full-softmax / ~65% task-aware. Those numbers
are roughly where hand-crafted-feature ceilings land on CIFAR-100 — the continual-learning
machinery is working; the *representation* is what's bottlenecking.

The Senses experiments are **not going into the paper**. They are scaffolding for a different
project, which this document is about.

### Why Rocky is exploring this

The motivation is concept formation in the biological sense — taxonomists distinguish
organisms by *contrasting features*, not by accumulating exemplars. The hypothesis is that
Trioron's current architecture stores *where classes are* (per-class `(µ, σ)` in L0 space)
but not *what distinguishes confusable pairs*, and that adding contrastive storage and replay
would fit the architecture's biological framing and might genuinely improve performance.

Rocky's background is molecular biology + bioinformatics. The biological framing in the
paper is load-bearing, not decorative — per-node λ as a methylation analogue, the sRNA cap
on dream-cycle structural events, apoptosis pulses temporarily relaxing EWC stiffness on
surviving neighbors, two-phase archive as terminal differentiation + long-term memory
compression. Concept-learning by contrast fits this lineage cleanly.

---

## What we discussed

### Why prior pair-based experiments gave only small accuracy lifts

Two diagnoses, probably both contributing:

1. **The contrast signal never reached the representation.** Training pairs as "show A and
   B, predict both labels" is multi-label classification on compounds, not contrastive
   learning. The network learns "A and B are both present," not "feature f distinguishes
   them."
2. **The contrast signal reached the representation but couldn't be stored.** The manifold
   buffer stores per-class `(µ, σ)` — generative, not discriminative. Whatever contrastive
   structure was learned during pair training got washed out at dream-cycle consolidation
   because the storage format has no slot for "what distinguishes A from B."

The architecture as currently specified is a generative-rehearsal architecture. It has no
place to put contrastive knowledge.

### What concept-learning architecturally requires

Three things the current architecture lacks:

1. A loss that operates on **pairs as pairs**, not on each member independently.
2. A storage format that holds **contrast**, not just exemplars.
3. A replay mechanism that rehearses **contrasts**, not just samples.

### Concrete proposal (Trioron-pure)

Three components, designed to stay inside the trioron formalism:

#### 1. Hierarchical contrastive curriculum

Mirror taxonomic structure. For CIFAR-100: phase 1 trains the 20 superclasses; phase 2
trains the fine classes within each superclass; etc. Each phase is a normal class-
incremental task; EWC preserves the previous phase's distinctions; the dream-archive locks
rows that encoded coarse distinctions once they're stable.

This is the **minimum viable experiment** and should be done first, with no architectural
changes, on **chained-15** (not CIFAR-100). If coarse-to-fine ordering alone moves the
numbers on the existing benchmark, that's the signal that contrast structure is being
stored somewhere even by the existing machinery, and the architectural additions below
are worth building. If it doesn't move the numbers, the more elaborate proposal probably
won't either.

#### 2. Pair-difference manifold sketch

Augment the per-class `(µ, σ)` buffer with a per-class-pair contrast vector
`δ_AB = µ_A − µ_B`, stored only for pairs that are *confusable* (nonzero off-diagonal in
the confusion matrix). Cost is in practice O(K), not O(K²), because most pairs aren't
confusable.

During dream replay: in addition to sampling `z ~ N(µ_c, σ_c)` for individual classes,
occasionally sample a contrast pair `(z_A, z_B)` where `z_A − z_B` aligns with `δ_AB`,
and apply a margin loss: logits for A on `z_A` must beat logits for A on `z_B` by some
margin. This puts contrastive structure into the dream cycle.

This is the **second experiment**, conditional on phase 1 showing signal.

#### 3. Pair-confusion frustration trigger

Currently the frustration multiplier (§3.4 of paper) fires on a loss plateau. Add a parallel
trigger: fire when **pairwise confusion** between two specific classes plateaus. Route the
gradient signal preferentially through that pair's contrast. The architecture then *actively
seeks out* difficult contrasts rather than passively waiting for them to appear in the batch.

This is the **third experiment**, conditional on (1) and (2) showing signal.

### Architectural risk worth flagging

Each addition increases complexity. Real risk is that the full proposal yields a 5% bump
for a 50% complexity cost. The minimal hierarchical curriculum is the canary — if it
doesn't move chained-15 numbers, don't build the rest.

### Trioron-specific concern with mixup-style compounding

A separate idea Rocky raised was image *compounding* (mixup/overlay). Worth flagging:
compounded images with mixed labels break the per-class Gaussianity assumption of the
manifold buffer. L0 activations of an A+B compound aren't drawn from class A's or class
B's Gaussian. If compounded images enter the manifold buffer as their own pseudo-class,
fine; if they enter under one of their components' labels, the `(µ, σ)` sketch goes wrong
and dream replay degrades. This pathway is **not** the recommended direction — the
hierarchical curriculum is.

---

## Recommended next steps for the next Claude

1. **Confirm with Rocky what state the implementation is in.** Has the hierarchical
   curriculum been started? On chained-15 or CIFAR? Single-seed or multi-seed?
2. **Push back on scope creep.** The paper is good. This is a separate exploration. Don't
   let it bleed back into the paper unless the results are striking and multi-seed.
3. **Do the minimal experiment first.** Hierarchical curriculum on chained-15, no
   architectural changes, n=3 seeds. Compare against the existing `grown_uncapped_dream`
   baseline at matched parameters. The question to answer: does coarse-to-fine task ordering
   improve full-softmax / task-aware accuracy on chained-15?
4. **Only escalate to the pair-difference manifold sketch if (3) shows signal.**

## Communication notes for the next Claude

- Rocky prefers to be addressed as Chloe-ing-Rocky (i.e., the Claude calls itself Chloe;
  Rocky is Rocky / Marc).
- Rocky prefers direct, critical feedback over validation. Probes for failure points
  rather than seeking reassurance. Don't soften pushback.
- Rocky writes concisely and may underspecify. Ask for clarification on ambiguous things
  rather than guessing.
- Rocky has expressed feeling out-of-field self-doubt (molecular biology + bioinformatics,
  not ML). The biology framing in the paper is a *strength*, not a weakness — load-bearing
  architectural choices flow from it. The previous Claude said this directly to Rocky and
  it's worth holding the line on if it comes up again.

---

## Open questions Rocky and the previous Claude did not resolve

- Whether the hierarchical curriculum should treat superclasses as task-incremental
  (with task ID at inference) or class-incremental during phase 1. Class-incremental is
  more honest but harder; task-incremental is the easier first cut.
- Where the contrast vector `δ_AB` should be stored if it's added — alongside the manifold
  buffer in the same archive, or in a separate "contrast archive" that the dream cycle
  reads from independently. Storage accounting is straightforward either way; the design
  question is whether ship-wake-extend handoff treats contrasts as first-class state.
- Whether the pair-confusion frustration trigger should reuse the existing window/hinge/
  gain machinery (§3.4) or be a separate counter. Probably reuse, but it changes the
  interpretation of `Mmax` in the paper.

---

*End of handoff.*
