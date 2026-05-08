# Pong primitive vocabulary — design doc

This doc enumerates the primitive concepts we'll train as Mode-E donors
(see `docs/learning_methods.md` §3) before extending onto Pong itself.
Each primitive is a small Mode-A donor on a synthetic vector-state
distribution where per-class clustering is verifiable by construction.
The set of donors is then `absorb`ed into one **vocabulary organism**
that serves as the substrate for all subsequent Pong work.

> **Design pivot 2026-05-08.** Earlier drafts of this doc operated on
> raw 28224-d pixel frames. Empirical clustering probe showed that
> ball-related primitives don't survive the random L0 projection
> (BALL_LEFT vs BALL_RIGHT probe_acc = 0.60 at l0_dim=128). Diagnosis:
> **trioron is not a perception system.** Its random L0 projection is
> spatially blind and works on small structured vectors, not on raw
> high-dimensional images. The honest fix (per
> `device_conscience_pattern.md` and the original handoff's "option
> α") is to **lift perception out of the substrate**: extract a
> compact state vector via RAM and feed *that* to trioron. All
> primitives below operate on the state vector, not pixels.

## State-vector format

Each timestep, a perception module emits a **standardized 7-d float32
vector**:

```
state_t = (ball_x, ball_y, ball_dx, ball_dy, paddle_y, paddle_dy, ball_speed)
          where ball_speed = ‖(ball_dx, ball_dy)‖
```

- `ball_x, ball_y` — ball center position, frame coords `[0, 84]`
- `ball_dx, ball_dy` — per-frame velocity, `≈ [-5, 5]`
- `paddle_y` — paddle (agent's, right side in Pong) center, `[0, 84]`
- `paddle_dy` — paddle's per-frame velocity, `≈ [-3, 3]`
- `ball_speed` — pre-computed `‖(ball_dx, ball_dy)‖` so a linear
  probe over L0 codes can recover speed-based primitives. The
  substrate's L0 ReLU non-linearity is too weak to recover a norm
  from raw velocities — moving the non-linear transformation
  upstream of L0 is the load-bearing fix.

Each axis is then **standardized** (zero-mean, unit-variance) using
the calibration constants in `synthetic_env.STANDARDIZE`. This keeps
the random L0 projection from being dominated by high-magnitude
axes (positions in `[0, 84]`) at the expense of low-magnitude axes
(velocities in `[-5, 5]`).

The perception module for Pong is `experiments/atari_trioron/features.py`'s
`extract_pong(ale)`, which reads the ALE RAM (positions are at fixed
RAM offsets), computes velocities by differencing against the
previous timestep, computes ball_speed, and applies the same
standardization. The extractor lives outside trioron — trioron does
**not** consume raw pixels and does **not** compute speed inside L0.

L0 width on this 7-d standardized input is small. **Empirically
verified L0 width: 64.** At 16 some adjacent-bin primitives
(HIGH vs MID) come in at 0.81-0.90 probe_acc; at 64 all 12 probed
pairs clear the 0.95 threshold (most at 1.00).

## Goals

- Each primitive must respect the per-class clustering assumption in
  6-d state space — verified by the probe in §6.
- Primitives are **task-agnostic in their definition**: "ball moving
  upward" is a property of the velocity sign, not of Pong-specific
  context.
- Primitives are **composable**: the Pong-extension layer reads
  multiple primitive activations per timestep and learns the
  game-specific composition.
- Class IDs live in a global namespace shared across the vocabulary so
  `absorb` works without collision.

## Class ID layout

Disjoint integer ranges per primitive group.

| Range | Primitive group | Classes |
|-------|-----------------|---------|
| 100–102 | Ball vertical position | BALL_HIGH, BALL_MID, BALL_LOW |
| 103–105 | Ball horizontal position | BALL_LEFT, BALL_CENTER, BALL_RIGHT |
| 110–112 | Paddle vertical position | PADDLE_HIGH, PADDLE_MID, PADDLE_LOW |
| 120–123 | Ball motion direction | BALL_GOING_UP, BALL_GOING_DOWN, BALL_GOING_LEFT, BALL_GOING_RIGHT |
| 130–131 | Ball speed | BALL_FAST, BALL_SLOW |
| 140–142 | Ball-paddle relative vertical | BALL_ABOVE_PADDLE, BALL_ALIGNED_WITH_PADDLE, BALL_BELOW_PADDLE |
| 150–151 | Ball-paddle approach | BALL_APPROACHING_PADDLE, BALL_RECEDING_FROM_PADDLE |

20 primitive classes total. Pong-extension (task #10) opens a
separate disjoint range (200+) for game-specific compound classes.

## Tier 1 — position primitives

Single-timestep, defined by which third of the screen the relevant
quantity falls in. Boundaries: thirds of `[0, 84]` → `[0, 28)`,
`[28, 56)`, `[56, 84]`.

| ID | Class | Definition (state-vector predicate) |
|----|-------|-------------------------------------|
| 100 | BALL_HIGH | `ball_y < 28` |
| 101 | BALL_MID  | `28 ≤ ball_y < 56` |
| 102 | BALL_LOW  | `ball_y ≥ 56` |
| 103 | BALL_LEFT   | `ball_x < 28` |
| 104 | BALL_CENTER | `28 ≤ ball_x < 56` |
| 105 | BALL_RIGHT  | `ball_x ≥ 56` |
| 110 | PADDLE_HIGH | `paddle_y < 28` |
| 111 | PADDLE_MID  | `28 ≤ paddle_y < 56` |
| 112 | PADDLE_LOW  | `paddle_y ≥ 56` |

**Clustering argument:** these are axis-aligned partitions of a 6-d
space. A random linear projection preserves linear separability with
overwhelming probability for K-class linear partitions when L0 is at
least a few × log(K). Probe should give probe_acc ≈ 1.0.

## Tier 2 — motion-direction primitives

Defined by sign of velocity components.

| ID | Class | Definition |
|----|-------|------------|
| 120 | BALL_GOING_UP    | `ball_dy < -ε` |
| 121 | BALL_GOING_DOWN  | `ball_dy > +ε` |
| 122 | BALL_GOING_LEFT  | `ball_dx < -ε` |
| 123 | BALL_GOING_RIGHT | `ball_dx > +ε` |

`ε = 0.5` pixels/frame to suppress numerical jitter as a class
positive.

**Clustering argument:** half-space partitions in 6-d. Probe should
give probe_acc ≈ 1.0.

**Note on diagonal motion:** a single sample can be both
BALL_GOING_UP and BALL_GOING_LEFT simultaneously. We train each
primitive's donor against an independent negative class (e.g.
BALL_GOING_UP vs not-going-up), so co-occurrence is fine — both
branches will fire at inference, and the action head composes them.

## Tier 3 — speed primitives

| ID | Class | Definition |
|----|-------|------------|
| 130 | BALL_FAST | `‖(ball_dx, ball_dy)‖ ≥ 3.0` |
| 131 | BALL_SLOW | `0.5 ≤ ‖(ball_dx, ball_dy)‖ < 3.0` |

Stationary ball (norm < 0.5) is treated as neither — represents
between-rallies states; rolled into a separate game-state primitive
if needed.

**Clustering argument:** annular partition in (dx, dy) subspace.
Linear in `‖.‖²` so a degree-2 projection separates them; in practice
the L0 ReLU non-linearity gives enough non-linear capacity for the
linear probe to separate. Probe should give probe_acc ≥ 0.9.

## Tier 4 — relational primitives

Capture the ball–paddle relationship that's most directly useful for
Pong action selection.

| ID | Class | Definition |
|----|-------|------------|
| 140 | BALL_ABOVE_PADDLE        | `ball_y < paddle_y - 4` |
| 141 | BALL_ALIGNED_WITH_PADDLE | `|ball_y - paddle_y| ≤ 4` |
| 142 | BALL_BELOW_PADDLE        | `ball_y > paddle_y + 4` |
| 150 | BALL_APPROACHING_PADDLE  | `ball_dx > 0` (Pong: agent's paddle on right) |
| 151 | BALL_RECEDING_FROM_PADDLE | `ball_dx < 0` |

**Clustering argument:** linear partitions on a difference variable
(`ball_y - paddle_y`) and on velocity component. Both clean linear
partitions in 6-d. Probe should give probe_acc ≈ 1.0.

**Pong-specific approach direction.** "Approaching" depends on which
side the paddle is on. For Pong's right-side agent paddle, ball
moving with `ball_dx > 0` is approaching. Breakout's bottom paddle
would invert this — keep this primitive Pong-only or generalize via
a `paddle_x` field (defer until Breakout).

## Verification protocol

Before training each primitive donor, run a clustering probe on ~200
synthetic state vectors (100 class-positive, 100 class-negative):

1. Project each sample through a fresh frozen L0 layer at the shared
   seed (`l0_width=32`, `freeze_l0=True`).
2. Compute mean intra-class distance, mean inter-class distance,
   and ratio (this is mostly diagnostic in low-d; the probe accuracy
   is the load-bearing check).
3. Train a 1-step linear probe (logistic regression on L0 codes);
   accuracy on a held-out 80/20 split must be `≥ 0.95`.

In 6-d input space with 32-d L0, almost all primitives in the
vocabulary above should saturate the probe at 1.0. Any primitive
falling below 0.95 is a sign the class definition is too noisy or
the threshold needs widening.

## Synthetic data generation

Per-class samplers operate directly in state-vector space (no
rendering needed). For each primitive:

1. Sample the relevant axes from their natural ranges, with the rest
   sampled uniformly from full range.
2. Filter to the class-positive half-space / region.
3. Add small Gaussian noise (σ = 0.5 px) to make samples non-degenerate.

Example for `BALL_HIGH`:
```python
ball_y = uniform(0, 28)            # class-positive region
ball_x = uniform(0, 84)            # full range on irrelevant axis
ball_dx, ball_dy = uniform(-5, 5)  # full range
paddle_y = uniform(0, 84)
paddle_dy = uniform(-3, 3)
state = (ball_x, ball_y, ball_dx, ball_dy, paddle_y, paddle_dy)
```

Negative-class samples for a primitive are drawn uniformly from the
*complement* of the class-positive region on the relevant axis, with
all other axes sampled uniformly. This gives the linear probe a clean
binary classification problem.

## Composition pattern

After `absorb`, the vocabulary organism's branches all fire in
parallel on a state vector. The action head reads the per-branch
log-likelihoods (or the L1 codes) and learns the Pong-specific
composition:

```
inputs to action head per timestep:
  [BALL_HIGH-prob, BALL_MID-prob, BALL_LOW-prob,
   PADDLE_HIGH-prob, PADDLE_MID-prob, PADDLE_LOW-prob,
   BALL_GOING_UP-prob, BALL_GOING_DOWN-prob, ...,
   BALL_ABOVE_PADDLE-prob, BALL_BELOW_PADDLE-prob,
   BALL_APPROACHING-prob, ...]

learned composition (example):
  if BALL_ABOVE_PADDLE-prob is high AND BALL_APPROACHING:
    emit UP-action
  elif BALL_BELOW_PADDLE-prob is high AND BALL_APPROACHING:
    emit DOWN-action
  else:
    emit NO-OP
```

The composition is **learned**, not coded. The action head is a tiny
MLP trained by self-imitation on top-k rallies (Mode D). Trioron's
substrate stays frozen during action-head training.

## Cross-game transfer notes (Breakout)

Most primitives transfer trivially because they're defined on a
generic state vector:
- All Tier 1 position primitives (with paddle-x added for Breakout's
  horizontal paddle)
- All Tier 2 motion-direction primitives
- Tier 3 speed primitives
- Tier 4 relational — needs generalization via paddle position

Breakout-specific additions (defer until Pong arms work):
- BRICKS_REMAINING_HIGH / LOW (a separate scalar from RAM)
- BALL_NEAR_BRICK_LAYER
- (The Pong PADDLE-VERTICAL primitives may not transfer cleanly;
  Breakout's paddle is horizontal, so PADDLE_LEFT/MID/RIGHT
  primitives in `paddle_x` are the correct mirror.)

## Empirical verification (2026-05-08)

Probe results at the final design (STATE_DIM=7, standardized,
l0_dim=64, n=500 per class):

| Pair | probe_acc |
|------|-----------|
| BALL_HIGH vs LOW | 1.00 |
| BALL_HIGH vs MID | 0.98 |
| BALL_LEFT vs RIGHT | 1.00 |
| BALL_LEFT vs CENTER | 0.96 |
| PADDLE_HIGH vs LOW | 1.00 |
| PADDLE_HIGH vs MID | 0.97 |
| BALL_GOING_UP vs DOWN | 1.00 |
| BALL_GOING_LEFT vs RIGHT | 1.00 |
| BALL_FAST vs SLOW | 0.97 |
| BALL_ABOVE_PADDLE vs BELOW | 1.00 |
| BALL_ABOVE_PADDLE vs ALIGNED | 0.98 |
| BALL_APPROACHING vs RECEDING | 1.00 |

All clear the 0.95 threshold. Vocabulary is sound; donor training
(task #8) can proceed.

## Open questions

1. **Cap_bytes per primitive**. Each primitive donor is tiny — start
   at `cap_bytes=2_000` and only grow if a primitive fails its own
   held-out test.
2. **Negative-class definition**. Currently uniform over the
   complement region. Could stratify by sampling negatives from
   *other* primitives' positive regions to make the donor learn the
   primitive concept rather than a shortcut. Default plan: uniform
   complement until donor-training results suggest otherwise.
3. **Standardization constants vs real Pong distribution**.
   `STANDARDIZE` is calibrated to the synthetic uniform distribution.
   Real RAM-extracted Pong states are non-uniform (ball spends more
   time mid-field; paddle velocities cluster near 0). The substrate
   should still recognize the primitives — the standardization is
   approximate, not load-bearing — but worth re-calibrating from a
   small real-Pong sample if extension (task #10) underperforms.

## Status

This is a design doc. Implementation tasks:
- `#7` — synthetic primitive-env framework (DONE in vector-space form)
- `#8` — train primitive donors (one per group above)
- `#9` — absorb into vocabulary organism
- `#10` — extend onto Pong (with `experiments/atari_trioron/features.py`
  as the perception module at inference time)

The verification probe (§6) is the gate before each donor is committed
to `outputs/atari_primitive_donors/`.
