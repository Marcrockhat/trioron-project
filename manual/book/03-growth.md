# 3. Growth
`status: skeleton — protocol and formulas final; worked notebook to come`

A substrate that cannot fit its task should not be redesigned — it should **grow**. This chapter will walk cell division and the growth controller on a task with a provable depth requirement (parity: with $\sigma(z)=z+z^2$, $k$-bit parity needs $\lceil \log_2 k \rceil$ quad layers).

## The protocol (final — validated)
The order of operations decides everything:

1. **Stall** — a plateau detector (patience $\geq 8$ epochs without train-loss or test-accuracy progress) fires. Growing earlier than the plateau breaks yields half-trained parts.
2. **Grow** — new capacity is added (a divided cell, a new link, one more re-application of a tied link).
3. **Joint train** — *everything stays trainable*: the new capacity learns **with** the old, not on top of frozen garbage.
4. **Lock after settlement** — only when the target is reached does credit lock ($\lambda$ rises on the settled region).

The failed alternative is worth remembering: *frozen-greedy* growth (lock **before** the new part settles) never recovers — on tasks with no low-order signal the first part learns nothing and everything later builds on it.

## What will be in the worked example
- `StallTrigger` and `fit_grow` from `trioron.lifecycle.chain` on parity-4: 1 link = chance, growth to 3 links = 1.00.
- Watching $\lambda$ rise on settled links.
- The overshoot phenomenon (one extra growth event before the plateau breaks) and why it is harmless.
