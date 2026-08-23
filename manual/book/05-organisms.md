# 5. Organisms
`status: skeleton`

An **organism** is a saveable composition of substrates plus its calibration statistics, with a uniform contract (`trioron.organisms.base`):

- `fit`-style stage methods return `self` (chainable acquisition schedules);
- inference methods are `no_grad` and take native inputs;
- `save(path)`, `load(path)` (type-checked), `n_params()`.

Shipped organisms:

- **Logophosis** — the language organism: a gradient-free symbol layer (Phasecyte over chord-encoded tokens) feeding a sequential link chain whose re-application depth = nesting depth; the head is substrate too. Demonstrates zero-shot generalisation of operators (negation) to depths never trained.
- **Kinopsis** — the motion organism: leaves over feature streams, settled heads, a router, a velocity readout.

This chapter will walk building a small organism end-to-end and transferring a skill between worlds (load, do **not** re-adapt the symbol layer, train free; lock only what is fragile).
