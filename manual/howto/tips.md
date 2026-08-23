# How-to & tips
`status: living page`

- **Loss parked at exactly ln 2 ≈ 0.693** → signal scale (use fan-in init) or missing cross-terms (inputs that must interact are not sharing a dendritic branch).
- **Quad loss spikes/explodes while accuracy still climbs** → you are surviving on grad-clip; bound the state (e.g. tanh between repeated applications) instead of hoping.
- **Seed before `construct`** — structure reproducibility, not just weights.
- **Budget first**: set the `Envelope` before designing; growth must fit inside it.
- **Never re-adapt a settled symbol layer** under a trained consumer; new symbols can be read as their consistent projections onto the old inventory.
- **Grad-clip 1.0** is part of the standard recipe with quad substrates.
- **CPU is enough** for Phase-1-sized substrates (50K params); parallel *processes* beat intra-op threads for sweeps.
