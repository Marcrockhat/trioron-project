"""Trioron 2.0 — substrate profile presets.

A `TrioronProfile` is a frozen bundle of substrate-level configuration
choices that line up with a deployment regime: depth-heavy reasoning,
classification (width-scaling), constrained edge hardware, or fully
unrestricted learning. Picking a regime sets sane defaults across:

  - Axis 5 dendritic compartmentalization (`branch_activation`, `B_max`)
  - structural plasticity gates (`allow_grow_node`, `allow_grow_branch`,
    `allow_insert_layer`)
  - hardware ceilings (`memory_cap_bytes`, `time_cap_seconds`)
  - donor-load override behavior (`re_apply_after_donor_load`)

The profile is an *active default* — `TrioronLayer.__init__` consults
the current active profile when its `branch_activation` / `B_max`
kwargs are left None. Passing an explicit kwarg always overrides the
profile. The default active profile is `OPEN` (everything on, no cap),
which reproduces 1.0-era construction defaults byte-for-byte.

Typical use:

  >>> import trioron.profile as tp
  >>> tp.TrioronProfile.set_active(tp.REASONING)   # for the rest of this process
  >>> # or, scoped to a block:
  >>> with tp.TrioronProfile.use(tp.EDGE):
  ...     layer = TrioronLayer(fan_in=4, n_nodes=2)
  ...     # layer was constructed with EDGE defaults: B_max=1, identity σ_branch
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import ClassVar, Iterator, Optional


@dataclass(frozen=True)
class TrioronProfile:
    """A named bundle of substrate-level configuration defaults.

    Fields:
      name: Human-readable identifier (shown in repr, logs, telemetry).
      branch_activation: Axis 5 per-branch nonlinearity for fresh
          layers. Live default = "quad" (NMDA-style supralinear);
          "identity" disables Axis 5 functionally even at K>1.
      B_max: Per-cell branch budget cap. 1 = Axis 5 hard-disabled,
          no branch_weight memory allocated beyond the K=1 slot.
      allow_grow_node: Whether population-level width growth is
          permitted. Triggers consult this gate; False = freeze
          n_nodes per layer.
      allow_grow_branch: Whether Axis 5 dendritic growth is
          permitted. Independent of branch_activation (you can have
          quad branches but freeze their count).
      allow_insert_layer: Whether between-cell depth growth is
          permitted (Axis 3). Phase 2 in the spec; not yet wired
          to a trigger, so this gate is documentary for now.
      memory_cap_bytes: Hard upper bound on substrate growth in
          bytes (consumed by ceilings.py). None = no cap.
      time_cap_seconds: Per-stabilization-window time ceiling. None =
          no cap. Consumed by CeilingsController.
      re_apply_after_donor_load: When loading a v1 donor that auto-
          flips branch_activation to "identity" for back-compat, this
          flag (True by default) restores the active profile's
          branch_activation post-load. Set False to honor the v1
          override and treat the loaded layer as functionally
          point-neuron forever.
    """
    name: str
    branch_activation: str = "quad"
    B_max: int = 8
    allow_grow_node: bool = True
    allow_grow_branch: bool = True
    allow_insert_layer: bool = True
    memory_cap_bytes: Optional[int] = None
    time_cap_seconds: Optional[float] = None
    re_apply_after_donor_load: bool = True

    # Class-level active profile state. Default OPEN is set below
    # after the preset constants are defined.
    _active: ClassVar[Optional["TrioronProfile"]] = None

    @classmethod
    def active(cls) -> "TrioronProfile":
        """Return the currently-active profile. Falls back to OPEN if
        nothing has been set."""
        if cls._active is None:
            return OPEN
        return cls._active

    @classmethod
    def set_active(cls, profile: "TrioronProfile") -> None:
        """Install `profile` as the process-wide default. Subsequent
        TrioronLayer constructions that don't specify branch_activation
        / B_max will consult this profile."""
        cls._active = profile

    @classmethod
    @contextmanager
    def use(cls, profile: "TrioronProfile") -> Iterator["TrioronProfile"]:
        """Scoped active-profile override. Restores the previous active
        profile on exit even if an exception fires.

            with TrioronProfile.use(EDGE):
                build_donor(...)         # gets EDGE defaults
            # active profile back to whatever it was before
        """
        prev = cls._active
        cls._active = profile
        try:
            yield profile
        finally:
            cls._active = prev


# ---------------------------------------------------------------------
# Named presets
# ---------------------------------------------------------------------

# Reasoning / depth-heavy regime. Dendrites live, all growth open, no
# hardware cap. Use for tasks where per-cell expressive depth matters
# (Phase 6 dendrite-delta-style fine discrimination, hierarchical
# reasoning probes, anything that benefits from supralinear branch
# pools).
REASONING = TrioronProfile(
    name="reasoning",
    branch_activation="quad",
    B_max=8,
    allow_grow_node=True,
    allow_grow_branch=True,
    allow_insert_layer=True,
    memory_cap_bytes=None,
    time_cap_seconds=None,
    re_apply_after_donor_load=True,
)

# Classification regime. Axis 5 functionally off (identity σ_branch +
# B_max=1 hard cap = no dendrite memory at all), but width and depth
# growth stay live so the substrate scales capacity through cell count.
# Matches Rocky's "for classification, it will be growth" framing and
# the [[feedback_cl_machinery_scope]] memory's "classification benches:
# freeze trioron as a feature bank, vanilla NN downstream" guidance.
CLASSIFICATION = TrioronProfile(
    name="classification",
    branch_activation="identity",
    B_max=1,
    allow_grow_node=True,
    allow_grow_branch=False,
    allow_insert_layer=True,
    memory_cap_bytes=None,
    time_cap_seconds=None,
    re_apply_after_donor_load=True,
)

# Constrained-hardware regime. Dendrites off (no branch_weight memory),
# growth allowed but ceiling-gated, both memory and time caps active.
# Sized for Orange Pi 5B / ESP32-class deployment per the device
# conscience pattern. Caps are conservative defaults; override on the
# profile copy if your specific hardware is tighter or looser.
EDGE = TrioronProfile(
    name="edge",
    branch_activation="identity",
    B_max=1,
    allow_grow_node=True,
    allow_grow_branch=False,
    allow_insert_layer=False,
    memory_cap_bytes=256 * 1024 * 1024,        # 256 MB
    time_cap_seconds=30.0,
    re_apply_after_donor_load=True,
)

# Fully unrestricted regime. Everything on, nothing capped. This is
# the default active profile — matches 1.0-era construction defaults
# byte-for-byte (the architectural-knob fields just expose the
# pre-existing defaults under a named label rather than changing them).
# re_apply_after_donor_load=False here so the v1-load silent override
# behavior matches prior semantics; the named regimes opt INTO
# re-application to enforce their chosen branch_activation across
# donor loads.
OPEN = TrioronProfile(
    name="open",
    branch_activation="quad",
    B_max=8,
    allow_grow_node=True,
    allow_grow_branch=True,
    allow_insert_layer=True,
    memory_cap_bytes=None,
    time_cap_seconds=None,
    re_apply_after_donor_load=False,
)


# Install OPEN as the default active profile at import time. Subsequent
# calls to set_active() override; the class-level fallback in active()
# still returns OPEN if nothing was installed (resilient under test
# isolation that resets _active to None).
TrioronProfile._active = OPEN


# Convenience map for log/CLI lookup. Iterate over named presets:
#   for name, profile in PRESETS.items(): ...
PRESETS: dict[str, TrioronProfile] = {
    "reasoning": REASONING,
    "classification": CLASSIFICATION,
    "edge": EDGE,
    "open": OPEN,
}
