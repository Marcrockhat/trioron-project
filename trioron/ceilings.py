"""Trioron — hard ceilings (§4.2 + §7 + §8 step 7).

Preflight checks evaluated before every proposed cellular division.
Two independent gates:

    1. Memory ceiling.  The estimated memory footprint after the
       proposed division must not exceed `M_max`. We compare
       (current allocated bytes + division Δ bytes) to M_max.

    2. Stabilization-time plateau.  If the previous division required
       more than `T_div_max` wall-clock seconds to re-stabilize, the
       network is *mature*: no further divisions are allowed.

Either failed gate "arrests" the controller permanently. Once arrested,
all subsequent preflights return `allowed=False` with the original
arrest reason — per §4.2: "Once arrested, the network is *mature* and
may only update via plasticity (weights change, topology does not)."

The orchestrator's contract:

    decision = ceilings.preflight(net, layer_idx)
    if not decision.allowed:
        # Skip division; log decision.reason; continue training under
        # whatever EWC / plasticity regime is active.
        ...
    else:
        net.grow_layer(layer_idx, ...)
        rebuild optimizer
        ceilings.mark_stabilization_start()
        for _ in range(T_stabilize):
            train one step
        ceilings.mark_stabilization_end()
        # mark_stabilization_end records elapsed seconds; if it
        # exceeded T_div_max the *next* preflight will arrest.

Persistence: state_dict() / load_state_dict() lets the orchestrator
checkpoint arrest state across reboots — Orange Pi will reboot.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch

from .network import TrioronNetwork


# ---------------------------------------------------------------------
# Memory-provider helpers
# ---------------------------------------------------------------------


def _cuda_allocated_bytes() -> Optional[int]:
    """torch.cuda.memory_allocated() if CUDA is available *and* initialized,
    else None. Calling this is cheap; it doesn't initialize CUDA on its own."""
    try:
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            return int(torch.cuda.memory_allocated())
    except Exception:
        return None
    return None


def _proc_self_statm_rss_bytes() -> Optional[int]:
    """Linux RSS in bytes via /proc/self/statm. None if unavailable."""
    try:
        with open("/proc/self/statm", "r") as fh:
            fields = fh.read().split()
        # fields = [size, resident, shared, text, lib, data, dt] (in pages)
        rss_pages = int(fields[1])
        return rss_pages * os.sysconf("SC_PAGE_SIZE")
    except (FileNotFoundError, OSError, ValueError, IndexError):
        return None


def default_memory_provider() -> int:
    """Best available current-allocation signal, in bytes.

    Preference order: CUDA allocator → /proc RSS → 0. The 0 fallback is
    intentional — on platforms where neither signal is available, the
    memory ceiling check effectively reduces to "Δ alone must fit under
    M_max", which is conservative-leaning-permissive but still useful.
    """
    cuda = _cuda_allocated_bytes()
    if cuda is not None:
        return cuda
    rss = _proc_self_statm_rss_bytes()
    if rss is not None:
        return rss
    return 0


# ---------------------------------------------------------------------
# Division Δ-bytes estimator
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class DivisionDelta:
    """Breakdown of the float counts added by a proposed grow_layer call.

    Fields are in floats; bytes() converts using the supplied dtype size.
    """
    layer_idx: int
    fan_in: int
    has_next_layer: bool
    next_layer_n_nodes: int
    params_floats: int
    buffers_floats: int
    optimizer_floats: int

    @property
    def total_floats(self) -> int:
        return self.params_floats + self.buffers_floats + self.optimizer_floats

    def bytes(self, dtype_bytes: int = 4) -> int:
        return self.total_floats * int(dtype_bytes)


def division_param_delta(
    net: TrioronNetwork,
    layer_idx: int,
    optimizer_state_per_param: int = 2,
) -> DivisionDelta:
    """Compute the exact tensor-element delta of net.grow_layer(layer_idx).

    Mirrors what TrioronLayer.grow_node + grow_input actually allocate
    (see trioron/node.py).

    grow_node(layer i) adds:
        params:   W += 1 row of fan_in_i floats   →   fan_in_i
                  b += 1                          →   1
        buffers:  lam (+1), u (+1), b_anchor (+1), fisher_b (+1),
                  W_anchor (+row), fisher_W (+row)
                  → 4 + 2*fan_in_i

    grow_input(layer i+1) adds (only if i+1 exists):
        params:   W += 1 col of n_nodes_{i+1}     →   n_nodes_{i+1}
        buffers:  W_anchor (+col), fisher_W (+col)
                  → 2 * n_nodes_{i+1}

    optimizer_state_per_param scales params_floats. Adam = 2 (m, v);
    SGD = 0; SGD-with-momentum = 1.
    """
    if not (0 <= layer_idx < len(net.layers)):
        raise IndexError(
            f"layer_idx {layer_idx} out of range [0, {len(net.layers)})"
        )
    if optimizer_state_per_param < 0:
        raise ValueError("optimizer_state_per_param must be >= 0")

    target = net.layers[layer_idx]
    fan_in = int(target.fan_in)
    has_next = layer_idx + 1 < len(net.layers)
    next_n_nodes = int(net.layers[layer_idx + 1].n_nodes) if has_next else 0

    params = (fan_in + 1) + (next_n_nodes if has_next else 0)
    buffers = (4 + 2 * fan_in) + (2 * next_n_nodes if has_next else 0)
    opt_state = optimizer_state_per_param * params

    return DivisionDelta(
        layer_idx=layer_idx,
        fan_in=fan_in,
        has_next_layer=has_next,
        next_layer_n_nodes=next_n_nodes,
        params_floats=params,
        buffers_floats=buffers,
        optimizer_floats=opt_state,
    )


# ---------------------------------------------------------------------
# Decision dataclass
# ---------------------------------------------------------------------


# Reason codes — kept as bare strings so they're trivially loggable / CSV-able.
REASON_OK = "ok"
REASON_MEMORY_CEILING = "memory_ceiling"
REASON_TIME_CEILING = "time_ceiling"
REASON_ARRESTED = "arrested"
REASON_INVALID_LAYER = "invalid_layer"


@dataclass(frozen=True)
class PreflightDecision:
    allowed: bool
    reason: str
    layer_idx: int
    current_bytes: int
    delta_bytes: int
    projected_bytes: int
    M_max_bytes: int
    last_stab_seconds: Optional[float]
    T_div_max_seconds: float

    def __str__(self) -> str:
        verdict = "ALLOW" if self.allowed else "DENY"
        mb = 1024 * 1024
        stab = (
            f"{self.last_stab_seconds:.2f}s"
            if self.last_stab_seconds is not None
            else "n/a"
        )
        return (
            f"[{verdict}] reason={self.reason} layer={self.layer_idx}  "
            f"mem {self.current_bytes / mb:.1f}MB + Δ{self.delta_bytes / mb:.3f}MB "
            f"= {self.projected_bytes / mb:.1f}MB "
            f"(M_max={self.M_max_bytes / mb:.0f}MB)  "
            f"last_stab={stab} (T_div_max={self.T_div_max_seconds:.0f}s)"
        )


# ---------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------


class CeilingsController:
    """Per-network arbiter that gates cellular division on §4.2 ceilings.

    Lifecycle:
        c = CeilingsController(M_max_bytes=2 * 1024**3, T_div_max_seconds=60.0)
        decision = c.preflight(net, layer_idx)
        if decision.allowed:
            net.grow_layer(...)
            c.mark_stabilization_start()
            ...                              # T_stabilize training steps
            elapsed = c.mark_stabilization_end()
            if elapsed > T_div_max:
                # Next preflight will arrest with REASON_TIME_CEILING.
                pass
        else:
            log(decision.reason)             # MEMORY / TIME / ARRESTED

    Sticky arrest:
        Once any preflight returns allowed=False due to a hard ceiling
        violation (memory or time), `arrested` flips to True and stays
        True for the lifetime of this controller. Subsequent preflights
        return REASON_ARRESTED with the original arrest reason kept in
        `arrest_reason` for diagnostics.
    """

    def __init__(
        self,
        M_max_bytes: int,
        T_div_max_seconds: float,
        optimizer_state_per_param: int = 2,
        dtype_bytes: int = 4,
        memory_provider: Optional[Callable[[], int]] = None,
        time_provider: Optional[Callable[[], float]] = None,
    ):
        if M_max_bytes < 1:
            raise ValueError("M_max_bytes must be >= 1")
        if T_div_max_seconds <= 0:
            raise ValueError("T_div_max_seconds must be > 0")

        self.M_max_bytes = int(M_max_bytes)
        self.T_div_max_seconds = float(T_div_max_seconds)
        self.optimizer_state_per_param = int(optimizer_state_per_param)
        self.dtype_bytes = int(dtype_bytes)
        self._memory_provider = memory_provider or default_memory_provider
        self._time_provider = time_provider or time.monotonic

        self.arrested: bool = False
        self.arrest_reason: Optional[str] = None
        self.last_stab_seconds: Optional[float] = None
        self.divisions_attempted: int = 0
        self.divisions_allowed: int = 0
        self._stab_start: Optional[float] = None

    # ----- preflight -----

    def preflight(self, net: TrioronNetwork, layer_idx: int) -> PreflightDecision:
        """Evaluate all gates. Sets arrested=True on any hard veto."""
        self.divisions_attempted += 1

        # Validate layer_idx without raising — we want to surface this as
        # a non-allowing decision rather than crash mid-orchestrator.
        if not (0 <= layer_idx < len(net.layers)):
            return self._deny(
                REASON_INVALID_LAYER,
                layer_idx=layer_idx,
                current_bytes=self._memory_provider(),
                delta_bytes=0,
                arrest=False,
            )

        # Already arrested → return the arrest reason, no further checks.
        if self.arrested:
            current = self._memory_provider()
            return PreflightDecision(
                allowed=False,
                reason=REASON_ARRESTED,
                layer_idx=layer_idx,
                current_bytes=current,
                delta_bytes=0,
                projected_bytes=current,
                M_max_bytes=self.M_max_bytes,
                last_stab_seconds=self.last_stab_seconds,
                T_div_max_seconds=self.T_div_max_seconds,
            )

        # Time gate first — if the previous stabilization blew the budget,
        # arrest now regardless of any memory headroom.
        if (
            self.last_stab_seconds is not None
            and self.last_stab_seconds > self.T_div_max_seconds
        ):
            current = self._memory_provider()
            return self._deny(
                REASON_TIME_CEILING,
                layer_idx=layer_idx,
                current_bytes=current,
                delta_bytes=0,
                arrest=True,
            )

        # Memory gate.
        delta = division_param_delta(
            net, layer_idx, optimizer_state_per_param=self.optimizer_state_per_param
        )
        delta_bytes = delta.bytes(dtype_bytes=self.dtype_bytes)
        current = self._memory_provider()
        projected = current + delta_bytes
        if projected > self.M_max_bytes:
            return self._deny(
                REASON_MEMORY_CEILING,
                layer_idx=layer_idx,
                current_bytes=current,
                delta_bytes=delta_bytes,
                arrest=True,
            )

        self.divisions_allowed += 1
        return PreflightDecision(
            allowed=True,
            reason=REASON_OK,
            layer_idx=layer_idx,
            current_bytes=current,
            delta_bytes=delta_bytes,
            projected_bytes=projected,
            M_max_bytes=self.M_max_bytes,
            last_stab_seconds=self.last_stab_seconds,
            T_div_max_seconds=self.T_div_max_seconds,
        )

    # ----- stabilization timing -----

    def mark_stabilization_start(self) -> None:
        """Begin timing the post-division stabilization phase. Call AFTER
        net.grow_layer has returned and the optimizer has been rebuilt."""
        self._stab_start = self._time_provider()

    def mark_stabilization_end(self) -> float:
        """End timing and record the elapsed wall-clock seconds. Returns the
        elapsed value. Future preflights consult `last_stab_seconds`.

        Raises if mark_stabilization_start was not called first — that's a
        usage bug worth surfacing, not silently masking."""
        if self._stab_start is None:
            raise RuntimeError(
                "mark_stabilization_end called without a prior "
                "mark_stabilization_start"
            )
        elapsed = self._time_provider() - self._stab_start
        self._stab_start = None
        self.last_stab_seconds = float(elapsed)
        return self.last_stab_seconds

    # ----- internals -----

    def _deny(
        self,
        reason: str,
        layer_idx: int,
        current_bytes: int,
        delta_bytes: int,
        arrest: bool,
    ) -> PreflightDecision:
        if arrest:
            self.arrested = True
            self.arrest_reason = reason
        return PreflightDecision(
            allowed=False,
            reason=reason,
            layer_idx=layer_idx,
            current_bytes=current_bytes,
            delta_bytes=delta_bytes,
            projected_bytes=current_bytes + delta_bytes,
            M_max_bytes=self.M_max_bytes,
            last_stab_seconds=self.last_stab_seconds,
            T_div_max_seconds=self.T_div_max_seconds,
        )

    # ----- persistence (Orange Pi will reboot — preserve arrest state) -----

    def state_dict(self) -> dict:
        return {
            "M_max_bytes": self.M_max_bytes,
            "T_div_max_seconds": self.T_div_max_seconds,
            "optimizer_state_per_param": self.optimizer_state_per_param,
            "dtype_bytes": self.dtype_bytes,
            "arrested": self.arrested,
            "arrest_reason": self.arrest_reason,
            "last_stab_seconds": self.last_stab_seconds,
            "divisions_attempted": self.divisions_attempted,
            "divisions_allowed": self.divisions_allowed,
        }

    def load_state_dict(self, state: dict) -> None:
        self.M_max_bytes = int(state["M_max_bytes"])
        self.T_div_max_seconds = float(state["T_div_max_seconds"])
        self.optimizer_state_per_param = int(state["optimizer_state_per_param"])
        self.dtype_bytes = int(state["dtype_bytes"])
        self.arrested = bool(state["arrested"])
        self.arrest_reason = state.get("arrest_reason")
        self.last_stab_seconds = state.get("last_stab_seconds")
        self.divisions_attempted = int(state.get("divisions_attempted", 0))
        self.divisions_allowed = int(state.get("divisions_allowed", 0))
        self._stab_start = None

    def __repr__(self) -> str:
        mb = 1024 * 1024
        return (
            f"CeilingsController(M_max={self.M_max_bytes / mb:.0f}MB, "
            f"T_div_max={self.T_div_max_seconds:.0f}s, "
            f"arrested={self.arrested}"
            + (f"/{self.arrest_reason}" if self.arrested else "")
            + f", attempts={self.divisions_attempted}, "
            f"allowed={self.divisions_allowed})"
        )
