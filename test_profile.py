"""Tests for trioron.profile — the 4-preset regime API.

Covers:
  - dataclass construction + frozen-ness
  - active-profile global state with set_active / use() context manager
  - TrioronLayer construction consults the active profile when kwargs
    are None; explicit kwargs always override
  - 4 named presets carry the documented defaults
  - re_apply_after_donor_load semantics: v1 load under a named regime
    restores the regime's branch_activation; under OPEN it leaves the
    v1 'identity' override in place (backward-compat)
"""

from __future__ import annotations

import pytest
import torch

from trioron.node import TrioronLayer
import trioron.profile as tp
from trioron.profile import (
    TrioronProfile,
    REASONING,
    CLASSIFICATION,
    EDGE,
    OPEN,
    PRESETS,
)


# ---------- dataclass + presets ----------

def test_profile_is_frozen():
    raised = False
    try:
        REASONING.branch_activation = "tanh"  # type: ignore[misc]
    except Exception:
        raised = True
    assert raised, "TrioronProfile must be frozen"


def test_reasoning_preset_values():
    assert REASONING.name == "reasoning"
    assert REASONING.branch_activation == "quad"
    assert REASONING.B_max == 8
    assert REASONING.allow_grow_node
    assert REASONING.allow_grow_branch
    assert REASONING.allow_insert_layer
    assert REASONING.memory_cap_bytes is None
    assert REASONING.re_apply_after_donor_load


def test_classification_preset_disables_axis5_memory():
    assert CLASSIFICATION.branch_activation == "identity"
    assert CLASSIFICATION.B_max == 1
    assert not CLASSIFICATION.allow_grow_branch
    assert CLASSIFICATION.allow_grow_node  # width growth stays on
    assert CLASSIFICATION.memory_cap_bytes is None


def test_edge_preset_caps_hardware():
    assert EDGE.branch_activation == "identity"
    assert EDGE.B_max == 1
    assert not EDGE.allow_grow_branch
    assert EDGE.memory_cap_bytes is not None and EDGE.memory_cap_bytes > 0
    assert EDGE.time_cap_seconds is not None and EDGE.time_cap_seconds > 0


def test_open_preset_matches_pre_profile_defaults():
    """OPEN must reproduce the construction defaults present before the
    profile API existed — branch_activation='quad', B_max=8, no caps,
    and re_apply_after_donor_load=False so the v1 silent-override
    behavior is unchanged."""
    assert OPEN.branch_activation == "quad"
    assert OPEN.B_max == 8
    assert OPEN.memory_cap_bytes is None
    assert OPEN.time_cap_seconds is None
    assert OPEN.re_apply_after_donor_load is False


def test_presets_dict_holds_all_four():
    assert set(PRESETS.keys()) == {"reasoning", "classification", "edge", "open"}
    assert PRESETS["reasoning"] is REASONING
    assert PRESETS["open"] is OPEN


# ---------- active-profile state ----------

def test_default_active_profile_is_open():
    # Reset to neutral and verify the fallback path.
    TrioronProfile._active = None
    assert TrioronProfile.active() is OPEN


def test_set_active_installs_profile():
    try:
        TrioronProfile.set_active(EDGE)
        assert TrioronProfile.active() is EDGE
    finally:
        TrioronProfile.set_active(OPEN)


def test_use_context_manager_restores_previous():
    TrioronProfile.set_active(OPEN)
    with TrioronProfile.use(REASONING):
        assert TrioronProfile.active() is REASONING
        with TrioronProfile.use(EDGE):
            assert TrioronProfile.active() is EDGE
        assert TrioronProfile.active() is REASONING
    assert TrioronProfile.active() is OPEN


def test_use_restores_on_exception():
    TrioronProfile.set_active(OPEN)
    try:
        with TrioronProfile.use(REASONING):
            assert TrioronProfile.active() is REASONING
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert TrioronProfile.active() is OPEN


# ---------- TrioronLayer consults active profile ----------

def test_layer_default_under_open_profile_matches_legacy():
    """Under the default (OPEN) profile, a layer constructed with no
    Axis 5 kwargs gets branch_activation='quad', B_max=8 — byte-for-
    byte identical to pre-profile construction."""
    TrioronProfile.set_active(OPEN)
    layer = TrioronLayer(fan_in=4, n_nodes=2)
    assert layer.branch_activation == "quad"
    assert layer.B_max == 8


def test_layer_under_edge_profile_picks_up_constrained_defaults():
    try:
        with TrioronProfile.use(EDGE):
            layer = TrioronLayer(fan_in=4, n_nodes=2)
            assert layer.branch_activation == "identity"
            assert layer.B_max == 1
            assert layer.branch_weight.shape == (2, 1)
    finally:
        # Verify the active profile was restored.
        assert TrioronProfile.active() is OPEN


def test_explicit_kwargs_override_active_profile():
    with TrioronProfile.use(EDGE):
        # EDGE says identity / B_max=1, but the caller asks for tanh / 4.
        layer = TrioronLayer(
            fan_in=4, n_nodes=2,
            branch_activation="tanh", B_max=4,
        )
        assert layer.branch_activation == "tanh"
        assert layer.B_max == 4


def test_layer_under_reasoning_profile_picks_up_quad():
    with TrioronProfile.use(REASONING):
        layer = TrioronLayer(fan_in=4, n_nodes=2)
        assert layer.branch_activation == "quad"
        assert layer.B_max == 8


def test_layer_under_classification_profile_disables_dendrite():
    with TrioronProfile.use(CLASSIFICATION):
        layer = TrioronLayer(fan_in=4, n_nodes=2)
        assert layer.branch_activation == "identity"
        assert layer.B_max == 1


# ---------- re_apply_after_donor_load ----------

def _make_v1_state_dict() -> dict:
    """Build a state_dict that simulates a pre-Axis-5 donor: present in
    1.0 era, missing every 2.0 buffer."""
    layer = TrioronLayer(fan_in=4, n_nodes=3, activation="relu")
    sd = layer.state_dict()
    for k in (
        "branch_id", "branch_weight", "branch_weight_anchor",
        "fisher_branch_weight", "B_per_node", "internal_stress",
        "branch_utility", "dendrite_orphan",
        "input_sources", "input_archived",
        "axonal_gain", "axonal_gain_anchor",
    ):
        sd.pop(k, None)
    return sd


def test_v1_load_under_open_keeps_silent_override_to_identity():
    """OPEN.re_apply_after_donor_load=False → v1 load flips branch_activation
    to 'identity' and that flip is left in place (backward-compat with the
    pre-profile semantics)."""
    sd = _make_v1_state_dict()
    with TrioronProfile.use(OPEN):
        dst = TrioronLayer(fan_in=4, n_nodes=3, activation="relu")
        # The construct-time default under OPEN is 'quad'.
        assert dst.branch_activation == "quad"
        dst.load_state_dict(sd)
        # OPEN doesn't re-apply, so the v1 silent override wins.
        assert dst.branch_activation == "identity"


def test_v1_load_under_reasoning_restores_profile_branch_activation():
    """REASONING.re_apply_after_donor_load=True → after the v1 load
    auto-flips to 'identity', the profile's choice ('quad') is restored."""
    sd = _make_v1_state_dict()
    with TrioronProfile.use(REASONING):
        dst = TrioronLayer(fan_in=4, n_nodes=3, activation="relu")
        dst.load_state_dict(sd)
        assert dst.branch_activation == "quad"


def test_v1_load_under_classification_restores_identity_explicitly():
    """CLASSIFICATION's branch_activation is 'identity' AND re_apply=True.
    The v1 flip and the re-apply both land on 'identity' — no net change
    but the semantic is now 'classification regime explicitly wants
    identity', not 'v1 silent override happens to match'."""
    sd = _make_v1_state_dict()
    with TrioronProfile.use(CLASSIFICATION):
        dst = TrioronLayer(fan_in=4, n_nodes=3, activation="relu")
        dst.load_state_dict(sd)
        assert dst.branch_activation == "identity"


def test_2_0_state_dict_load_unaffected_by_re_apply_flag():
    """A 2.0 state_dict carries branch_id, so the v1 fallback path is
    never hit. re_apply_after_donor_load is moot in that case."""
    src = TrioronLayer(
        fan_in=4, n_nodes=3, activation="relu",
        branch_activation="sigmoid",
    )
    sd = src.state_dict()
    with TrioronProfile.use(REASONING):
        dst = TrioronLayer(fan_in=4, n_nodes=3, activation="relu",
                           branch_activation="tanh")
        dst.load_state_dict(sd)
        # The construct-time 'tanh' stays; the 2.0 load doesn't tug it.
        # (branch_activation isn't in state_dict at all — it's a config
        # attribute, not a buffer.)
        assert dst.branch_activation == "tanh"


# ---------- GrowthTrigger profile gate ----------

def _make_warm_trigger_at_conditions() -> "GrowthTrigger":
    """Build a GrowthTrigger and hand-fill its histories with values
    that satisfy all three conditions, so observe()'s next call will
    have conditions_met=True. Returns the trigger ready for one
    observation."""
    from trioron.triggers import GrowthTrigger
    trig = GrowthTrigger(latent_dim=8, window=4)
    # Histories must be full before warmup clears. window=4 → need
    # 2*window=8 loss samples, 4 rank samples, 4 grad samples. We
    # feed values that produce conditions_met on the NEXT observation:
    #   loss: high stable plateau (small improvement < eps_loss).
    #   rank: near latent_dim (saturated).
    #   grad: in [g_min, g_max] (stable).
    # Pre-fill via direct deque manipulation so we can construct the
    # state cleanly without running 8 fake observations.
    trig._loss_hist.extend([1.0] * 8)
    trig._rank_hist.extend([8.0] * 4)
    trig._grad_hist.extend([1e-2] * 4)
    trig._t = 8
    return trig


def test_growth_trigger_fires_under_open_profile():
    from trioron.triggers import GrowthTrigger
    TrioronProfile.set_active(OPEN)
    trig = _make_warm_trigger_at_conditions()
    state = trig.observe(loss=1.0, hidden=torch.eye(8), grad_norm=1e-2)
    assert state.conditions_met
    assert state.fire, "OPEN profile must permit growth → fire follows conditions"


def test_growth_trigger_suppressed_under_no_grow_profile():
    """A profile with allow_grow_node=False must suppress fire even
    when all three conditions are met. conditions_met still reports
    True so logs can distinguish 'didn't fire because conditions' vs
    'didn't fire because policy'."""
    no_grow = TrioronProfile(
        name="no_grow",
        branch_activation="quad",
        B_max=8,
        allow_grow_node=False,
        allow_grow_branch=True,
        allow_insert_layer=True,
    )
    TrioronProfile.set_active(no_grow)
    trig = _make_warm_trigger_at_conditions()
    state = trig.observe(loss=1.0, hidden=torch.eye(8), grad_norm=1e-2)
    assert state.conditions_met, "conditions still observable"
    assert not state.fire, "profile gate must suppress fire"


# ---------- internal_frustration_candidates profile gate ----------

def test_internal_frustration_candidates_active_under_open():
    TrioronProfile.set_active(OPEN)
    layer = TrioronLayer(fan_in=4, n_nodes=3)
    with torch.no_grad():
        layer.internal_stress.copy_(torch.tensor([0.20, 0.01, 0.15]))
    cands = layer.internal_frustration_candidates(
        threshold=0.05, overall_saliency_ceiling=1.0,
    )
    assert cands == [0, 2]  # both above threshold, sorted descending


def test_internal_frustration_candidates_empty_under_no_branch_profile():
    """CLASSIFICATION and EDGE both forbid grow_branch — the candidate
    generator must return [] regardless of internal_stress values."""
    layer = TrioronLayer(fan_in=4, n_nodes=3)
    with torch.no_grad():
        layer.internal_stress.copy_(torch.tensor([0.50, 0.50, 0.50]))
    with TrioronProfile.use(CLASSIFICATION):
        assert layer.internal_frustration_candidates() == []
    with TrioronProfile.use(EDGE):
        assert layer.internal_frustration_candidates() == []
    with TrioronProfile.use(REASONING):
        cands = layer.internal_frustration_candidates(
            threshold=0.05, overall_saliency_ceiling=1.0,
        )
        assert cands == [0, 1, 2]


def test_grow_branch_callable_even_under_no_branch_profile():
    """The structural mutator grow_branch stays callable directly
    under any profile — the policy gate lives at the candidate
    generator, not the mutator. This lets explicit callers (manual
    probes, tests) operate the substrate regardless of regime."""
    with TrioronProfile.use(CLASSIFICATION):
        # CLASSIFICATION has B_max=1, so grow_branch refuses
        # immediately. Build a layer with B_max=4 by explicit kwarg
        # override, then call grow_branch under the same profile.
        layer = TrioronLayer(
            fan_in=4, n_nodes=1, B_max=4, branch_activation="quad",
        )
        new_b = layer.grow_branch(node_idx=0, source_cols=[1])
        assert new_b == 1
        assert layer.B_per_node[0].item() == 2


# ---------- CeilingsController profile fallback ----------

def test_ceilings_controller_uses_profile_caps_when_none():
    from trioron.ceilings import CeilingsController
    with TrioronProfile.use(EDGE):
        c = CeilingsController()  # both args None → fall to profile
        assert c.M_max_bytes == EDGE.memory_cap_bytes
        assert c.T_div_max_seconds == EDGE.time_cap_seconds


def test_ceilings_controller_falls_back_to_uncapped_under_open():
    """OPEN profile has both caps None → controller should fall back
    to effectively-uncapped sentinels (sys.maxsize / inf)."""
    import math
    import sys as _sys
    from trioron.ceilings import CeilingsController
    with TrioronProfile.use(OPEN):
        c = CeilingsController()
        assert c.M_max_bytes == _sys.maxsize
        assert math.isinf(c.T_div_max_seconds)


def test_ceilings_controller_explicit_args_override_profile():
    from trioron.ceilings import CeilingsController
    with TrioronProfile.use(EDGE):
        c = CeilingsController(M_max_bytes=10_000, T_div_max_seconds=5.0)
        assert c.M_max_bytes == 10_000
        assert c.T_div_max_seconds == 5.0


# ---------- test isolation hygiene ----------

@pytest.fixture(autouse=True)
def _reset_active_profile_after_each_test():
    """Some tests in this module mutate the active profile directly via
    set_active(). Restore OPEN after each so we don't leak into other
    test files."""
    yield
    TrioronProfile.set_active(OPEN)
