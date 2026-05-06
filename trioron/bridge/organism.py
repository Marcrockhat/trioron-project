"""BridgedOrganism — wraps a MultiBranchOrganism behind any Encoder
and dispatches its decision to a ToolDispatcher.

Pipeline:

    raw input ──► Encoder ──► L0Adapter ──► MultiBranchOrganism
                                                     │
                                                     ▼
                                            class-id argmax
                                                     │
                                                     ▼
                                            ToolDispatcher.dispatch

The class-id → tool-name mapping is owned by the bridge (not by the
donors), so donors stay tool-agnostic and reusable across deployments
with different tool registries.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional

import torch

from .base import Encoder, L0Adapter
from .tools import ToolDispatcher
from ..multibranch import MultiBranchOrganism


@dataclass
class Decision:
    """A single routing+selection result, before any tool is invoked."""
    union_class: int
    tool_name: Optional[str]
    confidence: float
    gates: Dict[str, float]    # branch_label → gate weight
    topk: List[Dict[str, Any]] # [{"union_class": c, "tool_name": s, "prob": p}, ...]


class BridgedOrganism:
    """Pluggable encoder + organism + tool dispatcher.

    Construction options:

      class_to_tool: maps a union-class index to a tool name. Classes
          missing from the map produce decisions with tool_name=None
          (the orchestrator can fall back to a default policy).

      args_resolver: optional callable
          (raw_input, tool_name, decision) -> dict of args
          used by ``act`` to populate tool args from context.
    """

    def __init__(
        self,
        encoder: Encoder,
        organism: MultiBranchOrganism,
        dispatcher: ToolDispatcher,
        class_to_tool: Optional[Mapping[int, str]] = None,
        args_resolver: Optional[
            Callable[[Any, str, "Decision"], Dict[str, Any]]
        ] = None,
        adapter: Optional[L0Adapter] = None,
        temperature: float = 1.0,
    ):
        self.encoder = encoder
        self.organism = organism
        self.dispatcher = dispatcher
        self.class_to_tool: Dict[int, str] = dict(class_to_tool or {})
        self.args_resolver = args_resolver
        self.temperature = float(temperature)
        l0_dim = self.organism.l0_W.shape[0]
        self.adapter = adapter or L0Adapter(
            encoder_dim=encoder.encode_dim,
            l0_dim=l0_dim,
            l0_seed=int(self.organism.l0_seed) if self.organism.l0_seed is not None else 42,
            activation=self.organism.l0_activation,
        )
        if encoder.encode_dim != l0_dim and self.adapter.is_identity():
            raise ValueError(
                f"Encoder dim {encoder.encode_dim} != L0 dim {l0_dim} "
                "but adapter is identity. Construct an L0Adapter or "
                "let BridgedOrganism build a random projection."
            )

    # ----- decision -----

    def decide(
        self,
        raw_input: Any,
        *,
        topk: int = 5,
    ) -> Decision:
        """Encode, project, route, argmax. Returns a Decision; does NOT
        dispatch a tool. Use ``act`` for end-to-end dispatch."""
        e = self._encode(raw_input)
        z = self.adapter(e)
        with torch.no_grad():
            logits, extras = self.organism.forward_from_z(
                z, routing="soft",
                temperature=self.temperature,
                normalize_per_branch=True,
                return_extras=True,
            )
        # Single-input convention: take row 0.
        probs = torch.softmax(logits[0], dim=-1)
        topk = min(topk, probs.numel())
        top = torch.topk(probs, k=topk)
        union = self.organism.union_classes
        gates = extras["gates"][0].tolist()
        gate_dict = {b.label: g for b, g in zip(self.organism.branches, gates)}
        topk_records = [
            {
                "union_class": int(union[int(idx)]),
                "tool_name": self.class_to_tool.get(int(union[int(idx)])),
                "prob": float(p),
            }
            for p, idx in zip(top.values.tolist(), top.indices.tolist())
        ]
        chosen = topk_records[0]
        return Decision(
            union_class=chosen["union_class"],
            tool_name=chosen["tool_name"],
            confidence=chosen["prob"],
            gates=gate_dict,
            topk=topk_records,
        )

    # ----- act -----

    def act(
        self,
        raw_input: Any,
        *,
        args: Optional[Dict[str, Any]] = None,
        topk: int = 5,
    ) -> Dict[str, Any]:
        """Decide then dispatch. ``args`` is either passed in directly
        or resolved by the configured ``args_resolver``. Returns
        ``{"decision": ..., "tool_call": {...}, "tool_result": ...}``.

        If the chosen union-class has no registered tool, returns
        the decision with ``tool_call=None`` and no dispatch."""
        decision = self.decide(raw_input, topk=topk)
        if decision.tool_name is None:
            return {"decision": decision, "tool_call": None,
                    "tool_result": None}
        if args is None and self.args_resolver is not None:
            args = self.args_resolver(raw_input, decision.tool_name, decision)
        elif args is None:
            args = {}
        call = {"tool": decision.tool_name, "args": args}
        return {
            "decision": decision,
            "tool_call": call,
            "tool_result": self.dispatcher.dispatch(call),
        }

    # ----- internals -----

    def _encode(self, raw_input: Any) -> torch.Tensor:
        e = self.encoder(raw_input)
        if e.dim() == 1:
            e = e.unsqueeze(0)
        return e


__all__ = ["BridgedOrganism", "Decision"]
