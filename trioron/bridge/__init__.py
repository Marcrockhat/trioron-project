"""Cross-modal bridge for trioron.

Encoders translate raw input (text, image, audio) into a fixed-dim
feature vector that lands in trioron's shared L0 code space. Tools
turn trioron's output into structured external actions.

Public API:
    Encoder              — Protocol any frozen encoder can satisfy.
    L0Adapter            — Projects encoder output into L0 code space.
    ToolDispatcher       — Tool registry; JSON-schema (default) or
                           Python decorator with type-hint inference.
    Tool                 — Single tool record.
    BridgedOrganism      — Encoder + MultiBranchOrganism + dispatcher.
    Decision             — One routing+selection record.

Modality reference encoders (gated behind optional pip extras; import
errors surface a clear ``pip install trioron[bridge-...]`` hint):

    trioron.bridge.encoders.text.TextEncoder    (sentence-transformers)
    trioron.bridge.encoders.image.ImageEncoder  (open-clip-torch)
    trioron.bridge.encoders.audio.AudioEncoder  (openai-whisper)
"""
from .base import Encoder, L0Adapter
from .tools import Tool, ToolDispatcher
from .organism import BridgedOrganism, Decision

__all__ = [
    "Encoder", "L0Adapter",
    "Tool", "ToolDispatcher",
    "BridgedOrganism", "Decision",
]
