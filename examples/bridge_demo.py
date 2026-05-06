"""End-to-end bridge demo.

Wires the existing chained-15 multi-branch organism (digits + fashion
donors, both at L0 seed 42) behind a stub encoder, a tool dispatcher,
and a BridgedOrganism. The real-encoder version (sentence-transformers
+ user-domain donors) is the same code path with the stub swapped for
``trioron.bridge.encoders.text.TextEncoder``.

This demo runs WITHOUT any bridge optional extras installed — the stub
encoder just feeds a fake 128-dim feature vector that lands directly
in L0 space (identity adapter), so we can validate the orchestration
plumbing without downloading sentence-transformers.

Run:
    # 1. Train donors first if you haven't:
    trioron train --donor digits  --out donor_digits.pt
    trioron train --donor fashion --out donor_fashion.pt
    trioron absorb --donors donor_digits.pt,donor_fashion.pt \\
                   --out organism.pt
    # 2. Run the demo:
    python3 examples/bridge_demo.py
"""
from __future__ import annotations
import os
import sys
from typing import Sequence

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trioron.bridge import (
    Encoder, L0Adapter, ToolDispatcher, BridgedOrganism, Decision,
)
from trioron.cli import _load_organism


# ---------------------------------------------------------------------
# A stub encoder — so the demo runs without bridge extras installed.
# Swap this for `from trioron.bridge.encoders.text import TextEncoder`
# in real deployments.
# ---------------------------------------------------------------------


class StubEncoder:
    """Fake encoder for demo purposes — emits a deterministic 128-dim
    vector keyed by a coarse 'kind' field. In a real deployment this
    would be a sentence-transformer / CLIP / Whisper output."""

    encode_dim = 128

    def __call__(self, batch):
        # Accept either a single string or a list of strings.
        single = isinstance(batch, str)
        items = [batch] if single else list(batch)
        out = []
        for s in items:
            # Coarse "kind" — caller is expected to pass strings like
            # "digit-style:7" or "fashion-style:shirt" so the stub can
            # produce a vector in the right region of code-space. In
            # reality this would be the encoder's learned hidden state.
            seed = abs(hash(s)) % (2**31 - 1)
            gen = torch.Generator().manual_seed(seed)
            v = torch.randn(self.encode_dim, generator=gen)
            out.append(v / v.norm().clamp_min(1e-8))
        return torch.stack(out, dim=0)


# ---------------------------------------------------------------------
# Tool dispatcher — register two tools via decorator, one via JSON.
# ---------------------------------------------------------------------


tools = ToolDispatcher()


@tools.tool
def render_digit(class_id: int) -> str:
    """Render an ASCII placeholder for a digit class."""
    return f"[digit-renderer] showing class {class_id}"


@tools.tool
def describe_fashion(class_id: int) -> str:
    """Return a one-line description of a fashion class."""
    LABELS = {
        10: "T-shirt/top", 11: "Trouser", 12: "Pullover", 13: "Dress",
        14: "Coat", 15: "Sandal", 16: "Shirt", 17: "Sneaker",
        18: "Bag", 19: "Ankle boot",
    }
    return f"[fashion-describer] class {class_id}: {LABELS.get(class_id, '?')}"


# ---------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------


def main():
    organism_path = "organism.pt"
    if not os.path.exists(organism_path):
        organism_path_alt = "/tmp/smoke_organism.pt"
        if os.path.exists(organism_path_alt):
            organism_path = organism_path_alt
        else:
            print("Need an organism checkpoint. Run:")
            print("  trioron train --donor digits  --out donor_digits.pt")
            print("  trioron train --donor fashion --out donor_fashion.pt")
            print("  trioron absorb --donors donor_digits.pt,donor_fashion.pt \\")
            print("                 --out organism.pt")
            return 2

    organism = _load_organism(organism_path)
    print(f"Loaded organism with {len(organism.branches)} branch(es); "
          f"union = {organism.union_classes}")

    # Class-id → tool-name mapping. Digits → render_digit, fashion →
    # describe_fashion. In a richer deployment each class might map to
    # its own tool (e.g. one tool per class).
    class_to_tool = {}
    for c in organism.union_classes:
        if c < 10:
            class_to_tool[c] = "render_digit"
        else:
            class_to_tool[c] = "describe_fashion"

    encoder = StubEncoder()
    bridged = BridgedOrganism(
        encoder=encoder,
        organism=organism,
        dispatcher=tools,
        class_to_tool=class_to_tool,
        # Resolve args as {"class_id": <chosen union class>} from the
        # decision object. Real deployments would pull args from the
        # raw input or the encoder's intermediate state.
        args_resolver=lambda raw, name, dec: {"class_id": dec.union_class},
        temperature=1.0,
    )
    print(f"BridgedOrganism wired:")
    print(f"  encoder        = {type(encoder).__name__}  "
          f"(encode_dim={encoder.encode_dim})")
    print(f"  adapter        = {'identity' if bridged.adapter.is_identity() else 'random projection'}")
    print(f"  tools          = {tools.names()}")
    print()

    # Three demo inputs (the stub encoder uses a hash of the string to
    # produce deterministic codes — actual class-routing would depend on
    # the organism's archive, not the stub's hash).
    inputs = [
        "digit-style:7",
        "fashion-style:shirt",
        "ambiguous:?",
    ]
    print("=" * 78)
    print("Routing each input through encoder → adapter → organism → tool")
    print("=" * 78)
    for inp in inputs:
        result = bridged.act(inp)
        d: Decision = result["decision"]
        print(f"\ninput          = {inp!r}")
        print(f"  routed class = {d.union_class}  conf={d.confidence:.4f}")
        print(f"  branch gates = {d.gates}")
        print(f"  tool         = {d.tool_name}")
        if result["tool_call"] is not None:
            print(f"  tool call    = {result['tool_call']}")
            print(f"  tool result  = {result['tool_result']}")
        else:
            print("  (no tool registered for this class — would fall back)")

    print()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
