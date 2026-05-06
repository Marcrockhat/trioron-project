# Trioron Bridge

The bridge layer (`trioron.bridge`) wraps a multi-branch trioron
organism behind a pluggable encoder + tool dispatcher, turning it
into a cross-modal orchestrator. The shared-L0 invariant from
absorption (paper §3.10) generalizes to **shared encoder + shared
random projection** across all donors and recipients in a population.

## Architecture

```
raw input ──► Encoder ──► L0Adapter ──► MultiBranchOrganism
                                                 │
                                                 ▼
                                        class-id argmax
                                                 │
                                                 ▼
                                        ToolDispatcher.dispatch
```

Three pieces, each independently swappable:

| component | role | reference impls |
|---|---|---|
| `Encoder` | raw input → fixed-dim feature vector | text / image / audio |
| `L0Adapter` | encoder dim → trioron L0 code-space | random projection (seed-bound) or identity |
| `ToolDispatcher` | named tools with JSON-schema args | OpenAI/Anthropic-compatible |

## Install with optional extras

The reference encoders are NOT installed by default — pick the
modality you need:

```bash
pip install trioron[bridge-text]    # sentence-transformers
pip install trioron[bridge-image]   # open-clip-torch + Pillow
pip install trioron[bridge-audio]   # openai-whisper
pip install trioron[bridge-all]     # all three
```

Each extra is a transitive dependency on the upstream package; if you
don't install one, the corresponding `trioron.bridge.encoders.*`
module raises a clear `ImportError` with the pip command needed.

## Registering tools

### Path 1 — JSON-schema (OpenAI / Anthropic compatible)

```python
from trioron.bridge import ToolDispatcher

tools = ToolDispatcher()

tools.register_json_schema(
    name="describe_scene",
    description="Generate a brief description of the scene in the image.",
    parameters={
        "type": "object",
        "properties": {
            "image_path": {"type": "string"},
            "max_words": {"type": "integer"},
        },
        "required": ["image_path"],
    },
    fn=lambda image_path, max_words=20: f"...{image_path}...",
)

# Drop directly into an LLM API call:
print(tools.to_openai_dicts())
```

### Path 2 — Python decorator (type-hint inference)

```python
from trioron.bridge import ToolDispatcher

tools = ToolDispatcher()

@tools.tool
def describe_scene(image_path: str, max_words: int = 20) -> str:
    """Generate a brief description of the scene in the image."""
    ...

@tools.tool(name="add", description="Add two numbers.")
def add_numbers(a: int, b: int) -> int:
    return a + b
```

Both paths produce identical `Tool` records and dispatch behavior. Use
whichever feels right for the project — JSON-schema for migration from
existing OpenAI/Anthropic agent code, decorator for Python-native
authoring.

## Example: text encoder + LLM-style tool dispatch

Replace `StubEncoder` from `examples/bridge_demo.py` with the real
text encoder once `trioron[bridge-text]` is installed:

```python
from trioron.bridge import (
    BridgedOrganism, ToolDispatcher, L0Adapter,
)
from trioron.bridge.encoders.text import TextEncoder
from trioron.cli import _load_organism

# 1. Frozen text encoder (sentence-transformers all-MiniLM-L6-v2 by default)
encoder = TextEncoder()  # encode_dim = 384

# 2. Load a multi-branch organism (any number of donors)
organism = _load_organism("organism.pt")

# 3. L0 adapter: text encoder is 384-dim, organism's L0 is 128-dim, so
#    a random projection is applied. Seeded by the organism's L0 seed
#    so all bridged organisms in this population agree on the projection.
adapter = L0Adapter(
    encoder_dim=encoder.encode_dim,
    l0_dim=organism.l0_W.shape[0],
    l0_seed=organism.l0_seed,
)

# 4. Tools — register whatever your deployment exposes
tools = ToolDispatcher()

@tools.tool
def lookup_user_profile(user_id: str) -> dict:
    """Return the profile for a given user."""
    return {"user_id": user_id, "name": "...", "preferences": []}

@tools.tool
def query_database(table: str, where: str) -> list:
    """Run a read-only SQL-like query."""
    return [{"...": "..."}]

# 5. Map each branch's class-coverage to a tool. In this toy example,
#    classes 0..9 (digits branch) → user-profile lookup,
#    classes 10..19 (fashion branch) → database query.
class_to_tool = {}
for c in organism.union_classes:
    class_to_tool[c] = "lookup_user_profile" if c < 10 else "query_database"

# 6. Args resolver — populate tool args from the raw input + decision
def resolve_args(raw_input, tool_name, decision):
    if tool_name == "lookup_user_profile":
        return {"user_id": str(decision.union_class)}
    if tool_name == "query_database":
        return {"table": "items", "where": f"class_id={decision.union_class}"}
    return {}

# 7. Wire it together
bridged = BridgedOrganism(
    encoder=encoder, adapter=adapter,
    organism=organism, dispatcher=tools,
    class_to_tool=class_to_tool,
    args_resolver=resolve_args,
)

# 8. Use it
result = bridged.act("Tell me about user #42")
print(f"chose tool: {result['tool_call']['tool']}")
print(f"tool result: {result['tool_result']}")
```

## Example: image encoder

```python
from trioron.bridge.encoders.image import ImageEncoder

encoder = ImageEncoder(model_name="ViT-B-32", pretrained="openai")
# encode_dim = 512

# Pass either a path string or a PIL.Image:
features = encoder("path/to/image.png")            # (1, 512)
features = encoder(["img1.png", "img2.png"])       # (2, 512)

# Wire into BridgedOrganism the same way as TextEncoder.
```

## Example: audio encoder

```python
from trioron.bridge.encoders.audio import AudioEncoder

encoder = AudioEncoder(model_name="tiny")  # encode_dim = 384

features = encoder("audio.wav")            # (1, 384)
features = encoder(["a.wav", "b.wav"])     # (2, 384)
```

Whisper's audio encoder is mean-pooled across time so the result is a
single fixed-dim vector per clip.

## When to NOT use the bridge

The bridge adds two layers of indirection (encoder + adapter) over a
multi-branch organism. If your input is already in the same shape as
trioron's L0 expects (e.g., 28×28 grayscale flattened, like
chained-15), use the `trioron infer` CLI directly — no bridge needed.

The bridge is for the **deployment story**: trioron as an inner-voice
orchestrator that decides which tool / persona / sub-network handles a
given input from any modality.

## What about non-trioron donors?

The bridge encoder slot is for the **input substrate** (frozen feature
extractor common to all donors and recipients). Donors themselves are
still trioron — paste-and-go absorption only works between
architecturally compatible branches under the shared-substrate
invariant. To absorb a non-trioron model (e.g., a transformer LM),
you would first need to either wrap it as a frozen feature extractor
(then the donor on top of it is trioron-shaped) or distill it into a
trioron-shaped donor; both paths are covered in the discussion section
of the paper.

## Reference

| module | exports |
|---|---|
| `trioron.bridge` | `Encoder`, `L0Adapter`, `Tool`, `ToolDispatcher`, `BridgedOrganism`, `Decision` |
| `trioron.bridge.base` | `Encoder` Protocol, `L0Adapter` |
| `trioron.bridge.tools` | `Tool` dataclass, `ToolDispatcher` |
| `trioron.bridge.organism` | `BridgedOrganism`, `Decision` |
| `trioron.bridge.encoders.text` | `TextEncoder` |
| `trioron.bridge.encoders.image` | `ImageEncoder` |
| `trioron.bridge.encoders.audio` | `AudioEncoder` |

End-to-end smoke test (no extras required):

```bash
python3 examples/bridge_demo.py
```
