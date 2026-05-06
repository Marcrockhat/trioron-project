"""Tool registry + dispatcher for the trioron bridge.

Two registration interfaces:

  1. JSON-schema (default — OpenAI / Anthropic compatible).
     The registry stores a tool spec following the function-calling
     convention shared across major LLM APIs:

         {"name": "describe_scene",
          "description": "Generate a brief description ...",
          "parameters": {<JSON Schema>}}

     plus a Python callable that implements the tool. Reviewers
     familiar with OpenAI / Anthropic agent code recognize this
     immediately.

  2. Python decorator with type-hint inference (alternative).
     Author tools as plain Python functions; the decorator builds the
     JSON schema from type hints and the docstring. Best of both
     worlds — JSON for interop, Python ergonomics for authoring.

Both paths land in the same `ToolDispatcher` and produce identical
behavior at dispatch time.
"""
from __future__ import annotations
import inspect
import json
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, List, Mapping, Optional, get_args, get_origin,
    get_type_hints, Union,
)


# ---------------------------------------------------------------------
# Tool record — what the registry stores
# ---------------------------------------------------------------------


@dataclass
class Tool:
    """One registered tool: name + JSON-schema parameters + callable."""
    name: str
    description: str
    parameters: Dict[str, Any]   # JSON Schema describing the args object
    fn: Callable[..., Any]

    def to_openai_dict(self) -> Dict[str, Any]:
        """Render in the OpenAI / Anthropic tools schema shape."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


# ---------------------------------------------------------------------
# Type-hint → JSON Schema (decorator alt)
# ---------------------------------------------------------------------


_PRIMITIVE_TO_JSON = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def _type_to_schema(tp) -> Dict[str, Any]:
    """Convert a Python type annotation to a (very small) JSON Schema
    fragment. Supports str/int/float/bool, Optional[T], List[T], and a
    fallback to string for unrecognized types."""
    origin = get_origin(tp)
    if origin is Union:
        # Optional[T] = Union[T, None]; treat as the inner type.
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return _type_to_schema(args[0])
        return {"type": "string"}  # heterogenous — skip strict typing
    if origin in (list, List):
        item_tp = get_args(tp)[0] if get_args(tp) else str
        return {"type": "array", "items": _type_to_schema(item_tp)}
    if tp in _PRIMITIVE_TO_JSON:
        return {"type": _PRIMITIVE_TO_JSON[tp]}
    return {"type": "string"}


def _build_schema_from_signature(fn: Callable) -> Dict[str, Any]:
    """Build a JSON Schema 'object' fragment from a function signature."""
    hints = get_type_hints(fn)
    sig = inspect.signature(fn)
    properties: Dict[str, Any] = {}
    required: List[str] = []
    for name, param in sig.parameters.items():
        if name == "self" or name == "cls":
            continue
        ann = hints.get(name, str)
        properties[name] = _type_to_schema(ann)
        if param.default is inspect.Parameter.empty:
            required.append(name)
    schema: Dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


# ---------------------------------------------------------------------
# Registry / dispatcher
# ---------------------------------------------------------------------


class ToolDispatcher:
    """Registry + dispatcher for tools available to a bridged trioron
    organism. Use either ``register_json_schema`` (default; ugly but
    standard) or the ``@dispatcher.tool`` decorator (clean; type-hint
    inference)."""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    # ----- registration -----

    def register_json_schema(
        self,
        name: str,
        description: str,
        parameters: Mapping[str, Any],
        fn: Callable[..., Any],
    ) -> Tool:
        """Register a tool by passing the JSON schema directly. Use
        when migrating from existing OpenAI / Anthropic tool definitions
        or when you want the schema to live separately from the code."""
        if name in self._tools:
            raise ValueError(f"tool '{name}' already registered")
        tool = Tool(name=name, description=description,
                    parameters=dict(parameters), fn=fn)
        self._tools[name] = tool
        return tool

    def tool(
        self,
        _fn: Optional[Callable] = None,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """Decorator. Schema is inferred from type hints; description
        from the docstring (first line). Use as ``@dispatcher.tool`` or
        ``@dispatcher.tool(name="custom_name")``.

        Example:
            @dispatcher.tool
            def describe_scene(image_path: str, max_words: int = 20) -> str:
                \"\"\"Brief description of the scene in the image.\"\"\"
                ...
        """
        def deco(fn: Callable) -> Callable:
            tool_name = name or fn.__name__
            doc = (description or (fn.__doc__ or "").strip().split("\n")[0]
                   or fn.__name__)
            schema = _build_schema_from_signature(fn)
            self.register_json_schema(
                name=tool_name, description=doc,
                parameters=schema, fn=fn,
            )
            return fn
        if _fn is not None and callable(_fn):
            return deco(_fn)
        return deco

    # ----- lookup / dispatch -----

    def names(self) -> List[str]:
        return list(self._tools)

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"tool '{name}' not registered")
        return self._tools[name]

    def to_openai_dicts(self) -> List[Dict[str, Any]]:
        """Render the whole registry in the OpenAI / Anthropic tools
        format — drop directly into LLM API calls."""
        return [t.to_openai_dict() for t in self._tools.values()]

    def to_json(self) -> str:
        return json.dumps(self.to_openai_dicts(), indent=2)

    def dispatch(
        self,
        call: Mapping[str, Any],
    ) -> Any:
        """Invoke a tool. ``call`` is ``{"tool": <name>, "args": {...}}``.

        Returns whatever the tool callable returns. Raises KeyError /
        TypeError if the call doesn't match a registered tool / its
        argument schema.
        """
        if "tool" not in call:
            raise KeyError("tool call missing 'tool' field")
        name = call["tool"]
        args = call.get("args") or {}
        tool = self.get(name)
        return tool.fn(**args)


__all__ = ["Tool", "ToolDispatcher"]
