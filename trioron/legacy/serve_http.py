"""Optional FastAPI wrapper for ``trioron serve --http``.

Imported lazily — the only reason this lives in its own module is to
keep the import of ``trioron.cli`` from pulling in FastAPI at startup.
The CLI checks for the ``[serve]`` extra and surfaces a clean error
before this module is imported, so users without the extra never see
an ImportError from here.
"""
from __future__ import annotations

from typing import Any, Dict


def build_app(bridge):
    """Build a FastAPI app exposing /decide, /act, /tools, /classes."""
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    class DecideRequest(BaseModel):
        input: Any
        topk: int = 5

    class ActRequest(BaseModel):
        input: Any
        args: Dict[str, Any] | None = None
        topk: int = 5

    app = FastAPI(
        title="trioron serve",
        version="0.0.3",
        description="Bridge an absorbed trioron organism to tools.",
    )

    @app.get("/health")
    def health():
        return {
            "ok": True,
            "branches": [b.label for b in bridge.organism.branches],
            "union_classes": list(bridge.organism.union_classes),
        }

    @app.get("/tools")
    def list_tools():
        return bridge.dispatcher.to_openai_dicts()

    @app.get("/classes")
    def list_classes():
        return {
            "union_classes": list(bridge.organism.union_classes),
            "class_to_tool": bridge.class_to_tool,
        }

    @app.post("/decide")
    def decide(req: DecideRequest):
        d = bridge.decide(req.input, topk=req.topk)
        return {
            "union_class": d.union_class,
            "tool_name": d.tool_name,
            "confidence": d.confidence,
            "gates": d.gates,
            "topk": d.topk,
        }

    @app.post("/act")
    def act(req: ActRequest):
        try:
            result = bridge.act(req.input, args=req.args, topk=req.topk)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        d = result["decision"]
        return {
            "decision": {
                "union_class": d.union_class,
                "tool_name": d.tool_name,
                "confidence": d.confidence,
                "gates": d.gates,
                "topk": d.topk,
            },
            "tool_call": result["tool_call"],
            "tool_result": result["tool_result"],
        }

    return app


def run_uvicorn(app, port: int) -> None:
    """Blocking. Started in a thread by trioron serve."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(port), log_level="info")
