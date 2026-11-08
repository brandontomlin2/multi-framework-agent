"""FastAPI app for any AgentSpec: A2A (the agents-cli contract) plus a JSON route for mfa evals.

Routes:
    GET  /health
    POST /a2a/<A2A_NAME>                         A2A JSON-RPC (v1.0 + v0.3 compat)
    GET  /a2a/<A2A_NAME>/.well-known/agent-card.json
    POST /mfa/query  {"input": str, "thread_id": str?} -> AgentResult JSON

``/mfa/query`` returns the same normalized result (reply + tool trajectory) as
every other target, which is what lets one eval suite grade them all. A failed
turn is still a 200 with ``error`` set, exactly as the local runner reports it.
"""

from __future__ import annotations

import asyncio
import os
import uuid

from fastapi import FastAPI
from pydantic import BaseModel

from mfa.runtime.graph import run_turn
from mfa.runtime.result import AgentResult
from mfa.serving import load_root_agent
from mfa.serving.a2a import add_a2a_routes
from mfa.spec import CLOUD_RUN


class QueryRequest(BaseModel):
    input: str
    thread_id: str | None = None


def create_app() -> FastAPI:
    spec, model, graph = load_root_agent()
    app = FastAPI(title=spec.name, description=spec.description)
    target = os.environ.get("MFA_TARGET", CLOUD_RUN)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "agent": spec.name, "model": model}

    @app.post("/mfa/query")
    async def query(req: QueryRequest) -> dict:
        thread_id = req.thread_id or uuid.uuid4().hex
        try:
            result = await asyncio.to_thread(
                run_turn, graph, req.input, thread_id=thread_id, target=target, max_iterations=spec.max_iterations
            )
        except Exception as exc:  # run_turn has its own boundary; this is the last resort
            result = AgentResult(target=target, input=req.input, output="", error=f"{type(exc).__name__}: {exc}")
            result.metadata["thread_id"] = thread_id
        result.metadata["model"] = model
        return result.model_dump()

    add_a2a_routes(app, spec, graph)
    return app


app = create_app()
