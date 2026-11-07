"""FastAPI app for any AgentSpec: A2A (the agents-cli contract) plus a JSON route for mfa evals.

Routes:
    GET  /health
    POST /a2a/<A2A_NAME>                         A2A JSON-RPC (v1.0 + v0.3 compat)
    GET  /a2a/<A2A_NAME>/.well-known/agent-card.json
    POST /mfa/query  {"input": str, "thread_id": str?} -> AgentResult JSON

``/mfa/query`` returns the same normalized result (reply + tool trajectory) as
every other target, which is what lets one eval suite grade them all.
"""

from __future__ import annotations

import asyncio
import os
import uuid

from fastapi import FastAPI
from pydantic import BaseModel

from mfa.runtime.graph import run_turn
from mfa.serving import load_root_agent
from mfa.serving.a2a import add_a2a_routes
from mfa.spec import CLOUD_RUN


class QueryRequest(BaseModel):
    input: str
    thread_id: str | None = None


def create_app() -> FastAPI:
    spec, model, graph = load_root_agent()
    app = FastAPI(title=spec.name, description=spec.description)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "agent": spec.name, "model": model}

    @app.post("/mfa/query")
    async def query(req: QueryRequest) -> dict:
        result = await asyncio.to_thread(
            run_turn, graph, req.input,
            thread_id=req.thread_id or uuid.uuid4().hex,
            target=os.environ.get("MFA_TARGET", CLOUD_RUN),
            recursion_limit=spec.max_iterations * 2 + 1,
        )
        result.metadata["model"] = model
        return result.model_dump()

    add_a2a_routes(app, spec, graph)
    return app


app = create_app()
