"""Target 1: run the LangGraph graph in-process."""

from __future__ import annotations

import uuid
from typing import Any

from mfa.runtime.graph import build_agent_graph, run_turn
from mfa.runtime.models import make_chat_model
from mfa.runtime.result import AgentResult
from mfa.spec import LOCAL, AgentSpec


class LocalRunner:
    target = LOCAL

    def __init__(self, spec: AgentSpec, *, model: str | None = None, chat_model: Any | None = None):
        self.spec = spec
        self.model = spec.model_for(LOCAL, model)
        # chat_model lets tests (or callers) inject any LangChain chat model.
        self.graph = build_agent_graph(spec, chat_model or make_chat_model(self.model, effort=spec.effort))

    def run(self, user_input: str, *, thread_id: str | None = None) -> AgentResult:
        result = run_turn(
            self.graph,
            user_input,
            thread_id=thread_id or uuid.uuid4().hex,
            target=self.target,
            max_iterations=self.spec.max_iterations,
        )
        result.metadata["model"] = self.model
        return result


class LocalTarget:
    name = LOCAL

    def deploy(self, spec: AgentSpec, *, model: str | None = None, **options: Any) -> dict[str, Any]:
        # Nothing to provision: building the graph validates the spec.
        build_agent_graph(spec, make_chat_model(spec.model_for(LOCAL, model), effort=spec.effort))
        return {"target": LOCAL, "model": spec.model_for(LOCAL, model)}

    def runner(self, spec: AgentSpec, *, model: str | None = None, **options: Any) -> LocalRunner:
        return LocalRunner(spec, model=model, chat_model=options.get("chat_model"))
