"""The object that is pickled and served by Vertex AI Agent Engine.

It holds only strings (agent reference + model) so it pickles cleanly; the
LangGraph graph is built in ``set_up()`` on the Agent Engine instance, using
the same ``build_agent_graph`` as the local target. That is what makes the
local and Agent Engine runs the *same* agent.
"""

from __future__ import annotations

import uuid
from typing import Any

from mfa.spec import AGENT_ENGINE


class LangGraphAgentApp:
    def __init__(self, agent_ref: str, model: str):
        self.agent_ref = agent_ref
        self.model = model
        self._graph = None
        self._spec = None

    def set_up(self) -> None:
        from mfa.registry import load_agent
        from mfa.runtime.graph import build_agent_graph
        from mfa.runtime.models import make_chat_model

        self._spec = load_agent(self.agent_ref)
        # InMemorySaver keeps threads per serving instance. For durable
        # multi-turn state across instances, pass a persistent checkpointer
        # (e.g. AlloyDB / Cloud SQL / Firestore) here.
        self._graph = build_agent_graph(self._spec, make_chat_model(self.model))

    def query(self, *, input: str, thread_id: str | None = None) -> dict[str, Any]:
        """Run one user turn. Returns an ``AgentResult`` as a plain dict."""
        from mfa.runtime.graph import run_turn

        if self._graph is None:
            self.set_up()
        result = run_turn(
            self._graph,
            input,
            thread_id=thread_id or uuid.uuid4().hex,
            target=AGENT_ENGINE,
            recursion_limit=self._spec.max_iterations * 2 + 1,
        )
        result.metadata["model"] = self.model
        return result.model_dump()
