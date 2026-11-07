"""Offline stand-ins for the LLM, the Agent Engine remote, and the Claude Managed Agents API.

They let the test suite exercise every target's real code path (graph build,
pickling, event loop, tool execution, grading) without credentials.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class ScriptedChatModel(BaseChatModel):
    """Returns pre-scripted AIMessages in order. Each script step is either text or a tool call."""

    script: list[dict[str, Any]]
    cursor: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        step = self.script[self.cursor]
        self.cursor += 1
        if "tool" in step:
            msg = AIMessage(
                content="",
                tool_calls=[{"name": step["tool"], "args": step["args"], "id": f"call_{uuid.uuid4().hex[:8]}"}],
            )
        else:
            msg = AIMessage(content=step["text"])
        return ChatResult(generations=[ChatGeneration(message=msg)])


class FakeCMAClient:
    """Mimics client.beta.{environments,agents,sessions} enough for deploy + runner tests.

    For each session turn it replays the same script format as ScriptedChatModel:
    tool steps become ``agent.custom_tool_use`` events (and wait for the
    runner's ``user.custom_tool_result``), text steps become ``agent.message``.
    """

    def __init__(self, script: list[dict[str, Any]] | None = None):
        self.script = list(script or [])
        self.sent: list[dict[str, Any]] = []
        self.existing_envs: list[Any] = []
        self.env_creates = 0
        self.created_agents: list[dict[str, Any]] = []
        self.updated_agents: list[tuple[str, dict[str, Any]]] = []
        self.beta = SimpleNamespace(
            environments=SimpleNamespace(create=self._env_create, list=lambda **kw: list(self.existing_envs)),
            agents=SimpleNamespace(create=self._agent_create, update=self._agent_update, retrieve=self._agent_retrieve),
            sessions=SimpleNamespace(
                create=self._session_create,
                events=SimpleNamespace(stream=self._stream, send=self._send),
            ),
        )
        self._pending: list[Any] = []
        self._version = 0

    # control plane
    def _env_create(self, **kw):
        self.env_creates += 1
        return SimpleNamespace(id="env_fake", name=kw["name"])

    def _agent_create(self, **kw):
        self.created_agents.append(kw)
        self._version = 1
        return SimpleNamespace(id="agent_fake", version=self._version)

    def _agent_retrieve(self, agent_id, **kw):
        return SimpleNamespace(id=agent_id, version=7, model=SimpleNamespace(id="claude-opus-5-5"))

    def _agent_update(self, agent_id, **kw):
        self.updated_agents.append((agent_id, kw))
        self._version += 1
        return SimpleNamespace(id=agent_id, version=self._version)

    def _session_create(self, **kw):
        return SimpleNamespace(id=f"sesn_{uuid.uuid4().hex[:6]}")

    # data plane
    def _send(self, *, session_id, events):
        self.sent.extend(events)
        for e in events:
            if e["type"] in ("user.message", "user.custom_tool_result"):
                self._advance()

    def _advance(self):
        if not self.script:
            self._pending.append(_idle("end_turn"))
            return
        step = self.script.pop(0)
        if "tool" in step:
            self._pending.append(
                SimpleNamespace(type="agent.custom_tool_use", id=f"sevt_{uuid.uuid4().hex[:6]}",
                                name=step["tool"], input=step["args"])
            )
            self._pending.append(_idle("requires_action"))
        else:
            self._pending.append(
                SimpleNamespace(type="agent.message", content=[SimpleNamespace(type="text", text=step["text"])])
            )
            if step.get("end_turn", True):
                self._pending.append(_idle("end_turn"))

    def _stream(self, *, session_id):
        client = self

        class _S:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def __iter__(self):
                while client._pending:
                    yield client._pending.pop(0)

        return _S()


def _idle(reason: str):
    return SimpleNamespace(type="session.status_idle", stop_reason=SimpleNamespace(type=reason))


class FakeRemoteAgentEngine:
    """Stands in for the deployed Agent Engine: calls a real LangGraphAgentApp in-process."""

    def __init__(self, app):
        self.app = app

    def query(self, **kwargs):
        return self.app.query(**kwargs)
