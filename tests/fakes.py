"""Offline stand-ins for the LLM, the Agent Engine remote, and the Claude Managed Agents API.

They let the test suite exercise every target's real code path (graph build,
pickling, event loop, tool execution, grading) without credentials.

Both fakes replay the same *script* format, so one scenario can be run through
every target and the results compared:

* ``{"tool": name, "args": {...}}``          the model calls a tool
* ``{"text": "..."}``                         the model answers
* ``{"text": "...", "tool": name, "args": {...}}``  text and a tool call in one message
* ``{"error": {"type": ..., "retry_status": ..., "message": ...}}``  (CMA only) a
  ``session.error`` event before the next step
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


class ScriptedChatModel(BaseChatModel):
    """Returns pre-scripted AIMessages in order and records what it was given.

    ``bound_tools`` holds the tools passed to ``bind_tools`` and ``seen_messages``
    every prompt, so tests can assert the system prompt and tool schemas the
    graph really sends, not just the scripted replies.
    """

    script: list[dict[str, Any]]
    cursor: int = 0
    bound_tools: list[Any] = Field(default_factory=list)
    seen_messages: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        self.bound_tools = list(tools)
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.seen_messages.append(list(messages))
        if self.cursor >= len(self.script):
            raise RuntimeError(f"ScriptedChatModel: script exhausted after {self.cursor} steps")
        step = self.script[self.cursor]
        self.cursor += 1
        step = {k: v for k, v in step.items() if k != "error"}  # CMA-only steps are no-ops here
        if "tool" in step:
            msg = AIMessage(
                content=step.get("text", ""),
                tool_calls=[{"name": step["tool"], "args": step["args"], "id": f"call_{uuid.uuid4().hex[:8]}"}],
            )
        else:
            msg = AIMessage(content=step["text"])
        return ChatResult(generations=[ChatGeneration(message=msg)])


class FakeCMAClient:
    """Mimics client.beta.{environments,agents,sessions} enough for deploy + runner tests.

    For each session turn it replays the script above: tool steps become
    ``agent.custom_tool_use`` events (and wait for the runner's
    ``user.custom_tool_result``), text steps become ``agent.message``, error
    steps become ``session.error``. ``user.interrupt`` ends the turn.
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
            if e["type"] == "user.interrupt":
                self._pending.clear()
                self._pending.append(_idle("end_turn"))
            elif e["type"] in ("user.message", "user.custom_tool_result"):
                self._advance()

    def _advance(self):
        if not self.script:
            self._pending.append(_idle("end_turn"))
            return
        step = self.script.pop(0)
        if "error" in step:
            err = step["error"]
            self._pending.append(
                SimpleNamespace(
                    type="session.error",
                    id=f"sevt_{uuid.uuid4().hex[:6]}",
                    error=SimpleNamespace(
                        type=err.get("type", "unknown_error"),
                        message=err.get("message", "boom"),
                        retry_status=SimpleNamespace(type=err.get("retry_status", "exhausted")),
                    ),
                )
            )
            if err.get("retry_status", "exhausted") == "retrying":
                self._advance()  # the server retries and the turn goes on
            else:
                self._pending.append(_idle("retries_exhausted"))
            return
        if step.get("text"):
            self._pending.append(
                SimpleNamespace(type="agent.message", content=[SimpleNamespace(type="text", text=step["text"])])
            )
        if "tool" in step:
            self._pending.append(
                SimpleNamespace(type="agent.custom_tool_use", id=f"sevt_{uuid.uuid4().hex[:6]}",
                                name=step["tool"], input=step["args"])
            )
            self._pending.append(_idle("requires_action"))
        elif step.get("end_turn", True):
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
