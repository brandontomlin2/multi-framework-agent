"""Offline stand-ins for the LLM, the Agent Engine remote, and the Claude Managed Agents API.

They let the test suite exercise every target's real code path (graph build,
pickling, event loop, tool execution, grading) without credentials.

Both fakes replay the same *script* format, so one scenario can be run through
every target and the results compared:

* ``{"tool": name, "args": {...}}``          the model calls a tool
* ``{"calls": [{"tool": ..., "args": ...}, ...]}``  several tool calls in one model message
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

from mfa.spec import BUILTIN_TOOL_NAMES


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
        calls = step_calls(step)
        if calls:
            msg = AIMessage(
                content=step.get("text", ""),
                tool_calls=[{"name": c["tool"], "args": c["args"], "id": f"call_{uuid.uuid4().hex[:8]}"} for c in calls],
            )
        else:
            msg = AIMessage(content=step["text"])
        return ChatResult(generations=[ChatGeneration(message=msg)])


def step_calls(step: dict[str, Any]) -> list[dict[str, Any]]:
    """The tool calls a script step makes (zero, one, or several)."""
    if "calls" in step:
        return list(step["calls"])
    return [{"tool": step["tool"], "args": step["args"]}] if "tool" in step else []


class FakeCMAClient:
    """Mimics client.beta.{environments,agents,sessions} enough for deploy + runner tests.

    For each session turn it replays the script above the way the service
    streams it: a custom tool step becomes one ``agent.custom_tool_use`` event per
    call followed by a single ``session.status_idle(requires_action)``; the model
    only continues (next script step) once every call has a ``user.custom_tool_result``;
    text steps become ``agent.message``; error steps become ``session.error``.
    ``user.interrupt`` stops a running model step; while tool calls are still
    unresolved it does nothing, as on the service.

    Built-in tools (``read``, ``glob``, ``grep``, ``bash``, ...) follow the
    ``always_ask`` protocol the mfa deploy configures: ``agent.tool_use`` with
    ``evaluated_permission="ask"``, then ``requires_action``; a
    ``user.tool_confirmation`` allow runs the call through the same skill tools
    the LangGraph graph binds (pass ``spec``) and emits ``agent.tool_result``; a
    deny emits an error result carrying the deny message. A tool not in
    ``builtin_enabled`` is refused by the server (``evaluated_permission="deny"``).
    """

    def __init__(self, script: list[dict[str, Any]] | None = None, *, spec: Any | None = None,
                 builtin_enabled: set[str] | None = None):
        self.script = list(script or [])
        self.sent: list[dict[str, Any]] = []
        self.batches: list[list[dict[str, Any]]] = []  # each events=[...] request as sent
        self.existing_envs: list[Any] = []
        self.env_creates = 0
        self.created_agents: list[dict[str, Any]] = []
        self.updated_agents: list[tuple[str, dict[str, Any]]] = []
        self.sessions_created: list[dict[str, Any]] = []
        self.uploaded: dict[str, list[str]] = {}  # skill slug -> uploaded file names
        self.skill_versions: dict[str, int] = {}
        self.beta = SimpleNamespace(
            environments=SimpleNamespace(create=self._env_create, list=lambda **kw: list(self.existing_envs)),
            agents=SimpleNamespace(create=self._agent_create, update=self._agent_update, retrieve=self._agent_retrieve),
            sessions=SimpleNamespace(
                create=self._session_create,
                events=SimpleNamespace(stream=self._stream, send=self._send),
            ),
        )
        self.skills = SimpleNamespace(
            create=self._skill_create,
            list=lambda **kw: [],
            versions=SimpleNamespace(create=self._skill_version_create),
        )
        self._pending: list[Any] = []
        self._unresolved: set[str] = set()
        self._builtin_calls: dict[str, tuple[str, dict[str, Any]]] = {}
        self._version = 0
        self._builtin_tools: dict[str, Any] = {}
        if spec is not None:
            from mfa.skills import skill_tools

            self._builtin_tools = {t.name: t for t in skill_tools(spec)}
        self.builtin_enabled = set(builtin_enabled) if builtin_enabled is not None else set(self._builtin_tools)

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
        self.sessions_created.append(kw)
        return SimpleNamespace(id=f"sesn_{uuid.uuid4().hex[:6]}")

    # skills API
    def _skill_create(self, *, files, display_name=None, **kw):
        names = [f[0] if isinstance(f, tuple) else str(f) for f in files]
        slug = names[0].split("/")[0]
        self.uploaded.setdefault(slug, []).extend(names)
        self.skill_versions[slug] = 1
        return SimpleNamespace(id=f"skill_fake_{slug}", latest_version_id="skillver_1", display_name=display_name or slug)

    def _skill_version_create(self, skill_id, *, files, **kw):
        names = [f[0] if isinstance(f, tuple) else str(f) for f in files]
        slug = names[0].split("/")[0]
        self.uploaded.setdefault(slug, []).extend(names)
        self.skill_versions[slug] = self.skill_versions.get(slug, 1) + 1
        return SimpleNamespace(id=f"skillver_{self.skill_versions[slug]}", skill_id=skill_id, name=slug)

    # data plane
    def _send(self, *, session_id, events):
        self.sent.extend(events)
        self.batches.append(list(events))
        interrupt = any(e["type"] == "user.interrupt" for e in events)
        for e in events:
            if e["type"] == "user.message":
                self._advance()
            elif e["type"] == "user.custom_tool_result":
                self._unresolved.discard(e["custom_tool_use_id"])
                if not self._unresolved and not interrupt:
                    self._advance()  # every call answered: the model runs again
            elif e["type"] == "user.tool_confirmation":
                self._confirm(e)
                if not self._unresolved and not interrupt:
                    self._advance()
        if interrupt and not self._unresolved:
            # Stops the running step before it produced anything; the turn ends idle.
            self._pending.clear()
            self._pending.append(_idle("end_turn"))

    def _confirm(self, e):
        tool_use_id = e["tool_use_id"]
        name, args = self._builtin_calls.pop(tool_use_id, (None, {}))
        self._unresolved.discard(tool_use_id)
        if e["result"] == "deny":
            text, is_error = e.get("deny_message") or f"Permission to use {name} has been denied.", True
        else:
            # The server runs the built-in tool in the sandbox; here the same ToolSpec the graph binds runs it.
            text, is_error = self._builtin_tools[name].run(args)
        self._pending.append(_tool_result(tool_use_id, text, is_error))

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
        calls = step_calls(step)
        if calls:
            for c in calls:
                ev_id = f"sevt_{uuid.uuid4().hex[:6]}"
                if c["tool"] in BUILTIN_TOOL_NAMES:
                    if c["tool"] not in self.builtin_enabled:
                        # Not enabled on the agent: the server refuses before any policy applies.
                        self._pending.append(SimpleNamespace(
                            type="agent.tool_use", id=ev_id, name=c["tool"], input=c["args"],
                            evaluated_permission="deny", evaluation=None))
                        self._pending.append(_tool_result(ev_id, f"Permission to use {c['tool']} has been denied.", True))
                        continue
                    self._unresolved.add(ev_id)
                    self._builtin_calls[ev_id] = (c["tool"], dict(c["args"]))
                    self._pending.append(SimpleNamespace(
                        type="agent.tool_use", id=ev_id, name=c["tool"], input=c["args"],
                        evaluated_permission="ask", evaluation=SimpleNamespace(type="always_ask")))
                else:
                    self._unresolved.add(ev_id)
                    self._pending.append(
                        SimpleNamespace(type="agent.custom_tool_use", id=ev_id, name=c["tool"], input=c["args"])
                    )
            if self._unresolved:
                self._pending.append(_idle("requires_action", sorted(self._unresolved)))
            else:
                self._advance()  # every call was refused by the server: the model goes on
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


def _idle(reason: str, event_ids: list[str] | None = None):
    stop = SimpleNamespace(type=reason)
    if event_ids is not None:
        stop.event_ids = event_ids
    return SimpleNamespace(type="session.status_idle", stop_reason=stop)


def _tool_result(tool_use_id: str, text: str, is_error: bool):
    return SimpleNamespace(type="agent.tool_result", id=f"sevt_{uuid.uuid4().hex[:6]}", tool_use_id=tool_use_id,
                           content=[SimpleNamespace(type="text", text=text)], is_error=is_error)


class FakeRemoteAgentEngine:
    """Stands in for the deployed Agent Engine: calls a real LangGraphAgentApp in-process."""

    def __init__(self, app):
        self.app = app

    def query(self, **kwargs):
        return self.app.query(**kwargs)
