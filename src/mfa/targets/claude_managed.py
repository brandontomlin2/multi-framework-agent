"""Target 3: Claude Managed Agents.

Managed Agents runs its own agent loop on Anthropic's side, so the LangGraph
graph is not uploaded. Instead the *spec* is translated:

* ``system_prompt``  -> the agent's ``system``
* each ``ToolSpec``  -> a ``custom`` tool with the same name/description/schema
* the spec's Claude model -> the agent's ``model``

Custom tools execute in this process: the runner answers each
``agent.custom_tool_use`` event with ``ToolSpec.run`` - the same validation,
coercion, serialization and error text the LangGraph targets use - so tool
behaviour is identical across all targets. The runner also enforces the spec's
``max_iterations`` (Managed Agents has no equivalent of LangGraph's recursion
limit) and reports the same ``iteration_limit_error`` text when it is hit.
The built-in agent toolset (bash, files, web) is left off by default so the
agent has exactly the tools the spec declares; set ``MFA_CMA_ENABLE_TOOLSET=1``
to add it.

Credentials: ``MFA_ANTHROPIC_API_KEY`` or ``ANTHROPIC_API_KEY`` (or an ``ant auth login`` profile).
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Any

from mfa import deployments
from mfa.runtime.result import AgentResult, ToolCall
from mfa.spec import CLAUDE_MANAGED, AgentSpec, ToolResult, iteration_limit_error, unknown_tool_error

DEFAULT_TURN_TIMEOUT_S = float(os.environ.get("MFA_CMA_TURN_TIMEOUT_S", "600"))


def build_agent_params(spec: AgentSpec, model: str) -> dict[str, Any]:
    """Pure translation from AgentSpec to ``client.beta.agents.create`` kwargs."""
    tools: list[dict[str, Any]] = [
        {
            "type": "custom",
            "name": t.name,
            "description": t.description,
            "input_schema": t.input_schema,
        }
        for t in spec.tools
    ]
    if os.environ.get("MFA_CMA_ENABLE_TOOLSET") == "1":
        tools.insert(
            0,
            {
                "type": "agent_toolset_20260401",
                "default_config": {"enabled": True, "permission_policy": {"type": "auto"}},
                "configs": [
                    {"name": "web_fetch", "enabled": False},
                    {"name": "web_search", "enabled": False},
                ],
            },
        )
    return {
        "name": spec.name,
        "description": spec.description,
        "model": {"id": model},
        "system": spec.system_prompt,
        "tools": tools,
        "metadata": {"mfa_agent": spec.name},
    }


def _client():
    from mfa.credentials import anthropic_client

    return anthropic_client()


class ClaudeManagedRunner:
    target = CLAUDE_MANAGED

    def __init__(self, spec: AgentSpec, record: dict[str, Any], client: Any | None = None):
        self.spec = spec
        self.record = record
        self.client = client or _client()
        self._sessions: dict[str, str] = {}  # thread_id -> session_id

    def _session_for(self, thread_id: str) -> str:
        if thread_id not in self._sessions:
            session = self.client.beta.sessions.create(
                agent={"type": "agent", "id": self.record["agent_id"], "version": self.record["agent_version"]},
                environment_id=self.record["environment_id"],
                title=f"{self.spec.name} [{thread_id[:12]}]",
                metadata={"mfa_agent": self.spec.name, "mfa_thread": thread_id[:64]},
            )
            self._sessions[thread_id] = session.id
        return self._sessions[thread_id]

    def _tool_result(self, name: str, args: dict[str, Any]) -> ToolResult:
        try:
            tool = self.spec.tool_by_name(name)
        except KeyError:
            return ToolResult(unknown_tool_error(name, [t.name for t in self.spec.tools]), True)
        return tool.run(args)

    def run(self, user_input: str, *, thread_id: str | None = None) -> AgentResult:
        thread_id = thread_id or uuid.uuid4().hex
        start = time.monotonic()
        try:
            session_id = self._session_for(thread_id)
            result = self._drive_turn(session_id, user_input, start)
        except Exception as exc:
            return AgentResult(target=self.target, input=user_input, output="", error=repr(exc))
        result.latency_s = round(time.monotonic() - start, 3)
        result.metadata.update(
            {
                "thread_id": thread_id,
                "session_id": session_id,
                "model": self.record["model"],
                "agent_version": self.record["agent_version"],
            }
        )
        return result

    def _drive_turn(self, session_id: str, user_input: str, start: float) -> AgentResult:
        sessions = self.client.beta.sessions
        calls: list[ToolCall] = []
        final_text = ""
        error: str | None = None
        rounds = 0  # model messages that called tools this turn (one message may hold several calls)
        in_round = False
        refused: list[dict[str, Any]] = []  # results for the round past the cap, sent with the interrupt
        server_retries = 0
        limit_text = iteration_limit_error(self.spec.max_iterations)

        def send(*events: dict[str, Any]) -> None:
            sessions.events.send(session_id=session_id, events=list(events))

        # Stream-first: open the stream, then send, so no early events are missed.
        with sessions.events.stream(session_id=session_id) as stream:
            send({"type": "user.message", "content": [{"type": "text", "text": user_input}]})
            for event in stream:
                if time.monotonic() - start > DEFAULT_TURN_TIMEOUT_S:
                    send({"type": "user.interrupt"})
                    error = f"turn exceeded {DEFAULT_TURN_TIMEOUT_S}s"
                    break

                if event.type == "agent.message":
                    text = "".join(b.text for b in event.content if b.type == "text")
                    if text:
                        final_text = text  # last message of the turn, same rule as LangGraph
                elif event.type == "agent.custom_tool_use":
                    # Managed Agents emits one event per tool_use block of a model message,
                    # then one requires_action idle; the idle closes the round, so parallel
                    # calls count as one round, exactly like one AIMessage in the graph.
                    if not in_round:
                        rounds += 1
                        in_round = True
                    args = dict(event.input or {})
                    past_cap = rounds > self.spec.max_iterations
                    if past_cap:
                        # Same rule as iteration_cap_route: the round past the cap is not executed.
                        output, is_error = limit_text, True
                        error = limit_text
                    else:
                        output, is_error = self._tool_result(event.name, args)
                    calls.append(ToolCall(name=event.name, args=args, output=output))
                    result_event = {
                        "type": "user.custom_tool_result",
                        "custom_tool_use_id": event.id,
                        "content": [{"type": "text", "text": output}],
                        "is_error": is_error,
                    }
                    if past_cap:
                        refused.append(result_event)  # answered together with the interrupt below
                    else:
                        send(result_event)
                elif event.type in ("agent.tool_use", "agent.mcp_tool_use"):
                    # Only present when the built-in toolset is enabled. Eval runs are
                    # unattended, so a call the server paused for approval is denied.
                    calls.append(ToolCall(name=event.name, args=dict(getattr(event, "input", {}) or {})))
                    if getattr(event, "evaluated_permission", None) == "ask":
                        send({"type": "user.tool_confirmation", "tool_use_id": event.id, "result": "deny"})
                elif event.type == "session.error":
                    err = getattr(event, "error", None)
                    retry = getattr(getattr(err, "retry_status", None), "type", None)
                    if retry == "retrying":
                        # The server retries on its own; the LangGraph targets' SDK retries
                        # are just as invisible, so this is not a failure of the turn.
                        server_retries += 1
                        continue
                    kind = getattr(err, "type", "error")
                    message = getattr(err, "message", None) or str(err if err is not None else event)
                    error = error or f"{kind}: {message}"
                elif event.type == "session.status_idle":
                    reason = event.stop_reason.type
                    if reason == "requires_action":
                        in_round = False  # all tool calls of this model message have been delivered
                        if refused:
                            # Answer every refused call so the session is not left waiting on
                            # them, and stop the turn in the same batch (an interrupted turn
                            # ends with end_turn).
                            send(*refused, {"type": "user.interrupt"})
                            refused = []
                        continue
                    if reason != "end_turn":
                        error = error or f"session stopped: {reason}"
                    break
                elif event.type == "session.status_terminated":
                    error = error or "session terminated"
                    break

        result = AgentResult(target=self.target, input=user_input, output=final_text, tool_calls=calls, error=error)
        if server_retries:
            result.metadata["server_retries"] = server_retries
        return result


def environment_params(spec: AgentSpec) -> dict[str, Any]:
    # Tools run host-side, so the sandbox needs no outbound network.
    return {"name": f"{spec.name}-env", "config": {"type": "cloud", "networking": {"type": "limited"}}}


def _ensure_environment(client: Any, spec: AgentSpec) -> str:
    """Environment names are unique per workspace, so reuse one that already exists."""
    params = environment_params(spec)
    for env in client.beta.environments.list():
        if env.name == params["name"]:
            return env.id
    return client.beta.environments.create(**params).id


def write_ant_files(spec: AgentSpec, model: str, out_dir: str | Path) -> list[Path]:
    """Write the agent + environment as files for ``ant apply`` (Anthropic's recommended control plane).

    Layout: ``<out_dir>/<agent-name>/agent.md`` (frontmatter = config, body = system prompt)
    and ``environment.yaml``. Run ``ant apply <dir>/agent.md <dir>/environment.yaml`` and then
    ``mfa register <agent> -t claude_managed --agent-id ... --environment-id ...``.
    """
    import yaml

    params = build_agent_params(spec, model)
    system = params.pop("system")
    d = Path(out_dir) / spec.name.replace("_", "-")
    d.mkdir(parents=True, exist_ok=True)
    front = yaml.safe_dump(params, sort_keys=False, allow_unicode=True)
    agent_md = d / "agent.md"
    agent_md.write_text(f"---\n{front}---\n\n{system}")
    env_yaml = d / "environment.yaml"
    env_yaml.write_text(yaml.safe_dump(environment_params(spec), sort_keys=False))
    return [agent_md, env_yaml]


def register(spec: AgentSpec, agent_id: str, environment_id: str, client: Any | None = None) -> dict[str, Any]:
    """Record an agent/environment created elsewhere (``ant apply``, Console) so mfa can run evals on it."""
    client = client or _client()
    agent = client.beta.agents.retrieve(agent_id)
    model = agent.model if isinstance(agent.model, str) else agent.model.id
    record = {
        "target": CLAUDE_MANAGED,
        "agent_id": agent.id,
        "agent_version": agent.version,
        "environment_id": environment_id,
        "model": model,
        "managed_by": "external",
        "deployed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    deployments.save(spec.name, CLAUDE_MANAGED, record)
    return record


class ClaudeManagedTarget:
    name = CLAUDE_MANAGED

    def deploy(self, spec: AgentSpec, *, model: str | None = None, **options: Any) -> dict[str, Any]:
        if spec.graph_builder is not None:
            print(
                f"note: {spec.name} uses a custom LangGraph topology; Claude Managed Agents runs "
                "its own loop, so only the system prompt and tools are deployed."
            )
        model = spec.model_for(CLAUDE_MANAGED, model)
        client = options.get("client") or _client()
        params = build_agent_params(spec, model)
        existing = deployments.load(spec.name, CLAUDE_MANAGED) or {}

        environment_id = existing.get("environment_id") or _ensure_environment(client, spec)

        if existing.get("agent_id"):
            # This deploy step owns the agent, so update unconditionally; an
            # unchanged spec returns the current version.
            agent = client.beta.agents.update(existing["agent_id"], **params)
        else:
            agent = client.beta.agents.create(**params)

        record = {
            "target": CLAUDE_MANAGED,
            "agent_id": agent.id,
            "agent_version": agent.version,
            "environment_id": environment_id,
            "model": model,
            "deployed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        deployments.save(spec.name, CLAUDE_MANAGED, record)
        return record

    def runner(self, spec: AgentSpec, *, model: str | None = None, **options: Any) -> ClaudeManagedRunner:
        record = deployments.require(spec.name, CLAUDE_MANAGED)
        return ClaudeManagedRunner(spec, record, client=options.get("client"))
