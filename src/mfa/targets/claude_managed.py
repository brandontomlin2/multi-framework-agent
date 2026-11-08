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

Skills: local skill folders are uploaded with the Skills API at deploy time (a
new version whenever their content changes) and attached to the agent; the
built-in ``read``/``glob``/``grep`` (and ``bash`` when the spec allows scripts)
are enabled with the ``always_ask`` policy, so every built-in call pauses the
session and the runner approves it with the same :class:`mfa.skills.SkillSandbox`
rules the LangGraph targets enforce, counting it toward ``max_iterations`` too.

Credentials: ``MFA_ANTHROPIC_API_KEY`` or ``ANTHROPIC_API_KEY`` (or an ``ant auth login`` profile).
"""

from __future__ import annotations

import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from mfa import deployments
from mfa.runtime.result import AgentResult, ToolCall
from mfa.skills import SkillSandbox, skill_root_of, skills_used
from mfa.spec import (
    CLAUDE_MANAGED,
    AgentSpec,
    ToolResult,
    format_tool_error,
    iteration_limit_error,
    unknown_tool_error,
)

DEFAULT_TURN_TIMEOUT_S = float(os.environ.get("MFA_CMA_TURN_TIMEOUT_S", "600"))


def model_params(spec: AgentSpec, model: str) -> dict[str, Any]:
    """``model`` as an object so effort (and optional speed / inference geo) travel with the agent."""
    m: dict[str, Any] = {"id": model}
    if spec.effort:
        m["effort"] = spec.effort
    if speed := os.environ.get("MFA_CMA_SPEED"):
        m["speed"] = speed
    if geo := os.environ.get("MFA_CMA_INFERENCE_GEO"):
        m["inference_geo"] = geo
    return m


SKILL_TOOLS = ("read", "glob", "grep")


def toolset_params(spec: AgentSpec) -> dict[str, Any] | None:
    """The built-in toolset entry, or None when the agent needs none.

    With skills, only the file tools are on and every call is ``always_ask`` so the
    runner can apply the sandbox rules and the iteration cap before the server runs
    it. ``MFA_CMA_ENABLE_TOOLSET=1`` keeps its meaning (all file/bash tools on, web
    off, policy ``auto``): a parity-breaking knob for experiments.
    """
    if os.environ.get("MFA_CMA_ENABLE_TOOLSET") == "1":
        return {
            "type": "agent_toolset_20260401",
            "default_config": {"enabled": True, "permission_policy": {"type": "auto"}},
            "configs": [{"name": "web_fetch", "enabled": False}, {"name": "web_search", "enabled": False}],
        }
    if not spec.skills:
        return None
    names = [*SKILL_TOOLS, *(["bash"] if spec.allow_skill_scripts else [])]
    return {
        "type": "agent_toolset_20260401",
        "default_config": {"enabled": False, "permission_policy": {"type": "always_ask"}},
        "configs": [
            *({"name": n, "enabled": True, "permission_policy": {"type": "always_ask"}} for n in names),
            {"name": "web_fetch", "enabled": False},
            {"name": "web_search", "enabled": False},
        ],
    }


def skill_params(spec: AgentSpec, uploaded: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """The agent's ``skills`` array. Custom skills need the ids the deploy recorded; ``mfa show`` prints placeholders."""
    refs: list[dict[str, Any]] = []
    for sk in spec.skills:
        if sk.source == "anthropic":
            ref: dict[str, Any] = {"type": "anthropic", "skill_id": sk.name}
            if sk.version:
                ref["version"] = sk.version
        else:
            rec = (uploaded or {}).get(sk.name)
            ref = {"type": "custom", "skill_id": rec["skill_id"] if rec else f"<uploaded at deploy: {sk.name}>"}
            if rec and rec.get("version"):
                ref["version"] = rec["version"]
        refs.append(ref)
    return refs


def build_agent_params(spec: AgentSpec, model: str, *, uploaded_skills: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
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
    if toolset := toolset_params(spec):
        tools.insert(0, toolset)
    params: dict[str, Any] = {
        "name": spec.name,
        "description": spec.description,
        "model": model_params(spec, model),
        "system": spec.system_prompt,
        "tools": tools,
        "metadata": {"mfa_agent": spec.name},
    }
    if spec.skills:
        params["skills"] = skill_params(spec, uploaded_skills)
    return params


def sync_skills(client: Any, spec: AgentSpec, existing: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Upload local skills that are new or changed; return ``{name: {skill_id, version, sha256, files}}``.

    Files are posted as ``<skill-name>/<relative path>`` (the Skills API wants one
    top-level directory holding SKILL.md; that directory is the skill's slug). A
    skill whose content hash is unchanged is not re-uploaded. Set
    ``MFA_CMA_SKILL_VERSION=latest`` to attach ``latest`` instead of pinning the
    uploaded version id.
    """
    known = dict((existing or {}).get("skills", {}))
    pin_latest = os.environ.get("MFA_CMA_SKILL_VERSION") == "latest"
    out: dict[str, dict[str, Any]] = {}
    for sk in spec.local_skills:
        digest = sk.content_hash()
        files = [(f"{sk.name}/{rel}", data) for rel, data in sk.files()]
        prev = known.get(sk.name)
        if prev is None:
            created = client.skills.create(files=files, display_name=sk.name)
            skill_id, version = created.id, created.latest_version_id
        elif prev.get("sha256") != digest:
            created = client.skills.versions.create(prev["skill_id"], files=files)
            skill_id, version = prev["skill_id"], created.id
        else:
            skill_id, version = prev["skill_id"], prev.get("version_id") or prev.get("version")
        out[sk.name] = {
            "skill_id": skill_id,
            "version_id": version,
            "version": "latest" if pin_latest else version,
            "sha256": digest,
            "files": [rel for rel, _ in sk.files()],
        }
    return out


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
        # Decides whether a built-in call the server pauses for confirmation may run:
        # the same path and script rules the LangGraph targets enforce.
        self.sandbox = SkillSandbox.for_spec(spec)

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
        queued: list[dict[str, Any]] = []  # confirmations and refused results, sent at the round boundary
        pending: dict[str, int] = {}  # built-in tool_use id -> index in calls (output filled by tool_result)
        server_retries = server_denials = builtin_errors = 0
        skill_root: str | None = None
        limit_text = iteration_limit_error(self.spec.max_iterations)

        def send(*events: dict[str, Any]) -> None:
            sessions.events.send(session_id=session_id, events=list(events))

        def open_round() -> bool:
            nonlocal rounds, in_round
            if not in_round:
                rounds += 1
                in_round = True
            return rounds > self.spec.max_iterations

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
                    past_cap = open_round()
                    args = dict(event.input or {})
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
                        queued.append(result_event)  # answered together with the interrupt below
                    else:
                        send(result_event)
                elif event.type in ("agent.tool_use", "agent.mcp_tool_use"):
                    # Built-in (or MCP) tools run on the server. With the always_ask policy the
                    # call waits for our confirmation, which applies the sandbox rules and the
                    # iteration cap; its output arrives later as agent.tool_result.
                    past_cap = open_round()
                    args = dict(getattr(event, "input", {}) or {})
                    call = ToolCall(name=event.name, args=args)
                    calls.append(call)
                    pending[event.id] = len(calls) - 1
                    if event.name == "read" and skill_root is None:
                        skill_root = skill_root_of(str(args.get("file_path", "")))
                    permission = getattr(event, "evaluated_permission", None)
                    if past_cap:
                        call.output = limit_text
                        error = limit_text
                        if permission == "ask":
                            queued.append(_confirmation(event.id, "deny", limit_text))
                    elif permission == "ask":
                        try:
                            self.sandbox.authorize(event.name, args)
                            queued.append(_confirmation(event.id, "allow"))
                        except Exception as exc:
                            call.output = format_tool_error(event.name, exc)
                            queued.append(_confirmation(event.id, "deny", call.output))
                    elif permission == "deny":
                        call.output = f"Permission to use {event.name} has been denied."
                        server_denials += 1
                elif event.type in ("agent.tool_result", "agent.mcp_tool_result"):
                    idx = pending.get(getattr(event, "tool_use_id", None))
                    if idx is not None:
                        if getattr(event, "is_error", False):
                            builtin_errors += 1
                        if calls[idx].output is None:
                            calls[idx].output = _result_text(getattr(event, "content", None))
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
                        if queued:
                            # One request answers every paused call of the message; when the
                            # round was past the cap it also stops the turn (an interrupted
                            # turn ends with end_turn).
                            send(*queued, *([{"type": "user.interrupt"}] if error == limit_text else []))
                            queued = []
                        continue
                    if reason != "end_turn":
                        error = error or f"session stopped: {reason}"
                    break
                elif event.type == "session.status_terminated":
                    error = error or "session terminated"
                    break

        result = AgentResult(target=self.target, input=user_input, output=final_text, tool_calls=calls, error=error)
        for key, value in (
            ("server_retries", server_retries),
            ("server_denials", server_denials),
            ("builtin_errors", builtin_errors),
            ("skill_root", skill_root),
            ("skills_used", skills_used(calls)),
        ):
            if value:
                result.metadata[key] = value
        return result


def _confirmation(tool_use_id: str, result: str, deny_message: str | None = None) -> dict[str, Any]:
    event: dict[str, Any] = {"type": "user.tool_confirmation", "tool_use_id": tool_use_id, "result": result}
    if deny_message:
        event["deny_message"] = deny_message
    return event


def _result_text(content: Any) -> str:
    parts = []
    for block in content or []:
        kind = getattr(block, "type", None)
        parts.append(getattr(block, "text", "") if kind == "text" else f"[{kind}]")
    return "".join(parts)


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

    record = deployments.load(spec.name, CLAUDE_MANAGED) or {}
    params = build_agent_params(spec, model, uploaded_skills=record.get("skills"))
    system = params.pop("system")
    d = Path(out_dir) / spec.name.replace("_", "-")
    d.mkdir(parents=True, exist_ok=True)
    front = yaml.safe_dump(params, sort_keys=False, allow_unicode=True)
    agent_md = d / "agent.md"
    agent_md.write_text(f"---\n{front}---\n\n{system}")
    env_yaml = d / "environment.yaml"
    env_yaml.write_text(yaml.safe_dump(environment_params(spec), sort_keys=False))
    paths = [agent_md, env_yaml]
    for sk in spec.local_skills:
        # The skill folders travel with the files; upload them with the Skills API (or
        # `mfa deploy -t claude_managed`) and put the ids in agent.md's skills list.
        dest = d / "skills" / sk.name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(sk.path, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        paths.append(dest / "SKILL.md")
    return paths


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
    attached = getattr(agent, "skills", None)
    if attached:
        record["agent_skills"] = [
            {k: getattr(a, k, None) for k in ("type", "skill_id", "version")} if not isinstance(a, dict) else a
            for a in attached
        ]
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
        existing = deployments.load(spec.name, CLAUDE_MANAGED) or {}
        uploaded = sync_skills(client, spec, existing)
        params = build_agent_params(spec, model, uploaded_skills=uploaded)

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
        if spec.effort:
            record["effort"] = spec.effort
        if uploaded:
            record["skills"] = uploaded
        if toolset := toolset_params(spec):
            record["toolset"] = [c["name"] for c in toolset["configs"] if c.get("enabled")]
        if existing.get("skill_root"):
            record["skill_root"] = existing["skill_root"]
        deployments.save(spec.name, CLAUDE_MANAGED, record)
        return record

    def runner(self, spec: AgentSpec, *, model: str | None = None, **options: Any) -> ClaudeManagedRunner:
        record = deployments.require(spec.name, CLAUDE_MANAGED)
        return ClaudeManagedRunner(spec, record, client=options.get("client"))
