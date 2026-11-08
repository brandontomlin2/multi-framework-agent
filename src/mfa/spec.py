"""Framework-neutral agent definition.

An ``AgentSpec`` is the single source of truth for an agent. Every target
(local LangGraph, Vertex AI Agent Engine, Cloud Run, Claude Managed Agents) is
built from the same spec, so the agent's prompt, tools and behaviour stay
identical no matter where it is deployed. To lift-and-shift a new agent, write
a new spec; nothing in ``mfa.targets`` needs to change.

The turn contract
-----------------
Every target must hand the model, and the eval report, the same text for the
same tool call. ``ToolSpec.run`` is the single place where that text is made:

* arguments are validated and coerced against the tool's schema (so ``"2"``
  becomes ``2`` and ``"2026-11-08"`` becomes a ``date`` everywhere);
* a result that is not already a string is JSON-encoded;
* an exception, including invalid arguments, becomes an ``Error: ...`` string
  handed back to the model instead of aborting the turn.

LangGraph targets reach it through ``to_langchain()``; Claude Managed Agents
through the runner answering ``agent.custom_tool_use``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, NamedTuple

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import ValidationError

# Target identifiers used across the CLI, deployment records and eval reports.
LOCAL = "local"
AGENT_ENGINE = "agent_engine"
CLAUDE_MANAGED = "claude_managed"
CLOUD_RUN = "cloud_run"  # Gemini platform via a container (Google's recommended path for LangGraph)
TARGETS = (LOCAL, AGENT_ENGINE, CLOUD_RUN, CLAUDE_MANAGED)

# A target with no model of its own borrows one: both Gemini platform targets
# serve the same LangGraph graph, so a spec only has to name one Gemini model.
MODEL_FALLBACKS = {CLOUD_RUN: AGENT_ENGINE}


class ToolResult(NamedTuple):
    text: str
    is_error: bool


def format_tool_error(name: str, exc: BaseException) -> str:
    """The one error string every target hands the model when a tool call fails."""
    if isinstance(exc, ValidationError):
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or 'input'}: {e['msg']}" for e in exc.errors()
        )
        return f"Error: invalid arguments for {name}: {problems}"
    return f"Error: tool {name} failed: {type(exc).__name__}: {exc}"


def unknown_tool_error(name: str, available: list[str]) -> str:
    """The error string every target hands the model when it calls a tool the spec does not define."""
    return f"Error: unknown tool {name}; available tools: {', '.join(available) or '(none)'}"


def iteration_limit_error(max_iterations: int) -> str:
    """The error text every target reports when the model exceeds ``max_iterations`` tool rounds."""
    return f"Error: iteration limit reached: max_iterations={max_iterations}"


@dataclass(frozen=True)
class ToolSpec:
    """A plain Python function plus the JSON schema every framework needs."""

    name: str
    description: str
    input_schema: dict[str, Any]
    func: Callable[..., Any]
    # The pydantic model behind input_schema; used to validate + coerce arguments.
    args_model: Any = field(default=None, compare=False, repr=False)

    @classmethod
    def from_function(cls, func: Callable[..., Any]) -> "ToolSpec":
        # LangChain derives a JSON schema from type hints + a Google-style
        # docstring; we reuse it so there is exactly one schema per tool.
        lc_tool = StructuredTool.from_function(func, parse_docstring=True)
        schema = lc_tool.args_schema.model_json_schema()
        schema.pop("title", None)
        schema.pop("description", None)
        for prop in schema.get("properties", {}).values():
            prop.pop("title", None)
        schema.setdefault("required", [])
        return cls(
            name=lc_tool.name,
            description=lc_tool.description,
            input_schema=schema,
            func=func,
            args_model=lc_tool.args_schema,
        )

    def validate_args(self, args: dict[str, Any]) -> dict[str, Any]:
        """Validate and coerce model-supplied arguments; raises pydantic.ValidationError."""
        if self.args_model is None:
            return dict(args)
        parsed = self.args_model.model_validate(args)
        # Only pass what the model supplied, so the function's own defaults still apply.
        return {k: getattr(parsed, k) for k in type(parsed).model_fields if k in args}

    def run(self, args: dict[str, Any]) -> ToolResult:
        """Validate, run and serialize. Never raises: failures become an error string for the model."""
        try:
            result = self.func(**self.validate_args(args))
        except Exception as exc:
            return ToolResult(format_tool_error(self.name, exc), True)
        if isinstance(result, str):
            return ToolResult(result, False)
        return ToolResult(json.dumps(result, default=str, ensure_ascii=False), False)

    def invoke(self, args: dict[str, Any]) -> str:
        """The text the model sees for this call (see ``run`` for the error flag)."""
        return self.run(args).text

    def to_langchain(self) -> BaseTool:
        # The JSON schema is passed as a dict so LangChain binds the same schema the
        # other targets see but does *not* validate: validation, coercion and error
        # text all happen in run(), identically on every target.
        return StructuredTool(
            name=self.name,
            description=self.description,
            args_schema=self.input_schema,
            func=lambda **kwargs: self.invoke(kwargs),
        )


def tool(func: Callable[..., Any]) -> ToolSpec:
    """Decorator: turn a typed, docstring'd function into a portable ToolSpec."""
    return ToolSpec.from_function(func)


@dataclass
class AgentSpec:
    """Everything needed to build, deploy and evaluate one agent.

    Attributes:
        name: Stable identifier (used for deployment records and display names).
        description: One-line description shown in the target consoles.
        system_prompt: The agent's instructions; identical on every target.
        tools: Portable tools. Executed in-process for LangGraph targets and
            host-side (as custom tools) for Claude Managed Agents.
        models: Model per target. Values for LangGraph targets use the
            ``provider:model`` form understood by ``mfa.runtime.models``;
            Claude Managed Agents takes a bare Claude model ID.
        requirements: Extra pip requirements the agent's tools need at runtime
            (shipped to Agent Engine alongside the framework's own).
        eval_dataset: Path to the YAML eval cases shared by all targets.
        max_iterations: Cap on model<->tool round-trips per query, enforced on
            every target; exceeding it ends the turn with
            ``iteration_limit_error`` as the result's ``error``.
        graph_builder: Optional ``(spec, model, *, checkpointer) -> CompiledGraph``
            for agents whose LangGraph topology is not the default tool loop.
            Used by the local and Agent Engine targets. Claude Managed Agents
            runs its own loop, so only the prompt + tools carry over there.
    """

    name: str
    description: str
    system_prompt: str
    tools: list[ToolSpec]
    models: dict[str, str] = field(default_factory=dict)
    requirements: list[str] = field(default_factory=list)
    eval_dataset: str | None = None
    max_iterations: int = 10
    graph_builder: Callable[..., Any] | None = None

    def tool_by_name(self, name: str) -> ToolSpec:
        for t in self.tools:
            if t.name == name:
                return t
        raise KeyError(f"Agent {self.name!r} has no tool named {name!r}")

    def model_for(self, target: str, override: str | None = None) -> str:
        if override:
            return override
        try:
            return self.models[target] if target in self.models else self.models[MODEL_FALLBACKS[target]]
        except KeyError as exc:
            raise KeyError(
                f"Agent {self.name!r} defines no model for target {target!r}; "
                f"set AgentSpec.models[{target!r}] or pass --model."
            ) from exc
