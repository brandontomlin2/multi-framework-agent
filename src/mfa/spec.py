"""Framework-neutral agent definition.

An ``AgentSpec`` is the single source of truth for an agent. Every target
(local LangGraph, Vertex AI Agent Engine, Claude Managed Agents) is built from
the same spec, so the agent's prompt, tools and behaviour stay identical no
matter where it is deployed. To lift-and-shift a new agent, write a new spec;
nothing in ``mfa.targets`` needs to change.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from langchain_core.tools import BaseTool, StructuredTool

# Target identifiers used across the CLI, deployment records and eval reports.
LOCAL = "local"
AGENT_ENGINE = "agent_engine"
CLAUDE_MANAGED = "claude_managed"
TARGETS = (LOCAL, AGENT_ENGINE, CLAUDE_MANAGED)


@dataclass(frozen=True)
class ToolSpec:
    """A plain Python function plus the JSON schema every framework needs."""

    name: str
    description: str
    input_schema: dict[str, Any]
    func: Callable[..., Any]

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
        )

    def invoke(self, args: dict[str, Any]) -> str:
        """Run the tool and return a string result (what every framework sends back to the model)."""
        result = self.func(**args)
        if isinstance(result, str):
            return result
        return json.dumps(result, default=str)

    def to_langchain(self) -> BaseTool:
        return StructuredTool.from_function(self.func, name=self.name, parse_docstring=True)


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
        max_iterations: Cap on model<->tool round-trips per query.
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
            return self.models[target]
        except KeyError as exc:
            raise KeyError(
                f"Agent {self.name!r} defines no model for target {target!r}; "
                f"set AgentSpec.models[{target!r}] or pass --model."
            ) from exc
