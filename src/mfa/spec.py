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

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
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

# Claude Managed Agents' built-in toolset (agent_toolset_20260401). The LangGraph
# targets mirror the file tools for skills under the same names, so an agent's own
# tools may not use them.
BUILTIN_TOOL_NAMES = frozenset({"bash", "read", "write", "edit", "glob", "grep", "web_fetch", "web_search"})
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")


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


# --- Agent Skills -----------------------------------------------------------
#
# A skill is a folder with SKILL.md (YAML frontmatter + instructions) and optional
# scripts/, references/ and assets/ (the Agent Skills format, agentskills.io). The
# model sees only each skill's name, description and location until a task matches,
# then reads SKILL.md and the files it references. Claude Managed Agents attaches
# skills natively; the LangGraph targets get the same behaviour from mfa.skills.

_SKILL_NAME = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
_SKILL_KEYS = {"name", "description", "license", "allowed-tools", "metadata", "compatibility"}
_SKIP_FILES = {"__pycache__", "node_modules", ".DS_Store"}


@dataclass(frozen=True)
class SkillSpec:
    """One Agent Skill: a local folder (``source="custom"``) or an Anthropic-managed skill."""

    name: str
    description: str
    source: str = "custom"  # "custom" | "anthropic"
    path: Path | None = None  # custom: the folder holding SKILL.md
    version: str | None = None  # anthropic: optional pin; custom: set by the deploy, not the author
    license: str | None = None
    compatibility: str | None = None
    allowed_tools: tuple[str, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict, compare=False)

    @property
    def is_local(self) -> bool:
        return self.path is not None

    def files(self) -> list[tuple[str, bytes]]:
        """Every file in the folder as (relative posix path, bytes), sorted; caches and a root evals/ excluded."""
        if self.path is None:
            return []
        out: list[tuple[str, bytes]] = []
        for p in sorted(self.path.rglob("*")):
            rel = p.relative_to(self.path)
            if not p.is_file() or p.suffix == ".pyc" or _SKIP_FILES & set(rel.parts):
                continue
            if rel.parts[0] == "evals":
                continue
            out.append((rel.as_posix(), p.read_bytes()))
        return out

    def content_hash(self) -> str:
        return hash_skill_files(self.files())

    @property
    def has_scripts(self) -> bool:
        return self.path is not None and any(p.is_file() for p in (self.path / "scripts").glob("*"))


def parse_skill_frontmatter(text: str) -> dict[str, Any]:
    """The YAML block between the leading ``---`` lines of a SKILL.md, validated like Anthropic's quick_validate."""
    import yaml

    if not text.startswith("---"):
        raise ValueError("SKILL.md must start with a '---' YAML frontmatter block")
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise ValueError("SKILL.md frontmatter is not closed with '---'")
    data = yaml.safe_load(parts[1]) or {}
    if not isinstance(data, dict):
        raise ValueError("SKILL.md frontmatter must be a YAML mapping")
    unknown = set(data) - _SKILL_KEYS
    if unknown:
        raise ValueError(f"SKILL.md frontmatter has unexpected keys: {sorted(unknown)}")
    name, description = data.get("name"), data.get("description")
    if not isinstance(name, str) or not _SKILL_NAME.match(name) or len(name) > 64:
        raise ValueError(f"skill name {name!r} must be 1-64 lowercase letters, digits and single hyphens")
    if not isinstance(description, str) or not description.strip() or len(description) > 1024:
        raise ValueError(f"skill {name!r} needs a description of at most 1024 characters")
    if "<" in description or ">" in description:
        raise ValueError(f"skill {name!r}: description may not contain '<' or '>'")
    if "compatibility" in data and len(str(data["compatibility"])) > 500:
        raise ValueError(f"skill {name!r}: compatibility must be at most 500 characters")
    return data


def skill(path: str | Path) -> SkillSpec:
    """Load a skill folder. The frontmatter name must equal the folder name (it is the skill's slug everywhere)."""
    folder = Path(path).resolve()
    skill_md = folder / "SKILL.md"
    if not skill_md.is_file():
        raise FileNotFoundError(f"{folder} has no SKILL.md")
    fm = parse_skill_frontmatter(skill_md.read_text(encoding="utf-8"))
    if fm["name"] != folder.name:
        raise ValueError(f"skill name {fm['name']!r} must match its folder name {folder.name!r}")
    tools = fm.get("allowed-tools") or ()
    if isinstance(tools, str):
        tools = tuple(tools.split())
    meta = fm.get("metadata") or {}
    return SkillSpec(
        name=fm["name"],
        description=fm["description"].strip(),
        source="custom",
        path=folder,
        license=fm.get("license"),
        compatibility=fm.get("compatibility"),
        allowed_tools=tuple(tools),
        metadata={str(k): str(v) for k, v in meta.items()} if isinstance(meta, dict) else {},
    )


def local_skills(folder: str | Path) -> list[SkillSpec]:
    """Every ``<folder>/<name>/SKILL.md`` one level deep (the layout Claude Managed Agents scans too)."""
    root = Path(folder)
    if not root.is_dir():
        return []
    return [skill(p.parent) for p in sorted(root.glob("*/SKILL.md"))]


def hash_skill_files(files: list[tuple[str, bytes]]) -> str:
    """Content hash of a skill's ``files()`` listing (path and bytes of every file, in order)."""
    h = hashlib.sha256()
    for rel, data in files:
        h.update(rel.encode())
        h.update(b"\0")
        h.update(data)
        h.update(b"\0")
    return h.hexdigest()


def anthropic_skill(skill_id: str, version: str | None = None) -> SkillSpec:
    """A skill Anthropic publishes (``xlsx``, ``docx``, ``pptx``, ``pdf``). Attached on Claude Managed Agents;
    the LangGraph targets can only mirror it from a local copy (``MFA_ANTHROPIC_SKILLS_DIR``)."""
    return SkillSpec(name=skill_id, description="Anthropic-managed skill", source="anthropic", path=None, version=version)


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
            ``iteration_limit_error`` as the result's ``error``. A round is one
            assistant message, however many tool calls it holds; reads of skill
            files count like any other round.
        graph_builder: Optional ``(spec, model, *, checkpointer) -> CompiledGraph``
            for agents whose LangGraph topology is not the default tool loop.
            Used by the local and Agent Engine targets. Claude Managed Agents
            runs its own loop, so only the prompt + tools carry over there.
        skills: Agent Skills the agent may use (``local_skills(...)`` and/or
            ``anthropic_skill(...)``). Attached natively on Claude Managed Agents;
            on the LangGraph targets the model gets the same name/description/
            location listing plus read-only ``read``/``glob``/``grep`` tools
            confined to the skill folders.
        allow_skill_scripts: Also expose ``bash``, limited on every target to a
            skill's own ``scripts/`` files. Off by default.
        effort: Model effort level (``low`` .. ``max``) applied on every target
            that supports it (Claude directly, Gemini approximated as a thinking
            budget).
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
    skills: list[SkillSpec] = field(default_factory=list)
    allow_skill_scripts: bool = False
    effort: str | None = None

    def __post_init__(self) -> None:
        clashes = sorted(t.name for t in self.tools if t.name in BUILTIN_TOOL_NAMES)
        if clashes:
            raise ValueError(
                f"Agent {self.name!r}: tool names {clashes} are reserved for the built-in toolset "
                f"({', '.join(sorted(BUILTIN_TOOL_NAMES))}); rename them."
            )
        if self.effort is not None and self.effort not in EFFORT_LEVELS:
            raise ValueError(f"Agent {self.name!r}: effort must be one of {EFFORT_LEVELS}, not {self.effort!r}")
        names = [sk.name for sk in self.skills]
        if len(set(names)) != len(names):
            raise ValueError(f"Agent {self.name!r}: skill names must be unique, got {names}")

    def tool_by_name(self, name: str) -> ToolSpec:
        """One of the agent's own tools (skill tools are separate: see ``runtime_tools``)."""
        for t in self.tools:
            if t.name == name:
                return t
        raise KeyError(f"Agent {self.name!r} has no tool named {name!r}")

    def runtime_tools(self) -> list[ToolSpec]:
        """The agent's tools plus the skill file tools the LangGraph targets bind."""
        from mfa.skills import skill_tools

        return [*self.tools, *skill_tools(self)]

    def runtime_tool_names(self) -> list[str]:
        return [t.name for t in self.runtime_tools()]

    @property
    def local_skills(self) -> list[SkillSpec]:
        return [sk for sk in self.skills if sk.is_local]

    def skill_by_name(self, name: str) -> SkillSpec:
        for sk in self.skills:
            if sk.name == name:
                return sk
        raise KeyError(f"Agent {self.name!r} has no skill named {name!r}")

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
