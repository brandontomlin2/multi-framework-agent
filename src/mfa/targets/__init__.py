"""Deployment targets. Each one turns the same AgentSpec into a running agent."""

from __future__ import annotations

from mfa.spec import AGENT_ENGINE, CLAUDE_MANAGED, LOCAL


def get_target(name: str):
    # Lazy imports: each target only needs its own SDK installed.
    if name == LOCAL:
        from mfa.targets.local import LocalTarget

        return LocalTarget()
    if name == AGENT_ENGINE:
        from mfa.targets.agent_engine import AgentEngineTarget

        return AgentEngineTarget()
    if name == CLAUDE_MANAGED:
        from mfa.targets.claude_managed import ClaudeManagedTarget

        return ClaudeManagedTarget()
    raise ValueError(f"Unknown target {name!r}; expected one of {LOCAL}, {AGENT_ENGINE}, {CLAUDE_MANAGED}")
