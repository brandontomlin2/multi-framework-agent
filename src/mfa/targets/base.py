"""Interface every deployment target implements."""

from __future__ import annotations

from typing import Any, Protocol

from mfa.runtime.result import AgentResult
from mfa.spec import AgentSpec


class Runner(Protocol):
    """Sends one user turn to a running agent and returns a normalized result.

    ``thread_id`` groups turns into one conversation (LangGraph thread,
    Agent Engine thread, or Claude Managed Agents session).
    """

    target: str

    def run(self, user_input: str, *, thread_id: str | None = None) -> AgentResult: ...


class Target(Protocol):
    name: str

    def deploy(self, spec: AgentSpec, *, model: str | None = None, **options: Any) -> dict[str, Any]:
        """Create or update the agent on this target; returns the deployment record."""

    def runner(self, spec: AgentSpec, *, model: str | None = None, **options: Any) -> Runner:
        """Return a Runner bound to the deployed agent."""
