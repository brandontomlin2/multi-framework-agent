"""Resolve an agent reference (``weather_agent`` or ``pkg.module:ATTR``) to its AgentSpec."""

from __future__ import annotations

import importlib

from mfa.spec import AgentSpec


def load_agent(ref: str) -> AgentSpec:
    module_name, _, attr = ref.partition(":")
    if "." not in module_name:
        # Short form: agents/<name>/agent.py exporting AGENT.
        module_name = f"agents.{module_name}.agent"
    module = importlib.import_module(module_name)
    spec = getattr(module, attr or "AGENT")
    if not isinstance(spec, AgentSpec):
        raise TypeError(f"{ref!r} resolved to {type(spec).__name__}, expected AgentSpec")
    return spec
