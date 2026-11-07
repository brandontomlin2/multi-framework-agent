"""Multi-framework agent toolkit: one agent spec, three deployment targets, one eval suite."""

from mfa.spec import AGENT_ENGINE, CLAUDE_MANAGED, LOCAL, TARGETS, AgentSpec, ToolSpec, tool

__all__ = ["AgentSpec", "ToolSpec", "tool", "LOCAL", "AGENT_ENGINE", "CLAUDE_MANAGED", "TARGETS"]
