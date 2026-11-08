"""Multi-framework agent toolkit: one agent spec, three deployment targets, one eval suite."""

from mfa.spec import (
    AGENT_ENGINE,
    CLAUDE_MANAGED,
    CLOUD_RUN,
    LOCAL,
    TARGETS,
    AgentSpec,
    SkillSpec,
    ToolSpec,
    anthropic_skill,
    local_skills,
    skill,
    tool,
)

__all__ = [
    "AgentSpec", "ToolSpec", "SkillSpec", "tool", "skill", "local_skills", "anthropic_skill",
    "LOCAL", "AGENT_ENGINE", "CLOUD_RUN", "CLAUDE_MANAGED", "TARGETS",
]
