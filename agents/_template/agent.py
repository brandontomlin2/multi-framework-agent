"""Copy this folder to agents/<your_agent>/, rename, and edit. Nothing else needs to change.

    cp -r agents/_template agents/my_agent
    # edit agents/my_agent/{agent.py,tools.py,evals/cases.yaml}, replacing `_template` imports
    mfa eval my_agent --target local
    mfa deploy my_agent --target agent_engine,claude_managed
    mfa eval my_agent --target all
"""

from pathlib import Path

from mfa import AGENT_ENGINE, CLAUDE_MANAGED, LOCAL, AgentSpec  # , local_skills

from agents._template.tools import TOOLS

AGENT = AgentSpec(
    name="template_agent",
    description="Starter agent: replace me.",
    system_prompt="You are a helpful assistant. When asked to repeat something, use the echo tool.",
    tools=TOOLS,
    models={
        LOCAL: "anthropic:claude-opus-5-5",
        AGENT_ENGINE: "google_vertexai:gemini-3.1-pro-preview",
        CLAUDE_MANAGED: "claude-opus-5-5",
    },
    eval_dataset=str(Path(__file__).parent / "evals" / "cases.yaml"),
    # Agent Skills (folders with a SKILL.md under skills/): uncomment to attach them.
    # skills=local_skills(Path(__file__).parent / "skills"),
)
