import pickle

import pytest

from mfa.registry import load_agent
from mfa.spec import AGENT_ENGINE, CLAUDE_MANAGED
from mfa.targets.agent_engine import AgentEngineRunner, agent_module_ref, build_requirements
from mfa.targets.agent_engine_app import LangGraphAgentApp
from mfa.targets.claude_managed import ClaudeManagedRunner, ClaudeManagedTarget, build_agent_params
from mfa.targets.local import LocalRunner
from tests.fakes import FakeCMAClient, FakeRemoteAgentEngine, ScriptedChatModel

SPEC = load_agent("weather_agent")

SCRIPT = [
    {"tool": "get_weather_today", "args": {"location": "Paris"}},
    {"text": "In Paris today: partly cloudy, 54°F (12°C), high 59°F (15°C), low 46°F (8°C), 10% chance of rain."},
]


def test_local_runner_runs_tool_loop():
    r = LocalRunner(SPEC, chat_model=ScriptedChatModel(script=list(SCRIPT))).run("weather in Paris today?")
    assert r.error is None
    assert r.tool_names == ["get_weather_today"]
    assert '"country": "France"' in r.tool_calls[0].output and 'Île-de-France' in r.tool_calls[0].output
    assert "partly cloudy" in r.output


def test_local_runner_keeps_thread_state_and_reports_only_current_turn():
    model = ScriptedChatModel(script=[*SCRIPT, {"text": "Anything else?"}])
    runner = LocalRunner(SPEC, chat_model=model)
    runner.run("weather in Paris today?", thread_id="t1")
    second = runner.run("thanks", thread_id="t1")
    assert second.tool_calls == []
    assert second.output == "Anything else?"


def test_agent_engine_app_pickles_and_queries(monkeypatch):
    app = LangGraphAgentApp(agent_ref=agent_module_ref(SPEC, "weather_agent"), model="anthropic:claude-opus-5-5")
    app = pickle.loads(pickle.dumps(app))  # what Agent Engine does with the object
    monkeypatch.setattr("mfa.runtime.models.make_chat_model", lambda m, **kw: ScriptedChatModel(script=list(SCRIPT)))
    app.set_up()
    runner = AgentEngineRunner(FakeRemoteAgentEngine(app), "projects/p/locations/l/reasoningEngines/1")
    r = runner.run("weather in Paris today?")
    assert r.target == AGENT_ENGINE
    assert r.tool_names == ["get_weather_today"]
    assert "partly cloudy" in r.output


def test_agent_engine_requirements_follow_model_provider():
    gem = build_requirements(SPEC, "google_vertexai:gemini-3.1-pro-preview")
    assert any(r.startswith("langchain-google-vertexai") for r in gem)
    assert not any(r.startswith("langchain-anthropic") for r in gem)
    cl = build_requirements(SPEC, "anthropic:claude-opus-5-5")
    assert any(r.startswith("langchain-anthropic") for r in cl)
    assert agent_module_ref(SPEC, "weather_agent") == "agents.weather_agent.agent:AGENT"


def test_claude_managed_translation_mirrors_spec():
    params = build_agent_params(SPEC, "claude-opus-5-5")
    assert params["system"] == SPEC.system_prompt
    assert params["model"] == {"id": "claude-opus-5-5"}
    custom = [t for t in params["tools"] if t["type"] == "custom"]
    assert [t["name"] for t in custom] == [t.name for t in SPEC.tools]
    assert custom[0]["input_schema"] == SPEC.tools[0].input_schema
    # The weather agent has a skill, so the file tools are on (always_ask) and the skill is attached.
    assert params["tools"][0]["type"] == "agent_toolset_20260401"
    assert params["skills"] == [{"type": "custom", "skill_id": "<uploaded at deploy: weather-briefing>"}]


def test_effort_reaches_each_provider(monkeypatch):
    from dataclasses import replace

    from mfa.runtime.models import GEMINI_THINKING_BUDGET, effective_effort, make_chat_model
    from mfa.targets.claude_managed import model_params

    monkeypatch.setenv("MFA_ANTHROPIC_API_KEY", "sk-test")
    assert make_chat_model("anthropic:claude-opus-5-5", effort="high").reasoning_effort == "high"
    assert make_chat_model("anthropic:claude-opus-5-5").reasoning_effort is None
    spec = replace(SPEC, effort="xhigh")
    assert model_params(spec, "claude-opus-5-5") == {"id": "claude-opus-5-5", "effort": "xhigh"}
    assert model_params(SPEC, "claude-opus-5-5") == {"id": "claude-opus-5-5"}
    assert GEMINI_THINKING_BUDGET["xhigh"] > GEMINI_THINKING_BUDGET["low"]
    monkeypatch.setenv("MFA_EFFORT", "low")
    assert effective_effort("xhigh") == "low"  # the env knob wins on every target
    with pytest.raises(ValueError):
        replace(SPEC, effort="turbo")


def test_claude_managed_deploy_creates_then_updates():
    client = FakeCMAClient()
    target = ClaudeManagedTarget()
    rec1 = target.deploy(SPEC, client=client)
    assert rec1["agent_id"] == "agent_fake" and rec1["agent_version"] == 1
    rec2 = target.deploy(SPEC, client=client)
    assert len(client.created_agents) == 1 and len(client.updated_agents) == 1
    assert rec2["agent_version"] == 2 and rec2["environment_id"] == rec1["environment_id"]


def test_claude_managed_runner_executes_custom_tools_host_side():
    client = FakeCMAClient(script=list(SCRIPT))
    record = {"agent_id": "agent_fake", "agent_version": 1, "environment_id": "env_fake", "model": "claude-opus-5-5"}
    r = ClaudeManagedRunner(SPEC, record, client=client).run("weather in Paris today?")
    assert r.error is None and r.target == CLAUDE_MANAGED
    assert r.tool_names == ["get_weather_today"]
    results = [e for e in client.sent if e["type"] == "user.custom_tool_result"]
    assert len(results) == 1 and "partly cloudy" in results[0]["content"][0]["text"]
    assert "partly cloudy" in r.output


# --- the weather-briefing skill, end to end on every target -----------------

from pathlib import Path  # noqa: E402

from mfa.evals import load_cases, run_eval  # noqa: E402
from mfa.spec import CLOUD_RUN, LOCAL  # noqa: E402

SKILL_DIR = Path(__file__).resolve().parents[1] / "agents" / "weather_agent" / "skills" / "weather-briefing"
BRIEFING_SCRIPT = [
    {"tool": "get_weather_today", "args": {"location": "Paris"}},
    {"tool": "read", "args": {"file_path": "/workspace/skills/weather-briefing/SKILL.md"}},
    {"tool": "read", "args": {"file_path": "/workspace/skills/weather-briefing/references/clothing-guide.md"}},
    {"text": "Briefing for Paris, Île-de-France, France: partly cloudy, 54°F (12°C), high 59°F (15°C), low 46°F (8°C), "
             "10% chance of rain.\nWear: a light jacket or sweater over a long-sleeve shirt.\nRain: no umbrella needed."},
]
ALL_TARGETS = (LOCAL, AGENT_ENGINE, CLOUD_RUN, CLAUDE_MANAGED)


@pytest.mark.parametrize("target", ALL_TARGETS)
def test_briefing_case_passes_on_every_target(target, make_runner):
    runner = make_runner(target, BRIEFING_SCRIPT, SPEC, "weather_agent")
    report = run_eval(agent=SPEC.name, runner=runner, cases=load_cases(SPEC.eval_dataset, ids=["briefing_uses_skill"]), judge=None)
    case = report.cases[0]
    assert case.passed, [c for c in case.checks if c.passed is False]
    assert {c.name: c.passed for c in case.checks}["skill:weather-briefing"] is True
    reads = [c for c in case.result.tool_calls if c.name == "read"]
    if target == CLAUDE_MANAGED:
        # The server renders what it read; the eval only needs to see that the skill was read.
        assert all(r.output for r in reads)
    else:
        assert reads[0].output == (SKILL_DIR / "SKILL.md").read_text()
        assert reads[1].output == (SKILL_DIR / "references" / "clothing-guide.md").read_text()
    assert case.result.metadata["skills_used"] == ["weather-briefing"]


def test_plain_weather_case_fails_if_the_skill_is_read(make_runner):
    curious = [
        {"tool": "read", "args": {"file_path": "/workspace/skills/weather-briefing/SKILL.md"}},
        {"tool": "get_weather_today", "args": {"location": "Paris"}},
        {"text": "In Paris today: partly cloudy, 54°F (12°C), high 59°F (15°C), low 46°F (8°C), 10% chance of rain."},
    ]
    runner = make_runner(LOCAL, curious, SPEC, "weather_agent")
    report = run_eval(agent=SPEC.name, runner=runner, cases=load_cases(SPEC.eval_dataset, ids=["weather_with_city"]), judge=None)
    failed = {c.name for c in report.cases[0].checks if c.passed is False}
    assert "skill_not:weather-briefing" in failed and "max_tool_calls" in failed


def test_cma_deploy_uploads_a_skill_once_and_versions_it_on_change(tmp_path):
    import shutil
    from dataclasses import replace

    from mfa.spec import local_skills

    skills_dir = tmp_path / "skills"
    shutil.copytree(SKILL_DIR, skills_dir / "weather-briefing")
    spec = replace(SPEC, skills=local_skills(skills_dir))
    client = FakeCMAClient(spec=spec)
    target = ClaudeManagedTarget()

    rec1 = target.deploy(spec, client=client)
    assert client.uploaded["weather-briefing"] == [
        "weather-briefing/SKILL.md", "weather-briefing/references/clothing-guide.md", "weather-briefing/scripts/briefing_lines.py"]
    assert rec1["skills"]["weather-briefing"]["version"] == "skillver_1"
    assert client.created_agents[0]["skills"] == [{"type": "custom", "skill_id": "skill_fake_weather-briefing", "version": "skillver_1"}]

    rec2 = target.deploy(spec, client=client)  # unchanged content: nothing re-uploaded
    assert len(client.uploaded["weather-briefing"]) == 3 and rec2["skills"]["weather-briefing"]["version"] == "skillver_1"

    (skills_dir / "weather-briefing" / "SKILL.md").write_text((skills_dir / "weather-briefing" / "SKILL.md").read_text() + "\nMore.\n")
    spec2 = replace(SPEC, skills=local_skills(skills_dir))
    rec3 = target.deploy(spec2, client=client)
    assert len(client.uploaded["weather-briefing"]) == 6 and rec3["skills"]["weather-briefing"]["version"] == "skillver_2"
    assert client.updated_agents[-1][1]["skills"][0]["version"] == "skillver_2"


def test_mfa_effort_reaches_claude_managed_and_is_validated(monkeypatch):
    from dataclasses import replace

    from mfa.runtime.models import effective_effort
    from mfa.targets.claude_managed import ClaudeManagedTarget, model_params

    spec = replace(SPEC, effort="xhigh")
    monkeypatch.setenv("MFA_EFFORT", "low")
    assert model_params(spec, "claude-opus-5-5")["effort"] == "low"  # the env knob wins on CMA too
    client = FakeCMAClient(spec=spec)
    assert ClaudeManagedTarget().deploy(spec, client=client)["effort"] == "low"
    monkeypatch.setenv("MFA_EFFORT", "hgih")
    with pytest.raises(ValueError, match="MFA_EFFORT"):
        effective_effort("high")
    with pytest.raises(ValueError, match="MFA_EFFORT"):
        model_params(spec, "claude-opus-5-5")


def test_cma_deploy_reuses_the_skill_a_registered_agent_already_has(tmp_path):
    """After ``mfa register`` the record knows the attached skill's id but not its content: the next deploy must
    give that skill a new version, not create a duplicate skill on the account."""
    import shutil
    from dataclasses import replace

    from mfa import deployments
    from mfa.spec import CLAUDE_MANAGED, local_skills

    skills_dir = tmp_path / "skills"
    shutil.copytree(SKILL_DIR, skills_dir / "weather-briefing")
    spec = replace(SPEC, skills=local_skills(skills_dir))
    deployments.save(spec.name, CLAUDE_MANAGED, {
        "target": CLAUDE_MANAGED, "agent_id": "agent_reg", "agent_version": 3, "environment_id": "env_reg", "model": "m",
        "managed_by": "external", "agent_skills": [{"type": "custom", "skill_id": "skill_pre", "version": "skillver_0"}],
    })
    client = FakeCMAClient(spec=spec, registered_skills={"skill_pre": "weather-briefing"})
    rec = ClaudeManagedTarget().deploy(spec, client=client)
    assert client.skill_versions == {"weather-briefing": 2} and len(client.uploaded["weather-briefing"]) == 3
    assert rec["skills"]["weather-briefing"]["skill_id"] == "skill_pre" and rec["skills"]["weather-briefing"]["version"] == "skillver_2"
    assert client.updated_agents[-1][1]["skills"] == [{"type": "custom", "skill_id": "skill_pre", "version": "skillver_2"}]
    rec2 = ClaudeManagedTarget().deploy(spec, client=client)  # content now known and unchanged: nothing uploaded
    assert client.skill_versions == {"weather-briefing": 2} and rec2["skills"]["weather-briefing"]["version"] == "skillver_2"


def test_cma_record_describes_the_toolset_it_configured(monkeypatch):
    from mfa.targets.claude_managed import enabled_builtins, toolset_params

    assert enabled_builtins(toolset_params(SPEC)) == {"policy": "always_ask", "enabled": ["read", "glob", "grep"], "disabled": []}
    monkeypatch.setenv("MFA_CMA_ENABLE_TOOLSET", "1")
    assert enabled_builtins(toolset_params(SPEC)) == {"policy": "auto", "enabled": "all", "disabled": ["web_fetch", "web_search"]}
    assert enabled_builtins(None) is None
