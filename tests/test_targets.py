import pickle

import pytest

from mfa import deployments
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


@pytest.fixture(autouse=True)
def _tmp_deployments(tmp_path, monkeypatch):
    monkeypatch.setattr(deployments, "DEFAULT_DIR", tmp_path / ".deployments")


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
    monkeypatch.setattr("mfa.runtime.models.make_chat_model", lambda m: ScriptedChatModel(script=list(SCRIPT)))
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
    assert [t["name"] for t in params["tools"]] == [t.name for t in SPEC.tools]
    assert all(t["type"] == "custom" for t in params["tools"])
    assert params["tools"][0]["input_schema"] == SPEC.tools[0].input_schema


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
