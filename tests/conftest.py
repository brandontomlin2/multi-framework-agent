import pytest
from fastapi.testclient import TestClient

from mfa.runtime.result import AgentResult
from mfa.spec import AGENT_ENGINE, CLAUDE_MANAGED, CLOUD_RUN, LOCAL


@pytest.fixture(autouse=True)
def _tmp_deployments(tmp_path, monkeypatch):
    # Deploy records never land in the repo's .deployments/ from a test.
    from mfa import deployments

    monkeypatch.setattr(deployments, "DEFAULT_DIR", tmp_path / ".deployments")


@pytest.fixture(autouse=True)
def _offline_weather(monkeypatch):
    # Tests never hit the network: the example agent's tool uses deterministic fixtures.
    monkeypatch.setenv("MFA_WEATHER_PROVIDER", "fixture")
    # Skill locations are announced under the default virtual root in every test.
    monkeypatch.delenv("MFA_SKILLS_ROOT", raising=False)
    monkeypatch.delenv("MFA_EFFORT", raising=False)


class HttpRunner:
    """cloud_run as the evals see it: the FastAPI route, parsed back into an AgentResult."""

    target = CLOUD_RUN

    def __init__(self, client: TestClient):
        self.client = client

    def run(self, user_input, *, thread_id=None):
        body = self.client.post("/mfa/query", json={"input": user_input, "thread_id": thread_id})
        assert body.status_code == 200, body.text
        return AgentResult.model_validate(body.json())


@pytest.fixture
def make_runner(monkeypatch):
    """Build a runner for any target that replays ``script`` through the real adapter code.

    ``spec`` is the AgentSpec and ``ref`` how ``mfa.registry.load_agent`` finds it
    (a short name under agents/ or ``module:ATTR``), which the Agent Engine and
    container paths need because they load the agent by reference.
    """
    from mfa.serving import load_root_agent
    from mfa.targets.agent_engine import AgentEngineRunner
    from mfa.targets.agent_engine_app import LangGraphAgentApp
    from mfa.targets.claude_managed import ClaudeManagedRunner
    from mfa.targets.local import LocalRunner
    from tests.fakes import FakeCMAClient, FakeRemoteAgentEngine, ScriptedChatModel

    made = []

    def _make(target: str, script: list[dict], spec, ref: str):
        if target == LOCAL:
            return LocalRunner(spec, chat_model=ScriptedChatModel(script=list(script)))
        if target == AGENT_ENGINE:
            monkeypatch.setattr("mfa.runtime.models.make_chat_model", lambda m, **kw: ScriptedChatModel(script=list(script)))
            app = LangGraphAgentApp(agent_ref=ref, model="x:y")
            app.set_up()
            return AgentEngineRunner(FakeRemoteAgentEngine(app), "projects/p/locations/l/reasoningEngines/1")
        if target == CLOUD_RUN:
            load_root_agent.cache_clear()
            monkeypatch.setenv("MFA_AGENT", ref)
            monkeypatch.setenv("MFA_MODEL", "x:y")
            monkeypatch.setattr("mfa.runtime.models.make_chat_model", lambda m, **kw: ScriptedChatModel(script=list(script)))
            from mfa.serving.fast_api_app import create_app

            made.append(load_root_agent)
            return HttpRunner(TestClient(create_app()))
        if target == CLAUDE_MANAGED:
            record = {"agent_id": "a", "agent_version": 1, "environment_id": "e", "model": "claude-test"}
            client = FakeCMAClient(script=list(script), spec=spec)
            runner = ClaudeManagedRunner(spec, record, client=client)
            runner.fake = client  # tests inspect what was sent
            return runner
        raise ValueError(target)

    yield _make
    for cache in made:
        cache.cache_clear()
