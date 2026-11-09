"""Gemini container target (agents-cli contract + A2A), CMA file export/registration, Gemini eval export."""

import json
import subprocess
import sys

import pytest
import yaml
from fastapi.testclient import TestClient

from mfa import deployments
from mfa.evals import EvalCase, load_cases, run_eval
from mfa.evals.export import to_gemini_eval_dataset
from mfa.registry import load_agent
from mfa.runtime.result import AgentResult
from mfa.spec import AGENT_ENGINE, CLAUDE_MANAGED, CLOUD_RUN
from mfa.targets.claude_managed import ClaudeManagedTarget, register, write_ant_files
from mfa.targets.cloud_run import CloudRunRunner
from mfa.targets.container import build_context
from tests.fakes import FakeCMAClient, ScriptedChatModel

SPEC = load_agent("weather_agent")
SCRIPT = [
    {"tool": "get_weather_today", "args": {"location": "Paris"}},
    {"text": "In Paris today: partly cloudy, 54°F (12°C), high 59°F (15°C), low 46°F (8°C), 10% chance of rain."},
]


@pytest.fixture
def client(monkeypatch):
    from mfa.serving import load_root_agent

    load_root_agent.cache_clear()
    monkeypatch.setenv("MFA_MODEL", "x:y")
    monkeypatch.setattr("mfa.runtime.models.make_chat_model", lambda m, **kw: ScriptedChatModel(script=SCRIPT * 5))
    from mfa.serving.fast_api_app import create_app

    yield TestClient(create_app())
    load_root_agent.cache_clear()


def test_cloud_run_borrows_the_gemini_model_without_spec_changes():
    assert SPEC.model_for(CLOUD_RUN) == SPEC.model_for(AGENT_ENGINE)


def test_serving_query_returns_normalized_result(client):
    body = client.post("/mfa/query", json={"input": "weather in Paris today?"}).json()
    r = AgentResult.model_validate(body)
    assert r.tool_names == ["get_weather_today"] and "partly cloudy" in r.output
    assert client.get("/health").json()["agent"] == "weather_agent"


def test_a2a_card_and_message_send(client):
    card = client.get("/a2a/app/.well-known/agent-card.json").json()
    assert {s["id"] for s in card["skills"]} == {t.name for t in SPEC.tools} | {f"skill:{sk.name}" for sk in SPEC.skills}
    assert client.get("/health").json()["skills"] == ["weather-briefing"]
    resp = client.post("/a2a/app", json={
        "jsonrpc": "2.0", "id": 1, "method": "message/send",
        "params": {"message": {"kind": "message", "messageId": "m1", "role": "user",
                               "parts": [{"kind": "text", "text": "weather in Paris today?"}]}},
    }).json()
    texts = [p["text"] for a in resp["result"]["artifacts"] for p in a["parts"]]
    # Only the assistant reply is streamed, never the raw tool output.
    assert "".join(texts) == "In Paris today: partly cloudy, 54°F (12°C), high 59°F (15°C), low 46°F (8°C), 10% chance of rain."


def test_cloud_run_runner_against_served_app(client, monkeypatch):
    class _Resp:
        def __init__(self, data):
            self.data = data

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return self.data

    def fake_urlopen(req, timeout):
        assert req.full_url == "http://svc/mfa/query"
        r = client.post("/mfa/query", content=req.data, headers={"Content-Type": "application/json"})
        return _Resp(r.content)

    monkeypatch.setattr("mfa.targets.cloud_run._identity_token", lambda aud: None)
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    report = run_eval(
        agent=SPEC.name, runner=CloudRunRunner("http://svc"),
        cases=load_cases(SPEC.eval_dataset, ids=["weather_with_city"]), judge=None,
    )
    assert report.pass_rate == 1.0 and report.target == CLOUD_RUN


def test_container_context_follows_agents_cli_contract(tmp_path):
    out = build_context(SPEC, "anthropic:claude-opus-5-5", tmp_path / "ctx")
    for f in ["app/agent.py", "app/fast_api_app.py", "Dockerfile", "requirements.txt", "agents/weather_agent/agent.py",
              "agents/weather_agent/skills/weather-briefing/SKILL.md",
              "agents/weather_agent/skills/weather-briefing/references/clothing-guide.md"]:
        assert (out / f).exists(), f
    assert "uvicorn app.fast_api_app:app" in (out / "Dockerfile").read_text()
    reqs = (out / "requirements.txt").read_text()
    assert "a2a-sdk[http-server]" in reqs and "langchain-anthropic" in reqs
    # The context is self-contained: root_agent imports from it with nothing but the copied sources.
    code = "from app.agent import root_agent; print(type(root_agent).__name__)"
    res = subprocess.run(
        [sys.executable, "-c", code], cwd=out, capture_output=True, text=True,
        env={"MFA_AGENT": "weather_agent", "MFA_MODEL": "anthropic:claude-opus-5-5",
             "ANTHROPIC_API_KEY": "dummy", "PATH": "/usr/bin:/bin"},
    )
    assert res.returncode == 0, res.stderr
    assert "CompiledStateGraph" in res.stdout


def test_cma_deploy_reuses_existing_environment_by_name():
    client = FakeCMAClient()
    from types import SimpleNamespace

    client.existing_envs = [SimpleNamespace(id="env_existing", name="weather_agent-env")]
    rec = ClaudeManagedTarget().deploy(SPEC, client=client)
    assert rec["environment_id"] == "env_existing" and client.env_creates == 0


def test_cma_ant_files_and_register(tmp_path):
    agent_md, env_yaml, *skill_files = write_ant_files(SPEC, "claude-opus-5-5", tmp_path)
    text = agent_md.read_text()
    front, body = text.split("---\n")[1], text.split("---\n", 2)[2]
    cfg = yaml.safe_load(front)
    assert cfg["model"] == {"id": "claude-opus-5-5"} and "system" not in cfg
    assert [t["name"] for t in cfg["tools"] if t["type"] == "custom"] == [t.name for t in SPEC.tools]
    assert cfg["tools"][0]["type"] == "agent_toolset_20260401"
    assert cfg["skills"] == [{"type": "custom", "skill_id": "<uploaded at deploy: weather-briefing>"}]
    assert body.strip() == SPEC.system_prompt.strip()
    assert yaml.safe_load(env_yaml.read_text())["config"]["networking"]["type"] == "limited"
    # The skill folder travels with the files for ant apply / the Skills API.
    assert (agent_md.parent / "skills" / "weather-briefing" / "SKILL.md").exists()

    rec = register(SPEC, "agent_x", "env_y", client=FakeCMAClient())
    assert rec["agent_version"] == 7 and deployments.load(SPEC.name, CLAUDE_MANAGED)["agent_id"] == "agent_x"


def test_gemini_eval_export_shape():
    class Multi:
        target = "claude_managed"

        def run(self, text, *, thread_id=None):
            from mfa.runtime.result import ToolCall

            if text == "a":
                return AgentResult(target=self.target, input=text, output="",
                                   tool_calls=[ToolCall(name="get_weather_today", args={"location": "Atlantis"},
                                                        output='{"found": false}')])
            return AgentResult(target=self.target, input=text, output="done")

    case = EvalCase(id="mt", turns=["a", "b"], judge="be right")
    report = run_eval(agent="weather_agent", runner=Multi(), cases=[case], judge=None)
    data = to_gemini_eval_dataset(report, cases_by_id={"mt": case})
    ec = data["eval_cases"][0]
    assert ec["eval_case_id"] == "mt" and ec["rubric"] == "be right"
    t0, t1 = ec["agent_data"]["turns"]
    assert [e["author"] for e in t0["events"]] == ["user", "weather_agent", "tool"]
    assert t0["events"][1]["content"]["parts"][0]["function_call"]["name"] == "get_weather_today"
    assert t0["events"][2]["content"]["parts"][0]["function_response"]["response"] == {"found": False}
    assert t1["events"][-1]["content"]["parts"][0]["text"] == "done"
    json.dumps(data)


def test_built_container_bakes_in_selected_model_and_agent(tmp_path):
    """The default runtime model must match the provider packages in requirements.txt."""
    out = build_context(SPEC, "anthropic:claude-opus-5-5", tmp_path / "ctx", agent_ref="weather_agent")
    dockerfile = (out / "Dockerfile").read_text()
    assert "MFA_MODEL=anthropic:claude-opus-5-5" in dockerfile
    assert "MFA_AGENT=weather_agent" in dockerfile
    assert "langchain-anthropic" in (out / "requirements.txt").read_text()


def test_a2a_route_applies_spec_iteration_limit(monkeypatch):
    from mfa.serving import load_root_agent

    load_root_agent.cache_clear()
    monkeypatch.setattr(SPEC, "max_iterations", 2)
    monkeypatch.setenv("MFA_MODEL", "x:y")
    looping = [{"tool": "get_weather_today", "args": {"location": "Paris"}}] * 10 + [{"text": "FINAL"}]
    monkeypatch.setattr("mfa.runtime.models.make_chat_model", lambda m, **kw: ScriptedChatModel(script=looping))
    from mfa.serving.fast_api_app import create_app

    try:
        c = TestClient(create_app(), raise_server_exceptions=False)
        resp = c.post("/a2a/app", json={
            "jsonrpc": "2.0", "id": 1, "method": "message/send",
            "params": {"message": {"kind": "message", "messageId": "m1", "role": "user",
                                   "parts": [{"kind": "text", "text": "loop"}]}},
        }).json()
    finally:
        load_root_agent.cache_clear()
    task = resp["result"]
    assert task["status"]["state"] == "failed"
    assert task["status"]["message"]["parts"][0]["text"] == "Error: iteration limit reached: max_iterations=2"


def test_a2a_reply_is_the_final_message_like_mfa_query(monkeypatch):
    """Intermediate assistant text ("let me check") is replaced by the final reply, as the evals see it."""
    from mfa.serving import load_root_agent

    load_root_agent.cache_clear()
    monkeypatch.setenv("MFA_MODEL", "x:y")
    script = [{"text": "Let me check.", "tool": "get_weather_today", "args": {"location": "Paris"}}, {"text": "Sunny in Paris."}] * 2
    monkeypatch.setattr("mfa.runtime.models.make_chat_model", lambda m, **kw: ScriptedChatModel(script=script))
    from mfa.serving.fast_api_app import create_app

    try:
        c = TestClient(create_app())
        a2a = c.post("/a2a/app", json={
            "jsonrpc": "2.0", "id": 1, "method": "message/send",
            "params": {"message": {"kind": "message", "messageId": "m1", "role": "user",
                                   "parts": [{"kind": "text", "text": "weather in Paris?"}]}},
        }).json()["result"]
        query = c.post("/mfa/query", json={"input": "weather in Paris?"}).json()
    finally:
        load_root_agent.cache_clear()
    texts = ["".join(p["text"] for p in a["parts"]) for a in a2a["artifacts"]]
    assert texts == ["Sunny in Paris."] and query["output"] == "Sunny in Paris."


def test_forwarded_env_lists_only_named_vars(monkeypatch):
    monkeypatch.setenv("MFA_FORWARD_ENV", "MFA_WEATHER_PROVIDER, MISSING_VAR")
    assert deployments.forwarded_env() == {"MFA_WEATHER_PROVIDER": "fixture"}


def test_cloud_run_runner_retries_transient_errors_and_refreshes_token(monkeypatch):
    import urllib.error

    attempts = []
    import itertools

    tokens = (f"tok-{i}" for i in itertools.count(1))
    monkeypatch.setattr("mfa.targets.cloud_run._identity_token", lambda aud: next(tokens))
    monkeypatch.setattr("mfa.targets.cloud_run.time.sleep", lambda s: None)

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"target": "cloud_run", "input": "q", "output": "ok", "latency_s": 0.1}).encode()

    def flaky_urlopen(req, timeout):
        attempts.append(req.get_header("Authorization"))
        if len(attempts) == 1:
            raise urllib.error.HTTPError(req.full_url, 503, "unavailable", {}, None)
        if len(attempts) == 2:
            raise urllib.error.HTTPError(req.full_url, 401, "expired", {}, None)
        return _Resp()

    monkeypatch.setattr("urllib.request.urlopen", flaky_urlopen)
    r = CloudRunRunner("http://svc").run("q")
    assert r.error is None and r.output == "ok" and r.metadata["attempts"] == 3
    assert attempts == ["Bearer tok-1", "Bearer tok-1", "Bearer tok-2"]  # 401 forced a fresh token

    # Anything that may have reached the app is not retried: a retry would run the turn twice.
    for status in (500, 502, 504, 403):
        def not_again(req, timeout, status=status):
            raise urllib.error.HTTPError(req.full_url, status, "boom", {}, None)

        monkeypatch.setattr("urllib.request.urlopen", not_again)
        r = CloudRunRunner("http://svc", retries=3).run("q")
        assert r.error.startswith(f"HTTP {status}") and r.metadata["attempts"] == 1

    def read_timeout(req, timeout):
        raise urllib.error.URLError(TimeoutError("timed out"))

    monkeypatch.setattr("urllib.request.urlopen", read_timeout)
    assert CloudRunRunner("http://svc", retries=3).run("q").metadata["attempts"] == 1

    def refused(req, timeout):
        raise urllib.error.URLError(ConnectionRefusedError(111, "refused"))

    monkeypatch.setattr("urllib.request.urlopen", refused)
    assert CloudRunRunner("http://svc", retries=2).run("q").metadata["attempts"] == 3  # never reached the app


def test_cloud_run_runner_caches_a_missing_token_too(monkeypatch):
    fetches = []
    monkeypatch.setattr("mfa.targets.cloud_run._identity_token", lambda aud: fetches.append(aud) or None)

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"target": "cloud_run", "input": "q", "output": "ok"}).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout: _Resp())
    runner = CloudRunRunner("http://localhost:8080")
    for _ in range(5):
        assert runner.run("q").output == "ok"
    assert len(fetches) == 1  # public / local service: no token, and no gcloud subprocess per request
