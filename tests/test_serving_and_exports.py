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

SPEC = load_agent("support_agent")
SCRIPT = [
    {"tool": "get_order_status", "args": {"order_id": "ORD-1001"}},
    {"text": "Shipped via UPS, tracking 1Z999AA10123456784."},
]


@pytest.fixture(autouse=True)
def _tmp_deployments(tmp_path, monkeypatch):
    monkeypatch.setattr(deployments, "DEFAULT_DIR", tmp_path / ".deployments")


@pytest.fixture
def client(monkeypatch):
    from mfa.serving import load_root_agent

    load_root_agent.cache_clear()
    monkeypatch.setenv("MFA_MODEL", "x:y")
    monkeypatch.setattr("mfa.runtime.models.make_chat_model", lambda m: ScriptedChatModel(script=SCRIPT * 5))
    from mfa.serving.fast_api_app import create_app

    yield TestClient(create_app())
    load_root_agent.cache_clear()


def test_cloud_run_borrows_the_gemini_model_without_spec_changes():
    assert SPEC.model_for(CLOUD_RUN) == SPEC.model_for(AGENT_ENGINE)


def test_serving_query_returns_normalized_result(client):
    body = client.post("/mfa/query", json={"input": "where is ORD-1001?"}).json()
    r = AgentResult.model_validate(body)
    assert r.tool_names == ["get_order_status"] and "1Z999AA10123456784" in r.output
    assert client.get("/health").json()["agent"] == "support_agent"


def test_a2a_card_and_message_send(client):
    card = client.get("/a2a/app/.well-known/agent-card.json").json()
    assert {s["id"] for s in card["skills"]} == {t.name for t in SPEC.tools}
    resp = client.post("/a2a/app", json={
        "jsonrpc": "2.0", "id": 1, "method": "message/send",
        "params": {"message": {"kind": "message", "messageId": "m1", "role": "user",
                               "parts": [{"kind": "text", "text": "where is ORD-1001?"}]}},
    }).json()
    texts = [p["text"] for a in resp["result"]["artifacts"] for p in a["parts"]]
    # Only the assistant reply is streamed, never the raw tool output.
    assert "".join(texts) == "Shipped via UPS, tracking 1Z999AA10123456784."


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
        cases=load_cases(SPEC.eval_dataset, ids=["order_status_shipped"]), judge=None,
    )
    assert report.pass_rate == 1.0 and report.target == CLOUD_RUN


def test_container_context_follows_agents_cli_contract(tmp_path):
    out = build_context(SPEC, "anthropic:claude-opus-5-5", tmp_path / "ctx")
    for f in ["app/agent.py", "app/fast_api_app.py", "Dockerfile", "requirements.txt", "agents/support_agent/agent.py"]:
        assert (out / f).exists(), f
    assert "uvicorn app.fast_api_app:app" in (out / "Dockerfile").read_text()
    reqs = (out / "requirements.txt").read_text()
    assert "a2a-sdk[http-server]" in reqs and "langchain-anthropic" in reqs
    # The context is self-contained: root_agent imports from it with nothing but the copied sources.
    code = "from app.agent import root_agent; print(type(root_agent).__name__)"
    res = subprocess.run(
        [sys.executable, "-c", code], cwd=out, capture_output=True, text=True,
        env={"MFA_AGENT": "support_agent", "MFA_MODEL": "anthropic:claude-opus-5-5",
             "ANTHROPIC_API_KEY": "dummy", "PATH": "/usr/bin:/bin"},
    )
    assert res.returncode == 0, res.stderr
    assert "CompiledStateGraph" in res.stdout


def test_cma_deploy_reuses_existing_environment_by_name():
    client = FakeCMAClient()
    from types import SimpleNamespace

    client.existing_envs = [SimpleNamespace(id="env_existing", name="support_agent-env")]
    rec = ClaudeManagedTarget().deploy(SPEC, client=client)
    assert rec["environment_id"] == "env_existing" and client.env_creates == 0


def test_cma_ant_files_and_register(tmp_path):
    agent_md, env_yaml = write_ant_files(SPEC, "claude-opus-5-5", tmp_path)
    text = agent_md.read_text()
    front, body = text.split("---\n")[1], text.split("---\n", 2)[2]
    cfg = yaml.safe_load(front)
    assert cfg["model"] == {"id": "claude-opus-5-5"} and "system" not in cfg
    assert [t["name"] for t in cfg["tools"]] == [t.name for t in SPEC.tools]
    assert body.strip() == SPEC.system_prompt.strip()
    assert yaml.safe_load(env_yaml.read_text())["config"]["networking"]["type"] == "limited"

    rec = register(SPEC, "agent_x", "env_y", client=FakeCMAClient())
    assert rec["agent_version"] == 7 and deployments.load(SPEC.name, CLAUDE_MANAGED)["agent_id"] == "agent_x"


def test_gemini_eval_export_shape():
    class Multi:
        target = "claude_managed"

        def run(self, text, *, thread_id=None):
            from mfa.runtime.result import ToolCall

            if text == "a":
                return AgentResult(target=self.target, input=text, output="",
                                   tool_calls=[ToolCall(name="get_order_status", args={"order_id": "ORD-1"},
                                                        output='{"found": false}')])
            return AgentResult(target=self.target, input=text, output="done")

    case = EvalCase(id="mt", turns=["a", "b"], judge="be right")
    report = run_eval(agent="support_agent", runner=Multi(), cases=[case], judge=None)
    data = to_gemini_eval_dataset(report, cases_by_id={"mt": case})
    ec = data["eval_cases"][0]
    assert ec["eval_case_id"] == "mt" and ec["rubric"] == "be right"
    t0, t1 = ec["agent_data"]["turns"]
    assert [e["author"] for e in t0["events"]] == ["user", "support_agent", "tool"]
    assert t0["events"][1]["content"]["parts"][0]["function_call"]["name"] == "get_order_status"
    assert t0["events"][2]["content"]["parts"][0]["function_response"]["response"] == {"found": False}
    assert t1["events"][-1]["content"]["parts"][0]["text"] == "done"
    json.dumps(data)


def test_built_container_bakes_in_selected_model_and_agent(tmp_path):
    """The default runtime model must match the provider packages in requirements.txt."""
    out = build_context(SPEC, "anthropic:claude-opus-5-5", tmp_path / "ctx", agent_ref="support_agent")
    dockerfile = (out / "Dockerfile").read_text()
    assert "MFA_MODEL=anthropic:claude-opus-5-5" in dockerfile
    assert "MFA_AGENT=support_agent" in dockerfile
    assert "langchain-anthropic" in (out / "requirements.txt").read_text()


def test_a2a_route_applies_spec_iteration_limit(monkeypatch):
    from mfa.serving import load_root_agent

    load_root_agent.cache_clear()
    monkeypatch.setattr(SPEC, "max_iterations", 2)
    monkeypatch.setenv("MFA_MODEL", "x:y")
    looping = [{"tool": "get_order_status", "args": {"order_id": "ORD-1001"}}] * 10 + [{"text": "FINAL"}]
    monkeypatch.setattr("mfa.runtime.models.make_chat_model", lambda m: ScriptedChatModel(script=looping))
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
    assert "error" in resp and "Recursion limit of 5" in resp["error"]["message"]
