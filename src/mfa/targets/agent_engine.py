"""Target 2: Gemini Enterprise Agent Platform / Vertex AI Agent Engine.

The LangGraph graph itself is deployed: ``LangGraphAgentApp`` is pickled and
shipped together with the ``mfa`` and ``agents`` source trees, and Agent
Engine calls ``set_up()`` then ``query()`` on it.

Required configuration (env vars or CLI options):
    GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION (default us-central1),
    MFA_STAGING_BUCKET (gs://...).
Optional:
    MFA_AGENT_ENGINE_SERVICE_ACCOUNT, MFA_ANTHROPIC_SECRET (Secret Manager
    secret ID holding an Anthropic API key, needed only when the Agent Engine
    model is ``anthropic:...``).
"""

from __future__ import annotations

import contextlib
import importlib.metadata as md
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator

from mfa import deployments
from mfa.runtime.models import parse_model
from mfa.runtime.result import AgentResult
from mfa.spec import AGENT_ENGINE, AgentSpec
from mfa.targets.agent_engine_app import LangGraphAgentApp

REPO_ROOT = Path(__file__).resolve().parents[3]
MFA_SRC = REPO_ROOT / "src" / "mfa"
AGENTS_SRC = REPO_ROOT / "agents"

_BASE_PACKAGES = ["langgraph", "langchain-core", "pydantic", "pyyaml", "cloudpickle"]
_PROVIDER_PACKAGES = {
    "google_vertexai": ["langchain-google-vertexai"],
    "anthropic_vertex": ["langchain-google-vertexai", "anthropic"],
    "anthropic": ["langchain-anthropic", "anthropic"],
}


def _pinned(pkg: str) -> str:
    """Pin to the locally installed version so the pickle deserializes identically remotely."""
    try:
        return f"{pkg}=={md.version(pkg)}"
    except md.PackageNotFoundError:
        return pkg


def build_requirements(spec: AgentSpec, model: str) -> list[str]:
    provider, _ = parse_model(model)
    pkgs = _BASE_PACKAGES + _PROVIDER_PACKAGES.get(provider, [])
    reqs = [_pinned(p) for p in dict.fromkeys(pkgs)]
    reqs.append(f"google-cloud-aiplatform[agent_engines]=={md.version('google-cloud-aiplatform')}")
    return reqs + list(spec.requirements)


def agent_module_ref(spec: AgentSpec, agent_ref: str) -> str:
    """Canonical ``module:attr`` reference, resolvable inside the Agent Engine container."""
    if ":" in agent_ref or "." in agent_ref:
        return agent_ref
    return f"agents.{agent_ref}.agent:AGENT"


@contextlib.contextmanager
def _staged_sources() -> Iterator[Path]:
    """Copy mfa/ and agents/ to a clean dir so they land top-level (importable) on Agent Engine."""
    with tempfile.TemporaryDirectory(prefix="mfa-stage-") as tmp:
        stage = Path(tmp)
        ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "eval_results")
        shutil.copytree(MFA_SRC, stage / "mfa", ignore=ignore)
        shutil.copytree(AGENTS_SRC, stage / "agents", ignore=ignore)
        cwd = os.getcwd()
        os.chdir(stage)
        try:
            yield stage
        finally:
            os.chdir(cwd)


def _client(project: str | None, location: str | None):
    import vertexai

    project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = location or os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    if not project:
        raise RuntimeError("Set GOOGLE_CLOUD_PROJECT (or pass --project) for the agent_engine target.")
    return vertexai.Client(project=project, location=location), project, location


class AgentEngineRunner:
    target = AGENT_ENGINE

    def __init__(self, remote: Any, resource_name: str):
        self.remote = remote
        self.resource_name = resource_name

    def run(self, user_input: str, *, thread_id: str | None = None) -> AgentResult:
        start = time.monotonic()
        try:
            payload = self.remote.query(input=user_input, thread_id=thread_id)
        except Exception as exc:
            return AgentResult(target=self.target, input=user_input, output="", error=repr(exc))
        result = AgentResult.model_validate(payload)
        result.metadata["server_latency_s"] = result.latency_s
        result.latency_s = round(time.monotonic() - start, 3)
        result.metadata["resource_name"] = self.resource_name
        return result


class AgentEngineTarget:
    name = AGENT_ENGINE

    def deploy(
        self,
        spec: AgentSpec,
        *,
        model: str | None = None,
        agent_ref: str | None = None,
        project: str | None = None,
        location: str | None = None,
        staging_bucket: str | None = None,
        **options: Any,
    ) -> dict[str, Any]:
        model = spec.model_for(AGENT_ENGINE, model)
        client, project, location = _client(project, location)
        staging_bucket = staging_bucket or os.environ.get("MFA_STAGING_BUCKET")
        if not staging_bucket:
            raise RuntimeError("Set MFA_STAGING_BUCKET=gs://... (or pass --staging-bucket).")

        app = LangGraphAgentApp(agent_ref=agent_module_ref(spec, agent_ref or spec.name), model=model)

        env_vars: dict[str, Any] = {"GOOGLE_CLOUD_LOCATION": location}
        if parse_model(model)[0] == "anthropic":
            secret = os.environ.get("MFA_ANTHROPIC_SECRET")
            if not secret:
                raise RuntimeError(
                    "Agent Engine model uses the Claude API; set MFA_ANTHROPIC_SECRET to the "
                    "Secret Manager secret ID that holds ANTHROPIC_API_KEY."
                )
            env_vars["ANTHROPIC_API_KEY"] = {"secret": secret, "version": "latest"}

        config: dict[str, Any] = {
            "display_name": spec.name,
            "description": spec.description,
            "staging_bucket": staging_bucket,
            "requirements": build_requirements(spec, model),
            "extra_packages": ["mfa", "agents"],
            "env_vars": env_vars,
            "agent_framework": "langgraph",
            "labels": {"mfa-agent": spec.name.replace("_", "-").lower()},
        }
        if sa := (options.get("service_account") or os.environ.get("MFA_AGENT_ENGINE_SERVICE_ACCOUNT")):
            config["service_account"] = sa

        existing = deployments.load(spec.name, AGENT_ENGINE)
        with _staged_sources():
            if existing:
                remote = client.agent_engines.update(name=existing["resource_name"], agent=app, config=config)
            else:
                remote = client.agent_engines.create(agent=app, config=config)

        record = {
            "target": AGENT_ENGINE,
            "resource_name": remote.api_resource.name,
            "project": project,
            "location": location,
            "model": model,
            "agent_ref": app.agent_ref,
            "deployed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        deployments.save(spec.name, AGENT_ENGINE, record)
        return record

    def runner(self, spec: AgentSpec, *, model: str | None = None, **options: Any) -> AgentEngineRunner:
        record = deployments.require(spec.name, AGENT_ENGINE)
        client, _, _ = _client(record.get("project"), record.get("location"))
        remote = client.agent_engines.get(name=record["resource_name"])
        return AgentEngineRunner(remote, record["resource_name"])
