"""Target: Gemini platform via Cloud Run (container, A2A).

This is the path Google's official agents-cli LangChain extension recommends
for LangGraph agents. The same build context also works with
``agents-cli deploy`` or ``docker build`` (see ``mfa build``).

Required: GOOGLE_CLOUD_PROJECT, gcloud authenticated.
Optional: MFA_CLOUD_RUN_REGION (default us-central1), MFA_CLOUD_RUN_SERVICE_ACCOUNT,
MFA_ANTHROPIC_SECRET (Secret Manager secret, only for anthropic:* models),
MFA_CLOUD_RUN_ALLOW_UNAUTHENTICATED=1 (public endpoint; off by default).
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
import urllib.request
from typing import Any

from mfa import deployments
from mfa.runtime.models import parse_model
from mfa.runtime.result import AgentResult
from mfa.spec import CLOUD_RUN, AgentSpec
from mfa.targets.container import build_context


def service_name(spec: AgentSpec) -> str:
    return spec.name.replace("_", "-").lower()[:49]


def _identity_token(audience: str) -> str | None:
    """ID token for a private Cloud Run service: service-account ADC first, then the gcloud user."""
    try:
        import google.auth.transport.requests
        import google.oauth2.id_token

        return google.oauth2.id_token.fetch_id_token(google.auth.transport.requests.Request(), audience)
    except Exception:
        pass
    try:
        out = subprocess.run(
            ["gcloud", "auth", "print-identity-token"], check=True, capture_output=True, text=True
        )
        return out.stdout.strip()
    except Exception:
        return None


class CloudRunRunner:
    target = CLOUD_RUN

    def __init__(self, url: str, timeout_s: float = 600):
        self.url = url.rstrip("/")
        self.timeout_s = timeout_s
        self._token = _identity_token(self.url)

    def run(self, user_input: str, *, thread_id: str | None = None) -> AgentResult:
        start = time.monotonic()
        body = json.dumps({"input": user_input, "thread_id": thread_id}).encode()
        req = urllib.request.Request(f"{self.url}/mfa/query", data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        if self._token:
            req.add_header("Authorization", f"Bearer {self._token}")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                payload = json.loads(resp.read())
        except Exception as exc:
            return AgentResult(target=self.target, input=user_input, output="", error=repr(exc))
        result = AgentResult.model_validate(payload)
        result.metadata["server_latency_s"] = result.latency_s
        result.latency_s = round(time.monotonic() - start, 3)
        result.metadata["url"] = self.url
        return result


class CloudRunTarget:
    name = CLOUD_RUN

    def deploy(self, spec: AgentSpec, *, model: str | None = None, agent_ref: str | None = None,
               project: str | None = None, location: str | None = None, **options: Any) -> dict[str, Any]:
        model = spec.model_for(CLOUD_RUN, model)
        project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        region = location or os.environ.get("MFA_CLOUD_RUN_REGION", "us-central1")
        if not project:
            raise RuntimeError("Set GOOGLE_CLOUD_PROJECT (or pass --project) for the cloud_run target.")

        env = {
            "MFA_AGENT": agent_ref or spec.name,
            "MFA_MODEL": model,
            "MFA_TARGET": CLOUD_RUN,
            "GOOGLE_CLOUD_PROJECT": project,
            "A2A_NAME": "app",
            **deployments.forwarded_env(),
        }
        cmd = [
            "gcloud", "run", "deploy", service_name(spec),
            f"--project={project}", f"--region={region}",
            "--quiet", "--memory=1Gi", "--timeout=900",
            "--set-env-vars=" + ",".join(f"{k}={v}" for k, v in env.items()),
            f"--labels=mfa-agent={service_name(spec)}",
        ]
        if parse_model(model)[0] == "anthropic":
            secret = os.environ.get("MFA_ANTHROPIC_SECRET")
            if not secret:
                raise RuntimeError("Model uses the Claude API; set MFA_ANTHROPIC_SECRET to a Secret Manager secret.")
            cmd.append(f"--set-secrets=ANTHROPIC_API_KEY={secret}:latest")
        if sa := os.environ.get("MFA_CLOUD_RUN_SERVICE_ACCOUNT"):
            cmd.append(f"--service-account={sa}")
        cmd.append(
            "--allow-unauthenticated" if os.environ.get("MFA_CLOUD_RUN_ALLOW_UNAUTHENTICATED") == "1"
            else "--no-allow-unauthenticated"
        )

        with tempfile.TemporaryDirectory(prefix="mfa-run-") as tmp:
            ctx = build_context(spec, model, tmp, agent_ref=agent_ref or spec.name)
            subprocess.run([*cmd, f"--source={ctx}"], check=True)

        url = subprocess.run(
            ["gcloud", "run", "services", "describe", service_name(spec), f"--project={project}",
             f"--region={region}", "--format=value(status.url)"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        # Advertise the real URL in the A2A agent card.
        subprocess.run(
            ["gcloud", "run", "services", "update", service_name(spec), f"--project={project}",
             f"--region={region}", "--quiet", f"--update-env-vars=APP_URL={url}"],
            check=True,
        )
        record = {
            "target": CLOUD_RUN,
            "service": service_name(spec),
            "url": url,
            "a2a_card": f"{url}/a2a/app/.well-known/agent-card.json",
            "project": project,
            "region": region,
            "model": model,
            "deployed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        deployments.save(spec.name, CLOUD_RUN, record)
        return record

    def runner(self, spec: AgentSpec, *, model: str | None = None, **options: Any) -> CloudRunRunner:
        url = options.get("url") or deployments.require(spec.name, CLOUD_RUN)["url"]
        return CloudRunRunner(url)
