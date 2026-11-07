"""HTTP serving for containerized deployments (Cloud Run, GKE, Agent Runtime containers).

Follows the contract of Google's official agents-cli LangChain template:

* ``app/agent.py`` exports ``root_agent``: a compiled LangGraph graph with ``messages`` state
* ``app/fast_api_app.py`` exposes ``app``, served with ``uvicorn app.fast_api_app:app``
* the agent is reachable over A2A at ``POST /a2a/<name>`` with its card at the well-known path

``mfa`` generates those two ``app/`` files as thin shims at deploy time, so the
base application (``agents/<name>/``) never changes. Which agent and model a
container serves is set by ``MFA_AGENT`` and ``MFA_MODEL``.
"""

from __future__ import annotations

import os
from functools import lru_cache


@lru_cache(maxsize=1)
def load_root_agent():
    """Build the served agent's spec and compiled graph from MFA_AGENT / MFA_MODEL."""
    from mfa.registry import load_agent
    from mfa.runtime.graph import build_agent_graph
    from mfa.runtime.models import make_chat_model
    from mfa.spec import CLOUD_RUN

    spec = load_agent(os.environ.get("MFA_AGENT", "support_agent"))
    model = spec.model_for(CLOUD_RUN, os.environ.get("MFA_MODEL"))
    return spec, model, build_agent_graph(spec, make_chat_model(model))
