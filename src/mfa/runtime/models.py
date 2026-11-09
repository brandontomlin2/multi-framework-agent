"""Chat-model factory for the LangGraph runtime.

Model strings use ``provider:model``:

* ``anthropic:claude-opus-5-5``          - Claude API (``MFA_ANTHROPIC_API_KEY`` or ``ANTHROPIC_API_KEY``)
* ``google_vertexai:gemini-3.1-pro-preview`` - Gemini on Vertex AI (ADC)
* ``anthropic_vertex:claude-opus-5-5``   - Claude on Vertex AI (ADC)

Imports are lazy so a deployment only needs the provider it actually uses.
``effort`` (``AgentSpec.effort`` or ``MFA_EFFORT``) maps to Claude's effort level
directly and to a Gemini thinking budget approximately.
"""

from __future__ import annotations

import os

from langchain_core.language_models import BaseChatModel


def parse_model(model: str) -> tuple[str, str]:
    if ":" not in model:
        raise ValueError(f"Model {model!r} must look like 'provider:model-id'")
    provider, model_id = model.split(":", 1)
    return provider, model_id


# Gemini has no effort levels; a thinking budget is the nearest knob. Override with
# MFA_GEMINI_THINKING_BUDGET.
GEMINI_THINKING_BUDGET = {"low": 1024, "medium": 8192, "high": 16384, "xhigh": 24576, "max": 32768}


def effective_effort(effort: str | None) -> str | None:
    """``MFA_EFFORT`` overrides the spec's effort on every target; it is validated like ``AgentSpec.effort``."""
    from mfa.spec import EFFORT_LEVELS

    env = os.environ.get("MFA_EFFORT")
    if env:
        if env not in EFFORT_LEVELS:
            raise ValueError(f"MFA_EFFORT must be one of {', '.join(EFFORT_LEVELS)}, not {env!r}")
        return env
    return effort


def make_chat_model(model: str, *, effort: str | None = None) -> BaseChatModel:
    provider, model_id = parse_model(model)
    effort = effective_effort(effort)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        # Current Claude models think adaptively by default and reject sampling
        # params; effort (output_config.effort) is the one depth control.
        from mfa.credentials import anthropic_api_key

        key = anthropic_api_key()
        extra: dict = {"api_key": key} if key else {}
        if effort:
            extra["effort"] = effort
        return ChatAnthropic(model=model_id, max_tokens=16000, **extra)

    if provider == "google_vertexai":
        from langchain_google_vertexai import ChatVertexAI

        extra = {}
        if effort:
            budget = os.environ.get("MFA_GEMINI_THINKING_BUDGET")
            extra["thinking_budget"] = int(budget) if budget else GEMINI_THINKING_BUDGET[effort]
        return ChatVertexAI(
            model=model_id,
            project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
            # Gemini 3.x models are served from the global endpoint.
            location=os.environ.get("MFA_GEMINI_LOCATION", "global"),
            **extra,
        )

    if provider == "anthropic_vertex":
        from langchain_google_vertexai.model_garden import ChatAnthropicVertex

        extra = {"model_kwargs": {"output_config": {"effort": effort}}} if effort else {}
        return ChatAnthropicVertex(
            model_name=model_id,
            project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
            location=os.environ.get("MFA_CLAUDE_VERTEX_REGION", "global"),
            max_tokens=16000,
            **extra,
        )

    raise ValueError(f"Unknown model provider {provider!r} in {model!r}")
