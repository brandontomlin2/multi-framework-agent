"""Chat-model factory for the LangGraph runtime.

Model strings use ``provider:model``:

* ``anthropic:claude-opus-5-5``          - Claude API (``ANTHROPIC_API_KEY``)
* ``google_vertexai:gemini-3.1-pro-preview`` - Gemini on Vertex AI (ADC)
* ``anthropic_vertex:claude-opus-5-5``   - Claude on Vertex AI (ADC)

Imports are lazy so a deployment only needs the provider it actually uses.
"""

from __future__ import annotations

import os

from langchain_core.language_models import BaseChatModel


def parse_model(model: str) -> tuple[str, str]:
    if ":" not in model:
        raise ValueError(f"Model {model!r} must look like 'provider:model-id'")
    provider, model_id = model.split(":", 1)
    return provider, model_id


def make_chat_model(model: str) -> BaseChatModel:
    provider, model_id = parse_model(model)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        # Current Claude models think adaptively by default and reject sampling
        # params, so only max_tokens is set.
        return ChatAnthropic(model=model_id, max_tokens=16000)

    if provider == "google_vertexai":
        from langchain_google_vertexai import ChatVertexAI

        return ChatVertexAI(
            model=model_id,
            project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
            # Gemini 3.x models are served from the global endpoint.
            location=os.environ.get("MFA_GEMINI_LOCATION", "global"),
        )

    if provider == "anthropic_vertex":
        from langchain_google_vertexai.model_garden import ChatAnthropicVertex

        return ChatAnthropicVertex(
            model_name=model_id,
            project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
            location=os.environ.get("MFA_CLAUDE_VERTEX_REGION", "global"),
            max_tokens=16000,
        )

    raise ValueError(f"Unknown model provider {provider!r} in {model!r}")
