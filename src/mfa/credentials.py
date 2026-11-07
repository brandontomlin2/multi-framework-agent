"""Anthropic credentials for mfa.

``MFA_ANTHROPIC_API_KEY`` takes precedence over ``ANTHROPIC_API_KEY``. Use it on
machines where other tools also read ``ANTHROPIC_API_KEY``, such as a Claude Code
cloud environment, where that name would also be picked up by Claude Code itself.
With neither set, the SDK's normal resolution applies (``ant auth login``
profile, workload identity federation, ...).
"""

from __future__ import annotations

import os


def anthropic_api_key() -> str | None:
    return os.environ.get("MFA_ANTHROPIC_API_KEY") or None


def anthropic_client():
    import anthropic

    key = anthropic_api_key()
    return anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()
