"""The normalized output every target returns, so one eval harness can grade them all."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    output: str | None = None


class AgentResult(BaseModel):
    target: str
    input: str
    output: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    latency_s: float | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def tool_names(self) -> list[str]:
        return [c.name for c in self.tool_calls]
