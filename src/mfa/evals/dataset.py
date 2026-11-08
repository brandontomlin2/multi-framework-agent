"""Eval cases, shared verbatim by every target.

YAML format::

    cases:
      - id: weather_in_paris
        input: "What's the weather in Paris today?"   # or `turns: [..]` for multi-turn
        tags: [weather]
        expect:
          tools_called: [get_weather_today]      # each must be called at least once
          tools_not_called: []
          no_tool_calls: false
          tool_args:                             # exact (case-insensitive) match on some call's args
            get_weather_today: {location: Paris}
          tool_args_contain:                     # substring match: "Paris, France" also passes
            get_weather_today: {location: paris}
          max_tool_calls: 2
          output_contains: ["Paris"]             # all, case-insensitive
          output_contains_any: ["°F", "°C"]
          output_not_contains: ["I don't know"]
          output_regex: "\\\\d+ ?°"
          skills_used: [weather-briefing]        # the skill's SKILL.md was read (any target)
          skills_not_used: []
        judge: >-                                # optional LLM-as-judge rubric
          The reply gives today's conditions and temperatures that match the tool output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


class Expectations(BaseModel):
    tools_called: list[str] = Field(default_factory=list)
    tools_not_called: list[str] = Field(default_factory=list)
    no_tool_calls: bool = False
    tool_args: dict[str, dict[str, Any]] = Field(default_factory=dict)
    tool_args_contain: dict[str, dict[str, str]] = Field(default_factory=dict)
    max_tool_calls: int | None = None
    output_contains: list[str] = Field(default_factory=list)
    output_contains_any: list[str] = Field(default_factory=list)
    output_not_contains: list[str] = Field(default_factory=list)
    output_regex: str | None = None
    skills_used: list[str] = Field(default_factory=list)  # a skill counts as used when its SKILL.md was read
    skills_not_used: list[str] = Field(default_factory=list)


class EvalCase(BaseModel):
    id: str
    input: str | None = None
    turns: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    expect: Expectations = Field(default_factory=Expectations)
    judge: str | None = None

    @model_validator(mode="after")
    def _one_of_input_or_turns(self) -> "EvalCase":
        if bool(self.input) == bool(self.turns):
            raise ValueError(f"case {self.id!r}: set exactly one of `input` or `turns`")
        if self.input:
            self.turns = [self.input]
        return self


def load_cases(path: str | Path, *, tags: list[str] | None = None, ids: list[str] | None = None) -> list[EvalCase]:
    data = yaml.safe_load(Path(path).read_text())
    cases = [EvalCase.model_validate(c) for c in data["cases"]]
    seen: set[str] = set()
    for c in cases:
        if c.id in seen:
            raise ValueError(f"duplicate case id {c.id!r} in {path}")
        seen.add(c.id)
    if tags:
        cases = [c for c in cases if set(tags) & set(c.tags)]
    if ids:
        cases = [c for c in cases if c.id in ids]
    return cases
