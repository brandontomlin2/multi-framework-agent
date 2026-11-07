"""Graders: deterministic checks on output + tool trajectory, plus an optional LLM judge.

All graders consume the normalized ``AgentResult``, so they never know (or
care) which framework produced it.
"""

from __future__ import annotations

import os
import re
from typing import Any, Callable

from pydantic import BaseModel

from mfa.evals.dataset import EvalCase, Expectations
from mfa.runtime.result import AgentResult


class Check(BaseModel):
    name: str
    passed: bool | None  # None = skipped (e.g. judge disabled)
    detail: str = ""


def _lc(s: str) -> str:
    return s.casefold()


def _args_match(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    for k, v in expected.items():
        got = actual.get(k)
        if isinstance(v, str) and isinstance(got, str):
            if _lc(v) != _lc(got):
                return False
        elif got != v:
            return False
    return True


def deterministic_checks(exp: Expectations, result: AgentResult) -> list[Check]:
    checks: list[Check] = []
    names = result.tool_names
    out = _lc(result.output)

    checks.append(Check(name="no_error", passed=result.error is None, detail=result.error or ""))

    for tool in exp.tools_called:
        checks.append(Check(name=f"called:{tool}", passed=tool in names, detail=f"calls={names}"))
    for tool in exp.tools_not_called:
        checks.append(Check(name=f"not_called:{tool}", passed=tool not in names, detail=f"calls={names}"))
    if exp.no_tool_calls:
        checks.append(Check(name="no_tool_calls", passed=not names, detail=f"calls={names}"))
    if exp.max_tool_calls is not None:
        checks.append(
            Check(name="max_tool_calls", passed=len(names) <= exp.max_tool_calls, detail=f"{len(names)} calls")
        )
    for tool, expected_args in exp.tool_args.items():
        actual = [c.args for c in result.tool_calls if c.name == tool]
        ok = any(_args_match(expected_args, a) for a in actual)
        checks.append(Check(name=f"args:{tool}", passed=ok, detail=f"expected⊆{expected_args} got={actual}"))

    for s in exp.output_contains:
        checks.append(Check(name=f"contains:{s}", passed=_lc(s) in out))
    if exp.output_contains_any:
        ok = any(_lc(s) in out for s in exp.output_contains_any)
        checks.append(Check(name="contains_any", passed=ok, detail=str(exp.output_contains_any)))
    for s in exp.output_not_contains:
        checks.append(Check(name=f"not_contains:{s}", passed=_lc(s) not in out))
    if exp.output_regex:
        ok = re.search(exp.output_regex, result.output, re.IGNORECASE | re.MULTILINE) is not None
        checks.append(Check(name="regex", passed=ok, detail=exp.output_regex))
    return checks


# --- LLM judge -------------------------------------------------------------

Judge = Callable[[EvalCase, AgentResult], Check]


class _Verdict(BaseModel):
    reasoning: str
    passed: bool


JUDGE_SYSTEM = (
    "You grade a customer-facing AI agent's reply against a rubric. "
    "Judge only what the rubric asks. Tool calls the agent made are listed so you can "
    "verify that claims in the reply are grounded in tool output. Be strict: if any "
    "rubric requirement is unmet, the verdict is fail."
)


def claude_judge(model: str | None = None) -> Judge:
    """LLM-as-judge on Claude. The same judge grades every target, so scores are comparable."""
    import anthropic

    client = anthropic.Anthropic()
    model = model or os.environ.get("MFA_JUDGE_MODEL", "claude-opus-5-5")

    def judge(case: EvalCase, result: AgentResult) -> Check:
        transcript = "\n".join(f"USER: {t}" for t in case.turns)
        tools = "\n".join(f"- {c.name}({c.args}) -> {c.output}" for c in result.tool_calls) or "(none)"
        prompt = (
            f"<conversation>\n{transcript}\n</conversation>\n"
            f"<tool_calls>\n{tools}\n</tool_calls>\n"
            f"<final_reply>\n{result.output}\n</final_reply>\n"
            f"<rubric>\n{case.judge}\n</rubric>\n"
            "Does the final reply satisfy the rubric?"
        )
        try:
            resp = client.beta.messages.parse(
                model=model,
                max_tokens=4000,
                system=JUDGE_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
                output_format=_Verdict,
                output_config={"effort": "low"},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.APIError as exc:
            return Check(name="judge", passed=None, detail=f"judge error: {exc!r}")
        if resp.stop_reason == "refusal" or resp.parsed_output is None:
            return Check(name="judge", passed=None, detail=f"judge returned no verdict ({resp.stop_reason})")
        v = resp.parsed_output
        return Check(name="judge", passed=v.passed, detail=v.reasoning)

    return judge


def grade(case: EvalCase, result: AgentResult, judge: Judge | None) -> list[Check]:
    checks = deterministic_checks(case.expect, result)
    if case.judge:
        if judge is None:
            checks.append(Check(name="judge", passed=None, detail="judge disabled"))
        elif result.error is None:
            checks.append(judge(case, result))
        else:
            checks.append(Check(name="judge", passed=False, detail="agent errored"))
    return checks
