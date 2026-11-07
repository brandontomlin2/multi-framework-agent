"""Export mfa eval runs to Google's Gemini platform evaluation dataset format.

The output follows the ``EvaluationDataset`` shape documented in the official
agents-cli eval skill (``eval_cases[].agent_data.turns[].events[]``), so the
same runs graded by ``mfa eval`` can also be graded with
``agents-cli eval grade --traces <file> --config <eval_config.yaml>`` or the
Vertex/Agent Platform evaluation SDK — for any target, including Claude
Managed Agents.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mfa.evals.harness import EvalReport


def _event(author: str, part: dict[str, Any], role: str) -> dict[str, Any]:
    return {"author": author, "content": {"role": role, "parts": [part]}}


def to_gemini_eval_dataset(report: EvalReport, *, cases_by_id: dict | None = None) -> dict[str, Any]:
    agent = report.agent
    eval_cases = []
    for c in report.cases:
        turns_out = []
        per_turn = c.turns or [c.result]
        inputs = c.result.input.split("\n") if not c.turns else [t.input for t in c.turns]
        for i, (user_text, turn) in enumerate(zip(inputs, per_turn)):
            events = [_event("user", {"text": user_text}, "user")]
            for call in turn.tool_calls:
                events.append(_event(agent, {"function_call": {"name": call.name, "args": call.args}}, "model"))
                try:
                    response: Any = json.loads(call.output) if call.output else {}
                except ValueError:
                    response = {"result": call.output}
                if not isinstance(response, dict):
                    response = {"result": response}
                events.append(_event("tool", {"function_response": {"name": call.name, "response": response}}, "user"))
            if turn.output:
                events.append(_event(agent, {"text": turn.output}, "model"))
            turns_out.append({"turn_index": i, "events": events})

        case: dict[str, Any] = {
            "eval_case_id": c.case_id,
            "agent_data": {
                "agents": {agent: {"agent_id": agent, "description": f"{agent} on {report.target}"}},
                "turns": turns_out,
            },
            # Extra fields are allowed and available to custom metrics.
            "mfa_target": report.target,
            "mfa_model": report.model,
            "mfa_passed": c.passed,
            "mfa_checks": [ch.model_dump() for ch in c.checks],
        }
        source = (cases_by_id or {}).get(c.case_id)
        if source is not None and source.judge:
            case["rubric"] = source.judge
        eval_cases.append(case)
    return {"eval_cases": eval_cases}


def export_report(report_path: str | Path, out_path: str | Path, dataset: str | Path | None = None) -> Path:
    from mfa.evals.dataset import load_cases

    report = EvalReport.model_validate(json.loads(Path(report_path).read_text()))
    cases_by_id = {c.id: c for c in load_cases(dataset)} if dataset else None
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(to_gemini_eval_dataset(report, cases_by_id=cases_by_id), indent=2, default=str))
    return out
