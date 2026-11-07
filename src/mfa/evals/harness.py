"""Runs one eval dataset against any Runner and writes comparable reports."""

from __future__ import annotations

import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean, median

from pydantic import BaseModel, Field

from mfa.evals.dataset import EvalCase
from mfa.evals.graders import Check, Judge, grade
from mfa.runtime.result import AgentResult
from mfa.targets.base import Runner


class CaseResult(BaseModel):
    case_id: str
    tags: list[str] = Field(default_factory=list)
    passed: bool
    checks: list[Check]
    result: AgentResult


class EvalReport(BaseModel):
    agent: str
    target: str
    model: str | None = None
    started_at: str
    duration_s: float
    cases: list[CaseResult]

    @property
    def pass_rate(self) -> float:
        return sum(c.passed for c in self.cases) / len(self.cases) if self.cases else 0.0

    def summary(self) -> dict:
        lat = [c.result.latency_s for c in self.cases if c.result.latency_s is not None]
        graded = [ch for c in self.cases for ch in c.checks if ch.passed is not None]
        judge = [ch for c in self.cases for ch in c.checks if ch.name == "judge" and ch.passed is not None]
        return {
            "agent": self.agent,
            "target": self.target,
            "model": self.model,
            "cases": len(self.cases),
            "passed": sum(c.passed for c in self.cases),
            "pass_rate": round(self.pass_rate, 3),
            "check_pass_rate": round(sum(ch.passed for ch in graded) / len(graded), 3) if graded else None,
            "judge_pass_rate": round(sum(ch.passed for ch in judge) / len(judge), 3) if judge else None,
            "errors": sum(c.result.error is not None for c in self.cases),
            "latency_p50_s": round(median(lat), 2) if lat else None,
            "latency_mean_s": round(mean(lat), 2) if lat else None,
            "avg_tool_calls": round(mean(len(c.result.tool_calls) for c in self.cases), 2) if self.cases else None,
        }


def run_case(runner: Runner, case: EvalCase, judge: Judge | None) -> CaseResult:
    thread_id = f"eval-{case.id}-{uuid.uuid4().hex[:8]}"
    turns = [runner.run(t, thread_id=thread_id) for t in case.turns]
    last = turns[-1]
    combined = AgentResult(
        target=last.target,
        input="\n".join(case.turns),
        output=last.output,
        tool_calls=[c for t in turns for c in t.tool_calls],
        latency_s=round(sum(t.latency_s or 0 for t in turns), 3),
        error=next((t.error for t in turns if t.error), None),
        metadata=last.metadata,
    )
    checks = grade(case, combined, judge)
    passed = all(ch.passed is not False for ch in checks)
    return CaseResult(case_id=case.id, tags=case.tags, passed=passed, checks=checks, result=combined)


def run_eval(
    *,
    agent: str,
    runner: Runner,
    cases: list[EvalCase],
    judge: Judge | None = None,
    concurrency: int = 4,
    model: str | None = None,
) -> EvalReport:
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        results = list(pool.map(lambda c: run_case(runner, c, judge), cases))
    return EvalReport(
        agent=agent,
        target=runner.target,
        model=model,
        started_at=started,
        duration_s=round(time.monotonic() - t0, 2),
        cases=results,
    )


def to_markdown(report: EvalReport) -> str:
    s = report.summary()
    lines = [
        f"# Eval: `{report.agent}` on `{report.target}`",
        "",
        f"Model: `{report.model}` · started {report.started_at} · {report.duration_s}s",
        "",
        f"**{s['passed']}/{s['cases']} cases passed ({s['pass_rate']:.0%})** · "
        f"checks {s['check_pass_rate']} · judge {s['judge_pass_rate']} · errors {s['errors']} · "
        f"p50 latency {s['latency_p50_s']}s",
        "",
        "| case | result | failed checks | tools called |",
        "|---|---|---|---|",
    ]
    for c in report.cases:
        failed = ", ".join(ch.name for ch in c.checks if ch.passed is False) or "-"
        lines.append(
            f"| {c.case_id} | {'PASS' if c.passed else 'FAIL'} | {failed} | {', '.join(c.result.tool_names) or '-'} |"
        )
    fails = [c for c in report.cases if not c.passed]
    if fails:
        lines += ["", "## Failures", ""]
        for c in fails:
            lines.append(f"### {c.case_id}")
            for ch in c.checks:
                if ch.passed is False:
                    lines.append(f"- `{ch.name}`: {ch.detail}")
            lines.append(f"- output: {c.result.output[:500]!r}")
            lines.append("")
    return "\n".join(lines) + "\n"


def save_report(report: EvalReport, out_dir: str | Path = "eval_results") -> tuple[Path, Path]:
    out = Path(out_dir) / report.agent
    out.mkdir(parents=True, exist_ok=True)
    stamp = report.started_at.replace(":", "").replace("-", "")
    base = out / f"{report.target}-{stamp}"
    json_path = base.with_suffix(".json")
    json_path.write_text(json.dumps({"summary": report.summary(), **report.model_dump()}, indent=2, default=str))
    md_path = base.with_suffix(".md")
    md_path.write_text(to_markdown(report))
    return json_path, md_path


def compare(report_paths: list[str | Path]) -> str:
    """Side-by-side case matrix for reports from different targets (same dataset)."""
    reports = [EvalReport.model_validate(json.loads(Path(p).read_text())) for p in report_paths]
    header = "| case | " + " | ".join(r.target for r in reports) + " |"
    lines = ["# Cross-target comparison", "", header, "|---|" + "---|" * len(reports)]
    ids = list(dict.fromkeys(c.case_id for r in reports for c in r.cases))
    by = [{c.case_id: c for c in r.cases} for r in reports]
    for cid in ids:
        cells = []
        for m in by:
            c = m.get(cid)
            cells.append("-" if c is None else ("PASS" if c.passed else "FAIL"))
        lines.append(f"| {cid} | " + " | ".join(cells) + " |")
    lines.append("| **pass rate** | " + " | ".join(f"**{r.pass_rate:.0%}**" for r in reports) + " |")
    lines.append(
        "| p50 latency (s) | " + " | ".join(str(r.summary()["latency_p50_s"]) for r in reports) + " |"
    )
    return "\n".join(lines) + "\n"
