"""Runs one eval dataset against any Runner and writes comparable reports.

Trust rules:

* one case failing in the harness (not the agent) is recorded as a failed
  case with a ``harness`` check, never an exception that discards the run;
* checks that could not be graded (``Check.error``) are reported separately
  and are not passes;
* ``trials`` runs every case N times; pass rates carry a 95% Wilson interval so
  one flaky case is not read as a platform difference.
"""

from __future__ import annotations

import json
import math
import os
import re
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

_SECRET_NAME = re.compile(r"KEY|SECRET|TOKEN|PASSWORD|CREDENTIAL", re.IGNORECASE)


def environment_snapshot() -> dict[str, str]:
    """Non-secret MFA_* settings in effect for this run, so a report says what data/providers it used."""
    return {k: v for k, v in sorted(os.environ.items()) if k.startswith("MFA_") and not _SECRET_NAME.search(k)}


def wilson_interval(passed: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a pass rate from ``passed``/``total`` Bernoulli trials."""
    if total == 0:
        return (0.0, 0.0)
    p = passed / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3))


class CaseResult(BaseModel):
    case_id: str
    tags: list[str] = Field(default_factory=list)
    passed: bool
    checks: list[Check]
    result: AgentResult  # all turns combined: last reply, every tool call
    turns: list[AgentResult] = Field(default_factory=list)  # per-turn detail (used by exports)
    trial: int = 0

    @property
    def ungraded(self) -> list[Check]:
        return [ch for ch in self.checks if ch.error]


class EvalReport(BaseModel):
    agent: str
    target: str
    model: str | None = None
    started_at: str
    duration_s: float
    cases: list[CaseResult]
    trials: int = 1
    environment: dict[str, str] = Field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        return sum(c.passed for c in self.cases) / len(self.cases) if self.cases else 0.0

    @property
    def case_ids(self) -> list[str]:
        return list(dict.fromkeys(c.case_id for c in self.cases))

    def passes_for(self, case_id: str) -> tuple[int, int]:
        runs = [c for c in self.cases if c.case_id == case_id]
        return sum(c.passed for c in runs), len(runs)

    def summary(self) -> dict:
        lat = [c.result.latency_s for c in self.cases if c.result.latency_s is not None]
        all_checks = [ch for c in self.cases for ch in c.checks]
        graded = [ch for ch in all_checks if ch.passed is not None]
        judge = [ch for ch in all_checks if ch.name == "judge" and ch.passed is not None]
        passed = sum(c.passed for c in self.cases)
        return {
            "agent": self.agent,
            "target": self.target,
            "model": self.model,
            "cases": len(self.case_ids),
            "trials": self.trials,
            "runs": len(self.cases),
            "passed": passed,
            "pass_rate": round(self.pass_rate, 3),
            "pass_rate_ci95": list(wilson_interval(passed, len(self.cases))),
            "check_pass_rate": round(sum(ch.passed for ch in graded) / len(graded), 3) if graded else None,
            "judge_pass_rate": round(sum(ch.passed for ch in judge) / len(judge), 3) if judge else None,
            "judge_errors": sum(ch.error for ch in all_checks if ch.name == "judge"),
            "harness_errors": sum(any(ch.name == "harness" for ch in c.checks) for c in self.cases),
            "errors": sum(c.result.error is not None for c in self.cases),
            "latency_p50_s": round(median(lat), 2) if lat else None,
            "latency_mean_s": round(mean(lat), 2) if lat else None,
            "avg_tool_calls": round(mean(len(c.result.tool_calls) for c in self.cases), 2) if self.cases else None,
        }


def run_case(runner: Runner, case: EvalCase, judge: Judge | None, trial: int = 0) -> CaseResult:
    """Run every turn of one case on one thread and grade it. Never raises."""
    thread_id = f"eval-{case.id}-{trial}-{uuid.uuid4().hex[:8]}"
    try:
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
        checks = grade(case, combined, judge, turns)
    except Exception as exc:
        # A bug in a runner or grader fails this case, not the whole run.
        detail = f"{type(exc).__name__}: {exc}"
        combined = AgentResult(target=runner.target, input="\n".join(case.turns), output="", error=detail)
        return CaseResult(
            case_id=case.id, tags=case.tags, passed=False, trial=trial,
            checks=[Check(name="harness", passed=False, detail=detail)], result=combined, turns=[],
        )
    passed = all(ch.passed is not False for ch in checks)
    return CaseResult(case_id=case.id, tags=case.tags, passed=passed, checks=checks, result=combined, turns=turns, trial=trial)


def run_eval(
    *,
    agent: str,
    runner: Runner,
    cases: list[EvalCase],
    judge: Judge | None = None,
    concurrency: int = 4,
    model: str | None = None,
    trials: int = 1,
) -> EvalReport:
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    t0 = time.monotonic()
    jobs = [(c, trial) for trial in range(max(1, trials)) for c in cases]
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        results = list(pool.map(lambda job: run_case(runner, job[0], judge, job[1]), jobs))
    return EvalReport(
        agent=agent,
        target=runner.target,
        model=model,
        started_at=started,
        duration_s=round(time.monotonic() - t0, 2),
        cases=results,
        trials=max(1, trials),
        environment=environment_snapshot(),
    )


def _ci(s: dict) -> str:
    lo, hi = s["pass_rate_ci95"]
    return f"{lo:.0%}–{hi:.0%}"


def to_markdown(report: EvalReport) -> str:
    s = report.summary()
    runs = f" over {s['runs']} runs ({s['trials']} trials)" if s["trials"] > 1 else ""
    lines = [
        f"# Eval: `{report.agent}` on `{report.target}`",
        "",
        f"Model: `{report.model}` · started {report.started_at} · {report.duration_s}s",
        "",
        f"**{s['passed']}/{s['runs']} passed ({s['pass_rate']:.0%}, 95% CI {_ci(s)}){runs}** · "
        f"checks {s['check_pass_rate']} · judge {s['judge_pass_rate']} · "
        f"judge errors {s['judge_errors']} · agent errors {s['errors']} · harness errors {s['harness_errors']} · "
        f"p50 latency {s['latency_p50_s']}s",
        "",
    ]
    if s["judge_errors"]:
        lines += [f"> {s['judge_errors']} judge check(s) could not be graded; they are not counted as passes.", ""]
    if report.environment:
        lines += ["Environment: " + ", ".join(f"`{k}={v}`" for k, v in report.environment.items()), ""]
    lines += ["| case | result | failed checks | tools called |", "|---|---|---|---|"]
    for cid in report.case_ids:
        runs_for = [c for c in report.cases if c.case_id == cid]
        p, n = report.passes_for(cid)
        verdict = f"{p}/{n}" if n > 1 else ("PASS" if p else "FAIL")
        failed = ", ".join(sorted({ch.name for c in runs_for for ch in c.checks if ch.passed is False})) or "-"
        tools = ", ".join(dict.fromkeys(t for c in runs_for for t in c.result.tool_names)) or "-"
        lines.append(f"| {cid} | {verdict} | {failed} | {tools} |")
    fails = [c for c in report.cases if not c.passed]
    if fails:
        lines += ["", "## Failures", ""]
        for c in fails:
            lines.append(f"### {c.case_id}" + (f" (trial {c.trial})" if report.trials > 1 else ""))
            for ch in c.checks:
                if ch.passed is False:
                    lines.append(f"- `{ch.name}`: {ch.detail}")
            lines.append(f"- output: {c.result.output[:500]!r}")
            lines.append("")
    ungraded = [(c, ch) for c in report.cases for ch in c.ungraded]
    if ungraded:
        lines += ["", "## Not graded", ""]
        for c, ch in ungraded:
            lines.append(f"- {c.case_id} `{ch.name}`: {ch.detail}")
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
    ids = list(dict.fromkeys(cid for r in reports for cid in r.case_ids))
    for cid in ids:
        cells = []
        for r in reports:
            p, n = r.passes_for(cid)
            cells.append("-" if n == 0 else (f"{p}/{n}" if n > 1 else ("PASS" if p else "FAIL")))
        lines.append(f"| {cid} | " + " | ".join(cells) + " |")
    sums = [r.summary() for r in reports]
    lines.append("| **pass rate** | " + " | ".join(f"**{s['pass_rate']:.0%}** ({_ci(s)})" for s in sums) + " |")
    lines.append("| runs | " + " | ".join(f"{s['runs']} ({s['trials']} trials)" for s in sums) + " |")
    lines.append("| judge errors | " + " | ".join(str(s["judge_errors"]) for s in sums) + " |")
    lines.append("| p50 latency (s) | " + " | ".join(str(s["latency_p50_s"]) for s in sums) + " |")
    if any(s["pass_rate_ci95"][0] > o["pass_rate_ci95"][1] for s in sums for o in sums):
        lines += ["", "Intervals do not overlap for at least one pair: the difference is unlikely to be sampling noise."]
    else:
        lines += ["", "All pass-rate intervals overlap: the differences are within sampling noise at these sample sizes."]
    return "\n".join(lines) + "\n"
