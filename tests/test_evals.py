import json

from mfa.evals import Check, EvalCase, compare, load_cases, run_eval, save_report, to_markdown
from mfa.evals.graders import deterministic_checks
from mfa.registry import load_agent
from mfa.runtime.result import AgentResult, ToolCall


class CannedRunner:
    """Returns a fixed result per input; records thread ids to check multi-turn plumbing."""

    def __init__(self, target, responses):
        self.target = target
        self.responses = responses
        self.threads = []

    def run(self, user_input, *, thread_id=None):
        self.threads.append(thread_id)
        out, calls = self.responses.get(user_input, ("", []))
        return AgentResult(target=self.target, input=user_input, output=out,
                           tool_calls=[ToolCall(name=n, args=a) for n, a in calls], latency_s=0.1)


def test_shared_dataset_is_valid():
    spec = load_agent("support_agent")
    cases = load_cases(spec.eval_dataset)
    assert len(cases) >= 10
    tool_names = {t.name for t in spec.tools}
    for c in cases:
        for name in [*c.expect.tools_called, *c.expect.tools_not_called, *c.expect.tool_args]:
            assert name in tool_names, f"{c.id} references unknown tool {name}"
    assert load_cases(spec.eval_dataset, tags=["smoke"])
    assert [c.id for c in load_cases(spec.eval_dataset, ids=["out_of_scope"])] == ["out_of_scope"]


def test_deterministic_checks():
    case = EvalCase.model_validate({
        "id": "x", "input": "hi",
        "expect": {
            "tools_called": ["a"], "tools_not_called": ["b"],
            "tool_args": {"a": {"order_id": "ord-1"}},
            "output_contains": ["Shipped"], "output_contains_any": ["ups", "fedex"],
            "output_not_contains": ["sorry"], "output_regex": r"TCK-\d+", "max_tool_calls": 1,
        },
    })
    good = AgentResult(target="t", input="hi", output="shipped via UPS, TCK-12",
                       tool_calls=[ToolCall(name="a", args={"order_id": "ORD-1", "extra": 1})])
    assert all(c.passed for c in deterministic_checks(case.expect, good))
    bad = AgentResult(target="t", input="hi", output="sorry", tool_calls=[ToolCall(name="b")])
    failed = {c.name for c in deterministic_checks(case.expect, bad) if not c.passed}
    assert {"called:a", "not_called:b", "args:a", "contains:Shipped", "regex"} <= failed


def test_harness_multi_turn_judge_and_reports(tmp_path):
    cases = [
        EvalCase(id="one", input="q1", expect={"tools_called": ["t"]}, judge="be nice"),
        EvalCase(id="two", turns=["a", "b"], expect={"tools_called": ["x", "y"], "output_contains": ["done"]}),
    ]
    runner = CannedRunner("local", {
        "q1": ("hello", [("t", {})]),
        "a": ("", [("x", {})]),
        "b": ("done", [("y", {})]),
    })
    judged = []

    def judge(case, result):
        judged.append(case.id)
        return Check(name="judge", passed=result.output == "hello")

    report = run_eval(agent="demo", runner=runner, cases=cases, judge=judge, concurrency=1)
    assert report.pass_rate == 1.0
    assert judged == ["one"]
    assert runner.threads[1] == runner.threads[2]  # both turns of case "two" share a thread
    assert report.summary()["judge_pass_rate"] == 1.0

    json_path, md_path = save_report(report, tmp_path)
    assert json.loads(json_path.read_text())["summary"]["passed"] == 2
    assert "2/2 cases passed" in md_path.read_text()

    other = run_eval(agent="demo", runner=CannedRunner("claude_managed", {}), cases=cases, judge=None)
    assert other.pass_rate == 0.0
    assert "FAIL" in to_markdown(other)
    p2, _ = save_report(other, tmp_path / "b")
    table = compare([json_path, p2])
    assert "| one | PASS | FAIL |" in table


def test_judge_skipped_when_disabled_does_not_fail_case():
    case = EvalCase(id="j", input="q", judge="anything")
    report = run_eval(agent="d", runner=CannedRunner("local", {"q": ("ok", [])}), cases=[case], judge=None)
    assert report.cases[0].passed
    assert report.cases[0].checks[-1].passed is None
