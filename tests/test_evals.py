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
    spec = load_agent("weather_agent")
    cases = load_cases(spec.eval_dataset)
    assert len(cases) >= 10
    tool_names = set(spec.runtime_tool_names())  # the agent's tools plus the skill file tools
    skill_names = {sk.name for sk in spec.skills}
    for c in cases:
        for name in [*c.expect.tools_called, *c.expect.tools_not_called, *c.expect.tool_args, *c.expect.tool_args_contain,
                     *(e.tool for e in c.expect.tool_calls)]:
            assert name in tool_names, f"{c.id} references unknown tool {name}"
        for name in [*c.expect.skills_used, *c.expect.skills_not_used]:
            assert name in skill_names, f"{c.id} references unknown skill {name}"
    assert [c.id for c in load_cases(spec.eval_dataset, tags=["skill"])]
    assert load_cases(spec.eval_dataset, tags=["smoke"])
    assert [c.id for c in load_cases(spec.eval_dataset, ids=["asks_for_location"])] == ["asks_for_location"]


def test_deterministic_checks():
    case = EvalCase.model_validate({
        "id": "x", "input": "hi",
        "expect": {
            "tools_called": ["a"], "tools_not_called": ["b"],
            "tool_args": {"a": {"order_id": "ord-1"}}, "tool_args_contain": {"a": {"order_id": "ord"}},
            "output_contains": ["Shipped"], "output_contains_any": ["ups", "fedex"],
            "output_not_contains": ["sorry"], "output_regex": r"TCK-\d+", "max_tool_calls": 1,
        },
    })
    good = AgentResult(target="t", input="hi", output="shipped via UPS, TCK-12",
                       tool_calls=[ToolCall(name="a", args={"order_id": "ORD-1", "extra": 1})])
    assert all(c.passed for c in deterministic_checks(case.expect, good))
    bad = AgentResult(target="t", input="hi", output="sorry", tool_calls=[ToolCall(name="b")])
    failed = {c.name for c in deterministic_checks(case.expect, bad) if not c.passed}
    assert {"called:a", "not_called:b", "args:a", "args_contain:a", "contains:Shipped", "regex"} <= failed


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

    def judge(case, result, turns):
        judged.append((case.id, len(turns)))
        return Check(name="judge", passed=result.output == "hello")

    report = run_eval(agent="demo", runner=runner, cases=cases, judge=judge, concurrency=1)
    assert report.pass_rate == 1.0
    assert judged == [("one", 1)]
    assert runner.threads[1] == runner.threads[2]  # both turns of case "two" share a thread
    assert report.summary()["judge_pass_rate"] == 1.0

    json_path, md_path = save_report(report, tmp_path)
    assert json.loads(json_path.read_text())["summary"]["passed"] == 2
    assert "2/2 passed" in md_path.read_text()

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


def test_judge_errors_are_not_passes_and_are_counted():
    case = EvalCase(id="j", input="q", judge="anything")
    errored = Check(name="judge", passed=None, error=True, detail="judge error: 429")
    report = run_eval(agent="d", runner=CannedRunner("local", {"q": ("ok", [])}), cases=[case],
                      judge=lambda c, r, t: errored)
    s = report.summary()
    assert report.cases[0].passed  # no check failed...
    assert s["judge_errors"] == 1 and s["judge_pass_rate"] is None  # ...but nothing was judged either
    assert "could not be graded" in to_markdown(report)


def test_one_crashing_case_does_not_discard_the_run():
    class Flaky(CannedRunner):
        def run(self, user_input, *, thread_id=None):
            if user_input == "crash":
                raise RuntimeError("runner bug")
            return super().run(user_input, thread_id=thread_id)

    cases = [EvalCase(id="ok", input="q"), EvalCase(id="bad", input="crash")]
    report = run_eval(agent="d", runner=Flaky("local", {"q": ("fine", [])}), cases=cases, judge=None)
    by_id = {c.case_id: c for c in report.cases}
    assert by_id["ok"].passed and not by_id["bad"].passed
    assert by_id["bad"].checks[0].name == "harness" and "runner bug" in by_id["bad"].checks[0].detail
    assert report.summary()["harness_errors"] == 1


def test_trials_give_per_case_counts_and_an_interval(tmp_path):
    class Alternating(CannedRunner):
        def __init__(self):
            super().__init__("local", {})
            self.n = 0

        def run(self, user_input, *, thread_id=None):
            self.n += 1
            return AgentResult(target=self.target, input=user_input, output="yes" if self.n % 2 else "no")

    case = EvalCase(id="flip", input="q", expect={"output_contains": ["yes"]})
    report = run_eval(agent="d", runner=Alternating(), cases=[case], judge=None, trials=4, concurrency=1)
    s = report.summary()
    assert s["runs"] == 4 and s["trials"] == 4 and s["pass_rate"] == 0.5
    lo, hi = s["pass_rate_ci95"]
    assert lo < 0.5 < hi
    assert "| flip | 2/4 |" in to_markdown(report)
    p, _ = save_report(report, tmp_path)
    assert "2/4" in compare([p]) and "sampling noise" not in compare([p])  # one report: nothing to compare
    assert "no evidence of a difference" in compare([p, p])


def test_report_records_non_secret_environment(monkeypatch):
    monkeypatch.setenv("MFA_WEATHER_PROVIDER", "fixture")
    monkeypatch.setenv("MFA_ANTHROPIC_API_KEY", "sk-secret")
    report = run_eval(agent="d", runner=CannedRunner("local", {}), cases=[EvalCase(id="e", input="q")], judge=None)
    assert report.environment["MFA_WEATHER_PROVIDER"] == "fixture"
    assert "MFA_ANTHROPIC_API_KEY" not in report.environment


def test_interval_is_over_cases_not_runs_and_no_data_means_no_information():
    from mfa.evals.harness import wilson_interval

    assert wilson_interval(0, 0) == (0.0, 1.0)
    cases = [EvalCase(id=f"c{i}", input="q", expect={"output_contains": ["ok"]}) for i in range(11)]
    one_run = run_eval(agent="d", runner=CannedRunner("local", {"q": ("ok", [])}), cases=cases, judge=None)
    many = run_eval(agent="d", runner=CannedRunner("local", {"q": ("ok", [])}), cases=cases, judge=None, trials=5)
    # 55 correlated runs must not look like 55 independent cases.
    assert many.summary()["pass_rate_ci95"] == one_run.summary()["pass_rate_ci95"]


def test_grader_crash_keeps_the_trajectory_and_is_ungraded():
    def exploding_judge(case, result, turns):
        raise ValueError("verdict was not JSON")

    case = EvalCase(id="j", input="q", judge="rubric")
    report = run_eval(agent="d", runner=CannedRunner("local", {"q": ("hello", [("t", {})])}), cases=[case], judge=exploding_judge)
    c = report.cases[0]
    assert not c.passed and c.result.output == "hello" and c.result.tool_names == ["t"] and c.result.error is None
    assert c.checks[0].name == "harness" and c.checks[0].error and "verdict was not JSON" in c.checks[0].detail
    assert "Not graded" in to_markdown(report)


def test_export_keeps_eval_case_ids_unique_across_trials():
    from mfa.evals.export import to_gemini_eval_dataset

    cases = [EvalCase(id="a", input="q"), EvalCase(id="b", input="q")]
    report = run_eval(agent="d", runner=CannedRunner("local", {"q": ("ok", [])}), cases=cases, judge=None, trials=2)
    ids = [c["eval_case_id"] for c in to_gemini_eval_dataset(report)["eval_cases"]]
    assert len(ids) == len(set(ids)) == 4 and {c["mfa_trial"] for c in to_gemini_eval_dataset(report)["eval_cases"]} == {0, 1}
    single = run_eval(agent="d", runner=CannedRunner("local", {"q": ("ok", [])}), cases=cases, judge=None)
    assert [c["eval_case_id"] for c in to_gemini_eval_dataset(single)["eval_cases"]] == ["a", "b"]


def test_cli_gate_fails_on_ungraded_checks_only_when_gated(monkeypatch, tmp_path, capsys):
    from mfa import cli

    class AlwaysAsks(CannedRunner):
        def run(self, user_input, *, thread_id=None):
            return AgentResult(target=self.target, input=user_input, output="Which city do you mean?", latency_s=0.1)

    monkeypatch.setattr(cli, "_runner", lambda spec, target, args: AlwaysAsks("local", {}))
    errored = Check(name="judge", passed=None, error=True, detail="judge error: 429")
    monkeypatch.setattr("mfa.evals.claude_judge", lambda model, effort: (lambda c, r, t: errored))
    base = ["eval", "weather_agent", "-t", "local", "--case", "asks_for_location", "--out", str(tmp_path)]
    assert cli.main(base) == 0  # exploratory run: reported, not failed
    assert cli.main([*base, "--min-pass-rate", "0.5"]) == 1  # gated run: an ungraded check is not a pass
    assert "could not be graded" in capsys.readouterr().err
    assert cli.main([*base, "--min-pass-rate", "0.5", "--allow-ungraded"]) == 0


def test_skill_checks_look_at_which_skill_md_was_read():
    exp = {"skills_used": ["weather-briefing"], "skills_not_used": ["other-skill"]}
    case = EvalCase.model_validate({"id": "s", "input": "q", "expect": exp})
    used = AgentResult(target="t", input="q", output="ok", tool_calls=[
        ToolCall(name="read", args={"file_path": "/mnt/anywhere/weather-briefing/SKILL.md"})])
    assert {c.name: c.passed for c in deterministic_checks(case.expect, used)}.items() >= {
        "skill:weather-briefing": True, "skill_not:other-skill": True}.items()
    wrong = AgentResult(target="t", input="q", output="ok", tool_calls=[
        ToolCall(name="read", args={"file_path": "other-skill/SKILL.md"})])
    assert {c.name: c.passed for c in deterministic_checks(case.expect, wrong)}.items() >= {
        "skill:weather-briefing": False, "skill_not:other-skill": False}.items()


def test_tool_calls_expectation_needs_distinct_matching_calls():
    exp = {"tool_calls": [{"tool": "w", "args_contain": {"location": "paris"}}, {"tool": "w", "args_contain": {"location": "tokyo"}}]}
    case = EvalCase.model_validate({"id": "p", "input": "q", "expect": exp})
    both = AgentResult(target="t", input="q", output="", tool_calls=[
        ToolCall(name="w", args={"location": "Paris, France"}), ToolCall(name="w", args={"location": "Tokyo"})])
    assert all(c.passed for c in deterministic_checks(case.expect, both) if c.name.startswith("call["))
    twice_paris = AgentResult(target="t", input="q", output="", tool_calls=[
        ToolCall(name="w", args={"location": "Paris"}), ToolCall(name="w", args={"location": "Paris"})])
    names = {c.name: c.passed for c in deterministic_checks(case.expect, twice_paris)}
    assert names["call[0]:w"] is True and names["call[1]:w"] is False  # the same call cannot satisfy both


def test_args_contain_reaches_nested_arguments_and_sequence_is_ordered():
    from mfa.evals.dataset import Expectations, ExpectedCall
    from mfa.evals.graders import deterministic_checks
    from mfa.runtime.result import AgentResult, ToolCall

    result = AgentResult(target="local", input="i", output="o", tool_calls=[
        ToolCall(name="lookup_order", args={"order_id": ""}),
        ToolCall(name="search_products", args={"query": "Running shoes", "filters": {"category": "shoes", "sizes": ["9", "10"], "max_price": 100.0}, "page": 1}),
        ToolCall(name="start_return", args={"order_id": "ORD-1001", "sku": "SH-RUN-9", "reason": "too_small"}),
    ])
    exp = Expectations(
        tool_args_contain={"search_products": {"filters.category": "shoes", "filters.sizes": ["10"], "filters.max_price": 100, "query": "running"}},
        tool_calls=[ExpectedCall(tool="lookup_order", args={"order_id": ""}),
                    ExpectedCall(tool="search_products", args_contain={"filters": {"sizes": ["9", "10"]}, "page": 1})],
        tool_sequence=["lookup_order", "start_return"],
    )
    checks = {c.name: c for c in deterministic_checks(exp, result)}
    assert all(c.passed for c in checks.values()), {n: c.detail for n, c in checks.items() if not c.passed}

    wrong = Expectations(
        tool_args_contain={"search_products": {"filters.sizes": ["11"], "filters.max_price": 99}},
        tool_sequence=["start_return", "lookup_order"],
        tool_calls=[ExpectedCall(tool="search_products", args_contain={"filters.category": "apparel"})],
    )
    checks = {c.name: c for c in deterministic_checks(wrong, result)}
    assert checks["args_contain:search_products"].passed is False and checks["sequence"].passed is False
    assert checks["call[0]:search_products"].passed is False
