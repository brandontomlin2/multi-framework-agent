"""Behavioural parity contract: the same scenario must produce the same AgentResult on every target.

The old parity test only showed a happy-path script came back. These scenarios
cover the places the targets used to diverge: a tool that raises, invalid and
coercible arguments, a model that loops past max_iterations, multi-turn state,
and recovering on the same thread after a failed turn. Every target runs the
real adapter code (graph, pickled Agent Engine app, FastAPI route, CMA event
loop) against the offline fakes.
"""

from __future__ import annotations

from datetime import date

import pytest
from fastapi.testclient import TestClient

from mfa.runtime.result import AgentResult
from mfa.spec import AGENT_ENGINE, CLAUDE_MANAGED, CLOUD_RUN, LOCAL, AgentSpec, format_tool_error, iteration_limit_error, tool
from mfa.runtime.graph import recursion_limit_for
from mfa.targets.agent_engine import AgentEngineRunner
from mfa.targets.agent_engine_app import LangGraphAgentApp
from mfa.targets.claude_managed import ClaudeManagedRunner
from mfa.targets.local import LocalRunner
from tests.fakes import FakeCMAClient, FakeRemoteAgentEngine, ScriptedChatModel


@tool
def boom(reason: str) -> str:
    """Always fails.

    Args:
        reason: Why it should fail.
    """
    raise RuntimeError(f"kaboom: {reason}")


@tool
def book(when: date, count: int = 1) -> dict:
    """Book seats for a day.

    Args:
        when: The day to book.
        count: Number of seats.
    """
    return {"when": when.isoformat(), "count": count, "types": [type(when).__name__, type(count).__name__]}


PARITY_SPEC = AgentSpec(
    name="parity_agent",
    description="Exercises the cross-target turn contract.",
    system_prompt="You are a booking assistant used in parity tests.",
    tools=[book, boom],
    models={LOCAL: "x:y", AGENT_ENGINE: "x:y", CLAUDE_MANAGED: "claude-test"},
    max_iterations=2,
)
SELF_REF = "tests.test_parity:PARITY_SPEC"
TARGETS = (LOCAL, AGENT_ENGINE, CLOUD_RUN, CLAUDE_MANAGED)

BOOKED = '{"when": "2026-11-08", "count": 2, "types": ["date", "int"]}'
BOOK_CALL = {"tool": "book", "args": {"when": "2026-11-08", "count": "2"}}

SCENARIOS = {
    # name: (script, expected (output, error, [(tool, args, output)]))
    "happy_path_coerces_arguments": (
        [BOOK_CALL, {"text": "Booked."}],
        ("Booked.", None, [("book", BOOK_CALL["args"], BOOKED)]),
    ),
    "tool_exception_goes_back_to_the_model": (
        [{"tool": "boom", "args": {"reason": "test"}}, {"text": "It failed, sorry."}],
        ("It failed, sorry.", None, [("boom", {"reason": "test"}, "Error: tool boom failed: RuntimeError: kaboom: test")]),
    ),
    "invalid_arguments_go_back_to_the_model": (
        [{"tool": "book", "args": {"when": "someday", "count": "two"}}, {"text": "I need a real date."}],
        ("I need a real date.", None, [(
            "book", {"when": "someday", "count": "two"},
            "Error: invalid arguments for book: "
            "when: Input should be a valid date or datetime, input is too short; "
            "count: Input should be a valid integer, unable to parse string as an integer",
        )]),
    ),
    "unknown_tool_goes_back_to_the_model": (
        [{"tool": "teleport", "args": {"to": "Mars"}}, {"text": "No such tool."}],
        ("No such tool.", None, [("teleport", {"to": "Mars"}, "Error: unknown tool teleport; available tools: book, boom")]),
    ),
    "loop_hits_max_iterations": (
        [BOOK_CALL] * 5 + [{"text": "never reached"}],
        ("", iteration_limit_error(2), [
            ("book", BOOK_CALL["args"], BOOKED),
            ("book", BOOK_CALL["args"], BOOKED),
            ("book", BOOK_CALL["args"], iteration_limit_error(2)),
        ]),
    ),
}


def shape(r: AgentResult):
    return r.output, r.error, [(c.name, c.args, c.output) for c in r.tool_calls]


class _HttpRunner:
    """cloud_run as the evals see it: the FastAPI route, parsed back into an AgentResult."""

    target = CLOUD_RUN

    def __init__(self, client: TestClient):
        self.client = client

    def run(self, user_input, *, thread_id=None):
        body = self.client.post("/mfa/query", json={"input": user_input, "thread_id": thread_id})
        assert body.status_code == 200, body.text
        return AgentResult.model_validate(body.json())


@pytest.fixture
def make_runner(monkeypatch):
    """Build a runner for any target that replays ``script`` through the real adapter code."""
    from mfa.serving import load_root_agent

    made = []

    def _make(target: str, script: list[dict]):
        if target == LOCAL:
            return LocalRunner(PARITY_SPEC, chat_model=ScriptedChatModel(script=list(script)))
        if target == AGENT_ENGINE:
            monkeypatch.setattr("mfa.runtime.models.make_chat_model", lambda m: ScriptedChatModel(script=list(script)))
            app = LangGraphAgentApp(agent_ref=SELF_REF, model="x:y")
            app.set_up()
            return AgentEngineRunner(FakeRemoteAgentEngine(app), "projects/p/locations/l/reasoningEngines/1")
        if target == CLOUD_RUN:
            load_root_agent.cache_clear()
            monkeypatch.setenv("MFA_AGENT", SELF_REF)
            monkeypatch.setenv("MFA_MODEL", "x:y")
            monkeypatch.setattr("mfa.runtime.models.make_chat_model", lambda m: ScriptedChatModel(script=list(script)))
            from mfa.serving.fast_api_app import create_app

            made.append(load_root_agent)
            return _HttpRunner(TestClient(create_app()))
        if target == CLAUDE_MANAGED:
            record = {"agent_id": "a", "agent_version": 1, "environment_id": "e", "model": "claude-test"}
            return ClaudeManagedRunner(PARITY_SPEC, record, client=FakeCMAClient(script=list(script)))
        raise ValueError(target)

    yield _make
    for cache in made:
        cache.cache_clear()


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_every_target_returns_the_same_result(scenario, make_runner):
    script, expected = SCENARIOS[scenario]
    results = {t: make_runner(t, script).run("go") for t in TARGETS}
    for target, r in results.items():
        assert shape(r) == expected, f"{target}: {shape(r)}"
        assert r.target == target


@pytest.mark.parametrize("target", TARGETS)
def test_second_turn_sees_the_first_and_reports_only_itself(target, make_runner):
    runner = make_runner(target, [BOOK_CALL, {"text": "Booked."}, {"text": "Anything else?"}])
    first = runner.run("book two seats tomorrow", thread_id="thread-1")
    second = runner.run("thanks", thread_id="thread-1")
    assert shape(first) == SCENARIOS["happy_path_coerces_arguments"][1]
    assert shape(second) == ("Anything else?", None, [])


@pytest.mark.parametrize("target", TARGETS)
def test_thread_recovers_after_a_failed_turn(target, make_runner):
    """A turn that hit the iteration limit must not poison the thread for the next turn."""
    runner = make_runner(target, [BOOK_CALL] * 3 + [{"text": "Recovered."}])  # 3rd call is the one past the limit
    failed = runner.run("loop", thread_id="t")
    assert failed.error == iteration_limit_error(2)
    again = runner.run("try again", thread_id="t")
    assert shape(again) == ("Recovered.", None, [])


def test_langgraph_fake_sees_the_system_prompt_and_tool_schemas():
    """The scripted model records what the graph sent, so a missing prompt or schema cannot hide."""
    model = ScriptedChatModel(script=[{"text": "hi"}])
    LocalRunner(PARITY_SPEC, chat_model=model).run("hello")
    assert [t.name for t in model.bound_tools] == ["book", "boom"]
    assert model.bound_tools[0].args_schema == PARITY_SPEC.tools[0].input_schema
    first_prompt = model.seen_messages[0]
    assert first_prompt[0].type == "system" and first_prompt[0].content == PARITY_SPEC.system_prompt
    assert first_prompt[-1].type == "human" and first_prompt[-1].content == "hello"


def test_cma_tool_result_flags_errors_and_binds_the_same_schema():
    client = FakeCMAClient(script=[{"tool": "boom", "args": {"reason": "x"}}, {"text": "ok"}])
    record = {"agent_id": "a", "agent_version": 1, "environment_id": "e", "model": "m"}
    ClaudeManagedRunner(PARITY_SPEC, record, client=client).run("go")
    results = [e for e in client.sent if e["type"] == "user.custom_tool_result"]
    assert results[0]["is_error"] is True and results[0]["content"][0]["text"].startswith("Error: tool boom failed")


def test_cma_server_retries_are_not_failures_but_exhausted_errors_are():
    record = {"agent_id": "a", "agent_version": 1, "environment_id": "e", "model": "m"}
    retrying = FakeCMAClient(script=[
        {"error": {"type": "model_overloaded_error", "retry_status": "retrying", "message": "busy"}},
        {"text": "Done after retry."},
    ])
    r = ClaudeManagedRunner(PARITY_SPEC, record, client=retrying).run("go")
    assert r.error is None and r.output == "Done after retry." and r.metadata["server_retries"] == 1

    exhausted = FakeCMAClient(script=[
        {"error": {"type": "model_overloaded_error", "retry_status": "exhausted", "message": "still busy"}},
    ])
    r = ClaudeManagedRunner(PARITY_SPEC, record, client=exhausted).run("go")
    assert r.error == "model_overloaded_error: still busy"


def test_tool_error_text_is_shared():
    assert format_tool_error("x", ValueError("bad")) == "Error: tool x failed: ValueError: bad"
    assert PARITY_SPEC.tool_by_name("book").run({"when": "2026-01-02"}) == ('{"when": "2026-01-02", "count": 1, "types": ["date", "int"]}', False)


@pytest.mark.parametrize("target", TARGETS)
def test_max_iterations_rounds_are_allowed_and_one_more_is_not(target, make_runner):
    """n tool rounds plus an answer must succeed; the (n+1)th round is refused everywhere."""
    exactly = make_runner(target, [BOOK_CALL] * PARITY_SPEC.max_iterations + [{"text": "Done."}]).run("go")
    assert exactly.error is None and exactly.output == "Done." and len(exactly.tool_calls) == PARITY_SPEC.max_iterations
    one_more = make_runner(target, [BOOK_CALL] * (PARITY_SPEC.max_iterations + 1) + [{"text": "Done."}]).run("go")
    assert one_more.error == iteration_limit_error(PARITY_SPEC.max_iterations)
    assert len(one_more.tool_calls) == PARITY_SPEC.max_iterations + 1


def test_recursion_limit_matches_langgraph_step_counting():
    # 2n+1 supersteps are needed for n rounds and LangGraph stops at the limit itself.
    assert recursion_limit_for(2) == 6 and recursion_limit_for(4) == 10
