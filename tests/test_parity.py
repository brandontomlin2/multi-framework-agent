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
from dataclasses import replace
from pathlib import Path

from mfa.runtime.graph import recursion_limit_for
from mfa.runtime.result import AgentResult
from mfa.skills import skills_prompt_block
from mfa.spec import (
    AGENT_ENGINE,
    CLAUDE_MANAGED,
    CLOUD_RUN,
    LOCAL,
    AgentSpec,
    format_tool_error,
    iteration_limit_error,
    local_skills,
    tool,
)
from mfa.targets.claude_managed import ClaudeManagedRunner
from mfa.targets.local import LocalRunner
from tests.fakes import FakeCMAClient, ScriptedChatModel


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
    "parallel_calls_count_as_one_round": (
        [{"calls": [BOOK_CALL, {"tool": "boom", "args": {"reason": "p"}}]}, {"text": "Both done."}],
        ("Both done.", None, [
            ("book", BOOK_CALL["args"], BOOKED),
            ("boom", {"reason": "p"}, "Error: tool boom failed: RuntimeError: kaboom: p"),
        ]),
    ),
    "parallel_round_past_the_cap_is_refused_whole": (
        [{"calls": [BOOK_CALL, BOOK_CALL]}] * 3 + [{"text": "never reached"}],
        ("", iteration_limit_error(2), [("book", BOOK_CALL["args"], BOOKED)] * 4
         + [("book", BOOK_CALL["args"], iteration_limit_error(2))] * 2),
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


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_every_target_returns_the_same_result(scenario, make_runner):
    script, expected = SCENARIOS[scenario]
    results = {t: make_runner(t, script, PARITY_SPEC, SELF_REF).run("go") for t in TARGETS}
    for target, r in results.items():
        assert shape(r) == expected, f"{target}: {shape(r)}"
        assert r.target == target


@pytest.mark.parametrize("target", TARGETS)
def test_second_turn_sees_the_first_and_reports_only_itself(target, make_runner):
    runner = make_runner(target, [BOOK_CALL, {"text": "Booked."}, {"text": "Anything else?"}], PARITY_SPEC, SELF_REF)
    first = runner.run("book two seats tomorrow", thread_id="thread-1")
    second = runner.run("thanks", thread_id="thread-1")
    assert shape(first) == SCENARIOS["happy_path_coerces_arguments"][1]
    assert shape(second) == ("Anything else?", None, [])


@pytest.mark.parametrize("target", TARGETS)
def test_thread_recovers_after_a_failed_turn(target, make_runner):
    """A turn that hit the iteration limit must not poison the thread for the next turn."""
    runner = make_runner(target, [BOOK_CALL] * 3 + [{"text": "Recovered."}], PARITY_SPEC, SELF_REF)  # 3rd call is past the cap
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
    exactly = make_runner(target, [BOOK_CALL] * PARITY_SPEC.max_iterations + [{"text": "Done."}], PARITY_SPEC, SELF_REF).run("go")
    assert exactly.error is None and exactly.output == "Done." and len(exactly.tool_calls) == PARITY_SPEC.max_iterations
    one_more = make_runner(target, [BOOK_CALL] * (PARITY_SPEC.max_iterations + 1) + [{"text": "Done."}], PARITY_SPEC, SELF_REF).run("go")
    assert one_more.error == iteration_limit_error(PARITY_SPEC.max_iterations)
    assert len(one_more.tool_calls) == PARITY_SPEC.max_iterations + 1


def test_recursion_limit_is_only_a_generous_safety_net():
    # n rounds need 2n+1 supersteps in the template; custom rounds may take more, so the
    # limit is well above that and never the thing that enforces max_iterations.
    assert recursion_limit_for(2) >= 2 * 2 + 2 and recursion_limit_for(30) >= 2 * 30 + 2


def test_langgraph_tool_messages_carry_the_error_flag_cma_sends():
    """The model must see the same is_error signal on every target, not only the same text."""
    runner = LocalRunner(PARITY_SPEC, chat_model=ScriptedChatModel(script=[
        {"calls": [BOOK_CALL, {"tool": "boom", "args": {"reason": "x"}}, {"tool": "nope", "args": {}}]}, {"text": "ok"},
    ]))
    runner.run("go", thread_id="t")
    messages = runner.graph.get_state({"configurable": {"thread_id": "t"}}).values["messages"]
    statuses = {m.name: m.status for m in messages if m.type == "tool"}
    assert statuses == {"book": "success", "boom": "error", "nope": "error"}

    client = FakeCMAClient(script=[{"calls": [BOOK_CALL, {"tool": "boom", "args": {"reason": "x"}}, {"tool": "nope", "args": {}}]}, {"text": "ok"}])
    record = {"agent_id": "a", "agent_version": 1, "environment_id": "e", "model": "m"}
    ClaudeManagedRunner(PARITY_SPEC, record, client=client).run("go")
    flags = [e["is_error"] for e in client.sent if e["type"] == "user.custom_tool_result"]
    assert flags == [False, True, True]


def _two_node_custom_graph(spec, model, *, checkpointer=None):
    """A graph_builder with its own node names, using the exported cap route."""
    from langchain_core.messages import SystemMessage
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode

    from mfa.runtime.graph import iteration_cap_route

    tools = [t.to_langchain() for t in spec.tools]
    bound = model.bind_tools(tools)
    g = StateGraph(MessagesState)
    g.add_node("model", lambda st: {"messages": [bound.invoke([SystemMessage(content=spec.system_prompt), *st["messages"]])]})
    g.add_node("run_tools", ToolNode(tools))
    g.add_node("post", lambda st: {})  # a third step per round, unlike the template
    g.add_edge(START, "model")
    g.add_conditional_edges("model", iteration_cap_route(spec, tools_node="run_tools"), {"run_tools": "run_tools", END: END})
    g.add_edge("run_tools", "post")
    g.add_edge("post", "model")
    return g.compile(checkpointer=checkpointer or InMemorySaver())


def test_custom_graph_builder_gets_the_same_cap_and_recovers():
    spec = replace(PARITY_SPEC, graph_builder=_two_node_custom_graph)
    ok = LocalRunner(spec, chat_model=ScriptedChatModel(script=[BOOK_CALL] * 2 + [{"text": "Done."}])).run("go")
    assert ok.error is None and ok.output == "Done." and len(ok.tool_calls) == 2  # n rounds + answer pass

    runner = LocalRunner(spec, chat_model=ScriptedChatModel(script=[BOOK_CALL] * 3 + [{"text": "Recovered."}]))
    capped = runner.run("go", thread_id="t")
    assert capped.error == iteration_limit_error(2) and len(capped.tool_calls) == 3
    assert runner.graph.get_state({"configurable": {"thread_id": "t"}}).next == ()  # repaired without an "agent" node
    assert shape(runner.run("again", thread_id="t")) == ("Recovered.", None, [])


def test_interrupt_before_graph_is_left_pending_not_reported_as_the_cap():
    def hitl(spec, model, *, checkpointer=None):
        from langchain_core.messages import SystemMessage
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.graph import END, START, MessagesState, StateGraph
        from langgraph.prebuilt import ToolNode, tools_condition

        tools = [t.to_langchain() for t in spec.tools]
        bound = model.bind_tools(tools)
        g = StateGraph(MessagesState)
        g.add_node("agent", lambda st: {"messages": [bound.invoke([SystemMessage(content=spec.system_prompt), *st["messages"]])]})
        g.add_node("tools", ToolNode(tools))
        g.add_edge(START, "agent")
        g.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
        g.add_edge("tools", "agent")
        return g.compile(checkpointer=checkpointer or InMemorySaver(), interrupt_before=["tools"])

    runner = LocalRunner(replace(PARITY_SPEC, graph_builder=hitl), chat_model=ScriptedChatModel(script=[BOOK_CALL, {"text": "x"}]))
    r = runner.run("go", thread_id="t")
    assert r.error is None and r.metadata["pending_nodes"] == ["tools"]
    assert [(c.name, c.output) for c in r.tool_calls] == [("book", None)]  # waiting for approval, not refused
    assert runner.graph.get_state({"configurable": {"thread_id": "t"}}).next == ("tools",)


# --- Agent Skills -----------------------------------------------------------
# The same skill folder is read through LangGraph's sandbox tools and, on Claude
# Managed Agents, through the built-in tools the runner approves with the same
# sandbox rules. Locations are announced under one virtual root on every target.

SKILLS_DIR = Path(__file__).parent / "skills"
SKILL_SPEC = replace(PARITY_SPEC, name="skill_parity_agent", skills=local_skills(SKILLS_DIR))
SCRIPT_SPEC = replace(SKILL_SPEC, name="script_parity_agent", allow_skill_scripts=True)
SKILL_REF = "tests.test_parity:SKILL_SPEC"
SCRIPT_REF = "tests.test_parity:SCRIPT_SPEC"
ROOT = "/workspace/skills/parity-skill"
SKILL_MD = (SKILLS_DIR / "parity-skill" / "SKILL.md").read_text()
NOTES = (SKILLS_DIR / "parity-skill" / "references" / "notes.md").read_text()
READ_SKILL = {"tool": "read", "args": {"file_path": f"{ROOT}/SKILL.md"}}
READ_NOTES = {"tool": "read", "args": {"file_path": f"{ROOT}/references/notes.md"}}
OUTSIDE = "Error: tool read failed: PermissionError: /etc/passwd is not inside a skill folder (/workspace/skills/<skill>/...)"

SKILL_SCENARIOS = {
    "skill_is_read_on_demand": (
        [READ_SKILL, READ_NOTES, {"text": "Noted."}],
        ("Noted.", None, [("read", READ_SKILL["args"], SKILL_MD), ("read", READ_NOTES["args"], NOTES)]),
    ),
    "view_range_is_coerced": (
        [{"tool": "read", "args": {"file_path": f"{ROOT}/references/notes.md", "view_range": ["1", "2"]}}, {"text": "Two lines."}],
        ("Two lines.", None, [("read", {"file_path": f"{ROOT}/references/notes.md", "view_range": ["1", "2"]}, "line one\nline two\n")]),
    ),
    "glob_and_grep_see_the_same_files": (
        [{"tool": "glob", "args": {"pattern": "**/*.md"}}, {"tool": "grep", "args": {"pattern": "hello"}}, {"text": "Found."}],
        ("Found.", None, [
            ("glob", {"pattern": "**/*.md"}, f"{ROOT}/SKILL.md\n{ROOT}/references/notes.md"),
            ("grep", {"pattern": "hello"}, f"{ROOT}/SKILL.md:8:Read `references/notes.md` for the notes. Run `scripts/hello.py` to say hello.\n"
                                           f"{ROOT}/scripts/hello.py:3:print(\"hello from parity-skill\" + (\" \" + \" \".join(sys.argv[1:]) if len(sys.argv) > 1 else \"\"))"),
        ]),
    ),
    "mixed_round_custom_and_skill_read": (
        [{"calls": [BOOK_CALL, READ_SKILL]}, {"text": "Both."}],
        ("Both.", None, [("book", BOOK_CALL["args"], BOOKED), ("read", READ_SKILL["args"], SKILL_MD)]),
    ),
    "missing_skill_file_is_an_error": (
        [{"tool": "read", "args": {"file_path": f"{ROOT}/references/missing.md"}}, {"text": "No file."}],
        ("No file.", None, [("read", {"file_path": f"{ROOT}/references/missing.md"},
                             f"Error: tool read failed: FileNotFoundError: no such file in skill folder: {ROOT}/references/missing.md")]),
    ),
    "path_outside_skills_is_refused": (
        [{"tool": "read", "args": {"file_path": "/etc/passwd"}}, {"text": "Refused."}],
        ("Refused.", None, [("read", {"file_path": "/etc/passwd"}, OUTSIDE)]),
    ),
    "skill_reads_count_toward_the_cap": (
        [READ_SKILL] * 3 + [{"text": "never"}],
        ("", iteration_limit_error(2), [("read", READ_SKILL["args"], SKILL_MD)] * 2 + [("read", READ_SKILL["args"], iteration_limit_error(2))]),
    ),
}


@pytest.mark.parametrize("scenario", list(SKILL_SCENARIOS))
def test_skill_scenarios_match_on_every_target(scenario, make_runner):
    script, expected = SKILL_SCENARIOS[scenario]
    for target in TARGETS:
        r = make_runner(target, script, SKILL_SPEC, SKILL_REF).run("go")
        assert shape(r) == expected, f"{target}: {shape(r)}"
        if expected[2] and expected[2][0][0] == "read" and "SKILL.md" in expected[2][0][1]["file_path"]:
            assert r.metadata.get("skills_used") == ["parity-skill"], target


def test_refused_skill_read_never_runs_on_cma(make_runner):
    runner = make_runner(CLAUDE_MANAGED, [{"tool": "read", "args": {"file_path": "/etc/passwd"}}, {"text": "x"}], SKILL_SPEC, SKILL_REF)
    runner.run("go")
    confirmations = [e for e in runner.fake.sent if e["type"] == "user.tool_confirmation"]
    assert confirmations == [{"type": "user.tool_confirmation", "tool_use_id": confirmations[0]["tool_use_id"],
                              "result": "deny", "deny_message": OUTSIDE}]


def test_cap_on_cma_denies_the_paused_read_and_interrupts_in_one_batch(make_runner):
    runner = make_runner(CLAUDE_MANAGED, [READ_SKILL] * 3 + [{"text": "never"}], SKILL_SPEC, SKILL_REF)
    assert runner.run("go").error == iteration_limit_error(2)
    last = runner.fake.batches[-1]
    assert [e["type"] for e in last] == ["user.tool_confirmation", "user.interrupt"]
    assert last[0]["result"] == "deny" and last[0]["deny_message"] == iteration_limit_error(2)


def test_mixed_round_sends_the_confirmation_at_the_round_boundary(make_runner):
    runner = make_runner(CLAUDE_MANAGED, [{"calls": [BOOK_CALL, READ_SKILL]}, {"text": "Both."}], SKILL_SPEC, SKILL_REF)
    assert runner.run("go").output == "Both."
    kinds = [[e["type"] for e in b] for b in runner.fake.batches]
    assert kinds == [["user.message"], ["user.custom_tool_result"], ["user.tool_confirmation"]]


def test_bash_runs_only_a_skills_own_scripts(make_runner):
    ok = {"tool": "bash", "args": {"command": f"python3 {ROOT}/scripts/hello.py there"}}
    bad = {"tool": "bash", "args": {"command": "cat /etc/passwd | head"}}
    expected = ("Ran.", None, [
        ("bash", ok["args"], "hello from parity-skill there\n"),
        ("bash", bad["args"], "Error: tool bash failed: PermissionError: only a skill's own scripts/ may be run, "
                              "without shell operators: cat /etc/passwd | head"),
    ])
    for target in TARGETS:
        r = make_runner(target, [ok, bad, {"text": "Ran."}], SCRIPT_SPEC, SCRIPT_REF).run("go")
        assert shape(r) == expected, f"{target}: {shape(r)}"


def test_bash_without_opt_in_is_refused_on_every_target(make_runner):
    """Named divergence: LangGraph does not bind bash at all; the server refuses the disabled tool."""
    script = [{"tool": "bash", "args": {"command": f"python3 {ROOT}/scripts/hello.py"}}, {"text": "x"}]
    lg = make_runner(LOCAL, script, SKILL_SPEC, SKILL_REF).run("go")
    assert lg.tool_calls[0].output == "Error: unknown tool bash; available tools: book, boom, read, glob, grep"
    cma = make_runner(CLAUDE_MANAGED, script, SKILL_SPEC, SKILL_REF).run("go")
    assert cma.tool_calls[0].output == "Permission to use bash has been denied."
    assert cma.metadata["server_denials"] == 1


@pytest.mark.parametrize("target", TARGETS)
def test_skill_spec_threads_recover_after_a_capped_turn(target, make_runner):
    runner = make_runner(target, [READ_SKILL] * 3 + [{"text": "Recovered."}], SKILL_SPEC, SKILL_REF)
    assert runner.run("loop", thread_id="t").error == iteration_limit_error(2)
    assert shape(runner.run("again", thread_id="t")) == ("Recovered.", None, [])


def test_langgraph_announces_skills_and_binds_the_toolset_schemas():
    model = ScriptedChatModel(script=[{"text": "hi"}])
    LocalRunner(SKILL_SPEC, chat_model=model).run("hello")
    system = model.seen_messages[0][0].content
    assert system == SKILL_SPEC.system_prompt + "\n\n" + skills_prompt_block(SKILL_SPEC)
    assert f"- parity-skill: {SKILL_SPEC.skills[0].description} (location: {ROOT})" in system
    bound = {t.name: t.args_schema for t in model.bound_tools}
    assert list(bound) == ["book", "boom", "read", "glob", "grep"]
    assert set(bound["read"]["properties"]) == {"file_path", "view_range"}
    assert set(bound["glob"]["properties"]) == {"pattern", "path"} == set(bound["grep"]["properties"])
    scripted = ScriptedChatModel(script=[{"text": "hi"}])
    LocalRunner(SCRIPT_SPEC, chat_model=scripted).run("hello")
    assert set({t.name: t for t in scripted.bound_tools}["bash"].args_schema["properties"]) == {"command", "restart", "timeout_ms"}


def test_cma_translation_attaches_skills_and_gates_the_file_tools():
    from mfa.targets.claude_managed import ClaudeManagedTarget

    client = FakeCMAClient(spec=SKILL_SPEC)
    rec = ClaudeManagedTarget().deploy(SKILL_SPEC, client=client)
    params = client.created_agents[0]
    assert params["skills"] == [{"type": "custom", "skill_id": "skill_fake_parity-skill", "version": "skillver_1"}]
    toolset = params["tools"][0]
    assert toolset["type"] == "agent_toolset_20260401"
    assert toolset["default_config"] == {"enabled": False, "permission_policy": {"type": "always_ask"}}
    assert [(c["name"], c.get("enabled")) for c in toolset["configs"]] == [
        ("read", True), ("glob", True), ("grep", True), ("web_fetch", False), ("web_search", False)]
    assert [t["name"] for t in params["tools"][1:]] == ["book", "boom"]
    assert params["system"] == SKILL_SPEC.system_prompt  # the server announces skills itself
    assert client.uploaded == {"parity-skill": ["parity-skill/SKILL.md", "parity-skill/references/notes.md", "parity-skill/scripts/hello.py"]}
    assert rec["skills"]["parity-skill"]["version"] == "skillver_1" and rec["toolset"] == ["read", "glob", "grep"]
