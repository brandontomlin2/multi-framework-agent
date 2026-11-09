"""Builds the LangGraph graph for any AgentSpec.

This is the LangGraph *template*: a tool-calling loop (model -> tools -> model)
that runs locally and, unchanged, inside Vertex AI Agent Engine and the Cloud
Run container. An agent that needs a different topology sets
``AgentSpec.graph_builder``; it must return a compiled graph whose state has a
``messages`` key (see ``build_agent_graph``).

``run_turn`` is the turn boundary shared by every LangGraph target. It never
raises: a failed turn returns an ``AgentResult`` whose ``error`` is set and
whose ``tool_calls`` still hold the partial trajectory, and the thread is left
in a state the next turn can continue from.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from mfa.runtime.result import AgentResult, ToolCall
from mfa.skills import skills_prompt_block, skills_used
from mfa.spec import AgentSpec, ToolSpec, iteration_limit_error, unknown_tool_error


def recursion_limit_for(max_iterations: int) -> int:
    """Safety-net ``recursion_limit`` for a turn of ``max_iterations`` tool rounds.

    The iteration cap itself is enforced by ``iteration_cap_route`` (used by the
    template graph; available to custom builders). This limit only stops a custom
    graph that loops without ever calling a tool, so it is deliberately generous:
    a round may take several supersteps in a custom topology.
    """
    return max(50, max_iterations * 4 + 2)


def tool_rounds_this_turn(messages: list[BaseMessage]) -> int:
    """Assistant messages with tool calls since the last user message."""
    rounds = 0
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            break
        if isinstance(m, AIMessage) and m.tool_calls:
            rounds += 1
    return rounds


def iteration_cap_route(spec: AgentSpec, *, tools_node: str = "tools") -> Callable[[MessagesState], str]:
    """Conditional edge after the model node: run tools, or end the turn.

    The call past ``spec.max_iterations`` tool rounds is not executed: the turn
    ends on the dangling tool call and ``run_turn`` reports ``iteration_limit_error``,
    the same way the Claude Managed Agents runner does. Custom ``graph_builder``
    graphs can use this in place of ``tools_condition`` to get the same cap.
    """

    def route(state: MessagesState) -> str:
        if tools_condition(state) == END:
            return END
        return END if tool_rounds_this_turn(state["messages"]) > spec.max_iterations else tools_node

    return route


def spec_tools(spec: AgentSpec) -> list[Any]:
    """The LangChain tools a graph binds for a spec: the agent's own plus the skill file tools.

    Custom ``graph_builder`` functions should bind these instead of ``spec.tools``
    so skills work for them too.
    """
    return [t.to_langchain() for t in spec.runtime_tools()]


def system_prompt_for(spec: AgentSpec) -> str:
    """The spec's prompt plus the skills listing (name, description, location) when it has skills."""
    block = skills_prompt_block(spec)
    return f"{spec.system_prompt.rstrip()}\n\n{block}" if block else spec.system_prompt


def build_graph(spec: AgentSpec, model: BaseChatModel, *, checkpointer: Any | None = None):
    tools: list[ToolSpec] = spec.runtime_tools()
    by_name = {t.name: t for t in tools}
    lc_tools = [t.to_langchain() for t in tools]
    bound = model.bind_tools(lc_tools) if lc_tools else model
    system = SystemMessage(content=system_prompt_for(spec))
    names = list(by_name)

    def call_model(state: MessagesState) -> dict:
        return {"messages": [bound.invoke([system, *state["messages"]])]}

    def run_call(tc: dict) -> ToolMessage:
        # Every target hands the model the text *and* the error flag ToolSpec.run
        # produced; the Claude Managed Agents runner sends the same pair as
        # user.custom_tool_result(is_error=...).
        if tc["name"] not in by_name:
            text, is_error = unknown_tool_error(tc["name"], names), True
        else:
            text, is_error = by_name[tc["name"]].run(dict(tc["args"]))
        return ToolMessage(
            content=text, name=tc["name"], tool_call_id=tc["id"], status="error" if is_error else "success"
        )

    def wrap_tool_call(request, execute):
        return run_call(request.tool_call)

    async def awrap_tool_call(request, execute):
        return await asyncio.to_thread(run_call, request.tool_call)  # keep sync tools off the event loop

    g = StateGraph(MessagesState)
    g.add_node("agent", call_model)
    g.add_edge(START, "agent")
    if lc_tools:
        # The LangChain tools carry the schema for bind_tools; execution goes through run_call.
        g.add_node("tools", ToolNode(lc_tools, wrap_tool_call=wrap_tool_call, awrap_tool_call=awrap_tool_call))
        g.add_conditional_edges("agent", iteration_cap_route(spec), {"tools": "tools", END: END})
        g.add_edge("tools", "agent")
    else:
        g.add_edge("agent", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())


def build_agent_graph(spec: AgentSpec, model: BaseChatModel, *, checkpointer: Any | None = None):
    """Use the spec's custom builder if it has one, otherwise the default template."""
    builder = spec.graph_builder or build_graph
    return builder(spec, model, checkpointer=checkpointer)


def _text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


def extract_result(messages: list[BaseMessage], *, target: str, user_input: str) -> AgentResult:
    """Convert the messages produced by one turn into the shared AgentResult shape."""
    outputs: dict[str, str] = {}
    errors: dict[str, bool] = {}
    for m in messages:
        if isinstance(m, ToolMessage):
            outputs[m.tool_call_id] = m.content if isinstance(m.content, str) else json.dumps(m.content)
            errors[m.tool_call_id] = getattr(m, "status", "success") == "error"

    calls: list[ToolCall] = []
    final = ""
    for m in messages:
        if isinstance(m, AIMessage):
            for tc in m.tool_calls:
                calls.append(ToolCall(name=tc["name"], args=tc["args"], output=outputs.get(tc["id"]),
                                      is_error=errors.get(tc["id"], False)))
            text = _text(m)
            if text:
                final = text
    return AgentResult(target=target, input=user_input, output=final, tool_calls=calls)


def turn_error_text(exc: BaseException, max_iterations: int, messages: list[BaseMessage] | None = None) -> str:
    """Normalize a failed turn's exception into the error string shared with the other targets."""
    if isinstance(exc, GraphRecursionError):
        if messages is None or tool_rounds_this_turn(messages) > max_iterations:
            return iteration_limit_error(max_iterations)
        return f"GraphRecursionError: {exc}"  # a custom graph looped without exceeding the tool cap
    return f"{type(exc).__name__}: {exc}"


def dangling_tool_calls(messages: list[BaseMessage]) -> list[dict]:
    """Tool calls in the last AI message that have no ToolMessage answering them."""
    if not messages or not isinstance(messages[-1], AIMessage):
        return []
    return list(messages[-1].tool_calls)


def _fillers(messages: list[BaseMessage], error_text: str) -> list[ToolMessage]:
    return [
        ToolMessage(content=error_text, name=tc["name"], tool_call_id=tc["id"], status="error")
        for tc in dangling_tool_calls(messages)
    ]


def _repair_node(graph) -> str | None:
    # Writing as the model node makes the template's conditional edge route to END,
    # so the checkpoint has no pending task and the next turn starts a fresh run.
    # Custom graphs name their nodes freely: let LangGraph infer the writer there.
    return "agent" if "agent" in getattr(graph, "nodes", {}) else None


def repair_thread(graph, config: dict, error_text: str) -> list[BaseMessage]:
    """After a failed turn, answer any unanswered tool call so the thread stays valid for the next turn.

    Returns the thread's messages after the repair. Never raises: a repair that
    fails leaves the thread as it was.
    """
    messages = list(graph.get_state(config).values.get("messages", []))
    fillers = _fillers(messages, error_text)
    if fillers:
        try:
            graph.update_state(config, {"messages": fillers}, as_node=_repair_node(graph))
            messages.extend(fillers)
        except Exception:
            pass
    return messages


async def arepair_thread(graph, config: dict, error_text: str) -> list[BaseMessage]:
    """``repair_thread`` for async callers (an async-only checkpointer must not be driven synchronously)."""
    messages = list((await graph.aget_state(config)).values.get("messages", []))
    fillers = _fillers(messages, error_text)
    if fillers:
        try:
            await graph.aupdate_state(config, {"messages": fillers}, as_node=_repair_node(graph))
            messages.extend(fillers)
        except Exception:
            pass
    return messages


def capped_turn(messages: list[BaseMessage], pending_nodes: tuple, max_iterations: int) -> bool:
    """True when a turn ended on a refused tool call because of the iteration cap.

    A graph compiled with ``interrupt_before`` also returns with a dangling tool
    call, but then it has a pending node and the cap was not exceeded; that turn
    is left untouched for the caller to resume.
    """
    return (
        bool(dangling_tool_calls(messages))
        and not pending_nodes
        and tool_rounds_this_turn(messages) > max_iterations
    )


def run_turn(graph, user_input: str, *, thread_id: str, target: str, max_iterations: int) -> AgentResult:
    """Run one user turn through a compiled graph and return only that turn's activity.

    Never raises. On failure the result carries ``error`` plus whatever tool
    calls happened before it, and unanswered tool calls in the thread are
    answered with the error text so the conversation can continue.
    """
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit_for(max_iterations)}
    start = time.monotonic()
    before = len(graph.get_state(config).values.get("messages", []))
    pending: tuple = ()
    try:
        state = graph.invoke({"messages": [HumanMessage(content=user_input)]}, config)
        messages = state["messages"]
        error = None
        pending = tuple(graph.get_state(config).next or ())
        if capped_turn(messages, pending, max_iterations):
            error = iteration_limit_error(max_iterations)
            messages = repair_thread(graph, config, error)
    except Exception as exc:
        messages = list(graph.get_state(config).values.get("messages", []))
        error = turn_error_text(exc, max_iterations, messages)
        messages = repair_thread(graph, config, error)
    result = extract_result(messages[before:], target=target, user_input=user_input)
    result.error = error
    result.latency_s = round(time.monotonic() - start, 3)
    result.metadata["thread_id"] = thread_id
    if used := skills_used(result.tool_calls):
        result.metadata["skills_used"] = used
    if pending:
        result.metadata["pending_nodes"] = list(pending)  # e.g. a custom graph waiting at interrupt_before
    return result
