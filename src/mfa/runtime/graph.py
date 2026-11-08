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

import json
import time
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from mfa.runtime.result import AgentResult, ToolCall
from mfa.spec import AgentSpec, iteration_limit_error, unknown_tool_error


def recursion_limit_for(max_iterations: int) -> int:
    """Safety-net ``recursion_limit`` for a turn of ``max_iterations`` tool rounds.

    The template graph enforces ``max_iterations`` itself (see ``build_graph``), so
    this only guards custom ``graph_builder`` topologies. A turn with n rounds takes
    2n+1 supersteps and LangGraph needs one spare tick to finish cleanly at the limit.
    """
    return max_iterations * 2 + 2


def tool_rounds_this_turn(messages: list[BaseMessage]) -> int:
    """Assistant messages with tool calls since the last user message."""
    rounds = 0
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            break
        if isinstance(m, AIMessage) and m.tool_calls:
            rounds += 1
    return rounds


def build_graph(spec: AgentSpec, model: BaseChatModel, *, checkpointer: Any | None = None):
    lc_tools = [t.to_langchain() for t in spec.tools]
    bound = model.bind_tools(lc_tools) if lc_tools else model
    system = SystemMessage(content=spec.system_prompt)

    def call_model(state: MessagesState) -> dict:
        return {"messages": [bound.invoke([system, *state["messages"]])]}

    names = [t.name for t in lc_tools]

    def _unknown(request):
        tc = request.tool_call
        return ToolMessage(
            content=unknown_tool_error(tc["name"], names), name=tc["name"], tool_call_id=tc["id"], status="error"
        )

    def wrap_tool_call(request, execute):
        # Same text as the Claude Managed Agents runner for a tool the spec does not define.
        return _unknown(request) if request.tool is None else execute(request)

    async def awrap_tool_call(request, execute):
        return _unknown(request) if request.tool is None else await execute(request)

    def route(state: MessagesState) -> str:
        if tools_condition(state) == END:
            return END
        # The call past the cap is not executed: the turn ends here and run_turn
        # reports iteration_limit_error, the same way the CMA runner does.
        return END if tool_rounds_this_turn(state["messages"]) > spec.max_iterations else "tools"

    g = StateGraph(MessagesState)
    g.add_node("agent", call_model)
    g.add_edge(START, "agent")
    if lc_tools:
        # Tool failures never reach here: ToolSpec.run turns them into error text.
        g.add_node("tools", ToolNode(lc_tools, wrap_tool_call=wrap_tool_call, awrap_tool_call=awrap_tool_call))
        g.add_conditional_edges("agent", route, {"tools": "tools", END: END})
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
    for m in messages:
        if isinstance(m, ToolMessage):
            outputs[m.tool_call_id] = m.content if isinstance(m.content, str) else json.dumps(m.content)

    calls: list[ToolCall] = []
    final = ""
    for m in messages:
        if isinstance(m, AIMessage):
            for tc in m.tool_calls:
                calls.append(ToolCall(name=tc["name"], args=tc["args"], output=outputs.get(tc["id"])))
            text = _text(m)
            if text:
                final = text
    return AgentResult(target=target, input=user_input, output=final, tool_calls=calls)


def turn_error_text(exc: BaseException, max_iterations: int) -> str:
    """Normalize a failed turn's exception into the error string shared with the other targets."""
    if isinstance(exc, GraphRecursionError):
        return iteration_limit_error(max_iterations)
    return f"{type(exc).__name__}: {exc}"


def dangling_tool_calls(messages: list[BaseMessage]) -> list[dict]:
    """Tool calls in the last AI message that have no ToolMessage answering them."""
    if not messages or not isinstance(messages[-1], AIMessage):
        return []
    return list(messages[-1].tool_calls)


def repair_thread(graph, config: dict, error_text: str) -> list[BaseMessage]:
    """After a failed turn, answer any unanswered tool call so the thread stays valid for the next turn.

    Returns the thread's messages after the repair.
    """
    messages = list(graph.get_state(config).values.get("messages", []))
    pending = dangling_tool_calls(messages)
    if pending:
        fillers = [
            ToolMessage(content=error_text, name=tc["name"], tool_call_id=tc["id"], status="error")
            for tc in pending
        ]
        # Written as the agent node: the conditional edge then routes to END, so the
        # checkpoint has no pending task and the next turn starts a fresh run with a
        # full recursion budget (as the tools node it would leave "agent" pending).
        graph.update_state(config, {"messages": fillers}, as_node="agent")
        messages.extend(fillers)
    return messages


def run_turn(graph, user_input: str, *, thread_id: str, target: str, max_iterations: int) -> AgentResult:
    """Run one user turn through a compiled graph and return only that turn's activity.

    Never raises. On failure the result carries ``error`` plus whatever tool
    calls happened before it, and unanswered tool calls in the thread are
    answered with the error text so the conversation can continue.
    """
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit_for(max_iterations)}
    start = time.monotonic()
    before = len(graph.get_state(config).values.get("messages", []))
    try:
        state = graph.invoke({"messages": [HumanMessage(content=user_input)]}, config)
        messages = state["messages"]
        error = None
        if dangling_tool_calls(messages):
            # The graph ended on a tool call it refused to run: the iteration cap.
            error = iteration_limit_error(max_iterations)
            messages = repair_thread(graph, config, error)
    except Exception as exc:
        error = turn_error_text(exc, max_iterations)
        messages = repair_thread(graph, config, error)
    result = extract_result(messages[before:], target=target, user_input=user_input)
    result.error = error
    result.latency_s = round(time.monotonic() - start, 3)
    result.metadata["thread_id"] = thread_id
    return result
