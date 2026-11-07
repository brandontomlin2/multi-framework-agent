"""Builds the LangGraph graph for any AgentSpec.

This is the LangGraph *template*: a tool-calling loop (model -> tools -> model)
that runs locally and, unchanged, inside Vertex AI Agent Engine. An agent that
needs a different topology sets ``AgentSpec.graph_builder``; it must return a
compiled graph whose state has a ``messages`` key (see ``build_agent_graph``).
"""

from __future__ import annotations

import json
import time
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from mfa.runtime.result import AgentResult, ToolCall
from mfa.spec import AgentSpec


def build_graph(spec: AgentSpec, model: BaseChatModel, *, checkpointer: Any | None = None):
    lc_tools = [t.to_langchain() for t in spec.tools]
    bound = model.bind_tools(lc_tools) if lc_tools else model
    system = SystemMessage(content=spec.system_prompt)

    def call_model(state: MessagesState) -> dict:
        return {"messages": [bound.invoke([system, *state["messages"]])]}

    g = StateGraph(MessagesState)
    g.add_node("agent", call_model)
    g.add_edge(START, "agent")
    if lc_tools:
        g.add_node("tools", ToolNode(lc_tools))
        g.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
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


def run_turn(graph, user_input: str, *, thread_id: str, target: str, recursion_limit: int) -> AgentResult:
    """Run one user turn through a compiled graph and return only that turn's activity."""
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit}
    before = len(graph.get_state(config).values.get("messages", []))
    start = time.monotonic()
    state = graph.invoke({"messages": [HumanMessage(content=user_input)]}, config)
    result = extract_result(state["messages"][before:], target=target, user_input=user_input)
    result.latency_s = round(time.monotonic() - start, 3)
    result.metadata["thread_id"] = thread_id
    return result
