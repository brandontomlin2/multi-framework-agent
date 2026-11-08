"""A2A protocol layer for any AgentSpec, modeled on Google's agents-cli LangChain template.

The reply an A2A client receives is the same text ``/mfa/query`` returns (and
the evals grade): the agent's final message of the turn. Earlier assistant text
in the same turn (for example "let me check") is streamed while the turn runs
and then replaced by the final message.
"""

from __future__ import annotations

import os
import uuid

from a2a.helpers import new_task_from_user_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import add_a2a_routes_to_fastapi, create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill, Part
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH, PROTOCOL_VERSION_0_3, PROTOCOL_VERSION_1_0
from a2a.utils.errors import UnsupportedOperationError
from fastapi import FastAPI

from mfa.runtime.graph import (
    _text,
    dangling_tool_calls,
    extract_result,
    recursion_limit_for,
    repair_thread,
    turn_error_text,
)
from mfa.spec import CLOUD_RUN, AgentSpec, iteration_limit_error

A2A_NAME = os.environ.get("A2A_NAME", "app")
A2A_RPC_PATH = f"/a2a/{A2A_NAME}"


def advertised_base_url() -> str:
    if app_url := os.environ.get("APP_URL"):
        return app_url
    host = os.environ.get("HOST", "127.0.0.1")
    if host in ("0.0.0.0", "::", ""):
        host = "127.0.0.1"
    return f"http://{host}:{os.environ.get('PORT', '8080')}"


class SpecAgentExecutor(AgentExecutor):
    """Bridges an A2A request to the compiled graph; the A2A context id is the LangGraph thread."""

    def __init__(self, graph, max_iterations: int):
        self.graph = graph
        self.max_iterations = max_iterations

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.current_task
        if task is None:
            task = new_task_from_user_message(context.message)
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.start_work()

        user_input = context.get_user_input()
        config = {
            "configurable": {"thread_id": task.context_id},
            "recursion_limit": recursion_limit_for(self.max_iterations),
        }
        artifact_id = uuid.uuid4().hex
        streamed = ""  # text of the assistant message currently being streamed
        new_message = True  # the next assistant chunk starts a new message
        final_messages = None
        before = len((await self.graph.aget_state(config)).values.get("messages", []))
        try:
            async for mode, data in self.graph.astream(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config,
                stream_mode=["messages", "values"],
            ):
                if mode == "messages":
                    chunk, _meta = data
                    if getattr(chunk, "type", "") not in ("AIMessageChunk", "ai"):
                        new_message = True  # a tool result separates assistant messages
                        continue
                    text = _text(chunk)
                    if not text:
                        continue
                    if new_message:
                        streamed = ""
                    streamed += text
                    # append=False starts the artifact over, so the client always
                    # holds only the latest assistant message of the turn.
                    await updater.add_artifact(
                        [Part(text=text)], artifact_id=artifact_id, name="response",
                        append=not new_message, last_chunk=False,
                    )
                    new_message = False
                elif mode == "values":
                    final_messages = data.get("messages") if isinstance(data, dict) else None
        except Exception as exc:
            error = turn_error_text(exc, self.max_iterations)
            repair_thread(self.graph, config, error)
            await updater.failed(updater.new_agent_message([Part(text=error)]))
            return
        if dangling_tool_calls(list(final_messages or [])):
            # Ended on a refused tool call: the iteration cap, reported like /mfa/query does.
            error = iteration_limit_error(self.max_iterations)
            repair_thread(self.graph, config, error)
            await updater.failed(updater.new_agent_message([Part(text=error)]))
            return

        final = extract_result(
            list(final_messages or [])[before:], target=os.environ.get("MFA_TARGET", CLOUD_RUN), user_input=user_input
        ).output
        if final != streamed:
            await updater.add_artifact([Part(text=final)], artifact_id=artifact_id, name="response", append=False)
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        # The request handler maps this to the protocol's TaskNotCancelableError.
        raise UnsupportedOperationError("This agent does not support cancellation.")


def agent_card(spec: AgentSpec) -> AgentCard:
    rpc_url = f"{advertised_base_url()}{A2A_RPC_PATH}"
    return AgentCard(
        name=A2A_NAME,
        description=spec.description,
        supported_interfaces=[
            AgentInterface(url=rpc_url, protocol_binding="JSONRPC", protocol_version=PROTOCOL_VERSION_1_0),
            # Gemini's registration validator still expects the 0.3 card shape.
            AgentInterface(url=rpc_url, protocol_binding="JSONRPC", protocol_version=PROTOCOL_VERSION_0_3),
        ],
        version=os.environ.get("AGENT_VERSION", "0.1.0"),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[
            AgentSkill(id=t.name, name=t.name, description=t.description, tags=[spec.name])
            for t in spec.tools
        ] or [AgentSkill(id="chat", name="chat", description=spec.description, tags=[spec.name])],
    )


def add_a2a_routes(app: FastAPI, spec: AgentSpec, graph) -> None:
    card = agent_card(spec)
    handler = DefaultRequestHandler(
        # Same cap as /mfa/query, local and Agent Engine runs.
        agent_executor=SpecAgentExecutor(graph, max_iterations=spec.max_iterations),
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )
    add_a2a_routes_to_fastapi(
        app,
        agent_card_routes=create_agent_card_routes(card, card_url=f"{A2A_RPC_PATH}{AGENT_CARD_WELL_KNOWN_PATH}"),
        jsonrpc_routes=create_jsonrpc_routes(handler, rpc_url=A2A_RPC_PATH, enable_v0_3_compat=True),
    )
