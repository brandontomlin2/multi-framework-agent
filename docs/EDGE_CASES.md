# Edge-case catalogue

_What a real agent will do that the example agent does not yet exercise, derived from Google's own agent samples (agents-cli templates, adk-samples recipes, agent-starter-pack) on 2026-11-08. The raw survey with per-sample evidence is `docs/research/google-agent-samples-2026-11-08.md`. Use this file to choose the next example agents and eval-format additions._

Status legend: **supported** (runtime and evals cover it), **runtime only** (works on every target but no case can assert it), **partial**, **missing**. Difficulty is for this template on all four targets: L (hours), M (a day), H (a design).

## 1. What Google's samples actually exercise

Google's *templates* (agents-cli scaffold, LangChain/Claude extensions, starter-pack) are weather/time toys with one or two string→string tools and evals that grade only final-response quality, like ours. The complexity lives in the adk-samples recipes: deep-search (planner → sequential/loop pipeline with `google_search`, `output_schema`, citations), staged-ui-widgets (six tools with enum-like arguments, sentinel values, `status: ok|not_found|rejected` results, user-scoped preferences, five domain rubrics), high-volume-document-analyzer (a cursor kept in tool state across calls), ambient-expense-agent (`RequestInput` human-in-the-loop, `input_schema`), small-business-loan-agent (`output_schema`, a `before_tool` callback that blocks calls by process state), on-brand-genmedia (loop with escalation, artifacts), oauth-user-consent-flow (`{"pending": true}` tools), safety-plugins (tool-output injection), long-horizon-harness (iteration budgets, skills toolset, compaction). Their eval assets are of three kinds: agents-cli `EvaluationDataset` + rubric metrics, ADK `.evalset.json` with `intermediate_data.tool_uses` (trajectory, often names-only and any-order), and bespoke pytest pipelines.

## 2. The catalogue

| # | Edge case | Google demo | Our status | Proposal | Diff. |
|---|---|---|---|---|---|
| 1 | Parallel tool calls across many entities in **one round** | deep-search "4-5 queries, execute all"; genmedia N variations | **supported** for 2 (`two_cities_in_parallel`); no N>2 case; cannot assert they happened in one round | `metadata.tool_rounds` on every target + `expect.max_tool_rounds`, `tool_call_counts: {tool: n}` | L |
| 2 | Nested / list / enum / `Literal` arguments, sentinel values (`order_id: ""` = latest) | staged-ui-widgets, high-volume-document-analyzer, genmedia `try_on(...)`, loan `UnderwritingReport` | **partial**: schema + coercion work (`book(when: date)` in the parity suite); no example tool uses them; `tool_args_contain` cannot reach nested keys | dotted paths in `args_contain` (`"filters.size"`), JSON-subset `args`, `tool_args_invalid: [tool]` (first attempt rejected by validation) | L–M |
| 3 | Very large tool outputs | high-volume analyzer summarises; long-horizon prunes; CMA offloads >100,000 chars | **missing** for agent tools (only skill reads truncate): a 2 MB result is sent whole on LangGraph and offloaded on CMA — a divergence | `AgentSpec.max_tool_output_chars` applied in `ToolSpec.run` on every target; `expect.tool_output_max_chars` | L |
| 4 | Binary / media results | retail-virtual-tryon strips bytes, returns URIs; genmedia never handles bytes | **missing**: `json.dumps(default=str)` would stringify bytes | rule "tools return references"; `ToolSpec.run` rejects `bytes` with a clear error | L |
| 5 | Long-running / async tools | `LongRunningFunctionTool` → `{"status": "pending"}`; OAuth `{"pending": true}` | **not modelled** | portable job-handle pattern: `start_x` returns `{status: pending, job_id}`, `check_x` polls; fixture flips on 2nd call | M |
| 6 | Tool errors mid-plan and recovery | `retrieve_docs` folds exceptions into text; `not_found` with alternatives | **runtime only** (shared error text, `is_error` flag, parity test `boom`) | `ToolCall.is_error`, `expect.tool_errors: [tool]`, `tool_output_contains: {tool: text}`, fault toggle `MFA_<AGENT>_FAULTS` | L |
| 7 | Prompt injection through a tool result | safety-plugins `fib_tool` injected message + LlmAsAJudge on tool output | **partial** (user-message injection only) | a fixture record carrying an injected instruction; `output_not_contains: [marker]` | L |
| 8 | Slot filling and corrections over 3+ turns, per-turn assertions | staged-ui-widgets multi-turn evalsets; brand-aligned `revision_prompt` | **supported** for 2 turns on the combined result; cannot assert a specific turn | `turns: [{input, expect}]`; harness grades each turn and the whole | M |
| 9 | State the tools carry across turns (cursor, pagination, preferences) | document analyzer cursor in `tool_context.state`; `user:` profile; `set_score` | **missing** (tools are stateless; no `ToolContext`) | per-thread state dict passed to tools identically on all targets (CMA tools run host-side); `initial_state`, `expect.state_contains` | M–H |
| 10 | Structured JSON output | `output_schema=Feedback/UnderwritingReport` (disables tools in ADK) | **missing** | (a) fenced JSON the grader parses, or (b) a `submit_answer(...)` tool whose arguments are the schema — visible in every trajectory; `expect.output_json` | M |
| 11 | Sub-agent delegation (`AgentTool`, `transfer_to_agent`) | financial-advisor ×4, deep-search planner, brand-search-optimization | **not in the template** (CMA-only in the parity matrix) | `agent_tool(sub_spec)` wrapping a nested `AgentSpec` as a `ToolSpec` run host-side — works on all four targets and appears as an ordinary tool call | M |
| 12 | Human-in-the-loop approval before a risky action | `require_confirmation`, `RequestInput`, prompt-level "once the user approves" | **partial**: `interrupt_before` graphs are left pending on LangGraph; CMA has no pause for custom tools | portable prompt pattern: a non-mutating `start_x` → ask → `confirm_x` next turn, asserted per turn (#8) | L / H |
| 13 | Grounding / search tools with citations | `google_search` (model-internal, invisible in traces), `VertexAiSearchTool` mocked under `INTEGRATION_TEST` | **by design** function tools only, so a fixture-corpus `search_docs` *is* visible on every target (an advantage) | `output_regex` for `[doc_id]` citations | L |
| 14 | Code execution | `BuiltInCodeExecutor` (invisible), Agent Engine sandbox | **not mirrored** beyond skill scripts | a `calculate(expression)` tool over an `ast`-restricted grammar | M |
| 15 | Artifacts / files produced and consumed by tools | `save_artifact`/`load_artifact`; PDFs from user parts | **missing** (skills read-only; `write`/`edit` not mirrored) | host-side per-thread artifact store with `read_artifact`; `initial_artifacts`, `expect.artifacts_created` | M–H |
| 16 | Multi-language input | "Reply in the same language as the user"; ACLI Translation metric | **missing** | a French/Spanish case with `tool_args_contain` + judge; `output_language` would need a dependency | L |
| 17 | Safety / refusal / unsupported category | `safety` metric; `unsupported_category` rubric | **supported** (`out_of_scope`, `prompt_injection`, `other_day`) | add #7 and an "unsupported, offers alternative" case | L |
| 18 | Budget / iteration caps and loops | `LoopAgent(max_iterations)` + escalation; `IterationBudget(max_tool_calls_per_iteration)` | **runtime only** (`max_iterations`, shared limit text, parity tests) | `expect.error_contains`, `expect_error: true`; `max_tool_calls_per_round` | L |
| 19 | Session resumption / seeded history | ADK `session_input.state`; agents-cli "N+1" seeded prior turns | **partial** (threads persist per instance; every case starts fresh) | `history: [{user, assistant}]` seeded via `update_state` on LangGraph, replayed on CMA (divergence) | M |
| 20 | Model-internal tools vanishing from the trajectory | agents-cli `builtin-tools-eval.md` | **not an issue** here | a spec guard that rejects provider-native tools | L |
| 21 | Ordered / counted trajectory assertions | ADK `tool_trajectory_avg_score`, name-only ANY_ORDER evaluator | **partial** (`tools_called`, `tool_calls` any-order, `max_tool_calls`) | `tool_sequence: [a, b, c]` (exact or subsequence), `tool_call_counts` | L |
| 22 | Reference answers and per-case rubric lists | `reference.response`, `rubric_groups`, EVAL.yaml MUST/MUST NOT lists | **partial** (one free-text `judge`) | `judge` as a list graded separately; `reference:` carried into the Gemini export | L–M |
| 23 | Deterministic fixtures and fault injection behind env toggles | `INTEGRATION_TEST=TRUE`, `USE_MOCK_API`, local JSON data | **supported** (`MFA_WEATHER_PROVIDER`, `MFA_FORWARD_ENV`) | add a fault toggle (#6) | L |

## 3. Eval-format gaps (ours vs Google's)

- **Seeded history and initial state** (`agent_data` prior turns, `session_input.state`): not expressible (#9, #19).
- **Per-turn expectations** and **intermediate agent text**: ours grade the combined result only (#8).
- **Reference answers, rubric lists, grounding `context`**: one free-text `judge` (#22).
- **Ordered sequences and counts**: not expressible (#21).
- **`mfa export -t gemini_eval`** should also emit `responses=[{response}]` (the single-turn metrics score nothing without it), `turn_id`, keep single-turn cases as `prompt`, carry `reference`/rubrics, and flag tool errors; an ADK `.evalset.json` exporter (`conversation[].user_content`, `final_response`, `intermediate_data.tool_uses`) would let `adk eval` grade our runs too.
- **An advantage to keep**: every tool here is a function tool, so trajectories are complete on all four targets, including Claude Managed Agents; Google's model-internal tools (`google_search`, code executor) never appear in theirs.

## 4. Recommended next example agents

Shared rules (each is something a Google sample does): plain `@tool` functions with typed arguments; a `fixture` provider behind an env toggle; tools return references, never bytes; "not found"/"rejected" are results with alternatives, not exceptions; approval is a conversational step before the mutating tool; a skill carries the policy/procedure.

1. **`order_support_agent`** (retail orders and returns; after staged-ui-widgets + loan HITL + oauth pending). Tools: `lookup_order(order_id="")` (sentinel = latest), `search_products(query, filters: ProductFilters | None, page)` (nested model with `Literal` category, `list[str]` sizes, pagination), `start_return(order_id, sku, reason: Literal[...])` → `needs_confirmation` with a token, `confirm_return(token)` → `pending` ticket, `check_return(ticket_id)`, `shipping_estimate(destination: Address, skus)`. Skill `returns-policy/`. Covers #2, #5, #6, #7, #8, #12, #16, #17, #21, #22. Twelve case ideas in the survey (E.1).
2. **`docs_research_agent`** (grounded Q&A with citations over a bundled corpus; after deep-search, rag recipes, llm-auditor). Tools: `search_docs`, `read_doc` (one 300,000-character doc to exercise the output cap), `list_docs` (pagination), `calculate(expression)` (restricted grammar), `submit_answer(answer, citations: list[Citation], confidence)` as the structured final output. Skill `cited-briefing/`, `max_iterations=6`. Covers #1, #3, #10, #13, #14, #18, #20, #21, #22. Eleven case ideas (E.2).
3. **`expense_approvals_agent`** (policy-driven intake with approval, pending jobs, references; after ambient-expense-agent, loan agent, starter-pack `request_user_input`). Tools: `submit_expense(expense: Expense)` (nested model with `date` and three `Literal`s), `convert_currency`, `request_manager_approval` → pending ticket, `check_ticket`, `attach_receipt` → URI reference, `list_my_expenses`. Skill `expense-policy/`. Covers #2, #4, #5, #6, #8, #12, #17, #18, #19, #23. Eleven case ideas (E.3).

Coverage: every catalogue row except #11 (sub-agents) and #15 (artifacts) is exercised by at least one of the three; #11 follows once `agent_tool()` exists.

## 5. Suggested order of work

1. Cheap format additions that the new agents need (#1, #6, #18, #21, #22, #3/#4 guards): `tool_rounds`, `tool_call_counts`, `tool_sequence`, `tool_errors`/`is_error`, `error_contains`, judge lists, output caps.
2. `order_support_agent` (the richest exercise of argument shapes, errors and conversation).
3. Per-turn expectations (#8), then `docs_research_agent` (#10 structured output, #3 large outputs).
4. `expense_approvals_agent`, the fault toggle and a French/Spanish case.
5. Tool state (#9), artifacts (#15), `agent_tool()` (#11): design items, each a small ADR before code.
