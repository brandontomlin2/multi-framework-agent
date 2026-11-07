# multi-framework-agent

Define an agent once, run it locally with **LangGraph**, deploy it to **Gemini Enterprise Agent Platform (Vertex AI Agent Engine)** or **Claude Managed Agents**, and grade every target with the **same eval suite**.

```
                    agents/<name>/agent.py  ── AgentSpec (prompt, tools, models, evals)
                                 │
          ┌──────────────────────┼─────────────────────────┐
          ▼                      ▼                         ▼
   local (LangGraph)    agent_engine (Vertex AI)    claude_managed (Anthropic)
   graph runs in-proc   same graph, pickled &       spec translated to agent config;
                        served by Agent Engine      tools run host-side as custom tools
          │                      │                         │
          └──────────── AgentResult (output + tool trajectory) ─┘
                                 ▼
                 mfa eval: same cases.yaml, same graders, same judge
```

## How "lift and shift" works

| Piece | Where it lives | Local | Agent Engine | Claude Managed Agents |
|---|---|---|---|---|
| System prompt | `AgentSpec.system_prompt` | LangGraph system message | same graph | agent `system` |
| Tools | `@tool` functions | `ToolNode` | `ToolNode` (same code ships to GCP) | `custom` tools, executed by the runner with the same functions |
| Loop | | `mfa.runtime.graph` (model → tools → model) | identical graph via `LangGraphAgentApp.set_up()` | Anthropic's managed loop |
| Model | `AgentSpec.models[target]` | any `provider:model` | any `provider:model` (Gemini by default) | Claude model ID |
| Result | `AgentResult` | built from graph messages | returned by `query()` | built from session events |

Every target returns the same normalized `AgentResult` (final text and each tool call with its args and output). The eval harness only ever sees that object, so a case means the same thing on every target.

**What carries over to Claude Managed Agents:** CMA runs its own agent loop, so the LangGraph graph isn't uploaded there. The portable parts are the prompt and the tools, and those are what the spec holds. If an agent sets a custom LangGraph topology (`AgentSpec.graph_builder`), that topology runs locally and on Agent Engine. On CMA you get the same prompt and tools on Anthropic's loop, and the deploy step prints a note saying so.

**Where CMA tools execute:** tools are registered as `custom` tools, so the `ClaudeManagedRunner` handles each `agent.custom_tool_use` event by calling the same Python function the other targets call. Tool behaviour is identical across targets, and secrets for your tools stay in your process. If you want the agent to work fully unattended with no client in the loop, expose the tools as an MCP server and add it to the agent instead.

## Layout

```
src/mfa/
  spec.py                    AgentSpec / ToolSpec / @tool: the framework-neutral definition
  registry.py                "support_agent" → agents.support_agent.agent:AGENT
  runtime/graph.py           the LangGraph template (shared by local + Agent Engine)
  runtime/models.py          provider:model → LangChain chat model
  targets/local.py           target 1
  targets/agent_engine.py    target 2: deploy/update + runner
  targets/agent_engine_app.py  the object Agent Engine serves (set_up/query)
  targets/claude_managed.py  target 3: spec→agent translation, deploy/update, session runner
  evals/                     dataset schema, graders, LLM judge, harness, reports, compare
  cli.py                     `mfa` command
agents/support_agent/        worked example (orders, refunds, policies, escalation) + 10 eval cases
agents/_template/            copy this to start a new agent
tests/                       offline tests with fake LLM / fake Agent Engine / fake CMA API
```

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[all]"        # or .[anthropic] / .[vertex] for just one provider
pytest -q                      # offline, no credentials needed
```

Credentials per target:

| Target | Needs |
|---|---|
| `local` | Credentials for the model in `models["local"]`: `ANTHROPIC_API_KEY` for `anthropic:*`, or `gcloud auth application-default login` + `GOOGLE_CLOUD_PROJECT` for `google_vertexai:*` / `anthropic_vertex:*` |
| `agent_engine` | ADC, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION` (default `us-central1`), `MFA_STAGING_BUCKET=gs://...`. Optional `MFA_AGENT_ENGINE_SERVICE_ACCOUNT`. If the Agent Engine model is `anthropic:*`, set `MFA_ANTHROPIC_SECRET` to a Secret Manager secret ID holding the key |
| `claude_managed` | `ANTHROPIC_API_KEY` (or an `ant auth login` profile) |
| LLM judge | `ANTHROPIC_API_KEY`. Default judge is `claude-opus-5-5` at low effort, override with `MFA_JUDGE_MODEL`. Use `--no-judge` to run only the deterministic checks |

Models are set per target in the spec and can be overridden with env vars (`MFA_LOCAL_MODEL`, `MFA_AGENT_ENGINE_MODEL`, `MFA_CLAUDE_MANAGED_MODEL`) or `--model`. The example defaults are Claude Opus 5.5 locally and on CMA, and `gemini-3.1-pro-preview` on Agent Engine, served from the `global` endpoint (`MFA_GEMINI_LOCATION`). Gemini 2.5 was retired on Vertex in October 2026. To run Claude inside Agent Engine, use `anthropic_vertex:claude-opus-5-5`.

## Usage

```bash
# 1. Local
mfa run  support_agent -t local "Where is order ORD-1001?"
mfa chat support_agent -t local
mfa eval support_agent -t local

# 2. Gemini Enterprise Agent Platform / Vertex AI Agent Engine
mfa show   support_agent -t agent_engine          # requirements + model that will ship
mfa deploy support_agent -t agent_engine          # create, or update if already deployed
mfa eval   support_agent -t agent_engine

# 3. Claude Managed Agents
mfa show   support_agent -t claude_managed        # the exact agent config that will be created
mfa deploy support_agent -t claude_managed        # creates env + agent; redeploy = new agent version
mfa eval   support_agent -t claude_managed

# All three, side by side, gated for CI
mfa deploy support_agent -t agent_engine,claude_managed
mfa eval   support_agent -t all --min-pass-rate 0.9
mfa compare eval_results/support_agent/*.json
```

Useful eval flags: `--tag smoke`, `--case refund_within_window`, `--no-judge`, `--concurrency 8`, `--out dir`.

Deploys write `.deployments/<agent>.<target>.json` (Agent Engine resource name, or CMA agent ID, version, and environment ID). Redeploying updates the same resource instead of creating a new one. For CMA, each redeploy that changes the spec creates a new immutable agent version, and eval sessions pin to the recorded version. These files hold no secrets. Commit them, by removing `.deployments/` from `.gitignore`, if CI or teammates need to evaluate the same deployments.

Each eval run writes `eval_results/<agent>/<target>-<timestamp>.{json,md}`. The JSON holds every case's checks, the full tool trajectory, latency, and CMA session IDs, which you can open in the Console session viewer.

## Writing evals

`agents/<name>/evals/cases.yaml` is shared by all targets:

```yaml
cases:
  - id: refund_within_window
    tags: [refunds, smoke]
    input: "I'd like a refund for order ORD-1002."     # or turns: [..] for multi-turn
    expect:
      tools_called: [check_refund_eligibility]          # trajectory checks
      tool_args: {check_refund_eligibility: {order_id: ORD-1002}}
      tools_not_called: [create_support_ticket]
      no_tool_calls: false
      max_tool_calls: 4
      output_contains: ["215"]                          # output checks (case-insensitive)
      output_contains_any: ["eligible", "refund"]
      output_not_contains: ["processed"]
      output_regex: "\\$?215(\\.00)?"
    judge: >-                                           # optional LLM-as-judge rubric
      Confirms ORD-1002 is eligible for $215.00 and does not claim it was already processed.
```

A case passes when every check passes. A skipped judge (`--no-judge`, or the judge returned no verdict) doesn't fail the case. Reports give the case pass rate, check pass rate, judge pass rate, error count, p50 latency, and average tool calls.

## Plugging in your own agent

```bash
cp -r agents/_template agents/my_agent
```

1. In `tools.py`, write typed functions with Google-style `Args:` docstrings and decorate them with `@tool`. The JSON schema for every framework is generated from these. List any extra pip dependencies in `AgentSpec.requirements` so they ship to Agent Engine.
2. In `agent.py`, set the prompt, tools, per-target models and `eval_dataset`, and export `AGENT`. Fix the `agents._template...` import.
3. In `evals/cases.yaml`, write the cases.
4. Run `mfa eval my_agent -t local`, then `mfa deploy my_agent -t agent_engine,claude_managed`, then `mfa eval my_agent -t all`.

To use an existing LangGraph graph, set `AgentSpec.graph_builder=lambda spec, model, *, checkpointer: <compiled graph>`. Its state needs a `messages` key. Local and Agent Engine run your graph as-is, and CMA gets the prompt and tools.

## Notes and limits

- **Agent Engine state:** the deployed app uses LangGraph's `InMemorySaver`, so multi-turn threads live on one serving instance. Swap in a persistent checkpointer in `LangGraphAgentApp.set_up()` for production multi-turn traffic.
- **Agent Engine packaging:** `mfa deploy` copies `src/mfa` and `agents/` into a staging directory and ships them as `extra_packages`. It pins LangGraph, LangChain and the aiplatform SDK to your local versions so the pickled app loads remotely. Run deploys from a repo checkout (`pip install -e .`).
- **CMA sandbox:** the environment is created with `limited` networking and no allowed hosts, because tools run host-side. The built-in toolset (bash, files) is off so the agent has exactly the spec's tools. Set `MFA_CMA_ENABLE_TOOLSET=1` to add it, with web tools still off. Any tool call the server pauses for approval during an eval is denied.
- **CMA turn timeout:** `MFA_CMA_TURN_TIMEOUT_S` (default 600). It is checked as events arrive, and the turn is interrupted when it is exceeded.
- **Cost:** live evals call real models. A full run is 10 cases per target plus one judge call per case.
