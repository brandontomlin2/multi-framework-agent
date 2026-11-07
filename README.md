# multi-framework-agent

Write an agent once in **LangGraph** style: a prompt, some typed Python tools, and a set of eval cases. Then run it **locally**, deploy it to **Gemini Enterprise Agent Platform**, or deploy it to **Claude Managed Agents**, and grade every deployment with the **same eval suite**. The base application (`agents/<name>/`) doesn't change between platforms. All the per-platform wiring is generated from the spec at deploy time.

```
                 agents/<name>/  ── agent.py (AgentSpec) · tools.py · evals/cases.yaml   ← your app, never edited per platform
                                 │
     ┌───────────────────┬───────┴─────────────┬──────────────────────────┐
     ▼                   ▼                     ▼                          ▼
  local             agent_engine           cloud_run                 claude_managed
  LangGraph         Gemini platform:       Gemini platform:          Anthropic: spec → agent config,
  in-process        Agent Runtime (SDK)    container + A2A           tools run host-side as custom tools
                    same graph, pickled    agents-cli contract       (SDK deploy, or ant apply files)
     │                   │                     │                          │
     └─────────── AgentResult (reply + tool trajectory) ──────────────────┘
                                 ▼
             mfa eval: same cases.yaml · same graders · same judge   (+ export to Gemini eval format)
```

## Platforms and targets

| Platform | Target | How the agent gets there | When to use |
|---|---|---|---|
| Your machine | `local` | The LangGraph graph runs in-process | Development, CI |
| Gemini Enterprise Agent Platform | `cloud_run` | Container built from the spec, served with `uvicorn app.fast_api_app:app`, A2A at `/a2a/app` | **Recommended for LangGraph on Gemini platform.** Matches Google's official agents-cli LangChain template |
| Gemini Enterprise Agent Platform | `agent_engine` | The same graph, pickled and deployed to Agent Runtime (formerly Agent Engine) with the `vertexai` SDK | You specifically need an Agent Runtime resource |
| Claude Managed Agents | `claude_managed` | Prompt and tools translated into a versioned CMA agent, tools executed host-side | Anthropic-hosted agent loop |

**Why both Gemini targets.** Google's official [`agents-cli`](https://github.com/google/agents-cli) LangChain extension says LangGraph agents on Agent Runtime are *degraded*: the agent serves requests, but the Console playground can't call it, it can't be published to Gemini, and Console sessions and traces stay empty. Those features need ADK routes. So for LangGraph that skill recommends Cloud Run or GKE over A2A. `cloud_run` follows that contract exactly: `app/agent.py` exports `root_agent`, and `app/fast_api_app.py` exposes `app`. You can also deploy the output of `mfa build` with `agents-cli deploy`, or run it on GKE. `agent_engine` stays available for teams that want an Agent Runtime resource.

**What carries over to Claude Managed Agents.** CMA runs its own agent loop, so the prompt and tools are what get deployed. The runner answers each `agent.custom_tool_use` event by calling the same Python function the other targets call, so tool behaviour is identical everywhere. If you set a custom LangGraph topology (`AgentSpec.graph_builder`), it runs on `local`, `agent_engine` and `cloud_run`. CMA gets only the prompt and tools, and the deploy prints a note saying so.

## Layout

```
agents/weather_agent/        worked example: today's weather, asks where if you don't say; 11 eval cases
agents/_template/            copy this to start a new agent
src/mfa/
  spec.py                    AgentSpec / ToolSpec / @tool: the framework-neutral definition
  runtime/graph.py           the LangGraph template (shared by local, agent_engine, cloud_run)
  runtime/models.py          provider:model → LangChain chat model
  serving/                   FastAPI + A2A app for containers (agents-cli contract) + /mfa/query
  targets/local.py           local
  targets/agent_engine*.py   Gemini platform: Agent Runtime via SDK
  targets/container.py       generated container build context (app/ shims, Dockerfile, requirements)
  targets/cloud_run.py       Gemini platform: Cloud Run via `gcloud run deploy --source`
  targets/claude_managed.py  Claude Managed Agents: translate, deploy/update, ant files, register, runner
  evals/                     dataset, graders, Claude judge, harness, reports, compare, Gemini export
  cli.py                     `mfa`
tests/                       offline: fake LLM, fake Agent Runtime, fake CMA API, in-process A2A server
```

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[all]"        # extras: anthropic, vertex, serving, dev
pytest -q                      # offline, no credentials needed
scripts/ci.sh                  # the CI check in a fresh venv (what GitHub Actions runs)
```

If GitHub Actions can't run (for example the account's Actions billing is blocked), `scripts/ci.sh --post-status` runs the same check and posts the result to the commit as a separate `local-ci (scripts/ci.sh)` status. The description says where it ran, so nobody mistakes it for the Actions check.

| Target | Needs |
|---|---|
| `local` | Credentials for `models["local"]`: `ANTHROPIC_API_KEY` for `anthropic:*`, or ADC + `GOOGLE_CLOUD_PROJECT` for `google_vertexai:*` / `anthropic_vertex:*` |
| `cloud_run` | `gcloud` logged in, `GOOGLE_CLOUD_PROJECT`. Optional: `MFA_CLOUD_RUN_REGION` (default `us-central1`), `MFA_CLOUD_RUN_SERVICE_ACCOUNT`, `MFA_CLOUD_RUN_ALLOW_UNAUTHENTICATED=1` (off by default, so callers need an identity token; the runner fetches one) |
| `agent_engine` | ADC, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION` (default `us-central1`), `MFA_STAGING_BUCKET=gs://...`. Optional: `MFA_AGENT_ENGINE_SERVICE_ACCOUNT` |
| `claude_managed` | `ANTHROPIC_API_KEY` (or an `ant auth login` profile). Optional: the `ant` CLI if you use the file-based flow |
| LLM judge | `ANTHROPIC_API_KEY`. Default judge is `claude-opus-5-5` at low effort, override with `MFA_JUDGE_MODEL`. `--no-judge` runs only the deterministic checks |

To use a Claude API model inside either Gemini target, set `MFA_ANTHROPIC_SECRET` to a Secret Manager secret holding the key.

Models are set per target in the spec. `cloud_run` reuses the spec's `agent_engine` model unless the spec names one of its own, so existing agents need no change. Overrides: `MFA_LOCAL_MODEL`, `MFA_AGENT_ENGINE_MODEL`, `MFA_CLAUDE_MANAGED_MODEL`, or `--model`. The example defaults are Claude Opus 5.5 locally and on CMA, and `gemini-3.1-pro-preview` on Gemini platform, served from the `global` endpoint (`MFA_GEMINI_LOCATION`). Gemini 2.5 was retired on Vertex in October 2026.

## Usage

```bash
# Local
mfa run  weather_agent -t local "What's the weather in Paris today?"
mfa chat weather_agent -t local                      # try "what's the weather today?" → it asks where
mfa eval weather_agent -t local

# Gemini platform, container (recommended for LangGraph)
mfa show   weather_agent -t cloud_run             # env, entrypoint and requirements that will ship
mfa build  weather_agent -o build/weather_agent   # inspect / docker build / agents-cli deploy it yourself
mfa deploy weather_agent -t cloud_run             # gcloud run deploy --source, private by default
mfa eval   weather_agent -t cloud_run
mfa eval   weather_agent -t cloud_run --url http://localhost:8080   # a container you run locally

# Gemini platform, Agent Runtime via SDK
mfa deploy weather_agent -t agent_engine
mfa eval   weather_agent -t agent_engine

# Claude Managed Agents: option A, SDK deploy
mfa show   weather_agent -t claude_managed        # the exact agent config
mfa deploy weather_agent -t claude_managed        # reuses the env by name; redeploy = new agent version
mfa eval   weather_agent -t claude_managed

# Claude Managed Agents: option B, version-controlled files with ant apply (Anthropic's recommended control plane)
mfa export   weather_agent -t claude_managed -o deploy/claude_managed
ant apply --dry-run deploy/claude_managed/support-agent/agent.md deploy/claude_managed/support-agent/environment.yaml
ant apply --yes     deploy/claude_managed/support-agent/agent.md deploy/claude_managed/support-agent/environment.yaml
mfa register weather_agent -t claude_managed --agent-id agent_... --environment-id env_...   # IDs from claude-lock.json
mfa eval     weather_agent -t claude_managed

# Everything that's deployed, side by side, gated for CI
mfa eval    weather_agent -t all --min-pass-rate 0.9   # 'all' = local + every target with a deployment record
mfa compare eval_results/weather_agent/*.json

# Grade any run with Google's tooling as well
mfa export weather_agent -t gemini_eval --report eval_results/weather_agent/claude_managed-<ts>.json -o traces.json
agents-cli eval grade --traces traces.json --config eval_config.yaml
```

Deploy records live in `.deployments/<agent>.<target>.json`: the Agent Runtime resource name, the Cloud Run URL and A2A card URL, or the CMA agent ID, version and environment. Redeploying updates the same resource. They hold no secrets. Commit them, by removing `.deployments/` from `.gitignore`, if CI or teammates evaluate the same deployments. With `ant apply`, also commit `claude-lock.json`.

## The example agent

`weather_agent` has one tool, `get_weather_today(location)`. It returns today's conditions, the current temperature, today's high and low, and the chance of precipitation, in °F and °C, plus other places with the same name when the name is ambiguous. If the user doesn't say where, the agent asks before calling the tool, and it uses a place given in a later turn.

Weather data comes from `MFA_WEATHER_PROVIDER`:
- `open-meteo` (default): live data from [Open-Meteo](https://open-meteo.com/). It needs no API key; the hosts are `geocoding-api.open-meteo.com` and `api.open-meteo.com`. On Cloud Run and Agent Runtime the container calls these hosts. On Claude Managed Agents the tool runs in the `mfa` runner, so the CMA sandbox stays without network access.
- `fixture`: deterministic offline data for Seattle, Paris, Tokyo, London and Portland. The tests always use it. For a fully reproducible eval run, set it where the tool runs: in your shell for `local` and `claude_managed`, and with `MFA_FORWARD_ENV=MFA_WEATHER_PROVIDER` at deploy time for the Gemini targets (`MFA_FORWARD_ENV` copies any listed env vars into the deployed container).

Because live weather changes, the eval cases never assert specific temperatures. The deterministic checks cover behaviour (did it ask where, did it call the tool with the right place, did it report temperatures and units), and the judge checks that the reply matches the tool output it saw.

## Serving surface (`cloud_run` and `mfa build`)

| Route | Purpose |
|---|---|
| `POST /a2a/app` | A2A JSON-RPC (v1.0, plus v0.3 compatibility). The A2A context ID is the LangGraph thread |
| `GET /a2a/app/.well-known/agent-card.json` | Agent card. One skill per tool. `APP_URL` is set to the service URL after deploy |
| `POST /mfa/query` | `{"input", "thread_id"}` → the normalized `AgentResult` (reply + tool trajectory) the eval harness grades |
| `GET /health` | Liveness, agent name, model |

## Evals

One `agents/<name>/evals/cases.yaml` is shared by every target:

```yaml
cases:
  - id: asks_for_location
    input: "What's the weather like today?"
    expect:
      no_tool_calls: true                                # it must ask, not guess
      output_contains_any: ["where", "which city", "location"]
  - id: location_in_follow_up
    turns: ["What's the weather today?", "Tokyo"]       # multi-turn
    expect:
      tools_called: [get_weather_today]                  # trajectory checks
      tool_args_contain: {get_weather_today: {location: tokyo}}   # substring; tool_args = exact
      max_tool_calls: 1
      output_contains: ["Tokyo"]                         # output checks (case-insensitive)
      output_regex: "\\d+ ?°"
    judge: >-                                            # optional LLM-as-judge rubric
      Gives today's weather for Tokyo, Japan, with temperatures that match the tool output.
```

- **Passing:** a case passes when every check passes. A skipped judge (`--no-judge`, or no verdict returned) doesn't fail a case.
- **Reports:** `eval_results/<agent>/<target>-<timestamp>.{json,md}` record per-case checks, per-turn tool trajectories, latency, and CMA session IDs (which you can open in the Console session viewer).
- **Gemini format:** `mfa export -t gemini_eval` converts any report, from any target, into the Gemini platform `EvaluationDataset` shape (`eval_cases[].agent_data.turns[].events[]`, with `function_call` / `function_response` parts). The rubric and mfa's own checks travel as extra fields for custom metrics.

## Plugging in your own agent

```bash
cp -r agents/_template agents/my_agent
```

1. In `tools.py`, write typed functions with Google-style `Args:` docstrings and decorate them with `@tool`. The JSON schema for every platform comes from these. List extra pip dependencies in `AgentSpec.requirements`; they ship to both Gemini targets.
2. In `agent.py`, set the prompt, tools, per-target models and `eval_dataset`, and export `AGENT`. Fix the `agents._template...` import.
3. In `evals/cases.yaml`, write the cases.
4. Run `mfa eval my_agent -t local`, then `mfa deploy my_agent -t cloud_run,claude_managed`, then `mfa eval my_agent -t all`.

Bringing an existing LangGraph graph: set `AgentSpec.graph_builder=lambda spec, model, *, checkpointer: <compiled graph>`. Its state needs a `messages` key. That graph is what `local`, `agent_engine` and `cloud_run` serve.

## Notes and limits

- **Conversation state:** the LangGraph targets use `InMemorySaver`, so a multi-turn thread lives on one serving instance. For production multi-turn traffic, swap in a persistent checkpointer in `mfa.serving.load_root_agent` / `LangGraphAgentApp.set_up()`.
- **Packaging:** both Gemini targets copy `src/mfa` and `agents/` and pin LangGraph, LangChain and SDK versions to your local install. Run deploys from a repo checkout (`pip install -e .`).
- **CMA sandbox:** `limited` networking with no allowed hosts, because tools run host-side. The built-in toolset is off so the agent has exactly the spec's tools. `MFA_CMA_ENABLE_TOOLSET=1` adds it, with web tools still off. Any tool call the server pauses for approval during an eval is denied. Environment names are unique per workspace, so a deploy reuses `<agent>-env` if it exists.
- **CMA turn timeout:** `MFA_CMA_TURN_TIMEOUT_S` (default 600). It is checked as events arrive, and the turn is interrupted when it is exceeded.
- **Cost:** live evals call real models. A full run is 10 cases per target plus one judge call per case.

## References

- Google [`agents-cli`](https://github.com/google/agents-cli): LangChain extension (`extensions/langchain`), deploy and eval skills
- Anthropic [`launch-your-agent`](https://github.com/anthropics/launch-your-agent) and [`skills`](https://github.com/anthropics/skills): Claude Managed Agents primitives, `ant apply`, outcome rubrics
