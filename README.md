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

**What carries over to Claude Managed Agents.** CMA runs its own agent loop, so the prompt, tools, skills and model effort are what get deployed. The runner answers each `agent.custom_tool_use` event by calling the same Python function the other targets call, so tool behaviour is identical everywhere; skills are uploaded with the Skills API and read through CMA's built-in file tools, which the runner approves with the same rules the LangGraph targets enforce (see *Agent Skills* below). If you set a custom LangGraph topology (`AgentSpec.graph_builder`), it runs on `local`, `agent_engine` and `cloud_run`. CMA gets only the prompt, tools and skills, and the deploy prints a note saying so.

## Layout

```
agents/weather_agent/        worked example: today's weather, asks where if you don't say; 15 eval cases
  skills/weather-briefing/   an Agent Skill (SKILL.md + references/ + scripts/) the agent reads on demand
agents/_template/            copy this to start a new agent
src/mfa/
  spec.py                    AgentSpec / ToolSpec / SkillSpec, @tool, skill(), local_skills(): the framework-neutral definition
  skills.py                  skills on LangGraph: virtual root, read/glob/grep(/bash) tools, prompt block; also gates CMA built-ins
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
| `claude_managed` | `MFA_ANTHROPIC_API_KEY` or `ANTHROPIC_API_KEY` (or an `ant auth login` profile). Optional: the `ant` CLI if you use the file-based flow |
| LLM judge | `MFA_ANTHROPIC_API_KEY` or `ANTHROPIC_API_KEY`. Default judge is `claude-opus-5-5` at `medium` effort; override with `MFA_JUDGE_MODEL` / `MFA_JUDGE_EFFORT` or `--judge-model` / `--judge-effort`. `--no-judge` runs only the deterministic checks |

**Anthropic key:** create it at platform.claude.com → Settings → API Keys. `mfa` reads `MFA_ANTHROPIC_API_KEY` first, then `ANTHROPIC_API_KEY`. Use `MFA_ANTHROPIC_API_KEY` wherever other tools also read `ANTHROPIC_API_KEY`, for example in a Claude Code cloud environment, where Claude Code would pick that name up too. In GitHub Actions, store it as the repository secret `ANTHROPIC_API_KEY`.

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
mfa eval    weather_agent -t all --min-pass-rate 0.9 --trials 3   # 'all' = local + every deployed target; 3 runs per case
mfa compare eval_results/weather_agent/*.json

# Grade any run with Google's tooling as well
mfa export weather_agent -t gemini_eval --report eval_results/weather_agent/claude_managed-<ts>.json -o traces.json
agents-cli eval grade --traces traces.json --config eval_config.yaml
```

Deploy records live in `.deployments/<agent>.<target>.json`: the Agent Runtime resource name, the Cloud Run URL and A2A card URL, or the CMA agent ID, version and environment. Redeploying updates the same resource. They hold no secrets. Commit them, by removing `.deployments/` from `.gitignore`, if CI or teammates evaluate the same deployments. With `ant apply`, also commit `claude-lock.json`.

## The example agent

`weather_agent` has one tool, `get_weather_today(location)`. It returns today's conditions, the current temperature, today's high and low, and the chance of precipitation, in °F and °C, plus other places with the same name when the name is ambiguous. If the user doesn't say where, the agent asks before calling the tool, and it uses a place given in a later turn.

It also has one skill, `skills/weather-briefing/`: when the user asks what to wear, whether to bring an umbrella, or for a briefing of their day, the agent reads `SKILL.md`, then `references/clothing-guide.md`, and answers in the skill's three-line shape with the guide's wording. The `skill`-tagged eval cases check that the skill is read on demand (`skills_used`) and that a plain forecast leaves it alone (`skills_not_used`). The fixture cities exercise every clothing band deterministically. `scripts/briefing_lines.py` produces the same lines from the command line; it runs only when an agent opts into `allow_skill_scripts`.

Weather data comes from `MFA_WEATHER_PROVIDER`:
- `open-meteo` (default): live data from [Open-Meteo](https://open-meteo.com/). It needs no API key; the hosts are `geocoding-api.open-meteo.com` and `api.open-meteo.com`. On Cloud Run and Agent Runtime the container calls these hosts. On Claude Managed Agents the tool runs in the `mfa` runner, so the CMA sandbox stays without network access.
- `fixture`: deterministic offline data for Seattle, Paris, Tokyo, London and Portland. The tests always use it. For a fully reproducible eval run, set it where the tool runs: in your shell for `local` and `claude_managed`, and with `MFA_FORWARD_ENV=MFA_WEATHER_PROVIDER` at deploy time for the Gemini targets (`MFA_FORWARD_ENV` copies any listed env vars into the deployed container).

Because live weather changes, the eval cases never assert specific temperatures. The deterministic checks cover behaviour (did it ask where, did it call the tool with the right place, did it report temperatures and units), and the judge checks that the reply matches the tool output it saw.

## The turn contract (what every target guarantees)

One eval suite can grade four runtimes only because they agree on what a turn returns. `tests/test_parity.py` runs each scenario below through the real adapter code of every target and asserts identical results.

| Situation | Every target does |
|---|---|
| Tool arguments | Validated and coerced against the tool's schema in `ToolSpec.run` (`"2"` → `2`, `"2026-11-08"` → `date`). Invalid arguments become `Error: invalid arguments for <tool>: ...` handed back to the model; the turn goes on. |
| Tool raises | `Error: tool <tool> failed: <Exception>: ...` handed back to the model, flagged as an error (`ToolMessage.status="error"` on LangGraph, `is_error: true` on CMA); the turn goes on. |
| Unknown tool name | `Error: unknown tool <name>; available tools: ...` handed back to the model. |
| More than `max_iterations` tool rounds | A round is one assistant message, however many tool calls it holds. The round past the cap is not executed. The turn ends with `AgentResult.error = "Error: iteration limit reached: max_iterations=N"`; the trajectory keeps the executed calls plus the refused ones (their `output` is that error). The thread stays usable for the next turn. |
| Any other failure | `AgentResult.error` is set and the partial trajectory is kept. `/mfa/query` still answers 200; A2A marks the task `failed` with the same text. |
| Final reply | The last non-empty assistant message of the turn, on `/mfa/query`, A2A and CMA alike. |
| Skill files | `read`/`glob`/`grep` see only the skill folders. Locations are announced under one virtual root (`/workspace/skills/<skill>/...`), and authorisation keys on the skill name after whatever root the server announces, so the same `file_path` is accepted on every target. A path outside the skill folders is refused with `Error: tool read failed: PermissionError: ...` everywhere; `glob`/`grep` must be given a skill directory as `path` (otherwise the CMA server would search its whole working directory), and patterns may not contain `..`. On CMA the runner denies a refused call before the server runs it. |
| Skill scripts | Only with `allow_skill_scripts=True`, and only `python3|sh|bash|node <skill>/scripts/<file> args` (bare interpreter names, plain arguments; no operators, expansions, comments, quotes or line breaks, because CMA runs the string in a real shell); anything else is `Error: tool bash failed: PermissionError: ...` on every target. |
| Transient platform retries | Invisible: the Claude Managed Agents runner ignores `session.error` events the server is retrying (counted in `metadata.server_retries`) and only reports exhausted or terminal errors. |

## Agent Skills, and the rest of the Claude Managed Agents feature set

[Agent Skills](https://github.com/anthropics/skills) are folders of instructions and resources an agent loads on demand: `SKILL.md` (YAML frontmatter with `name` and `description`, then the instructions) plus optional `references/`, `scripts/` and `assets/`. The model sees only each skill's name, description and location until a task matches; then it reads `SKILL.md` and the files it references. Claude Managed Agents supports them natively. This template gives the LangGraph targets the same behaviour, so a skill-using turn produces the same `AgentResult` everywhere.

**Adding skills to an agent**

```python
from mfa import AgentSpec, anthropic_skill, local_skills

AGENT = AgentSpec(
    ...,
    skills=[*local_skills(Path(__file__).parent / "skills"), anthropic_skill("xlsx")],
    allow_skill_scripts=False,   # True also exposes `bash`, limited to <skill>/scripts/ on every target
    effort="high",               # optional model effort, applied on every target that has it
)
```

`local_skills(folder)` loads every `<folder>/<name>/SKILL.md` one level deep (the same layout CMA scans in a repository's `.claude/skills/`); the frontmatter `name` must equal the folder name. Validation follows Anthropic's `quick_validate` rules. Tool names `bash`, `read`, `write`, `edit`, `glob`, `grep`, `web_fetch`, `web_search` are reserved for the built-in toolset.

**What each target does**

| | LangGraph targets (`local`, `agent_engine`, `cloud_run`) | `claude_managed` |
|---|---|---|
| Announcing skills | A `# Skills` block is appended to the system prompt: one line per skill with its description and location `/workspace/skills/<name>` (`MFA_SKILLS_ROOT` changes the root; it is virtual, nothing is copied) | The platform announces attached skills itself (its own wording and sandbox path) |
| Reading skills | `read`, `glob`, `grep` tools with the toolset's input fields (`file_path`/`view_range`, `pattern`/`path`), confined to the skill folders; `glob`/`grep` take the skill's location as `path` | Skill folders are uploaded with the Skills API at deploy (a new version whenever content changes; the agent pins the version id, or `latest` with `MFA_CMA_SKILL_VERSION=latest`). The built-in `read`/`glob`/`grep` are enabled with the `always_ask` policy, so every call pauses the session and the runner approves or denies it with the same sandbox rules (any call the server blocks on without an answer is denied, and a turn that times out denies its paused calls before interrupting), then records the server's `agent.tool_result` text as the call's output |
| Scripts | `bash` only with `allow_skill_scripts`; runs `<interpreter> <skill>/scripts/<file> args` as a fresh process with a minimal environment | `bash` enabled (also `always_ask`) only with `allow_skill_scripts`; the runner allows a command only if the same allowlist accepts it |
| Iteration cap | A read round is a model message like any other and counts toward `max_iterations` | Same: the round past the cap is denied before it runs and the turn is interrupted. Under a policy that lets the server run built-ins itself (`MFA_CMA_ENABLE_TOOLSET=1`), the round still counts and the turn is interrupted once its results are in |
| Anthropic skills (`xlsx`, `docx`, `pptx`, `pdf`) | Announced as *available on Claude Managed Agents only*, unless `MFA_ANTHROPIC_SKILLS_DIR` points at a checkout of `anthropics/skills`, whose copy is then mounted as a local skill | Attached by id |
| Packaging | Skill folders ship inside `agents/<name>/skills/` with the container and the Agent Runtime package (and in wheels via `package-data`) | Never packaged: uploaded, versioned, attached. `mfa export -t claude_managed` copies the folders beside `agent.md` for `ant` |

**Known divergences (named, never faked in tests):** the server renders `read`/`glob`/`grep` output in its own format (line numbers, newest-first ordering, offload of outputs over 100,000 characters), so the parity suite compares a CMA built-in call by name, arguments and error status (the offline fake deliberately renders results differently from the LangGraph tools), and live evals assert `skills_used` and `tool_args_contain`, never the text of a built-in tool's output; the announced skill location may differ from the virtual root (authorisation does not depend on it; the runner records what it sees in `metadata.skill_root`); CMA's `read` also renders images, PDFs and notebooks, ours is text-only; a tool the model names that the agent does not have is `Error: unknown tool ...` on LangGraph and `Permission to use ... has been denied.` on CMA; `bash` on CMA is a persistent shell in an isolated container, ours is a per-call process; `effort` is exact on Claude, a thinking-budget approximation on Gemini (`MFA_GEMINI_THINKING_BUDGET` overrides) and unverified on `anthropic_vertex`.

**Feature parity matrix.** Where the LangGraph base agent stands against Claude Managed Agents' features:

| CMA feature | Status | How |
|---|---|---|
| Custom tools | Mirrored | `@tool` functions run host-side on CMA and in-process elsewhere; one validation and error contract |
| Skills (custom) | Mirrored | this section |
| Skills (Anthropic-managed) | Partial | attached on CMA; local mirror only from a checkout of `anthropics/skills` with `allow_skill_scripts` and their Python dependencies |
| Model effort / speed / inference geo | Mirrored / CMA-only | `AgentSpec.effort` or `MFA_EFFORT` everywhere; `MFA_CMA_SPEED`, `MFA_CMA_INFERENCE_GEO` pass through to CMA only |
| Built-in toolset (`read`, `glob`, `grep`, `bash`) | Partial | mirrored for skills as above; `write`/`edit` not mirrored (skills are read-only); `web_search`/`web_fetch` off on CMA and not mirrored: express web access as a `@tool` |
| Permission policies | Partial | `always_ask` + the runner gate is the parity mode; `MFA_CMA_ENABLE_TOOLSET=1` (everything on, policy `auto`) is a parity-breaking knob for experiments |
| MCP servers | Not yet | `mcp_servers` + `mcp_toolset` on CMA, `langchain-mcp-adapters` on LangGraph; planned |
| Memory stores (`/mnt/memory`) | CMA-only | a versioned cross-session store; `InMemorySaver` is conversation state, not memory |
| Outcomes (rubric grader loop) | CMA-only | mfa's judge grades after the fact; the same rubrics could seed an outcome |
| Multiagent (subagents, advisor, workflows) | CMA-only | a custom `graph_builder` is author topology; CMA receives prompt + tools + skills from it |
| Session budgets, scheduled deployments, webhooks, vaults, repository mounts | CMA-only | control-plane features with no LangGraph equivalent in this template |

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

- **Passing:** a case passes when no check fails. A judge that is switched off (`--no-judge`) is a skip and doesn't fail a case. A judge that *could not run* (API error, refusal) is recorded as ungraded: it is not a pass, the report counts it under `judge_errors`, and `mfa eval` exits 1 when a gate is set unless you pass `--allow-ungraded`. A case whose runner or grader raises is a failed case with a `harness` check; it never discards the run.
- **Trials:** `--trials N` runs every case N times. The table shows `passed/runs` per case and `mfa compare` says whether two targets' pass-rate intervals overlap. The 95% Wilson interval is computed over *cases* (trials refine each case's pass estimate but are not independent samples, so they never shrink the interval); more distinct cases are what narrows it.
- **Environment:** every report records the non-secret `MFA_*` variables in effect (for example `MFA_WEATHER_PROVIDER`), so a cross-target difference can be traced to data, not only to the platform.
- **Skills:** `skills_used: [name]` passes when the skill's `SKILL.md` was read during the case (any root, any target); `skills_not_used` is the opposite. Tool expectations may name the skill tools (`read`, `glob`, `grep`, `bash`) like any other tool.
- **Reports:** `eval_results/<agent>/<target>-<timestamp>.{json,md}` record per-case checks, per-turn tool trajectories, latency, and CMA session IDs (which you can open in the Console session viewer).
- **Gemini format:** `mfa export -t gemini_eval` converts any report, from any target, into the Gemini platform `EvaluationDataset` shape (`eval_cases[].agent_data.turns[].events[]`, with `function_call` / `function_response` parts). The rubric and mfa's own checks travel as extra fields for custom metrics.

## Plugging in your own agent

```bash
cp -r agents/_template agents/my_agent
```

1. In `tools.py`, write typed functions with Google-style `Args:` docstrings and decorate them with `@tool`. The JSON schema for every platform comes from these. List extra pip dependencies in `AgentSpec.requirements`; they ship to both Gemini targets.
2. In `agent.py`, set the prompt, tools, per-target models and `eval_dataset`, and export `AGENT`. Fix the `agents._template...` import.
3. In `evals/cases.yaml`, write the cases.
4. Optional: add Agent Skills under `skills/<name>/SKILL.md` and attach them with `skills=local_skills(Path(__file__).parent / "skills")` (see *Agent Skills*). Set `effort` if the default depth is wrong for the job.
5. Run `mfa eval my_agent -t local`, then `mfa deploy my_agent -t cloud_run,claude_managed`, then `mfa eval my_agent -t all`.

Bringing an existing LangGraph graph: set `AgentSpec.graph_builder=lambda spec, model, *, checkpointer: <compiled graph>`. Its state needs a `messages` key. That graph is what `local`, `agent_engine` and `cloud_run` serve. Bind `mfa.runtime.graph.spec_tools(spec)` (the agent's tools plus the skill tools) and use `mfa.runtime.graph.system_prompt_for(spec)` so skills work in your graph too. To keep the iteration cap identical to the other targets, route after your model node with `mfa.runtime.graph.iteration_cap_route(spec, tools_node="<your tools node>")` instead of `tools_condition`; without it, only a generous `recursion_limit` safety net applies. A graph compiled with `interrupt_before` returns with the tool call pending: `run_turn` leaves it untouched and reports the waiting nodes in `metadata["pending_nodes"]`.

## Notes and limits

- **Conversation state:** the LangGraph targets use `InMemorySaver`, so a multi-turn thread lives on one serving instance. For production multi-turn traffic, swap in a persistent checkpointer in `mfa.serving.load_root_agent` / `LangGraphAgentApp.set_up()`.
- **Packaging:** both Gemini targets copy `src/mfa` and `agents/` and pin LangGraph, LangChain and SDK versions to your local install. Run deploys from a repo checkout (`pip install -e .`).
- **CMA sandbox:** `limited` networking with no allowed hosts, because tools run host-side. The built-in toolset is off so the agent has exactly the spec's tools. `MFA_CMA_ENABLE_TOOLSET=1` adds it, with web tools still off. Any tool call the server pauses for approval during an eval is denied. Environment names are unique per workspace, so a deploy reuses `<agent>-env` if it exists.
- **CMA turn timeout:** `MFA_CMA_TURN_TIMEOUT_S` (default 600). It is checked as events arrive, and the turn is interrupted when it is exceeded.
- **Capability knobs:** `MFA_EFFORT` (overrides `AgentSpec.effort` on every target), `MFA_SKILLS_ROOT` (virtual skill root announced to the model, default `/workspace/skills`), `MFA_ANTHROPIC_SKILLS_DIR` (checkout of `anthropics/skills` to mirror Anthropic skills locally), `MFA_GEMINI_THINKING_BUDGET` (replaces the effort→budget mapping for Gemini), `MFA_CMA_SKILL_VERSION=latest` (attach `latest` instead of the uploaded version id), `MFA_CMA_SPEED`, `MFA_CMA_INFERENCE_GEO`.
- **Cloud Run runner:** the identity token is refreshed before it expires (and once on a 401). Only failures that prove the request never reached the app are retried (connection refused, DNS, 429, 503), `MFA_CLOUD_RUN_RETRIES` times (default 3) with backoff; read timeouts and other 5xx are not, because a retry would run the same turn twice on the thread. The client timeout (960 s) is longer than the service's 900 s so the client never gives up first.
- **Cost:** live evals call real models. A full run is 11 cases per target per trial plus one judge call per case.

## References

- Google [`agents-cli`](https://github.com/google/agents-cli): LangChain extension (`extensions/langchain`), deploy and eval skills
- Anthropic [`launch-your-agent`](https://github.com/anthropics/launch-your-agent) and [`skills`](https://github.com/anthropics/skills): Claude Managed Agents primitives, `ant apply`, outcome rubrics
