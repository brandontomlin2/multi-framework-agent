# Hand-off: multi-framework-agent

_Last updated 2026-11-08. The README explains the design; this document is the state of the project, what has been verified where, and what to do next._

## 1. What this repository is

One agent definition (`agents/<name>/agent.py` exporting an `AgentSpec`: system prompt, typed `@tool` functions, Agent Skills, per-target models, YAML eval cases) runs unchanged on four targets, and one eval harness grades all of them:

| Target | Runtime | How it gets there |
|---|---|---|
| `local` | LangGraph in-process | `mfa run/eval -t local` |
| `cloud_run` | Gemini Enterprise Agent Platform, container (A2A + `/mfa/query`) | `mfa deploy -t cloud_run` (gcloud) or `mfa build` + `agents-cli deploy` |
| `agent_engine` | Vertex AI Agent Runtime, pickled LangGraph app | `mfa deploy -t agent_engine` |
| `claude_managed` | Claude Managed Agents (CMA), spec translated to a versioned agent | `mfa deploy -t claude_managed`, or `mfa export` + `ant apply` + `mfa register` |

Key modules: `src/mfa/spec.py` (the definition), `src/mfa/skills.py` (skills on LangGraph + the sandbox CMA's runner uses), `src/mfa/runtime/graph.py` (the turn boundary every LangGraph target shares), `src/mfa/targets/*.py` (one adapter per platform), `src/mfa/evals/*` (dataset, graders, harness, judge, exports), `src/mfa/cli.py`.

## 2. The promises the code makes (and how they are enforced)

1. **The turn contract** (README → "The turn contract"): tool validation/coercion, tool errors, unknown tools, the `max_iterations` cap, failed turns, and the final-reply rule behave identically on every target. Enforced by `tests/test_parity.py`, which runs each scenario through the real adapter code of all four targets with offline fakes and asserts identical `AgentResult`s.
2. **Skills parity** (README → "Agent Skills"): skills are announced, read and (optionally) scripted the same way everywhere; on CMA the built-in file tools run under `always_ask` and the runner approves each call with the same `SkillSandbox` rules LangGraph enforces.
3. **Eval trust** (README → "Evals"): ungraded checks are never passes; one crashing case never discards a run; `--trials` with Wilson intervals over cases; reports record the non-secret `MFA_*` environment.

If you change any of these, extend `tests/test_parity.py` first; it is the contract.

## 3. Verified state

### Offline (every commit)
`scripts/ci.sh` = fresh venv, `pip install -e ".[all]"`, `pytest -q`. **106 tests pass.** GitHub Actions (`ci.yml`) calls the same script but **jobs never start on this account** (Actions billing / spending limit; jobs show `runner_id: 0` and no logs). Until that is fixed in GitHub → Settings → Billing & plans, `scripts/ci.sh` is the CI.

### Live, Claude Managed Agents (2026-11-08)
- Agent `agent_01SD7xCKD87TXdWKDtrk3qji`, **version 2**, environment `env_01SVkmJeYVcPRMU3BhAQCvGd` (`weather_agent-env`, cloud, `limited` networking, no hosts).
- Skill `weather-briefing` uploaded via the Skills API as `skill_01FHTX7QQB9rRF4cb4JZsaXC`, version `skver_012cRsvrJn2bdqtk1cZh3qfR`; toolset `read`/`glob`/`grep` with `always_ask`.
- Confirmed on the service: skills mount at **`/workspace/skills/<name>`** (our default virtual root); the version-id pin and the `<slug>/<path>` multipart upload are accepted; the confirmation protocol (`agent.tool_use` → `requires_action` → `user.tool_confirmation` → `agent.tool_result`) works; the server renders `read` output with numbered lines.
- Evals, 15 cases, Claude judge: **15/15 on `local` and 15/15 on `claude_managed`** with `MFA_WEATHER_PROVIDER=fixture`. With live Open-Meteo under concurrency 2: 11/15 and 9/15, every failure a weather-service timeout/429 reported honestly by the agent (a data confound, not a platform difference).
- Full suite re-run from the session VM on 2026-11-08 (19 cases, Claude judge): **19/19 on `local` and 19/19 on `claude_managed`** with `MFA_WEATHER_PROVIDER=fixture` (p50 latency 2.9 s vs 5.5 s). Against live Open-Meteo from the same VM: 10/19 and 8/19, and all 20 failures were the weather service refusing the call (`429 Daily API request limit exceeded` or a connection reset) with the agent reporting the outage honestly on both targets; single turns on both targets succeeded whenever Open-Meteo served the request (live Paris weather on `local`, a live three-line Tokyo briefing through the skill on CMA). The VM's egress IP is shared, so Open-Meteo's per-IP free-tier budget was spent by other tenants; from a dedicated IP or with `MFA_OPEN_METEO_API_KEY` (customer hosts, per-key limits) the live suite is expected to match the fixture run.
- The four harder cases added afterwards (two cities in parallel, a correction in a follow-up turn, shouted input, two questions in one message): **4/4 on `local`, 3/4 on `claude_managed`**. The one miss was the judge, correctly, on `two_cities_in_parallel`: both cities' numbers were right and the two parallel tool calls worked, but the closing comparison ("about 9–11°F (5–6°C) higher all day") overstated the gap between the lows (4°C). Derived arithmetic over several tool results is a real edge class; keep the case and expect some variance there.
- The deployment record `.deployments/weather_agent.claude_managed.json` is gitignored and lived only in the sandbox that ran the deploy. Recreate it with  
  `mfa register weather_agent -t claude_managed --agent-id agent_01SD7xCKD87TXdWKDtrk3qji --environment-id env_01SVkmJeYVcPRMU3BhAQCvGd`  
  then `mfa deploy weather_agent -t claude_managed` to bring the skill ids into the record: it matches the attached custom skills by display name and uploads a new version of each (the attached content is unknown to it); later deploys re-upload only when content changes.

### Not yet run live
- `cloud_run` and `agent_engine`: no GCP credentials have been available. Deploy paths are covered by offline tests only (`tests/test_serving_and_exports.py`, `tests/test_targets.py`). First live checks to do: the container builds and serves `/health`, `/mfa/query` and the A2A card; Agent Runtime ships non-`.py` files (`SKILL.md`, references) with `extra_packages`; `ChatVertexAI(thinking_budget=...)` is accepted for the effort mapping.
- CMA `bash` for skill scripts (`allow_skill_scripts=True`): the weather agent keeps it off; exercised offline only.
- A CMA turn mixing a custom tool and a paused built-in in one model message, and a `requires_action` with several `event_ids`: handled by the runner and the fake, not yet observed live.

## 4. Credentials and environments

- Open-Meteo: no key needed for the free hosts. `MFA_OPEN_METEO_API_KEY` (optional, a paid Open-Meteo subscription) moves the weather tool to `customer-api.open-meteo.com` / `customer-geocoding-api.open-meteo.com`, which have per-key limits instead of per-IP ones; `MFA_OPEN_METEO_GEOCODE_URL` / `MFA_OPEN_METEO_FORECAST_URL` point at a self-hosted instance. Forward whichever you set into containers with `MFA_FORWARD_ENV`; the eval report's environment snapshot drops names containing KEY.
- Anthropic key: `MFA_ANTHROPIC_API_KEY` (read before `ANTHROPIC_API_KEY`, so a Claude Code cloud environment does not pick it up). In a Claude Code cloud session the platform refuses to store a credential for `api.anthropic.com` through the credential form; set it as an **environment variable** on the `multi-framework-agent` cloud environment (which also allows `api.open-meteo.com` and `geocoding-api.open-meteo.com`). Never paste a key into chat; the key used in the 2026-11-08 session was exposed that way and should be rotated.
- Gemini targets: ADC + `GOOGLE_CLOUD_PROJECT`; `MFA_STAGING_BUCKET` for Agent Runtime; `MFA_ANTHROPIC_SECRET` (Secret Manager) only if a Gemini target runs a Claude model.
- GitHub Actions live evals (`evals.yml`): repository secret `ANTHROPIC_API_KEY`, vars `GOOGLE_CLOUD_PROJECT`, `GCP_WIF_PROVIDER`, `GCP_SERVICE_ACCOUNT`.

## 5. Operating notes and gotchas

- **Pin the data source when comparing targets.** `MFA_WEATHER_PROVIDER=fixture` for the example agent (the `evals.yml` default). The tool runs host-side for CMA too, so the variable set in your shell covers both targets. Live Open-Meteo is a smoke test, not a gate: its free tier is capped per source IP per day, and a shared cloud egress IP (this session's VM, CI runners) is often already exhausted by other tenants, giving `429 Daily API request limit exceeded` and connection resets on both targets alike (the tool runs host-side for CMA too). The client retries 429/5xx twice and sends an identifying `User-Agent` (Open-Meteo throttles `Python-urllib` and generic agents harder), which helps between bursts but cannot buy back a spent daily budget.
- **`max_iterations` is rounds, not calls**: one assistant message with several parallel tool calls is one round. The template graph enforces it itself (`iteration_cap_route`); `recursion_limit` is only a generous safety net.
- **CMA built-ins must stay `always_ask`** for parity. `MFA_CMA_ENABLE_TOOLSET=1` (everything on, policy `auto`) is a parity-breaking experiment knob.
- **Skill paths**: the model is told `/workspace/skills/<name>`; authorisation keys on the skill name after any root, so a different mount path on the service would not break reads.
- **`glob`/`grep` need a skill directory as `path`**; without it the CMA server would search its whole working directory, so both targets refuse.
- **`bash` for skills** accepts only `python3|sh|bash|node <skill>/scripts/<file> args` with plain arguments; anything a shell would interpret is refused before CMA's real shell sees it.
- **Cloud Run runner retries** only failures that provably never reached the app (connection refused, DNS, 429, 503, one 401); a retry re-sends the turn on the same thread.
- **Judge**: `MFA_JUDGE_MODEL` (default `claude-opus-5-5`), `MFA_JUDGE_EFFORT` (default `medium`); errors and truncated verdicts are ungraded and fail a gated run unless `--allow-ungraded`.

## 6. Review history

- **Architecture review (2026-11-08)**, 7 lenses + adversarial verification: 49 confirmed findings. Priorities 1 (behavioural parity) and 3 (eval trust) are done (PR #1). Still open: priority 2 **serving hot path** (bounded/shareable conversation state: `InMemorySaver` grows per request, multi-turn threads pinned to one replica, no `--session-affinity`/`--concurrency`, no request deadline on the async path), priority 4 **platform fit** (CMA custom tools cannot run unattended → an MCP tool mode served from the Cloud Run container; `/mfa/query` has no application-level auth and accepts any `thread_id`), priority 5 **deploy unit** (only `agents/` can be deployed, records are cwd-relative and never reconciled, `agent_engine.py`/`container.py` duplicate staging).
- **Skills design panel + independent review (2026-11-08)**: implemented in PR #2; the review's seven findings (bash newline bypass, glob/grep without path, round counting for server-run built-ins, root-tied authorisation, unconfined glob, tautological CMA assertions, timeouts) are fixed.

## 7. Backlog (in the order I would do it)

1. **MCP servers on both targets** — the one "not yet" row in the README's parity matrix. Design is ready: `AgentSpec.mcp_servers=[McpServerSpec(name, url, tools?, permission)]` → CMA `mcp_servers` + `mcp_toolset` (auth via vaults), LangGraph via `langchain-mcp-adapters` (`MultiServerMCPClient(...).get_tools()`, awaited in a worker thread; wrap each tool as a `ToolSpec` so results go through the shared error contract). Offline tests inject the loader.
2. **Live Gemini runs** (`cloud_run` first, per Google's guidance), then `mfa eval -t all` and `mfa compare` across all four.
3. **Serving hot path** (review priority 2): drop the checkpointer for thread-less requests, make the checkpointer pluggable from env, `asyncio.wait_for` deadline, per-thread serialization, explicit Cloud Run flags.
4. **Unattended CMA**: an MCP tool mode so a Deployment/schedule/Console session does not stall on the first custom tool call.
5. **Deploy unit = the agent** (review priority 5) and record reconciliation.
6. **More example agents and edge cases** — see §8.

## 8. Expanding the example agents

The weather agent now has 19 eval cases: location handling, units, ambiguity, unknown places, scope guards, prompt injection, the `weather-briefing` skill (what to wear / umbrella / day plan), parallel calls for two cities, a correction in a follow-up turn, shouted input, and two questions answered by one call. The eval format gained `tool_calls` (distinct calls to one tool with different arguments) for the parallel case.

`docs/EDGE_CASES.md` is the catalogue of 23 edge cases derived from Google's agent samples (status in this template, proposal, difficulty), the eval-format gaps against agents-cli and ADK evalsets, and three fully specified next example agents (`order_support_agent`, `docs_research_agent`, `expense_approvals_agent`) with eval ideas; `docs/research/google-agent-samples-2026-11-08.md` is the raw survey with per-sample evidence. Start with the cheap format additions in its §5, then `order_support_agent`. When adding one: copy `agents/_template`, keep tools pure and fixture-backed so the suite stays offline, add a skill if the agent needs procedural knowledge, write cases with `tool_calls`/`skills_used` expectations, and extend `tests/test_parity.py` only when the agent exercises a behaviour the contract does not cover yet.

## 9. Conventions

- Commits carry the trailer lines used so far (see `git log`); PRs end with the Claude Code footer.
- Every change to `src/mfa` runs `scripts/ci.sh` before a push; a red PR is this repository's own problem unless the job never started (billing).
- Deployment records (`.deployments/`) and eval results (`eval_results/`) are gitignored; commit them deliberately if a team shares deployments.
