"""mfa - deploy, run and evaluate one agent locally, on Gemini platform, or on Claude Managed Agents.

Targets: local | agent_engine (Agent Runtime) | cloud_run (container, A2A) | claude_managed

    mfa deploy   weather_agent --target claude_managed
    mfa run      weather_agent --target local "What's the weather in Paris today?"
    mfa chat     weather_agent --target agent_engine
    mfa eval     weather_agent --target all --min-pass-rate 0.9
    mfa compare  eval_results/weather_agent/*.json
    mfa show     weather_agent --target claude_managed      # print the translated config
    mfa build    weather_agent -o build/weather_agent        # container context (docker / agents-cli)
    mfa export   weather_agent -t claude_managed -o deploy/claude_managed   # ant apply files
    mfa export   weather_agent -t gemini_eval --report eval_results/weather_agent/x.json -o traces.json
    mfa register weather_agent -t claude_managed --agent-id agent_... --environment-id env_...
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from mfa.registry import load_agent
from mfa.spec import TARGETS


def _targets(value: str, agent: str | None = None) -> list[str]:
    """'all' means every target for deploy; for run/eval it means local plus every deployed target."""
    if value != "all":
        return value.split(",")
    if agent is None:
        return list(TARGETS)
    from mfa import deployments
    from mfa.spec import LOCAL

    return [t for t in TARGETS if t == LOCAL or deployments.load(agent, t)]


def _runner(spec, target: str, args):
    from mfa.targets import get_target

    opts = {"url": args.url} if getattr(args, "url", None) else {}
    return get_target(target).runner(spec, model=args.model, **opts)


def cmd_deploy(args) -> int:
    from mfa.targets import get_target

    spec = load_agent(args.agent)
    for t in _targets(args.target):
        opts = {k: v for k, v in vars(args).items() if k in ("project", "location", "staging_bucket") and v}
        record = get_target(t).deploy(spec, model=args.model, agent_ref=args.agent, **opts)
        print(json.dumps(record, indent=2))
    return 0


def cmd_run(args) -> int:
    spec = load_agent(args.agent)
    runner = _runner(spec, args.target, args)
    result = runner.run(args.input, thread_id=args.thread_id)
    print(result.model_dump_json(indent=2) if args.json else result.output)
    return 1 if result.error else 0


def cmd_chat(args) -> int:
    spec = load_agent(args.agent)
    runner = _runner(spec, args.target, args)
    thread_id = uuid.uuid4().hex
    print(f"Chatting with {spec.name} on {args.target} (Ctrl-D to exit)")
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            return 0
        if not line:
            continue
        r = runner.run(line, thread_id=thread_id)
        for c in r.tool_calls:
            print(f"  · {c.name}({json.dumps(c.args)})")
        print(r.error or r.output)
    return 0


def cmd_eval(args) -> int:
    from mfa.evals import claude_judge, load_cases, run_eval, save_report, to_markdown

    spec = load_agent(args.agent)
    dataset = args.dataset or spec.eval_dataset
    if not dataset:
        print(f"Agent {spec.name} has no eval_dataset; pass --dataset", file=sys.stderr)
        return 2
    cases = load_cases(dataset, tags=args.tag, ids=args.case)
    judge = None if args.no_judge else claude_judge(args.judge_model)

    worst = 1.0
    paths = []
    for t in _targets(args.target, spec.name):
        runner = _runner(spec, t, args)
        report = run_eval(
            agent=spec.name,
            runner=runner,
            cases=cases,
            judge=judge,
            concurrency=args.concurrency,
            model=getattr(runner, "model", None) or _record_model(spec.name, t),
        )
        json_path, md_path = save_report(report, args.out)
        paths.append(json_path)
        print(to_markdown(report))
        print(f"saved {json_path} and {md_path}\n")
        worst = min(worst, report.pass_rate)

    if len(paths) > 1:
        from mfa.evals import compare

        print(compare(paths))
    if args.min_pass_rate is not None and worst < args.min_pass_rate:
        print(f"FAIL: pass rate {worst:.0%} < required {args.min_pass_rate:.0%}", file=sys.stderr)
        return 1
    return 0


def _record_model(agent: str, target: str) -> str | None:
    from mfa import deployments

    rec = deployments.load(agent, target)
    return rec.get("model") if rec else None


def cmd_compare(args) -> int:
    from mfa.evals import compare

    print(compare(args.reports))
    return 0


def cmd_show(args) -> int:
    from mfa.spec import AGENT_ENGINE, CLAUDE_MANAGED, CLOUD_RUN

    spec = load_agent(args.agent)
    if args.target == CLAUDE_MANAGED:
        from mfa.targets.claude_managed import build_agent_params

        print(json.dumps(build_agent_params(spec, spec.model_for(CLAUDE_MANAGED, args.model)), indent=2))
    elif args.target == CLOUD_RUN:
        from mfa.targets.container import container_requirements

        model = spec.model_for(CLOUD_RUN, args.model)
        print(json.dumps({"MFA_AGENT": args.agent, "MFA_MODEL": model, "entrypoint": "uvicorn app.fast_api_app:app",
                          "requirements": container_requirements(spec, model)}, indent=2))
    elif args.target == AGENT_ENGINE:
        from mfa.targets.agent_engine import agent_module_ref, build_requirements

        model = spec.model_for(AGENT_ENGINE, args.model)
        print(json.dumps({"agent_ref": agent_module_ref(spec, args.agent), "model": model,
                          "requirements": build_requirements(spec, model)}, indent=2))
    else:
        print(json.dumps({"name": spec.name, "model": spec.model_for(args.target, args.model),
                          "tools": [t.name for t in spec.tools]}, indent=2))
    return 0


def cmd_build(args) -> int:
    from mfa.spec import CLOUD_RUN
    from mfa.targets.container import build_context

    spec = load_agent(args.agent)
    out = build_context(
        spec, spec.model_for(CLOUD_RUN, args.model), args.out or f"build/{spec.name}", agent_ref=args.agent
    )
    print(f"Container context written to {out}")
    print(f"  docker build -t {spec.name.replace('_', '-')} {out}")
    print(f"  docker run -p 8080:8080 {spec.name.replace('_', '-')}   # model + agent are baked in")
    print(f"  mfa eval {args.agent} -t cloud_run --url http://localhost:8080")
    return 0


def cmd_export(args) -> int:
    spec = load_agent(args.agent)
    if args.target == "claude_managed":
        from mfa.spec import CLAUDE_MANAGED
        from mfa.targets.claude_managed import write_ant_files

        paths = write_ant_files(spec, spec.model_for(CLAUDE_MANAGED, args.model), args.out or "deploy/claude_managed")
        print("\n".join(str(p) for p in paths))
        print(f"Next: ant apply --dry-run {' '.join(str(p) for p in paths)}")
        return 0
    if args.target == "gemini_eval":
        from mfa.evals.export import export_report

        if not args.report:
            print("--report is required for gemini_eval", file=sys.stderr)
            return 2
        out = export_report(args.report, args.out or "eval_results/gemini_eval.json", spec.eval_dataset)
        print(f"Wrote {out}  (grade with: agents-cli eval grade --traces {out} --config <eval_config.yaml>)")
        return 0
    print("export target must be claude_managed or gemini_eval", file=sys.stderr)
    return 2


def cmd_register(args) -> int:
    from mfa.targets.claude_managed import register

    if args.target != "claude_managed":
        print("register currently supports -t claude_managed", file=sys.stderr)
        return 2
    print(json.dumps(register(load_agent(args.agent), args.agent_id, args.environment_id), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mfa", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp, multi: bool):
        sp.add_argument("agent", help="agent name under agents/ or module:ATTR")
        sp.add_argument(
            "--target", "-t", default="local",
            help=f"one of {', '.join(TARGETS)}" + (", a comma list, or 'all'" if multi else ""),
        )
        sp.add_argument("--model", help="override the spec's model for this target")

    d = sub.add_parser("deploy", help="create or update the agent on a target")
    common(d, True)
    d.add_argument("--project")
    d.add_argument("--location")
    d.add_argument("--staging-bucket")
    d.set_defaults(func=cmd_deploy)

    r = sub.add_parser("run", help="send one message")
    common(r, False)
    r.add_argument("input")
    r.add_argument("--thread-id")
    r.add_argument("--json", action="store_true", help="print the full normalized result")
    r.add_argument("--url", help="cloud_run: call this URL instead of the deployed one (e.g. a local container)")
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("chat", help="interactive multi-turn chat")
    common(c, False)
    c.add_argument("--url", help="cloud_run: call this URL instead of the deployed one")
    c.set_defaults(func=cmd_chat)

    e = sub.add_parser("eval", help="run the shared eval suite")
    common(e, True)
    e.add_argument("--dataset", help="defaults to the spec's eval_dataset")
    e.add_argument("--tag", action="append", help="only cases with this tag (repeatable)")
    e.add_argument("--case", action="append", help="only this case id (repeatable)")
    e.add_argument("--no-judge", action="store_true", help="skip LLM-as-judge checks")
    e.add_argument("--judge-model")
    e.add_argument("--concurrency", type=int, default=4)
    e.add_argument("--out", default="eval_results")
    e.add_argument("--min-pass-rate", type=float, help="exit 1 if any target scores below this (0-1)")
    e.add_argument("--url", help="cloud_run: evaluate this URL instead of the deployed one")
    e.set_defaults(func=cmd_eval)

    cmp_ = sub.add_parser("compare", help="compare saved eval reports across targets")
    cmp_.add_argument("reports", nargs="+", type=Path)
    cmp_.set_defaults(func=cmd_compare)

    s = sub.add_parser("show", help="print what would be deployed to a target")
    common(s, False)
    s.set_defaults(func=cmd_show)

    b = sub.add_parser("build", help="write a container build context (Cloud Run / GKE / agents-cli)")
    b.add_argument("agent")
    b.add_argument("--model")
    b.add_argument("--out", "-o")
    b.set_defaults(func=cmd_build)

    x = sub.add_parser("export", help="export ant-apply files (claude_managed) or a Gemini eval dataset")
    x.add_argument("agent")
    x.add_argument("--target", "-t", required=True, choices=["claude_managed", "gemini_eval"])
    x.add_argument("--model")
    x.add_argument("--report", help="gemini_eval: an mfa eval JSON report")
    x.add_argument("--out", "-o")
    x.set_defaults(func=cmd_export)

    g = sub.add_parser("register", help="record an agent created outside mfa (e.g. by ant apply)")
    g.add_argument("agent")
    g.add_argument("--target", "-t", required=True)
    g.add_argument("--agent-id", required=True)
    g.add_argument("--environment-id", required=True)
    g.set_defaults(func=cmd_register)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
