"""Agent Skills on the LangGraph targets, mirroring Claude Managed Agents.

Claude Managed Agents mounts a skill's folder in the session sandbox and lets
the model read it with the built-in ``read``/``glob``/``grep`` tools (and run
``scripts/`` with ``bash``). On local, Agent Runtime and Cloud Run the same
happens through :class:`SkillSandbox`:

* the model is told each skill's name, description and *location* (the same
  progressive disclosure: SKILL.md is read only when a task matches);
* ``read``, ``glob`` and ``grep`` carry the same names and input fields as
  the Managed Agents toolset and are confined to the skill folders;
* ``bash`` exists only when ``AgentSpec.allow_skill_scripts`` is set, and runs
  nothing but a skill's own ``scripts/`` files.

Locations are *virtual*: every target announces ``<root>/<skill-name>`` (root
``MFA_SKILLS_ROOT``, default ``/workspace/skills``) and resolves it to wherever
the folder really is, so a ``read`` call's ``file_path`` is identical in every
target's trajectory. The Claude Managed Agents runner uses the same object to
decide whether to allow a built-in call the server pauses for confirmation, so
path confinement and the script allowlist are identical by construction.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from mfa.spec import AgentSpec, SkillSpec, ToolSpec

DEFAULT_SKILLS_ROOT = "/workspace/skills"
MAX_OUTPUT_CHARS = 100_000  # Managed Agents offloads larger outputs to a file; we truncate
_INTERPRETERS = {"python", "python3", "sh", "bash", "node"}
# Anything bash itself would interpret: operators, expansions, quotes-that-hide-things, comments,
# and line breaks (a second line is a second command on the server's real shell).
_SHELL_META = re.compile(r"[|;&<>$`\\#*?\[\]{}~!\r\n]")
_SKILL_MD_PATH = re.compile(r"(?:^|/)(?P<skill>[a-z0-9]+(?:-[a-z0-9]+)*)/SKILL\.md$")


def skills_root() -> str:
    return os.environ.get("MFA_SKILLS_ROOT", DEFAULT_SKILLS_ROOT).rstrip("/") or DEFAULT_SKILLS_ROOT


def resolve_skills(spec: AgentSpec) -> list[SkillSpec]:
    """The spec's skills with Anthropic-managed ones replaced by a local copy when ``MFA_ANTHROPIC_SKILLS_DIR`` has one."""
    from mfa.spec import skill as load_skill

    base = os.environ.get("MFA_ANTHROPIC_SKILLS_DIR")
    out = []
    for sk in spec.skills:
        if sk.source == "anthropic" and base and (Path(base) / "skills" / sk.name / "SKILL.md").is_file():
            out.append(load_skill(Path(base) / "skills" / sk.name))
        elif sk.source == "anthropic" and base and (Path(base) / sk.name / "SKILL.md").is_file():
            out.append(load_skill(Path(base) / sk.name))
        else:
            out.append(sk)
    return out


class SkillSandbox:
    """Read-only view of an agent's skill folders behind one virtual root."""

    def __init__(self, skills: list[SkillSpec], *, root: str | None = None, allow_scripts: bool = False,
                 bash_timeout_ms: int = 120_000):
        self.root = (root or skills_root()).rstrip("/")
        self.skills = list(skills)
        self.local = {sk.name: sk for sk in skills if sk.path is not None}
        self.allow_scripts = allow_scripts
        self.bash_timeout_ms = bash_timeout_ms

    @classmethod
    def for_spec(cls, spec: AgentSpec) -> "SkillSandbox":
        return cls(resolve_skills(spec), allow_scripts=spec.allow_skill_scripts)

    # --- paths -----------------------------------------------------------
    def location(self, sk: SkillSpec) -> str:
        return f"{self.root}/{sk.name}"

    def announce(self, real: Path) -> str | None:
        """The virtual path of a real file inside a skill folder, or None when it is outside them."""
        for sk in self.local.values():
            try:
                rel = real.resolve().relative_to(sk.path.resolve())
            except ValueError:
                continue
            return f"{self.location(sk)}/{rel.as_posix()}".rstrip("/")
        return None

    def resolve(self, p: str) -> Path:
        """Map a model-supplied path to a real path inside a skill folder, or raise.

        Root-agnostic: the path may start with any root (``/workspace/skills``, the
        root a Managed Agents sandbox announces, ``skills/`` or nothing) as long as
        it then names one of this agent's skills followed by a path inside it.
        """
        text = str(p).strip()
        parts = [part for part in text.split("/") if part not in ("", ".")]
        idx = next((i for i, part in enumerate(parts) if part in self.local), None)
        if idx is None:
            raise PermissionError(f"{text} is not inside a skill folder ({self.root}/<skill>/...)")
        folder = self.local[parts[idx]].path.resolve()
        real = folder.joinpath(*parts[idx + 1:]).resolve()
        if real != folder and not real.is_relative_to(folder):
            raise PermissionError(f"{text} is not inside a skill folder ({self.root}/<skill>/...)")
        return real

    def check_bash(self, command: str | None) -> tuple[list[str], Path]:
        """Validate a bash command: a bare interpreter name (or the script itself) plus a file under a skill's scripts/.

        The string is what the Managed Agents server will run in a real shell, so
        anything bash would interpret (operators, expansions, comments, a second
        line) is refused rather than normalised away.
        """
        if not self.allow_scripts:
            raise PermissionError("running scripts is not enabled for this agent (allow_skill_scripts)")
        if not command or not command.strip():
            raise PermissionError("no command given")
        if _SHELL_META.search(command) or command.strip() != command.strip("\t "):
            raise PermissionError(f"only a skill's own scripts/ may be run, without shell operators: {command}")
        argv = shlex.split(command)
        if any(ch in tok for tok in argv for ch in "'\""):
            raise PermissionError(f"only a skill's own scripts/ may be run, without shell operators: {command}")
        script_index = 1 if argv and argv[0] in _INTERPRETERS else 0
        if len(argv) <= script_index:
            raise PermissionError(f"only a skill's own scripts/ may be run: {command}")
        script = self.resolve(argv[script_index])
        folder = next(sk.path.resolve() for sk in self.local.values() if script.is_relative_to(sk.path.resolve()))
        if not script.is_file() or script.parent != folder / "scripts":
            raise PermissionError(f"only a skill's own scripts/ may be run: {command}")
        argv[script_index] = str(script)
        return argv, folder

    def search_root(self, pattern: str | None, path: str | None) -> Path:
        """The one skill directory a glob/grep call may search: ``path`` inside a skill folder, or an absolute
        pattern under a skill's location. A call with neither is refused on every target, because the
        Managed Agents server would otherwise search its whole working directory."""
        if path:
            real = self.resolve(path)
            if not real.is_dir():
                raise NotADirectoryError(f"{path} is not a directory in a skill folder")
            return real
        if pattern and "/" in pattern.strip():
            head, _, _ = pattern.strip().rpartition("/")
            try:
                real = self.resolve(head)
                if real.is_dir():
                    return real
            except PermissionError:
                pass
        raise PermissionError(f"glob and grep need a path inside a skill folder, e.g. {self.root}/<skill>")

    def authorize(self, name: str, args: dict[str, Any]) -> None:
        """Raise if a built-in tool call falls outside the sandbox; the Managed Agents runner denies it then.

        Mirrors exactly what ``read``/``glob``/``grep``/``bash`` accept, so a call
        approved here is one the LangGraph targets would have run.
        """
        if name == "read":
            self.resolve(str(args.get("file_path", "")))
        elif name in ("glob", "grep"):
            self.search_root(str(args.get("pattern", "")), args.get("path"))
            if ".." in str(args.get("pattern", "")).split("/"):
                raise PermissionError("pattern may not contain '..'")
        elif name == "bash":
            if args.get("restart") and not args.get("command"):
                return
            self.check_bash(args.get("command"))
        else:
            raise PermissionError(f"tool {name} is not available to this agent")

    # --- the tools ---------------------------------------------------------
    def read(self, file_path: str, view_range: list[int] | None = None) -> str:
        real = self.resolve(file_path)
        if real.is_dir():
            raise IsADirectoryError(f"{file_path} is a directory; use glob to list it")
        if not real.is_file():
            raise FileNotFoundError(f"no such file in skill folder: {file_path}")
        text = real.read_text(encoding="utf-8")
        if view_range:
            lines = text.splitlines(keepends=True)
            start = max(1, int(view_range[0]))
            end = int(view_range[1]) if len(view_range) > 1 else 0
            end = len(lines) if end <= 0 else min(end, len(lines))
            text = "".join(lines[start - 1:end])
        return _truncate(text)

    def _files_under(self, base: Path, pattern: str):
        if ".." in pattern.split("/"):
            raise PermissionError("pattern may not contain '..'")
        for p in sorted(base.glob(pattern or "*")):
            if not p.is_file() or any(part in ("__pycache__", "node_modules") for part in p.parts):
                continue
            if not p.resolve().is_relative_to(base):
                continue  # a symlink pointing outside the skill folder is skipped, never followed
            yield p

    def glob(self, pattern: str, path: str | None = None) -> str:
        base = self.search_root(pattern, path)
        pat = pattern.strip()
        if path is None:
            pat = pat.rpartition("/")[2] or "*"  # the directory part chose the root
        hits = [self.announce(p) for p in self._files_under(base, pat)]
        return _truncate("\n".join(sorted(h for h in hits if h)) or "(no files match)")

    def grep(self, pattern: str, path: str | None = None) -> str:
        base = self.search_root(None, path) if path else self.search_root(pattern, None)
        rx = re.compile(pattern)
        hits: list[str] = []
        for p in self._files_under(base, "**/*"):
            try:
                lines = p.read_text(encoding="utf-8").splitlines()
            except (UnicodeDecodeError, OSError):
                continue
            for i, line in enumerate(lines, 1):
                if rx.search(line):
                    hits.append(f"{self.announce(p)}:{i}:{line}")
                    if len(hits) >= 200:
                        return _truncate("\n".join(hits) + "\n[more matches omitted]")
        return _truncate("\n".join(hits) or "(no matches)")

    def bash(self, command: str | None = None, restart: bool | None = None, timeout_ms: int | None = None) -> str:
        if restart and not command:
            return "bash session restarted"  # no persistent session here: each call is a fresh process
        argv, folder = self.check_bash(command)
        env = {k: v for k, v in os.environ.items() if k in ("PATH", "LANG", "LC_ALL", "PYTHONIOENCODING")}
        with tempfile.TemporaryDirectory(prefix="mfa-skill-") as home:
            env.update({"HOME": home, "MFA_SKILL_DIR": str(folder)})
            proc = subprocess.run(
                argv, cwd=folder, capture_output=True, text=True, env=env,
                timeout=(timeout_ms or self.bash_timeout_ms) / 1000,
            )
        out = proc.stdout + (("\n" + proc.stderr) if proc.stderr else "")
        if proc.returncode != 0:
            raise RuntimeError(f"exit status {proc.returncode}: {out.strip()[:2000]}")
        return _truncate(out)


def _truncate(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return text[:MAX_OUTPUT_CHARS] + f"\n[truncated: {len(text) - MAX_OUTPUT_CHARS} more characters]"


def skill_tools(spec: AgentSpec, sandbox: SkillSandbox | None = None) -> list[ToolSpec]:
    """``read``/``glob``/``grep`` (+ ``bash``) ToolSpecs with the Managed Agents toolset's input fields.

    Empty when the spec has no skills, so agents without skills are unchanged.
    """
    if not spec.skills:
        return []
    sb = sandbox or SkillSandbox.for_spec(spec)

    def read(file_path: str, view_range: list[int] | None = None) -> str:
        """Read a text file from one of this agent's skill folders.

        Args:
            file_path: Path of the file to read, e.g. "<skill location>/SKILL.md".
            view_range: Optional [start_line, end_line], 1-indexed inclusive; an end_line of 0 or less means to the end of the file.
        """
        return sb.read(file_path, view_range)

    def glob(pattern: str, path: str | None = None) -> str:
        """List files in a skill folder matching a glob pattern (``**`` matches directories), one path per line.

        Args:
            pattern: Glob pattern such as "**/*.md", relative to path.
            path: The skill directory to search under (a skill's location or a folder inside it). Required.
        """
        return sb.glob(pattern, path)

    def grep(pattern: str, path: str | None = None) -> str:
        """Search a skill folder's files for a regular expression; returns "path:line_number:line" for each match.

        Args:
            pattern: Regular expression to search for.
            path: The skill directory to search under (a skill's location or a folder inside it). Required.
        """
        return sb.grep(pattern, path)

    def bash(command: str | None = None, restart: bool | None = None, timeout_ms: int | None = None) -> str:
        """Run one of a skill's own scripts, e.g. "python3 <skill location>/scripts/x.py --flag value". Nothing else can be executed.

        Args:
            command: The script to run with its arguments.
            restart: When true (and no command), restart the shell session.
            timeout_ms: Optional per-call timeout in milliseconds.
        """
        return sb.bash(command, restart, timeout_ms)

    funcs = [read, glob, grep] + ([bash] if spec.allow_skill_scripts else [])
    return [ToolSpec.from_function(f) for f in funcs]


def skills_prompt_block(spec: AgentSpec, sandbox: SkillSandbox | None = None) -> str:
    """The system-prompt addition that lists skills the way Managed Agents announces them."""
    if not spec.skills:
        return ""
    sb = sandbox or SkillSandbox.for_spec(spec)
    lines = [
        "# Skills",
        "",
        "You have skills: folders of instructions and resources for specific kinds of task. Only each skill's "
        "name, description and location are listed here. When a task matches a skill's description, first call "
        "`read` on that skill's SKILL.md (for example {\"file_path\": \"<location>/SKILL.md\"}), then follow its "
        "instructions, reading the files it references (paths are relative to the skill's folder) with `read`, "
        "and finding files with `glob` or `grep` when needed, always passing the skill's location as `path`. "
        "Skill folders are read-only. Do not read a skill whose description does not match the task.",
        "",
    ]
    for sk in sb.skills:
        if sk.path is None:
            lines.append(f"- {sk.name}: {sk.description} (Anthropic-managed; available on Claude Managed Agents only)")
        else:
            lines.append(f"- {sk.name}: {sk.description} (location: {sb.location(sk)})")
    if spec.allow_skill_scripts:
        lines += ["", "You may run a skill's own scripts with `bash` (for example "
                  "{\"command\": \"python3 <location>/scripts/x.py\"}); nothing else can be executed."]
    return "\n".join(lines)


def skills_used(tool_calls: list[Any]) -> list[str]:
    """Skill names whose SKILL.md was read, in order of first read (root-agnostic)."""
    seen: list[str] = []
    for call in tool_calls:
        if getattr(call, "name", None) != "read":
            continue
        m = _SKILL_MD_PATH.search(str((getattr(call, "args", None) or {}).get("file_path", "")))
        if m and m.group("skill") not in seen:
            seen.append(m.group("skill"))
    return seen


def skill_root_of(file_path: str) -> str | None:
    """``/workspace/skills`` from ``/workspace/skills/<skill>/SKILL.md`` (used to learn the server's mount root)."""
    m = _SKILL_MD_PATH.search(str(file_path))
    return str(file_path)[: m.start()].rstrip("/") or None if m else None
