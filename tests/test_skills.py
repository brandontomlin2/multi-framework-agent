"""The skill sandbox: path confinement, the mirrored toolset tools, the prompt block."""

from pathlib import Path

import pytest

from mfa.skills import (
    DEFAULT_SKILLS_ROOT,
    MAX_OUTPUT_CHARS,
    SkillSandbox,
    resolve_skills,
    skill_root_of,
    skill_tools,
    skills_prompt_block,
    skills_used,
)
from mfa.runtime.result import ToolCall
from mfa.spec import AgentSpec, anthropic_skill, local_skills

SKILLS = Path(__file__).parent / "skills"
SPEC = AgentSpec(name="s", description="d", system_prompt="p", tools=[], skills=local_skills(SKILLS))
ROOT = f"{DEFAULT_SKILLS_ROOT}/parity-skill"


def test_paths_resolve_only_inside_skill_folders():
    sb = SkillSandbox.for_spec(SPEC)
    real = (SKILLS / "parity-skill" / "SKILL.md").resolve()
    for form in (f"{ROOT}/SKILL.md", "skills/parity-skill/SKILL.md", "parity-skill/SKILL.md", f"{ROOT}/./SKILL.md",
                 "/mnt/skills/custom/parity-skill/SKILL.md", "/workspace/skillsx/parity-skill/SKILL.md"):
        assert sb.resolve(form) == real, form  # any root the server announces, as long as the skill name follows
    assert sb.announce(real) == f"{ROOT}/SKILL.md"
    assert sb.announce(Path("/etc/passwd")) is None
    for bad in ("/etc/passwd", f"{ROOT}/../../etc/passwd", "other-skill/SKILL.md", "", "/workspace/skills"):
        with pytest.raises(PermissionError):
            sb.resolve(bad)
    with pytest.raises(FileNotFoundError):
        sb.read(f"{ROOT}/missing.md")
    with pytest.raises(IsADirectoryError):
        sb.read(f"{ROOT}/references")


def test_read_glob_grep_and_truncation(tmp_path, monkeypatch):
    sb = SkillSandbox.for_spec(SPEC)
    assert sb.read(f"{ROOT}/references/notes.md", [2, 0]) == "line two\nline three\n"
    assert sb.read(f"{ROOT}/references/notes.md", [1, 1]) == "line one\n"
    assert sb.glob("*.md", path=ROOT) == f"{ROOT}/SKILL.md"
    assert sb.glob("*", path=f"{ROOT}/references") == f"{ROOT}/references/notes.md"
    assert sb.glob(f"{ROOT}/references/*.md") == f"{ROOT}/references/notes.md"  # absolute pattern under a skill
    assert sb.glob("*.zzz", path=ROOT) == "(no files match)"
    assert sb.grep("line t", path=f"{ROOT}/references") == f"{ROOT}/references/notes.md:2:line two\n{ROOT}/references/notes.md:3:line three"
    assert sb.grep("nothing-here", path=ROOT) == "(no matches)"
    for call in (lambda: sb.glob("*"), lambda: sb.grep("x"), lambda: sb.glob("*", path="/etc"),
                 lambda: sb.glob("../*", path=ROOT), lambda: sb.glob("../../*.py", path=ROOT)):
        with pytest.raises(PermissionError):
            call()  # no skill path, a path outside, or a pattern that walks out
    # a symlink to a file outside the skill folder is skipped, not followed, and never named
    link_dir = tmp_path / "linked-skill"
    link_dir.mkdir()
    (link_dir / "SKILL.md").write_text("---\nname: linked-skill\ndescription: links\n---\nx")
    (tmp_path / "secret.txt").write_text("hunter2")
    (link_dir / "leak.txt").symlink_to(tmp_path / "secret.txt")
    linked = SkillSandbox.for_spec(AgentSpec(name="l", description="d", system_prompt="p", tools=[], skills=local_skills(tmp_path)))
    assert linked.glob("*", path="/workspace/skills/linked-skill") == "/workspace/skills/linked-skill/SKILL.md"
    assert "hunter2" not in linked.grep("hunter", path="/workspace/skills/linked-skill")
    with pytest.raises(PermissionError):
        linked.read("/workspace/skills/linked-skill/leak.txt")
    # a very large file is truncated the way the server offloads it
    monkeypatch.setenv("MFA_SKILLS_ROOT", "/srv/skills")
    big = tmp_path / "big-skill"
    big.mkdir()
    (big / "SKILL.md").write_text("---\nname: big-skill\ndescription: big\n---\n" + "x" * (MAX_OUTPUT_CHARS + 10))
    spec = AgentSpec(name="b", description="d", system_prompt="p", tools=[], skills=local_skills(tmp_path))
    out = SkillSandbox.for_spec(spec).read("/srv/skills/big-skill/SKILL.md")
    assert out.endswith("[truncated: 10 more characters]") or "[truncated" in out


def test_bash_is_an_allowlist_of_skill_scripts():
    locked = SkillSandbox.for_spec(SPEC)
    with pytest.raises(PermissionError, match="allow_skill_scripts"):
        locked.check_bash(f"python3 {ROOT}/scripts/hello.py")
    sb = SkillSandbox(SPEC.skills, allow_scripts=True)
    argv, folder = sb.check_bash(f"python3 {ROOT}/scripts/hello.py --flag 1")
    assert argv[0] == "python3" and argv[1].endswith("tests/skills/parity-skill/scripts/hello.py") and argv[2:] == ["--flag", "1"]
    assert folder == (SKILLS / "parity-skill").resolve()
    assert sb.bash(f"python3 {ROOT}/scripts/hello.py") == "hello from parity-skill\n"
    assert sb.bash(f"python3 /mnt/skills/custom/parity-skill/scripts/hello.py there") == "hello from parity-skill there\n"
    assert sb.bash(restart=True) == "bash session restarted"
    for bad in (f"python3 {ROOT}/SKILL.md", f"python3 {ROOT}/scripts/missing.py", "ls", "python3", f"python3 {ROOT}/scripts/hello.py; rm -rf /",
                f"python3 {ROOT}/scripts/hello.py > out", "/bin/sh -c id", "", f"/usr/bin/python3 {ROOT}/scripts/hello.py",
                f"python3 {ROOT}/scripts/hello.py\nid", f"python3 {ROOT}/scripts/hello.py # c", f"python3 {ROOT}/scripts/hello.py '$(id)'",
                f"python3 {ROOT}/scripts/hello.py ~ *", f"python3 {ROOT}/scripts/hello.py $(id)"):
        with pytest.raises(PermissionError):
            sb.check_bash(bad)
    sb.authorize("bash", {"restart": True})
    with pytest.raises(PermissionError):
        sb.authorize("write", {"file_path": f"{ROOT}/x", "content": "y"})  # never mirrored


def test_tool_schemas_match_the_managed_agents_toolset():
    tools = {t.name: t for t in skill_tools(SPEC)}
    assert list(tools) == ["read", "glob", "grep"]
    assert set(tools["read"].input_schema["properties"]) == {"file_path", "view_range"}
    assert tools["read"].input_schema["required"] == ["file_path"]
    assert set(tools["glob"].input_schema["properties"]) == {"pattern", "path"}
    assert set(tools["grep"].input_schema["properties"]) == {"pattern", "path"}
    scripted = AgentSpec(name="s", description="d", system_prompt="p", tools=[], skills=SPEC.skills, allow_skill_scripts=True)
    bash = {t.name: t for t in skill_tools(scripted)}["bash"]
    assert set(bash.input_schema["properties"]) == {"command", "restart", "timeout_ms"} and bash.input_schema["required"] == []
    assert skill_tools(AgentSpec(name="n", description="d", system_prompt="p", tools=[])) == []


def test_prompt_block_announces_locations_and_anthropic_skills(tmp_path, monkeypatch):
    spec = AgentSpec(name="s", description="d", system_prompt="p", tools=[], skills=[*SPEC.skills, anthropic_skill("xlsx")])
    block = skills_prompt_block(spec)
    assert f"- parity-skill: {SPEC.skills[0].description} (location: {ROOT})" in block
    assert "- xlsx: Anthropic-managed skill (Anthropic-managed; available on Claude Managed Agents only)" in block
    assert "bash" not in block
    # a local checkout of anthropics/skills turns the Anthropic skill into a readable local one
    (tmp_path / "skills" / "xlsx").mkdir(parents=True)
    (tmp_path / "skills" / "xlsx" / "SKILL.md").write_text("---\nname: xlsx\ndescription: Spreadsheets.\n---\nUse openpyxl.")
    monkeypatch.setenv("MFA_ANTHROPIC_SKILLS_DIR", str(tmp_path))
    resolved = resolve_skills(spec)
    assert resolved[1].is_local and resolved[1].description == "Spreadsheets."
    assert f"- xlsx: Spreadsheets. (location: {DEFAULT_SKILLS_ROOT}/xlsx)" in skills_prompt_block(spec)
    assert skills_prompt_block(AgentSpec(name="n", description="d", system_prompt="p", tools=[])) == ""


def test_skills_used_and_root_detection():
    calls = [ToolCall(name="get_weather_today", args={"location": "Paris"}),
             ToolCall(name="read", args={"file_path": "/mnt/skills/custom/weather-briefing/SKILL.md"}),
             ToolCall(name="read", args={"file_path": "/mnt/skills/custom/weather-briefing/references/x.md"}),
             ToolCall(name="read", args={"file_path": "/mnt/skills/custom/other-skill/SKILL.md"})]
    assert skills_used(calls) == ["weather-briefing", "other-skill"]
    assert skill_root_of("/mnt/skills/custom/weather-briefing/SKILL.md") == "/mnt/skills/custom"
    assert skill_root_of("weather-briefing/SKILL.md") is None and skill_root_of("notes.md") is None
