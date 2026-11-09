# Skills

Put Agent Skills here, one folder each with a `SKILL.md` (YAML frontmatter with `name`
and `description`, then instructions) and optional `references/`, `scripts/` and
`assets/`. The folder name must equal the frontmatter `name`. Attach them in
`agent.py` with `skills=local_skills(Path(__file__).parent / "skills")`.

See https://github.com/anthropics/skills for the format and examples.
