#!/usr/bin/env python3
"""
Skills Loader - Markdown-based Agent Skills (OpenClaw-inspired)
================================================================
Loads "skills" from `workspace/skills/<name>/SKILL.md`, where each skill is a
standalone markdown runbook the LLM can pull into its context on demand.

This is Beast's equivalent of OpenClaw's ClawHub skills — a filesystem-native,
zero-dependency skill registry. Skills are cheap: any folder with a SKILL.md
becomes an instantly-available capability.

SKILL.md format
~~~~~~~~~~~~~~~
Optional YAML-ish frontmatter at the top, then free-form markdown instructions:

    ---
    name: research
    description: Run a structured web research pass and cite every claim.
    triggers: research, deep-dive, summarize
    ---

    # Research Skill

    When this skill is invoked, follow these steps:

    1. ...
    2. ...

Anything under a single `SKILL.md` file counts. Sub-files (helper scripts,
templates) can live next to it and be referenced via relative paths; the LLM
will `read_file` them as needed.

Why skills instead of more tools?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tools are low-level verbs ("click", "fetch URL"). Skills are higher-level
*recipes* composed out of tools. Adding a new skill is just writing a markdown
file — no code, no restart, no MCP server to install.

Public API
~~~~~~~~~~
- `discover_skills()` → list of skill metadata dicts
- `get_skill(name)` → full SKILL.md content (or None)
- `get_skills_index()` → markdown bullet list for injection into the system prompt
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

SKILLS_DIR = Path(__file__).parent / "workspace" / "skills"


def _parse_frontmatter(text: str) -> dict:
    """Parse optional `---` YAML-ish frontmatter at the top of a SKILL.md file.

    Only supports flat `key: value` pairs. Values are unquoted. If there's no
    frontmatter block, returns an empty dict.
    """
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end < 0:
        return {}
    block = text[3:end].strip()
    out: dict = {}
    for line in block.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        out[key.strip().lower()] = value.strip().strip('"').strip("'")
    return out


def _strip_frontmatter(text: str) -> str:
    """Return SKILL.md body without the frontmatter block."""
    if not text.startswith("---"):
        return text
    end = text.find("\n---", 3)
    if end < 0:
        return text
    # Skip past the closing `---` and the following newline.
    return text[end + 4 :].lstrip("\n")


def discover_skills() -> list[dict]:
    """Return metadata for every SKILL.md under `workspace/skills/`.

    Each dict has: `name`, `description`, `triggers`, `path`. Skills without a
    `name:` in frontmatter fall back to their parent folder name. Skills without
    a `description:` get a placeholder. The scan is recursive so nested
    categories (e.g. `skills/research/web/SKILL.md`) are supported.
    """
    if not SKILLS_DIR.exists():
        return []
    skills: list[dict] = []
    for skill_file in sorted(SKILLS_DIR.rglob("SKILL.md")):
        try:
            text = skill_file.read_text()
        except Exception:
            continue
        meta = _parse_frontmatter(text)
        skills.append(
            {
                "name": meta.get("name") or skill_file.parent.name,
                "description": meta.get("description", "(no description)"),
                "triggers": meta.get("triggers", ""),
                "path": str(skill_file),
            }
        )
    return skills


def get_skill(name: str) -> Optional[str]:
    """Return the full SKILL.md body (without frontmatter) for the named skill.

    Name matching is case-insensitive. Returns None if not found.
    """
    target = name.strip().lower()
    for skill in discover_skills():
        if skill["name"].lower() == target:
            try:
                text = Path(skill["path"]).read_text()
                return _strip_frontmatter(text)
            except Exception:
                return None
    return None


def get_skills_index() -> str:
    """Return a markdown snippet suitable for injection into the system prompt.

    The LLM sees this at the top of every turn and uses it to decide whether
    to call `use_skill` for deeper instructions. Returns an empty string if
    there are no skills, so the caller can append it unconditionally.
    """
    skills = discover_skills()
    if not skills:
        return ""
    lines = [
        "## Available Skills",
        "",
        (
            "You have access to high-level skills (markdown runbooks) that live "
            "in `workspace/skills/`. Each one is a recipe built on top of your "
            "tools. When a user request matches a skill's description or triggers, "
            "call the `use_skill` tool with that skill's name to load its full "
            "instructions, then follow them."
        ),
        "",
    ]
    for skill in skills:
        line = f"- **{skill['name']}** — {skill['description']}"
        if skill["triggers"]:
            line += f"  _(triggers: {skill['triggers']})_"
        lines.append(line)
    lines.append("")
    return "\n".join(lines)
