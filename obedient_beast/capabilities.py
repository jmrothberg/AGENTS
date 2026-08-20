#!/usr/bin/env python3
"""
Capability Tiers - Backend-aware settings for Beast
====================================================
Reads LLM_BACKEND from environment and exposes tiered settings.

Cloud vs Local:
~~~~~~~~~~~~~~~
    Two brain modes: Cloud (Claude/OpenAI) and Local (your machine).
    The key difference is "depth" — how many tool-call steps the model
    can chain together before it must respond.

    +---------------------+-----------+-----------+
    | Setting             | Cloud     | Local     |
    +---------------------+-----------+-----------+
    | Default depth       | 10        | 8         |
    | Sequential thinking | On        | Off       |
    | Heartbeat interval  | 5 min     | 10 min    |
    | Tasks per cycle     | 3         | 2         |
    | Memory detail       | Full      | Full      |
    | MCP servers loaded  | All       | All       |
    +---------------------+-----------+-----------+

    Depth is adjustable at runtime: `/depth 3` sets 3 tool steps.
    Use `/depth` to see the current value.

"lfm" is a legacy name — it means any model served by the local server,
not just LFM-2.5 models. "depth" was previously called "max tool turns".

Inspired by Clawdbot/OpenClaw's tiered agent architecture.
"""

import os
from pathlib import Path
from dotenv import load_dotenv


def load_beast_env():
    """Load .env from repo root, then obedient_beast/, then cwd.

    First file wins for each key (override=False). Canonical location is the
    repo-root `.env` next to lfm_thinking.py.
    """
    beast_dir = Path(__file__).resolve().parent
    repo_root = beast_dir.parent
    load_dotenv(repo_root / ".env")
    load_dotenv(beast_dir / ".env")
    load_dotenv()


load_beast_env()

# ---------------------------------------------------------------------------
# Read the backend from the same env var as llm.py
# ---------------------------------------------------------------------------
_BACKEND = os.getenv("LLM_BACKEND_TEST") or os.getenv("LLM_BACKEND", "lfm")


def is_cloud() -> bool:
    """Check if we're running a cloud backend (claude/openai)."""
    return _BACKEND in ("claude", "openai")


# ---------------------------------------------------------------------------
# Tiered Settings — Cloud vs Local
# ---------------------------------------------------------------------------
# "Depth" = how many tool-call steps the model can chain before responding.
# Cloud models are faster and smarter, so they get more steps by default.
# Local (Qwen3.8-class) defaults to 8; user can change it with /depth.

if is_cloud():
    DEPTH = 10                              # aka MAX_TOOL_TURNS
    SEQUENTIAL_THINKING_ENABLED = True
    HEARTBEAT_INTERVAL_SEC = 300            # 5 min
    HEARTBEAT_TASKS_PER_CYCLE = 3
    MEMORY_DETAIL = "full"
    TIER_LABEL = f"Cloud ({_BACKEND})"
else:
    DEPTH = 8
    SEQUENTIAL_THINKING_ENABLED = False
    HEARTBEAT_INTERVAL_SEC = 600            # 10 min
    HEARTBEAT_TASKS_PER_CYCLE = 2
    MEMORY_DETAIL = "full"
    TIER_LABEL = "Local"

# Backward-compatible alias — older code references MAX_TOOL_TURNS
MAX_TOOL_TURNS = DEPTH

# No longer needed — strong local models handle multi-tool chains fine.
# Kept as a constant so beast.py doesn't break if it still references it.
SINGLE_TOOL_MODE = False

# MCP tier filtering — all tiers always loaded regardless of backend.
MCP_ALLOWED_TIERS = ["essential", "extended", "cloud"]

# ---------------------------------------------------------------------------
# Tool groups — shrink the tool list sent to a local 27B; all handlers stay.
# Env BEAST_TOOL_GROUPS=core,browser,art (or "all"). /tools also sets this
# for the process lifetime (same pattern as /lfm).
# ---------------------------------------------------------------------------
TOOL_GROUPS = {
    "core": [
        "shell", "read_file", "write_file", "list_dir", "edit_file",
        "fetch_url", "recall_memory", "add_task",
        "run_python", "run_html", "list_skills", "use_skill",
    ],
    "browser": [
        "browser_goto", "browser_read", "browser_click", "browser_type",
        "browser_screenshot", "browser_close",
    ],
    "desktop": [
        "screenshot", "mouse_click", "mouse_move", "keyboard_type",
        "keyboard_hotkey", "get_screen_size", "get_mouse_position",
    ],
    "art": ["generate_art"],
    "mcp_mgmt": ["install_mcp_server", "list_mcp_servers", "enable_mcp_server"],
    "spawn": ["spawn_agent"],
}


def get_active_tool_groups() -> list:
    """Resolve active tool groups from BEAST_TOOL_GROUPS (env or /tools)."""
    raw = os.getenv("BEAST_TOOL_GROUPS", "").strip()
    if raw.lower() == "all":
        return list(TOOL_GROUPS.keys())
    if raw:
        return [g.strip().lower() for g in raw.split(",") if g.strip()]
    if is_cloud():
        return list(TOOL_GROUPS.keys())
    return ["core", "browser", "art"]


def set_tool_groups(spec: str) -> list:
    """Set BEAST_TOOL_GROUPS for this process. Returns the resolved group list."""
    spec = (spec or "").strip()
    os.environ["BEAST_TOOL_GROUPS"] = spec
    return get_active_tool_groups()


def filter_tools_by_group(tools: list) -> list:
    """Keep built-in tools in active groups. MCP tools (mcp_*) always pass."""
    groups = get_active_tool_groups()
    if set(groups) >= set(TOOL_GROUPS.keys()):
        return tools
    allowed = set()
    for g in groups:
        allowed.update(TOOL_GROUPS.get(g, []))
    return [
        t for t in tools
        if t.get("name", "").startswith("mcp_") or t.get("name") in allowed
    ]


def set_depth(n: int):
    """Change depth (tool-call steps) at runtime. Called by /depth command."""
    global DEPTH, MAX_TOOL_TURNS
    DEPTH = max(1, min(n, 20))
    MAX_TOOL_TURNS = DEPTH


# ---------------------------------------------------------------------------
# Convenience: print tier info when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Backend:              {_BACKEND}")
    print(f"Mode:                 {TIER_LABEL}")
    print(f"Depth (tool steps):   {DEPTH}")
    print(f"Sequential thinking:  {SEQUENTIAL_THINKING_ENABLED}")
    print(f"Heartbeat interval:   {HEARTBEAT_INTERVAL_SEC}s")
    print(f"Tasks per cycle:      {HEARTBEAT_TASKS_PER_CYCLE}")
    print(f"Memory detail:        {MEMORY_DETAIL}")
    print(f"MCP servers:          {MCP_ALLOWED_TIERS}")
    print(f"Tool groups:          {get_active_tool_groups()}")
