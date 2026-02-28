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
    | Default depth       | 10        | 5         |
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
from dotenv import load_dotenv

load_dotenv()

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
# Local models default to 5 but the user can change it with /depth.

if is_cloud():
    DEPTH = 10                              # aka MAX_TOOL_TURNS
    SEQUENTIAL_THINKING_ENABLED = True
    HEARTBEAT_INTERVAL_SEC = 300            # 5 min
    HEARTBEAT_TASKS_PER_CYCLE = 3
    MEMORY_DETAIL = "full"
    TIER_LABEL = f"Cloud ({_BACKEND})"
else:
    DEPTH = 5
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
