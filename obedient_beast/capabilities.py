#!/usr/bin/env python3
"""
Capability Tiers - Backend-aware settings for Beast
====================================================
Reads LLM_BACKEND from environment and exposes tiered settings.

Three-Tier System:
~~~~~~~~~~~~~~~~~~~
    FULL mode  (Claude/OpenAI) — powerful cloud models, multi-tool capable
    LOCAL mode ("lfm" / local) — strong local models (Qwen3.5-122B, etc.)
    LITE mode  (small local)   — restricted, single-tool friendly (legacy)

    +---------------------+-----------+-----------+-----------+
    | Setting             | FULL      | LOCAL     | LITE      |
    +---------------------+-----------+-----------+-----------+
    | Max tool turns      | 10        | 5         | 2         |
    | Single-tool mode    | Off       | Off       | On        |
    | Sequential thinking | On        | Off       | Off       |
    | Heartbeat interval  | 5 min     | 10 min    | 10 min    |
    | Tasks per cycle     | 3         | 2         | 1         |
    | Memory detail       | Full      | Full      | Minimal   |
    | MCP tiers loaded    | All 3     | All 3     | All 3     |
    +---------------------+-----------+-----------+-----------+

"lfm" is a legacy name — it means any model served by the local server,
not just LFM-2.5 models.

LOCAL vs LITE — what changed?
    Newer local models (Qwen3.5-122B, etc.) are strong enough for multi-tool
    chains. They no longer loop infinitely on tool calls, so SINGLE_TOOL_MODE
    is off and they get 5 tool turns. LOCAL is now the default for "lfm" backend.
    Set LLM_LOCAL_TIER=lite to force the old restricted behavior.

Why do all MCP tiers load in every mode?
    The user's local LLM needs access to all MCP servers (including cloud ones
    like brave-search for web queries). MCP tier labels are organizational
    only — they don't restrict loading.

Inspired by Clawdbot/OpenClaw's tiered agent architecture.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Read the backend from the same env var as llm.py
# ---------------------------------------------------------------------------
_BACKEND = os.getenv("LLM_BACKEND_TEST") or os.getenv("LLM_BACKEND", "lfm")


def _is_cloud() -> bool:
    """Check if we're running a cloud backend (claude/openai)."""
    return _BACKEND in ("claude", "openai")


def _local_tier() -> str:
    """Determine local model tier: 'local' (default) or 'lite' (legacy restricted).

    Set LLM_LOCAL_TIER=lite to force old single-tool behavior for weaker models.
    """
    return os.getenv("LLM_LOCAL_TIER", "local").lower()


# ---------------------------------------------------------------------------
# Tiered Settings
# ---------------------------------------------------------------------------

if _is_cloud():
    # FULL mode — Claude/OpenAI cloud models
    MAX_TOOL_TURNS = 10
    SINGLE_TOOL_MODE = False
    SEQUENTIAL_THINKING_ENABLED = True
    HEARTBEAT_INTERVAL_SEC = 300        # 5 min
    HEARTBEAT_TASKS_PER_CYCLE = 3
    MEMORY_DETAIL = "full"
    TIER_LABEL = f"FULL ({_BACKEND})"
elif _local_tier() == "lite":
    # LITE mode — weak/small local models (legacy, opt-in via LLM_LOCAL_TIER=lite)
    MAX_TOOL_TURNS = 2
    SINGLE_TOOL_MODE = True             # Stop sending tools after first use
    SEQUENTIAL_THINKING_ENABLED = False
    HEARTBEAT_INTERVAL_SEC = 600        # 10 min
    HEARTBEAT_TASKS_PER_CYCLE = 1
    MEMORY_DETAIL = "minimal"
    TIER_LABEL = "LITE (local)"
else:
    # LOCAL mode — strong local models (Qwen3.5-122B, etc.) — new default
    MAX_TOOL_TURNS = 5
    SINGLE_TOOL_MODE = False            # Strong models handle multi-tool chains
    SEQUENTIAL_THINKING_ENABLED = False
    HEARTBEAT_INTERVAL_SEC = 600        # 10 min (still slower than cloud)
    HEARTBEAT_TASKS_PER_CYCLE = 2
    MEMORY_DETAIL = "full"
    TIER_LABEL = "LOCAL (local)"

# MCP tier filtering — all tiers always loaded regardless of backend.
# The tier labels in mcp_servers.json are for documentation/organization only.
MCP_ALLOWED_TIERS = ["essential", "extended", "cloud"]


# ---------------------------------------------------------------------------
# Convenience: print tier info when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Backend:              {_BACKEND}")
    print(f"Tier:                 {TIER_LABEL}")
    if not _is_cloud():
        print(f"Local tier override:  LLM_LOCAL_TIER={os.getenv('LLM_LOCAL_TIER', '(not set, default=local)')}")
    print(f"Max tool turns:       {MAX_TOOL_TURNS}")
    print(f"Single-tool mode:     {SINGLE_TOOL_MODE}")
    print(f"Sequential thinking:  {SEQUENTIAL_THINKING_ENABLED}")
    print(f"Heartbeat interval:   {HEARTBEAT_INTERVAL_SEC}s")
    print(f"Tasks per cycle:      {HEARTBEAT_TASKS_PER_CYCLE}")
    print(f"Memory detail:        {MEMORY_DETAIL}")
    print(f"MCP allowed tiers:    {MCP_ALLOWED_TIERS}")
