#!/usr/bin/env python3
"""
Capability Tiers - Backend-aware settings for Beast
====================================================
Reads LLM_BACKEND from environment and exposes tiered settings.

Claude/OpenAI     = FULL mode (powerful, multi-tool)
"lfm" (any local) = LITE mode (restricted, single-tool friendly)

"lfm" is a legacy name — it means any model served by the local server,
not just LFM-2.5 models.

Inspired by Clawdbot/OpenClaw's tiered agent architecture.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Read the backend from the same env var as llm.py
# ---------------------------------------------------------------------------
_BACKEND = os.getenv("LLM_BACKEND_TEST") or os.getenv("LLM_BACKEND", "claude")


def _is_local() -> bool:
    """Check if we're running a local backend (anything that isn't claude/openai)."""
    return _BACKEND not in ("claude", "openai")


# ---------------------------------------------------------------------------
# Tiered Settings
# ---------------------------------------------------------------------------

# Max tool-call turns before forcing a text response
MAX_TOOL_TURNS = 2 if _is_local() else 10

# After this many tool calls in one run(), stop sending tools to the LLM
# (forces it to summarize). For local LLMs this prevents infinite loops.
SINGLE_TOOL_MODE = _is_local()

# Sequential Thinking MCP: only useful if the LLM is strong enough
# and we can afford extra tool-call turns for reasoning steps
SEQUENTIAL_THINKING_ENABLED = not _is_local()

# Heartbeat interval (seconds) - how often the autonomous loop checks tasks
HEARTBEAT_INTERVAL_SEC = 600 if _is_local() else 300  # 10 min local, 5 min cloud

# How many tasks the heartbeat processes per wake-up
HEARTBEAT_TASKS_PER_CYCLE = 1 if _is_local() else 3

# Memory detail level: "full" saves rich context, "minimal" saves key facts only
MEMORY_DETAIL = "minimal" if _is_local() else "full"

# Label for logging ("LFM" in the label is legacy — means any local model)
TIER_LABEL = "LITE (local LFM)" if _is_local() else f"FULL ({_BACKEND})"


# ---------------------------------------------------------------------------
# Convenience: print tier info
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Backend:              {_BACKEND}")
    print(f"Tier:                 {TIER_LABEL}")
    print(f"Max tool turns:       {MAX_TOOL_TURNS}")
    print(f"Single-tool mode:     {SINGLE_TOOL_MODE}")
    print(f"Sequential thinking:  {SEQUENTIAL_THINKING_ENABLED}")
    print(f"Heartbeat interval:   {HEARTBEAT_INTERVAL_SEC}s")
    print(f"Tasks per cycle:      {HEARTBEAT_TASKS_PER_CYCLE}")
    print(f"Memory detail:        {MEMORY_DETAIL}")
