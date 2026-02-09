#!/usr/bin/env python3
"""
Capability Tiers - Backend-aware settings for Beast
====================================================
Reads LLM_BACKEND from environment and exposes tiered settings.

Two-Tier System:
~~~~~~~~~~~~~~~~~
    FULL mode (Claude/OpenAI) — powerful cloud models, multi-tool capable
    LITE mode ("lfm" / local)  — restricted, single-tool friendly

    +---------------------+-----------+-----------+
    | Setting             | FULL      | LITE      |
    +---------------------+-----------+-----------+
    | Max tool turns      | 10        | 2         |
    | Single-tool mode    | Off       | On        |
    | Sequential thinking | On        | Off       |
    | Heartbeat interval  | 5 min     | 10 min    |
    | Tasks per cycle     | 3         | 1         |
    | Memory detail       | Full      | Minimal   |
    | MCP tiers loaded    | All 3     | All 3     |
    +---------------------+-----------+-----------+

"lfm" is a legacy name — it means any model served by the local server,
not just LFM-2.5 models.

Why "2 tool turns" for LITE?
    Local LLMs (Qwen, GLM, Llama) tend to loop on tool calls instead of
    summarizing the result. Limiting to 2 turns + SINGLE_TOOL_MODE forces
    the model to: (1) call one tool, (2) see the result, (3) respond with text.

Why do all MCP tiers load in LITE mode too?
    The user's local LLM needs access to all MCP servers (including cloud ones
    like brave-search for web queries). The LITE restrictions (single-tool mode,
    2 max turns) prevent tool-call loops. MCP tier labels are organizational
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


def _is_local() -> bool:
    """Check if we're running a local backend (anything that isn't claude/openai)."""
    return _BACKEND not in ("claude", "openai")


# ---------------------------------------------------------------------------
# Tiered Settings
# ---------------------------------------------------------------------------

# Max tool-call turns before forcing a text response.
# FULL: 10 turns is enough for complex multi-step tasks (e.g., read file → edit → verify).
# LITE: 2 turns prevents infinite tool-call loops that local LLMs are prone to.
MAX_TOOL_TURNS = 2 if _is_local() else 10

# After first tool call in one run(), stop sending tools to the LLM.
# This forces the model to summarize the tool result as text.
# Only needed for local LLMs — cloud models handle multi-tool properly.
SINGLE_TOOL_MODE = _is_local()

# Sequential Thinking MCP: only useful if the LLM is strong enough to
# follow multi-step reasoning AND we can afford extra tool-call turns.
SEQUENTIAL_THINKING_ENABLED = not _is_local()

# Heartbeat interval (seconds) — how often the autonomous loop checks tasks.
# Longer interval for local models because they're slower and use local compute.
HEARTBEAT_INTERVAL_SEC = 600 if _is_local() else 300  # 10 min local, 5 min cloud

# How many tasks the heartbeat processes per wake-up.
# Limited for local models to avoid long blocking periods.
HEARTBEAT_TASKS_PER_CYCLE = 1 if _is_local() else 3

# Memory detail level: "full" saves rich context, "minimal" saves key facts only.
# Local models produce less reliable summaries, so we save less to avoid noise.
MEMORY_DETAIL = "minimal" if _is_local() else "full"

# MCP tier filtering — which tiers of MCP servers are loaded at startup.
# All tiers are always loaded regardless of backend. The tier labels in
# mcp_servers.json are for documentation/organization only.
# The LITE/FULL distinction controls tool-calling behavior (single-tool mode,
# max turns), NOT which MCP servers are available. A local LLM should still
# be able to use brave-search, github, etc.
MCP_ALLOWED_TIERS = ["essential", "extended", "cloud"]

# Label for logging ("LFM" in the label is legacy — means any local model)
TIER_LABEL = "LITE (local LFM)" if _is_local() else f"FULL ({_BACKEND})"


# ---------------------------------------------------------------------------
# Convenience: print tier info when run directly
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
    print(f"MCP allowed tiers:    {MCP_ALLOWED_TIERS}")
