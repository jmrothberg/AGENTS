#!/usr/bin/env python3
"""
Obedient Beast - CLI + Agent Loop + Tools
==========================================
A minimal agentic assistant with tool calling, autonomous task queue,
persistent memory, and two brain modes: Cloud (Claude/OpenAI) and Local
(any model via "lfm" backend). "Depth" controls how many tool steps
the model can chain per request (adjustable at runtime with /depth).

Architecture Overview:
~~~~~~~~~~~~~~~~~~~~~~
                    ┌──────────────────────┐
                    │    User Interface     │
                    │  (CLI or WhatsApp)    │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │     beast.run()       │  ◄── Agent loop: LLM ↔ Tools
                    │  (this file)          │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
    ┌─────────▼──────┐ ┌──────▼──────┐ ┌───────▼────────┐
    │   llm.py       │ │ Built-in    │ │  mcp_client.py │
    │  (3 backends)  │ │ Tools (18)  │ │  (MCP servers) │
    └────────────────┘ └─────────────┘ └────────────────┘

Data Flow:
~~~~~~~~~~
1. User input arrives (CLI input() or server.py HTTP POST)
2. Slash commands (/help, /status, etc.) are handled BEFORE the LLM
3. Non-slash input goes into the agent loop:
   a. Load conversation history from sessions/<id>.jsonl
   b. Append user message
   c. Call LLM with history + available tools
   d. If LLM returns tool_calls → execute them → add results → loop back to (c)
   e. If LLM returns text only → save to history → return to user
4. Auto-save key facts to memory (local JSON + MCP knowledge graph if running)

Tool Count: 18 built-in + N MCP tools (loaded dynamically)

Usage:
    python beast.py                     # Interactive CLI mode
    ./start.sh                          # 4 Terminal windows (server, WhatsApp, heartbeat, CLI)

Slash commands (work from CLI and WhatsApp):
    /help, /more, /status, /tasks, /done <id>, /drop <id>,
    /claude, /openai, /lfm, /depth <n>, /model, /heartbeat on|off,
    /clear, /clear tasks, /clear memory, /clear all, /tools, /skills
"""

import os
import sys
import json
import base64
import subprocess
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from llm import get_llm, ToolCall

# ---------------------------------------------------------------------------
# MCP Support (Phase 2 - Optional)
# ---------------------------------------------------------------------------
# MCP (Model Context Protocol) tools are loaded dynamically at startup.
# When MCP_ENABLED=true in .env, Beast spawns MCP server subprocesses
# (e.g., filesystem, memory, brave-search) and discovers their tools via
# JSON-RPC. These tools appear alongside built-in tools in the LLM's
# tool list, prefixed with "mcp_servername_" to avoid name collisions.
#
# If MCP is disabled or a server fails, Beast still works fine with
# its 18 built-in tools — MCP is purely additive.

# Default to true — if you have MCP servers configured, they should load.
# Set MCP_ENABLED=false in .env only if you explicitly want to disable MCP.
MCP_ENABLED = os.getenv("MCP_ENABLED", "true").lower() == "true"
_mcp_client = None  # Singleton MCP client, lazily initialized on first use


def get_mcp_tools() -> list[dict]:
    """
    Get MCP tools if MCP is enabled and servers are running.
    Returns tools in Beast's internal format (name/description/params dict).
    Lazily initializes the MCP client on first call.
    """
    global _mcp_client
    if not MCP_ENABLED:
        return []
    try:
        from mcp_client import get_mcp_client, init_mcp
        if _mcp_client is None:
            _mcp_client = init_mcp()  # Spawns server processes, discovers tools
        return _mcp_client.get_tools_for_llm()
    except Exception as e:
        print(f"[MCP] Not available: {e}", file=sys.stderr)
        return []


def execute_mcp_tool(name: str, args: dict) -> str:
    """
    Execute an MCP tool by its prefixed name (e.g., "mcp_memory_create_entities").
    Delegates to mcp_client.py which routes to the correct server subprocess.
    """
    global _mcp_client
    if _mcp_client is None:
        return "Error: MCP not initialized"
    try:
        from mcp_client import execute_mcp_tool as mcp_exec
        return mcp_exec(name, args)
    except Exception as e:
        return f"Error executing MCP tool: {e}"


def get_all_tools() -> list[dict]:
    """
    Get all available tools: 18 built-in + any MCP tools.
    This is what gets sent to the LLM so it knows what it can call.
    """
    return TOOLS + get_mcp_tools()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WORKSPACE = Path(__file__).parent / "workspace"
SESSIONS_DIR = Path(__file__).parent / "sessions"
SESSIONS_DIR.mkdir(exist_ok=True)

# Local memory file — the persistent memory store (see "Memory System" section below).
# Stores facts as {"facts": [{"text": "...", "timestamp": "..."}]}
# Capped at 200 entries (FIFO) to prevent unbounded growth.
MEMORY_FILE = WORKSPACE / "memory.json"

# Sandbox activity log — append-only log of every run_python/run_html execution.
# Easy to review what Beast generated and ran, without parsing LLM debug output.
SANDBOX_DIR = WORKSPACE / "Generated Code"
SANDBOX_LOG = SANDBOX_DIR / "activity.log"


def _sandbox_log(tool: str, folder: str, code_snippet: str, result_snippet: str):
    """Append a timestamped entry to the sandbox activity log."""
    try:
        SANDBOX_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = (
            f"\n{'='*60}\n"
            f"[{ts}] {tool}\n"
            f"Folder: {folder}\n"
            f"--- Code (first 500 chars) ---\n"
            f"{code_snippet[:500]}\n"
            f"--- Result (first 500 chars) ---\n"
            f"{result_snippet[:500]}\n"
        )
        with open(SANDBOX_LOG, "a") as f:
            f.write(entry)
    except Exception:
        pass  # Never block the response for a log failure


# Pending image to send with next response (for WhatsApp image sending).
# When the screenshot tool runs, it sets this path so server.py can
# include the image in the WhatsApp reply.
_pending_image: str = None

# Backend override — set via /claude /openai /lfm commands.
# Persists for the current process lifetime (not across restarts).
_backend_override: str = None


def set_pending_image(path: str):
    """Set an image to be sent with the next response (used by screenshot tool)."""
    global _pending_image
    _pending_image = path


def get_and_clear_pending_image() -> str:
    """Get and clear the pending image path. Called by server.py after each response."""
    global _pending_image
    path = _pending_image
    _pending_image = None
    return path


# ---------------------------------------------------------------------------
# System Prompt — loaded from SOUL.md + AGENTS.md
# ---------------------------------------------------------------------------
# SOUL.md defines Beast's personality, capabilities list, and boundaries.
# AGENTS.md defines task queue rules, reasoning templates, and standing goals.
# Both are optional — if missing, a minimal default prompt is used.

def _render_tools_manifest() -> str:
    """Auto-generate a TOOLS.md-style section for the system prompt.

    OpenClaw injects a TOOLS.md file that lists every capability alongside
    SOUL.md and AGENTS.md. We do the same dynamically: the manifest always
    reflects the *current* TOOLS list, so newly-added tools become visible to
    the LLM immediately without editing a markdown file by hand.
    """
    lines = [
        "## Tools Manifest",
        "",
        "These are the tools currently wired into your agent loop. Call them",
        "by name with the documented parameters. Tools are the atomic verbs;",
        "skills (see Available Skills) are recipes built on top of them.",
        "",
    ]
    for tool in TOOLS:
        lines.append(f"### `{tool['name']}`")
        lines.append(tool.get("description", ""))
        params = tool.get("params") or {}
        if params:
            lines.append("")
            lines.append("Parameters:")
            for key, desc in params.items():
                lines.append(f"- `{key}` — {desc}")
        lines.append("")
    return "\n".join(lines)


def load_system_prompt() -> str:
    """Compose the system prompt from SOUL.md + AGENTS.md + skills + tools.

    Layered prompt composition is one of the most powerful OpenClaw patterns:
    each layer answers a different question the LLM needs to know at every
    turn.

    - **SOUL.md** — who am I and how do I behave?
    - **AGENTS.md** — what standing goals, reasoning templates, and memory
      rules should I follow?
    - **Skills index** — what high-level recipes can I pull in on demand via
      `use_skill`?
    - **Tools manifest** — what low-level verbs do I have right now?
    - **Base prompt** — a last-resort fallback if nothing else is configured.
    """
    soul_file = WORKSPACE / "SOUL.md"
    agents_file = WORKSPACE / "AGENTS.md"
    base_prompt = """You are Obedient Beast, a helpful AI assistant that can execute commands and manage files.
You have access to tools to help the user. Use them when needed.
Be concise and helpful. When executing commands, explain what you're doing."""

    prompt = ""
    if soul_file.exists():
        prompt += soul_file.read_text() + "\n\n"
    # AGENTS.md: standing goals, reasoning templates, memory guidelines (Phase 4)
    if agents_file.exists():
        prompt += agents_file.read_text() + "\n\n"
    # Skills index: markdown runbooks loaded on demand via `use_skill`.
    try:
        from skills_loader import get_skills_index
        skills_block = get_skills_index()
        if skills_block:
            prompt += skills_block + "\n"
    except Exception as exc:  # pragma: no cover — never block startup on skills
        print(f"[Skills] Could not load skills index: {exc}", file=sys.stderr)
    # Tools manifest: auto-generated from the TOOLS list below.
    prompt += _render_tools_manifest() + "\n"
    prompt += base_prompt
    return prompt


# SYSTEM_PROMPT is deferred: TOOLS is defined further down, and the manifest
# renderer reads it at call time. We assign after TOOLS is defined.
SYSTEM_PROMPT = ""


# ---------------------------------------------------------------------------
# Memory System — Two-Tier Architecture
# ---------------------------------------------------------------------------
# Beast has two memory stores that work together:
#
# 1. LOCAL MEMORY (workspace/memory.json) — PRIMARY, PERSISTENT
#    Simple flat list of facts with timestamps. Survives restarts.
#    Capped at 200 entries (FIFO). Keyword search only.
#    This is the reliable backbone — always works, no dependencies.
#
# 2. MCP KNOWLEDGE GRAPH (optional) — RICHER, EPHEMERAL
#    Provided by @modelcontextprotocol/server-memory (an npx subprocess).
#    Stores entities as nodes with typed relationships/edges between them.
#    Example: entity "Jonathan" --[prefers]--> "dark mode"
#             entity "Jonathan" --[works_on]--> "Obedient Beast"
#    Supports semantic search and relationship traversal, so the LLM can
#    ask "what do I know about Jonathan?" and get connected facts back.
#    WHY IT'S USEFUL: flat memory can only keyword-match individual facts.
#    The knowledge graph connects facts together, enabling richer recall
#    (e.g., "what projects does the user work on?" traverses edges).
#    CAVEAT: ephemeral — the npx subprocess holds state in memory only.
#    When Beast restarts, the graph is empty. Key facts are independently
#    saved to local memory (memory.json) so nothing critical is lost.
#
# Both stores are written to on every conversation turn (_try_memory_save).
# /clear memory and /clear all wipe both stores.

def _load_local_memory() -> dict:
    """Load local memory from workspace/memory.json. Returns {"facts": [...]}."""
    if not MEMORY_FILE.exists():
        return {"facts": []}
    try:
        return json.loads(MEMORY_FILE.read_text())
    except (json.JSONDecodeError, IOError):
        return {"facts": []}


def _memory_category(text: str) -> str:
    """
    Heuristic category inference — used by smarter memory auto-save.
    Categories: preference | project | people | decision | conversation.
    """
    t = text.lower()
    if any(w in t for w in ("prefer", "like", "love", "hate", "favorite", "favourite")):
        return "preference"
    if any(w in t for w in ("decided", "chose", "will use", "going with", "pick")):
        return "decision"
    if any(w in t for w in ("project", "repo", "codebase", "build", "deploy")):
        return "project"
    if any(w in t for w in ("named", "is called", "works at", "@")):
        return "people"
    return "conversation"


def _fact_fingerprint(text: str) -> str:
    """Lowercase-normalized fingerprint for dedup. Strips punctuation + extra whitespace."""
    import re
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def _save_local_memory_fact(fact: str, category: str | None = None):
    """
    Save a single fact to local memory with dedup + categorization.
    Caps at 200 facts (FIFO). Skips if a near-duplicate exists in the
    last 50 facts (by normalized fingerprint).
    """
    data = _load_local_memory()
    facts = data.get("facts", [])

    # Dedup: skip if same fingerprint appeared in the last 50 facts
    fp = _fact_fingerprint(fact)
    if fp:
        recent_fps = {
            _fact_fingerprint(f.get("text", ""))
            for f in facts[-50:]
        }
        if fp in recent_fps:
            return  # Duplicate — don't grow memory

    facts.append({
        "text": fact,
        "timestamp": datetime.now().isoformat(),
        "category": category or _memory_category(fact),
    })
    # Cap at 200 facts — remove oldest when full (FIFO)
    if len(facts) > 200:
        facts = facts[-200:]
    data["facts"] = facts
    MEMORY_FILE.write_text(json.dumps(data, indent=2))


_MEM_TOKEN_RE = __import__("re").compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase word-tokenize for BM25. Drops stopwords of length <= 2."""
    return [t for t in _MEM_TOKEN_RE.findall(text.lower()) if len(t) > 2]


def _search_local_memory(query: str) -> str:
    """
    Hybrid BM25 + temporal decay memory search.

    Scoring (per fact):
        score = bm25(query, fact_tokens) * exp(-age_days / 30)

    BM25-lite: standard term-frequency / inverse-doc-frequency across the full
    memory set, with k1=1.5 and b=0.75 (standard defaults). Temporal decay
    multiplies the lexical score by a half-month half-life so recent facts
    outrank ancient ones on ties. No external dependencies — pure stdlib.
    """
    import math
    from collections import Counter

    data = _load_local_memory()
    facts = data.get("facts", [])
    if not facts:
        return f"No local memories found for '{query}'."

    q_tokens = _tokenize(query)
    if not q_tokens:
        # No searchable query — fall back to most recent 10
        recent = "\n".join(f.get("text", "") for f in facts[-10:])
        return recent or f"No local memories found for '{query}'."

    # Build BM25 stats across all facts
    docs = [_tokenize(f.get("text", "")) for f in facts]
    N = len(docs)
    avgdl = sum(len(d) for d in docs) / max(N, 1)
    # df: number of docs each query term appears in
    df = Counter()
    for d in docs:
        for t in set(q_tokens) & set(d):
            df[t] += 1

    k1, b = 1.5, 0.75
    now = datetime.now()
    scored = []
    for fact, tokens in zip(facts, docs):
        if not tokens:
            continue
        tf = Counter(tokens)
        score = 0.0
        for qt in q_tokens:
            n_qt = df.get(qt, 0)
            if n_qt == 0:
                continue
            idf = math.log((N - n_qt + 0.5) / (n_qt + 0.5) + 1)
            f_qt = tf.get(qt, 0)
            denom = f_qt + k1 * (1 - b + b * len(tokens) / avgdl)
            if denom > 0:
                score += idf * (f_qt * (k1 + 1)) / denom
        if score <= 0:
            continue
        # Temporal decay: e^(-age_days / 30)
        try:
            ts = datetime.fromisoformat(fact.get("timestamp", ""))
            age_days = max((now - ts).total_seconds() / 86400, 0)
            decay = math.exp(-age_days / 30)
        except Exception:
            decay = 0.5  # Unknown age → medium weight
        scored.append((score * decay, fact.get("text", "")))

    if not scored:
        return f"No local memories found for '{query}'."
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [text for _, text in scored[:10]]
    return "\n".join(top)


def _extract_atomic_facts(user_input: str, response_text: str) -> list[str]:
    """
    Extract declarative atomic facts from a conversation turn.
    Pulls sentences from the response that look like durable claims
    ("the user prefers X", "the project uses Y", "I installed Z").
    Returns a short list — empty if nothing fact-like was found.
    """
    import re
    facts = []
    # Grab sentences from response that contain fact-indicator verbs
    sentences = re.split(r"(?<=[.!?])\s+", response_text or "")
    indicators = (
        "prefer", "use", "uses", "installed", "configured", "lives in",
        "is called", "located", "decided", "chose", "named", "runs on",
        "set to", "will use", "has a", "wants",
    )
    for s in sentences:
        s = s.strip()
        if 10 < len(s) < 200 and any(ind in s.lower() for ind in indicators):
            facts.append(s)
        if len(facts) >= 3:
            break
    return facts


def _try_memory_save(session_id: str, user_input: str, response_text: str):
    """
    Auto-save key facts to memory at end of a conversation turn.
    Saves to BOTH local memory (persistent) AND MCP knowledge graph (if running).
    Local memory is the durable store; MCP graph adds richer entity/relationship
    recall during the current session but resets on restart.

    Strategy: extract atomic facts from the response when possible; otherwise
    fall back to a compact conversation snippet. Both paths go through
    _save_local_memory_fact which dedupes + categorizes.
    """
    # Always save to local memory fallback (works offline, no MCP needed)
    from capabilities import MEMORY_DETAIL

    # Prefer atomic facts pulled from the response text
    atomic = _extract_atomic_facts(user_input, response_text)
    if atomic:
        for f in atomic:
            _save_local_memory_fact(f)
        fact = atomic[0]  # Use first atomic fact for the MCP mirror below
    else:
        # Fallback: compact conversation snippet (dedup handles repeats)
        if MEMORY_DETAIL == "minimal":
            fact = f"user asked about '{user_input[:80]}'"
        else:
            fact = f"user asked '{user_input[:120]}'; beast replied '{response_text[:180]}'"
        _save_local_memory_fact(fact)

    # Also save to MCP memory if available (richer knowledge graph)
    if not MCP_ENABLED or _mcp_client is None:
        return
    try:
        execute_mcp_tool("mcp_memory_create_entities", {
            "entities": json.dumps([{
                "name": f"conversation_{session_id}_{datetime.now().strftime('%H%M%S')}",
                "entityType": "conversation",
                "observations": [fact]
            }])
        })
    except Exception as e:
        # Memory save is best-effort, never block the response
        print(f"[Memory] Auto-save failed (non-fatal): {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Tools Definition — 18 built-in tools
# ---------------------------------------------------------------------------
# Each tool is a dict with name, description, and params.
# The LLM sees these descriptions and decides which tools to call.
# Tools are grouped by category:
#   - File & System (5): shell, read_file, write_file, list_dir, edit_file
#   - Computer Control (7): screenshot, mouse_click/move, keyboard_type/hotkey, screen/mouse info
#   - Self-Upgrade (3): install/list/enable MCP servers
#   - Autonomous Agent (2): add_task, recall_memory
#   - Network (1): fetch_url

TOOLS = [
    # === File & System Tools ===
    # These are the bread-and-butter tools Beast uses most often.
    # shell is the most powerful — it can do anything the terminal can.
    {
        "name": "shell",
        "description": "Execute a shell command and return the output. Use for running programs, checking system status, etc.",
        "params": {
            "command": "The shell command to execute",
            "timeout": "Optional: max seconds to wait (default 60, max 300)"
        }
    },
    # read_file: Safe, read-only access to any file on the system.
    {
        "name": "read_file",
        "description": "Read and return the contents of a file.",
        "params": {"path": "Path to the file to read"}
    },
    # write_file: Creates parent directories automatically.
    {
        "name": "write_file",
        "description": "Write content to a file. Creates the file if it doesn't exist.",
        "params": {"path": "Path to the file", "content": "Content to write"}
    },
    # list_dir: Quick directory listing with file/folder icons.
    {
        "name": "list_dir",
        "description": "List files and directories in a path.",
        "params": {"path": "Directory path to list"}
    },
    # edit_file: Find-and-replace within a file. Only replaces first occurrence.
    {
        "name": "edit_file",
        "description": "Replace text in a file. Finds 'old_text' and replaces with 'new_text'.",
        "params": {"path": "Path to the file", "old_text": "Text to find", "new_text": "Text to replace with"}
    },
    # ---------------------------------------------------------------------------
    # Computer Control Tools (Phase 1)
    # ---------------------------------------------------------------------------
    # These tools give Beast the ability to see and interact with the desktop.
    # screenshot is especially useful via WhatsApp — Beast can capture the screen
    # and send the image back in the chat.
    # All coordinate-based tools (mouse_click, mouse_move) use absolute pixel coords.
    # Lazy imports: pyautogui and mss are imported only when these tools are called,
    # so Beast starts fast even if those packages aren't installed.
    {
        "name": "screenshot",
        "description": "Take a screenshot of the screen. Returns the file path to the saved image.",
        "params": {"filename": "Optional filename (default: screenshot_timestamp.png)"}
    },
    {
        "name": "mouse_click",
        "description": "Click the mouse at specific screen coordinates.",
        "params": {"x": "X coordinate", "y": "Y coordinate", "button": "Mouse button: left, right, or middle (default: left)"}
    },
    {
        "name": "mouse_move",
        "description": "Move the mouse to specific screen coordinates.",
        "params": {"x": "X coordinate", "y": "Y coordinate"}
    },
    {
        "name": "keyboard_type",
        "description": "Type text using the keyboard.",
        "params": {"text": "Text to type"}
    },
    {
        "name": "keyboard_hotkey",
        "description": "Press a keyboard shortcut (e.g., 'command+c' for copy on Mac).",
        "params": {"keys": "Keys to press, separated by + (e.g., 'command+shift+s')"}
    },
    {
        "name": "get_screen_size",
        "description": "Get the screen dimensions.",
        "params": {}
    },
    {
        "name": "get_mouse_position",
        "description": "Get the current mouse cursor position.",
        "params": {}
    },
    # ---------------------------------------------------------------------------
    # Self-Upgrade Tools (Phase 3) - Beast can add its own skills!
    # ---------------------------------------------------------------------------
    # These tools let Beast modify its own MCP server configuration.
    # When Beast discovers it needs a capability it doesn't have, it can
    # install a new MCP server, making itself more capable over time.
    # Changes take effect on next restart (MCP servers are spawned at startup).
    {
        "name": "install_mcp_server",
        "description": "Install a new MCP server to add new capabilities. Beast can use this to give itself new skills! Common servers: filesystem, git, memory, playwright, puppeteer, sqlite, slack, brave-search.",
        "params": {
            "name": "Short name for the server (e.g., 'playwright')",
            "command": "NPX command to run the server (e.g., 'npx -y @anthropic/mcp-server-playwright')",
            "description": "What this server does"
        }
    },
    {
        "name": "list_mcp_servers",
        "description": "List all configured MCP servers and their status.",
        "params": {}
    },
    {
        "name": "enable_mcp_server",
        "description": "Enable or disable an MCP server by name.",
        "params": {"name": "Server name", "enabled": "true or false"}
    },
    # ---------------------------------------------------------------------------
    # Autonomous Agent Tools (Phase 4 - Clawdbot-inspired)
    # ---------------------------------------------------------------------------
    # add_task: Queue work for later processing by the heartbeat scheduler.
    #   Users can say "remind me to..." and Beast queues a task.
    #   The heartbeat (heartbeat.py) picks up pending tasks on a timer.
    # recall_memory: Search persistent memory for past context.
    #   Uses MCP memory server (knowledge graph) when available,
    #   falls back to local JSON memory (workspace/memory.json) when not.
    {
        "name": "add_task",
        "description": "Add, update, or complete a task in the autonomous task queue. Beast and users can queue work for later. Supports scheduling: use scheduled_at for one-shot timed tasks, repeat_seconds for simple interval recurrence, or cron for cron-style recurrence.",
        "params": {
            "description": "What the task is (required for new tasks)",
            "priority": "low, medium, or high (default: medium)",
            "status": "pending, done, or failed (default: pending). Use 'done'/'failed' to close a task.",
            "task_id": "Optional: ID of existing task to update its status",
            "scheduled_at": "Optional: ISO timestamp for when to run (e.g., '2026-02-28T15:00:00'). Task won't be processed until this time.",
            "repeat_seconds": "Optional: interval in seconds for recurring tasks (e.g., 3600 for hourly, 86400 for daily). Task auto-resets after each run.",
            "cron": "Optional: 5-field cron expression (min hour dom month dow), e.g. '0 9 * * 1-5' for 9am weekdays. Overrides repeat_seconds when both are set."
        }
    },
    {
        "name": "recall_memory",
        "description": "Recall facts from persistent memory. Use to remember user preferences, past decisions, or project context.",
        "params": {
            "query": "What to search for in memory (e.g., 'user preferences', 'project config')"
        }
    },
    # ---------------------------------------------------------------------------
    # Network Tools — HTTP fetching (stdlib, no new dependencies)
    # ---------------------------------------------------------------------------
    # fetch_url: Simple HTTP GET/POST using urllib (Python stdlib).
    # Works offline for LAN APIs and local services.
    # Truncates response to 4000 chars to avoid blowing up the LLM context.
    # For full browser automation, recommend the Playwright MCP server.
    {
        "name": "fetch_url",
        "description": "Fetch a URL and return the response body. Works with any HTTP/HTTPS URL. Useful for checking APIs, downloading data, or reading web pages. Response truncated to 4000 chars.",
        "params": {
            "url": "The URL to fetch (http:// or https://)",
            "method": "HTTP method: GET or POST (default: GET)"
        }
    },
    # ---------------------------------------------------------------------------
    # Sub-Agent Spawning — run an isolated Beast sub-session for a subtask
    # ---------------------------------------------------------------------------
    # spawn_agent creates a fresh session and runs one subtask in it, then
    # returns just the final answer to the parent. Great for "research these
    # 3 things and summarize" — each subtask gets a clean context window,
    # so the parent doesn't drown in tool chatter. Sequential, not threaded.
    {
        "name": "spawn_agent",
        "description": "Spawn a sub-agent in a fresh session to handle an isolated subtask. Use for parallel-feel research ('look into X while I look into Y') or to keep the main context clean. Returns only the sub-agent's final answer, not its tool calls. Sub-agent runs with reduced depth (default 3).",
        "params": {
            "task": "The subtask for the sub-agent to perform (a complete instruction)",
            "depth": "Optional: max tool-chain steps for the sub-agent (default: 3, max: 10)"
        }
    },
    # ---------------------------------------------------------------------------
    # Skills (OpenClaw-inspired): markdown runbooks loaded on demand
    # ---------------------------------------------------------------------------
    # Skills live in workspace/skills/<name>/SKILL.md. `list_skills` shows the
    # catalog; `use_skill` loads a specific skill's full instructions into the
    # conversation so Beast can follow them step by step. This is the core of
    # Beast's ClawHub-style extensibility: adding capabilities = writing a
    # markdown file, no code changes required.
    {
        "name": "list_skills",
        "description": "List all available skills (markdown runbooks in workspace/skills/). Use this to discover what recipes you already have before writing new code.",
        "params": {}
    },
    {
        "name": "use_skill",
        "description": "Load a skill's full instructions by name. Returns the SKILL.md body — follow its steps exactly. Use when a user request matches a skill's description or triggers.",
        "params": {"name": "Skill name (see list_skills for valid names)"}
    },
    # ---------------------------------------------------------------------------
    # Persistent Browser (OpenClaw-inspired): Playwright with a profile on disk
    # ---------------------------------------------------------------------------
    # Unlike fetch_url, these tools drive a real headful Chromium context whose
    # cookies/localStorage live in workspace/browser_profile/. Log into GitHub /
    # Gmail / Slack once in the visible window and every subsequent run reuses
    # the session. Lazy-imported so Beast starts fine without Playwright.
    {
        "name": "browser_goto",
        "description": "Navigate the persistent browser to a URL. Cookies and logins are reused across runs (see workspace/browser_profile/). Returns the final URL and page title.",
        "params": {"url": "Absolute URL to open (http:// or https://)"}
    },
    {
        "name": "browser_read",
        "description": "Return the visible text of the current browser page (truncated to 4000 chars). Call after browser_goto.",
        "params": {"max_chars": "Optional: max chars to return (default 4000)"}
    },
    {
        "name": "browser_click",
        "description": "Click the first element matching a CSS selector in the browser (e.g. 'button[type=submit]', 'text=Sign in').",
        "params": {"selector": "CSS selector or Playwright text locator"}
    },
    {
        "name": "browser_type",
        "description": "Type text into an input element matching a CSS selector. Optionally submit by pressing Enter.",
        "params": {
            "selector": "CSS selector for the input",
            "text": "Text to type",
            "submit": "Optional: 'true' to press Enter after typing"
        }
    },
    {
        "name": "browser_screenshot",
        "description": "Screenshot the current browser page. Returns the saved file path (also queued for WhatsApp delivery).",
        "params": {"filename": "Optional filename (default: browser_<timestamp>.png)"}
    },
    {
        "name": "browser_close",
        "description": "Close the persistent browser context. Cookies remain on disk for the next session.",
        "params": {}
    },
    # ---------------------------------------------------------------------------
    # Sandbox — run generated code in an isolated workspace directory
    # ---------------------------------------------------------------------------
    # run_python: writes a script to workspace/Generated Code/py_<ts>/, runs it with
    # Beast's own Python, captures stdout/stderr, and lists any files created.
    # run_html: writes an HTML file to workspace/Generated Code/html_<ts>/ and opens
    # it in the default browser via stdlib webbrowser.open().
    {
        "name": "run_python",
        "description": "PREFERRED over shell for running Python code. Run a Python script in an isolated sandbox directory. Returns stdout, stderr, and a list of any files the script created (plots, CSVs, etc.). Uses Beast's own Python/venv so installed packages (numpy, matplotlib, etc.) are available. Always use this instead of shell+echo or write_file+shell for Python scripts.",
        "params": {
            "code": "The Python source code to run",
            "timeout": "Optional: max seconds to wait (default 30, max 120)",
            "filename": "Optional: script filename (default: script.py)"
        }
    },
    {
        "name": "run_html",
        "description": "PREFERRED over write_file for HTML pages. Create an HTML/CSS/JavaScript page, open it in the browser, AND auto-screenshot it for WhatsApp delivery. Perfect for visualizations, interactive demos, games, dashboards, charts. Always use this instead of write_file for HTML content.",
        "params": {
            "html": "The full HTML source (including <html>, <head>, <body>)",
            "filename": "Optional: filename (default: index.html)",
            "open_browser": "Optional: 'false' to skip opening the browser (default: true)"
        }
    },
]

# Now that TOOLS is defined, compose the full system prompt (SOUL + AGENTS +
# skills index + auto-generated tools manifest + fallback base prompt).
SYSTEM_PROMPT = load_system_prompt()


def execute_tool(name: str, args: dict) -> str:
    """
    Execute a tool and return the result as a string.

    Security model:
    - File tools (read/write/edit/list_dir) operate on any path the user specifies.
      Beast trusts the user — it's a personal assistant on your own machine.
    - Shell tool runs commands with the same permissions as the Beast process.
    - Computer control tools (pyautogui, mss) are lazy-imported to avoid startup
      crashes if those packages aren't installed (they're optional).
    - MCP tools are delegated to mcp_client.py which routes to the correct server.

    Error handling: All tools are wrapped in a try/except that returns error strings
    rather than raising — this lets the LLM see the error and decide what to do.
    """
    try:
        # === File & System Tools ===
        if name == "shell":
            # Shell tool: the most powerful tool. Runs any command.
            # Uses subprocess with capture_output for safety (no TTY).
            # Timeout is configurable (default 60s, max 300s) to handle long-running commands.
            timeout = min(int(args.get("timeout", 60)), 300)  # Cap at 5 minutes
            result = subprocess.run(
                args["command"],
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]: {result.stderr}"
            if result.returncode != 0:
                output += f"\n[exit code]: {result.returncode}"
            return output or "(no output)"

        elif name == "read_file":
            path = Path(args["path"]).expanduser()
            if not path.exists():
                return f"Error: File not found: {path}"
            return path.read_text()

        elif name == "write_file":
            path = Path(args["path"]).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)  # Auto-create parent dirs
            path.write_text(args["content"])
            return f"Successfully wrote {len(args['content'])} characters to {path}"

        elif name == "list_dir":
            path = Path(args["path"]).expanduser()
            if not path.exists():
                return f"Error: Directory not found: {path}"
            items = []
            for item in sorted(path.iterdir()):
                prefix = "📁 " if item.is_dir() else "📄 "
                items.append(f"{prefix}{item.name}")
            return "\n".join(items) if items else "(empty directory)"

        elif name == "edit_file":
            # Find-and-replace: only replaces the FIRST occurrence.
            # This is intentional — prevents accidental mass edits.
            path = Path(args["path"]).expanduser()
            if not path.exists():
                return f"Error: File not found: {path}"
            content = path.read_text()
            if args["old_text"] not in content:
                return f"Error: Text not found in file"
            new_content = content.replace(args["old_text"], args["new_text"], 1)
            path.write_text(new_content)
            return f"Successfully replaced text in {path}"

        # === Computer Control Tools ===
        # All use lazy imports — pyautogui and mss are only imported when needed.
        # This means Beast starts fine even if these packages aren't installed.
        elif name == "screenshot":
            import mss  # Lazy import: only needed when taking screenshots
            from datetime import datetime as dt
            filename = args.get("filename") or f"screenshot_{dt.now().strftime('%Y%m%d_%H%M%S')}.png"
            screenshot_dir = WORKSPACE / "screenshots"
            screenshot_dir.mkdir(exist_ok=True)
            filepath = screenshot_dir / filename
            with mss.mss() as sct:
                sct.shot(output=str(filepath))
            # Set as pending image so server.py includes it in WhatsApp reply
            set_pending_image(str(filepath))
            return f"Screenshot saved to {filepath}. Will send via WhatsApp."

        elif name == "mouse_click":
            import pyautogui  # Lazy import
            x = int(args["x"])
            y = int(args["y"])
            button = args.get("button", "left")
            pyautogui.click(x, y, button=button)
            return f"Clicked {button} button at ({x}, {y})"

        elif name == "mouse_move":
            import pyautogui  # Lazy import
            x = int(args["x"])
            y = int(args["y"])
            pyautogui.moveTo(x, y)
            return f"Moved mouse to ({x}, {y})"

        elif name == "keyboard_type":
            import pyautogui  # Lazy import
            text = args["text"]
            pyautogui.write(text)
            return f"Typed {len(text)} characters"

        elif name == "keyboard_hotkey":
            import pyautogui  # Lazy import
            keys = args["keys"].split("+")
            pyautogui.hotkey(*keys)
            return f"Pressed hotkey: {args['keys']}"

        elif name == "get_screen_size":
            import pyautogui  # Lazy import
            width, height = pyautogui.size()
            return f"Screen size: {width}x{height} pixels"

        elif name == "get_mouse_position":
            import pyautogui  # Lazy import
            x, y = pyautogui.position()
            return f"Mouse position: ({x}, {y})"

        # === Self-Upgrade Tools (MCP server management) ===
        # These modify config/mcp_servers.json. Changes take effect on next restart.
        elif name == "install_mcp_server":
            config_file = Path(__file__).parent / "config" / "mcp_servers.json"
            config_file.parent.mkdir(exist_ok=True)

            if config_file.exists():
                config = json.loads(config_file.read_text())
            else:
                config = {"servers": {}}

            server_name = args["name"]
            config["servers"][server_name] = {
                "enabled": True,
                "command": args["command"],
                "description": args["description"],
                "local": True,
                "tier": "extended"  # New servers default to extended tier
            }

            config_file.write_text(json.dumps(config, indent=2))
            return f"Installed MCP server '{server_name}'. Restart Beast to activate. Command: {args['command']}"

        elif name == "list_mcp_servers":
            config_file = Path(__file__).parent / "config" / "mcp_servers.json"
            if not config_file.exists():
                return "No MCP servers configured. Use install_mcp_server to add one."

            config = json.loads(config_file.read_text())
            servers = config.get("servers", {})

            if not servers:
                return "No MCP servers configured."

            result = "MCP Servers:\n"
            for name, info in servers.items():
                status = "enabled" if info.get("enabled", True) else "disabled"
                tier = info.get("tier", "essential")
                desc = info.get("description", "No description")
                result += f"  [{status}] [{tier}] {name}: {desc}\n"

            return result

        elif name == "enable_mcp_server":
            config_file = Path(__file__).parent / "config" / "mcp_servers.json"
            if not config_file.exists():
                return "No MCP servers configured."

            config = json.loads(config_file.read_text())
            server_name = args["name"]

            if server_name not in config.get("servers", {}):
                return f"Server '{server_name}' not found."

            enabled = args.get("enabled", "true").lower() == "true"
            config["servers"][server_name]["enabled"] = enabled
            config_file.write_text(json.dumps(config, indent=2))

            status = "enabled" if enabled else "disabled"
            return f"Server '{server_name}' is now {status}. Restart Beast to apply."

        # === Autonomous Agent Tools ===
        elif name == "add_task":
            # Task queue: stored in workspace/tasks.json as a flat list.
            # Each task has: id, description, priority, status, timestamps.
            # The heartbeat scheduler (heartbeat.py) processes pending tasks.
            tasks_file = WORKSPACE / "tasks.json"
            if tasks_file.exists():
                data = json.loads(tasks_file.read_text())
            else:
                data = {"tasks": []}

            task_id = args.get("task_id")
            if task_id:
                # Update existing task status (e.g., mark as done/failed)
                for t in data["tasks"]:
                    if str(t.get("id")) == str(task_id):
                        new_status = args.get("status", t.get("status", "pending"))
                        t["status"] = new_status
                        t["updated_at"] = datetime.now().isoformat()
                        tasks_file.write_text(json.dumps(data, indent=2))
                        return f"Task #{task_id} updated to '{new_status}'."
                return f"Error: Task #{task_id} not found."
            else:
                # Add new task
                max_id = max((t.get("id", 0) for t in data["tasks"]), default=0)
                new_task = {
                    "id": max_id + 1,
                    "description": args.get("description", "No description"),
                    "priority": args.get("priority", "medium"),
                    "status": args.get("status", "pending"),
                    "created_at": datetime.now().isoformat()
                }
                # Scheduling: one-shot, interval-recurring, or cron-recurring.
                scheduled_at = args.get("scheduled_at")
                repeat_seconds = args.get("repeat_seconds")
                cron_expr = args.get("cron")
                timing_info = ""
                if scheduled_at:
                    new_task["scheduled_at"] = scheduled_at
                    timing_info = f" (scheduled for {scheduled_at})"
                if repeat_seconds:
                    try:
                        interval = int(repeat_seconds)
                        new_task["repeat_seconds"] = interval
                        new_task["next_run_at"] = datetime.now().isoformat()
                        timing_info = f" (recurring every {interval}s)"
                    except ValueError:
                        pass
                if cron_expr:
                    # Validate by computing the first next_run_at. Errors are
                    # surfaced as task-add failures so the LLM can retry.
                    try:
                        from cron_schedule import next_run
                        nxt = next_run(cron_expr)
                        new_task["cron"] = cron_expr
                        new_task["next_run_at"] = nxt.isoformat()
                        timing_info = f" (cron '{cron_expr}', next run {nxt.isoformat()})"
                    except Exception as exc:
                        return f"Error: invalid cron expression {cron_expr!r}: {exc}"
                data["tasks"].append(new_task)
                tasks_file.write_text(json.dumps(data, indent=2))
                return f"Task #{new_task['id']} added: {new_task['description']} [{new_task['priority']}]{timing_info}"

        elif name == "recall_memory":
            # Memory recall: tries MCP memory server first (rich knowledge graph),
            # then falls back to local JSON memory (workspace/memory.json).
            query = args.get("query", "")

            if MCP_ENABLED and _mcp_client is not None:
                try:
                    # Call the MCP memory server to search for entities
                    result = execute_mcp_tool("mcp_memory_search_nodes", {"query": query})
                    if result and not result.startswith("Error"):
                        return f"Memory recall for '{query}':\n{result}"
                    # Fallback: try to read all entities from the knowledge graph
                    result = execute_mcp_tool("mcp_memory_read_graph", {})
                    if result and not result.startswith("Error"):
                        return f"Full memory graph:\n{result}"
                except Exception as e:
                    print(f"[Memory] MCP recall error (falling back to local): {e}", file=sys.stderr)

            # Fallback: search local JSON memory
            local_result = _search_local_memory(query)
            if local_result and "No local memories found" not in local_result:
                return f"Memory recall for '{query}' (local):\n{local_result}"
            return f"No memories found for '{query}'. Try saving some facts first."

        # === Network Tools ===
        elif name == "fetch_url":
            # fetch_url: Simple HTTP fetching using Python stdlib (urllib).
            # No new dependencies needed. Works for LAN APIs, local services, and web.
            # Response body is truncated to 4000 chars to avoid blowing up LLM context.
            # For full browser automation, recommend the Playwright MCP server.
            import urllib.request
            import urllib.error
            url = args.get("url", "")
            method = args.get("method", "GET").upper()

            if not url.startswith(("http://", "https://")):
                return "Error: URL must start with http:// or https://"

            try:
                import ssl
                try:
                    import certifi
                    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
                except ImportError:
                    ssl_ctx = ssl.create_default_context()
                req = urllib.request.Request(url, method=method)
                req.add_header("User-Agent", "ObedientBeast/1.0")
                with urllib.request.urlopen(req, timeout=30, context=ssl_ctx) as resp:
                    body = resp.read().decode("utf-8", errors="replace")
                    status = resp.status
                    # Truncate to 4000 chars to keep LLM context manageable
                    if len(body) > 4000:
                        body = body[:4000] + f"\n... [truncated, {len(body)} total chars]"
                    return f"HTTP {status}\n{body}"
            except urllib.error.HTTPError as e:
                return f"HTTP Error {e.code}: {e.reason}"
            except urllib.error.URLError as e:
                return f"URL Error: {e.reason}"
            except Exception as e:
                return f"Fetch error: {e}"

        # === Sub-Agent Spawning ===
        # Runs a full agent loop in a throwaway session, returns final text.
        # Uses a capped depth so sub-agents can't burn the whole budget.
        elif name == "spawn_agent":
            import uuid
            task = args.get("task", "").strip()
            if not task:
                return "Error: spawn_agent needs a 'task' parameter"
            try:
                sub_depth = min(int(args.get("depth", 3)), 10)
            except (TypeError, ValueError):
                sub_depth = 3
            sub_session = f"sub_{uuid.uuid4().hex[:8]}"
            # Temporarily override DEPTH for the sub-run
            import capabilities
            prev_depth = capabilities.DEPTH
            try:
                capabilities.DEPTH = sub_depth
                sub_result = run(task, session_id=sub_session)
            finally:
                capabilities.DEPTH = prev_depth
            return f"[sub-agent {sub_session} result]\n{sub_result}"

        # === Skills (markdown runbooks) ===
        elif name == "list_skills":
            from skills_loader import discover_skills
            skills = discover_skills()
            if not skills:
                return (
                    "No skills installed yet. Create one at "
                    "workspace/skills/<name>/SKILL.md with frontmatter "
                    "(name, description, triggers) and markdown instructions."
                )
            lines = ["Available skills:"]
            for s in skills:
                line = f"  • {s['name']} — {s['description']}"
                if s["triggers"]:
                    line += f" (triggers: {s['triggers']})"
                lines.append(line)
            lines.append("\nCall use_skill(name=...) to load a skill's full instructions.")
            return "\n".join(lines)

        elif name == "use_skill":
            from skills_loader import get_skill
            skill_name = args.get("name", "").strip()
            if not skill_name:
                return "Error: use_skill requires a 'name' argument."
            body = get_skill(skill_name)
            if body is None:
                return (
                    f"Error: skill {skill_name!r} not found. "
                    "Call list_skills to see what's available."
                )
            return (
                f"# Skill: {skill_name}\n\n"
                "Follow these instructions exactly:\n\n" + body
            )

        # === Persistent Browser (Playwright) ===
        elif name == "browser_goto":
            from browser_tools import browser_goto, BrowserUnavailable
            try:
                return browser_goto(args["url"])
            except BrowserUnavailable as exc:
                return f"Error: {exc}"

        elif name == "browser_read":
            from browser_tools import browser_read, BrowserUnavailable
            try:
                max_chars = int(args.get("max_chars") or 4000)
                return browser_read(max_chars=max_chars)
            except BrowserUnavailable as exc:
                return f"Error: {exc}"

        elif name == "browser_click":
            from browser_tools import browser_click, BrowserUnavailable
            try:
                return browser_click(args["selector"])
            except BrowserUnavailable as exc:
                return f"Error: {exc}"

        elif name == "browser_type":
            from browser_tools import browser_type, BrowserUnavailable
            submit = str(args.get("submit", "")).lower() in ("1", "true", "yes")
            try:
                return browser_type(args["selector"], args["text"], submit=submit)
            except BrowserUnavailable as exc:
                return f"Error: {exc}"

        elif name == "browser_screenshot":
            from browser_tools import browser_screenshot, BrowserUnavailable
            try:
                path = browser_screenshot(args.get("filename"))
                # Queue for WhatsApp delivery, matching the desktop screenshot tool.
                set_pending_image(path)
                return f"Browser screenshot saved to {path}. Will send via WhatsApp."
            except BrowserUnavailable as exc:
                return f"Error: {exc}"

        elif name == "browser_close":
            from browser_tools import browser_close, BrowserUnavailable
            try:
                return browser_close()
            except BrowserUnavailable as exc:
                return f"Error: {exc}"

        # === Sandbox Tools ===
        # run_python: isolated script execution in workspace/Generated Code/py_<ts>/
        # run_html:   write + open HTML in workspace/Generated Code/html_<ts>/
        # Activity log: workspace/Generated Code/activity.log — append-only,
        # one timestamped entry per run so users can review what happened.
        elif name == "run_python":
            code = args.get("code", "").strip()
            if not code:
                return "Error: run_python requires a 'code' parameter"
            timeout = min(int(args.get("timeout", 30)), 120)
            filename = args.get("filename", "script.py")
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            sandbox_dir = WORKSPACE / "Generated Code" / f"py_{ts}"
            sandbox_dir.mkdir(parents=True, exist_ok=True)
            script_path = sandbox_dir / filename
            script_path.write_text(code)
            try:
                result = subprocess.run(
                    [sys.executable, str(script_path)],
                    capture_output=True, text=True,
                    timeout=timeout,
                    cwd=str(sandbox_dir)
                )
                output = result.stdout
                if result.stderr:
                    output += f"\n[stderr]:\n{result.stderr}"
                if result.returncode != 0:
                    output += f"\n[exit code]: {result.returncode}"
                output = output or "(no output)"
            except subprocess.TimeoutExpired:
                output = f"Error: Script timed out after {timeout}s"
            # List files the script created (excluding the script itself)
            created = [
                f.name for f in sandbox_dir.iterdir()
                if f.name != filename
            ]
            if created:
                output += f"\n\n[Created files in {sandbox_dir}]:\n"
                output += "\n".join(f"  {f}" for f in sorted(created))
                # Auto-queue the first image for WhatsApp delivery
                for f in sorted(created):
                    if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".svg")):
                        set_pending_image(str(sandbox_dir / f))
                        output += f"\n(Image {f} queued for WhatsApp)"
                        break
            _sandbox_log("run_python", str(sandbox_dir), code, output)
            return output

        elif name == "run_html":
            html = args.get("html", "").strip()
            if not html:
                return "Error: run_html requires an 'html' parameter"
            filename = args.get("filename", "index.html")
            open_browser = str(args.get("open_browser", "true")).lower() != "false"
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            sandbox_dir = WORKSPACE / "Generated Code" / f"html_{ts}"
            sandbox_dir.mkdir(parents=True, exist_ok=True)
            filepath = sandbox_dir / filename
            filepath.write_text(html)
            file_url = f"file://{filepath.resolve()}"
            if open_browser:
                import webbrowser
                webbrowser.open(file_url)
            result_lines = [
                f"HTML page saved to {filepath}",
                f"URL: {file_url}",
                "Opened in default browser." if open_browser else "Browser not opened (open_browser=false).",
            ]
            # Auto-screenshot for WhatsApp: render the page in the persistent
            # browser, take a screenshot, and queue it so WhatsApp users can
            # actually see the HTML page on their phone.
            try:
                from browser_tools import browser_goto, browser_screenshot, BrowserUnavailable
                browser_goto(file_url)
                import time; time.sleep(1)  # let JS/CSS render
                screenshot_name = f"html_{ts}.png"
                shot_path = browser_screenshot(screenshot_name)
                set_pending_image(shot_path)
                result_lines.append(f"Screenshot captured and queued for WhatsApp ({screenshot_name}).")
            except Exception as e:
                # Browser not available — no screenshot, still works
                result_lines.append(f"(No screenshot — browser not available: {e})")
            final_result = "\n".join(result_lines)
            _sandbox_log("run_html", str(sandbox_dir), html, final_result)
            return final_result

        # === MCP Tools ===
        # Any tool name starting with "mcp_" is routed to the MCP client.
        # The naming convention is: mcp_<servername>_<toolname>
        # e.g., mcp_filesystem_read_file, mcp_memory_search_nodes
        elif name.startswith("mcp_"):
            return execute_mcp_tool(name, args)

        else:
            return f"Error: Unknown tool: {name}"

    except Exception as e:
        return f"Error executing {name}: {str(e)}"


# ---------------------------------------------------------------------------
# Session Management
# ---------------------------------------------------------------------------
# Conversations are stored as JSONL files in sessions/ directory.
# Each line is a JSON message (role: user/assistant/tool).
# Session IDs: "cli_YYYYMMDD_HHMMSS" for CLI, "wa_<phone>" for WhatsApp.
# This means WhatsApp users get persistent conversation continuity.

def get_session_path(session_id: str) -> Path:
    """Get path to session file."""
    return SESSIONS_DIR / f"{session_id}.jsonl"


def load_session(session_id: str) -> list:
    """Load conversation history from session file (JSONL format)."""
    path = get_session_path(session_id)
    if not path.exists():
        return []

    messages = []
    for line in path.read_text().strip().split("\n"):
        if line:
            messages.append(json.loads(line))
    return messages


def save_message(session_id: str, message: dict):
    """Append a message to the session file (append-only JSONL)."""
    path = get_session_path(session_id)
    with open(path, "a") as f:
        f.write(json.dumps(message) + "\n")


# ---------------------------------------------------------------------------
# Fallback helpers
# ---------------------------------------------------------------------------

# LLM_FALLBACK: comma-separated backend names to try if primary fails.
# Example: LLM_FALLBACK=claude,openai — if primary (lfm) fails, try Claude then OpenAI.
LLM_FALLBACK = [b.strip() for b in os.getenv("LLM_FALLBACK", "").split(",") if b.strip()]


def _text_only_history(history: list) -> list:
    """
    Strip tool-call and tool-result messages from history, keeping only
    plain user/assistant text messages. This avoids format incompatibility
    when falling back between backends (Claude tool format != OpenAI format).
    """
    clean = []
    for msg in history:
        role = msg.get("role")
        content = msg.get("content")
        # Skip tool-result messages
        if role == "tool":
            continue
        # Skip assistant messages with tool_calls (OpenAI format)
        if role == "assistant" and msg.get("tool_calls"):
            continue
        # Skip messages with structured content blocks (Claude format)
        if isinstance(content, list):
            # Extract text from Claude content blocks
            texts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
            if texts:
                clean.append({"role": role, "content": " ".join(texts)})
            continue
        # Keep plain text messages
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            clean.append({"role": role, "content": content})
    return clean


def _strip_images_from_history(history: list) -> list:
    """
    Replace multimodal content arrays with text-only equivalents.
    Used when an image was sent but the model doesn't support vision —
    we keep the text part and add a note about the dropped image.
    """
    clean = []
    for msg in history:
        content = msg.get("content")
        if isinstance(content, list):
            # Extract just the text blocks, drop image blocks
            texts = []
            had_image = False
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    texts.append(block["text"])
                elif block.get("type") in ("image", "image_url"):
                    had_image = True
            text = " ".join(texts)
            if had_image:
                text = "[An image was sent but this model cannot process images.]\n" + text
            clean.append({**msg, "content": text})
        else:
            clean.append(msg)
    return clean


def _summarize_dropped_context(dropped_messages: list) -> str:
    """
    Build a mechanical summary of dropped messages (no LLM call needed).
    Extracts user topics and tool names used to preserve key context.
    """
    user_topics = []
    tools_used = set()
    for msg in dropped_messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user" and isinstance(content, str):
            # First line of each user message as a topic
            first_line = content.split("\n")[0].strip()[:100]
            if first_line and not first_line.startswith("[EARLIER CONTEXT SUMMARY]"):
                user_topics.append(first_line)
        elif role == "assistant":
            # Extract tool names from Claude format
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        tools_used.add(block.get("name", ""))
            # Extract tool names from OpenAI format
            if isinstance(msg.get("tool_calls"), list):
                for tc in msg["tool_calls"]:
                    name = tc.get("function", {}).get("name", "") if isinstance(tc, dict) else ""
                    if name:
                        tools_used.add(name)
    lines = []
    if user_topics:
        # Keep last 10 topics
        recent = user_topics[-10:]
        lines.append("Topics discussed: " + "; ".join(recent))
    if tools_used:
        lines.append("Tools used: " + ", ".join(sorted(tools_used)))
    return "\n".join(lines) if lines else "Earlier conversation (details trimmed)"


# ---------------------------------------------------------------------------
# Agent Loop
# ---------------------------------------------------------------------------

def run(user_input: str, session_id: str = "default", llm=None, image_path: str = None) -> str:
    """
    Run the agent loop: call LLM, execute tools, repeat until done.
    Returns the final text response.

    The loop works like this:
    1. Load conversation history from sessions/<session_id>.jsonl
    2. Append user message to history
    3. Call LLM with: history + system prompt + available tools
    4. If LLM returns tool_calls:
       a. Execute each tool
       b. Add tool call + result to history
       c. Loop back to step 3 (LLM sees the results and decides next action)
    5. If LLM returns text only: save it and return to the user
    6. Auto-save key facts to memory (both MCP and local JSON)

    Depth (tool-chain limit):
    - "Depth" controls how many tool steps the model can chain (set via /depth).
    - Cloud default: 10, Local default: 5. Prevents infinite loops.
    - All tools are always available on every step.
    """
    # Apply backend override if set (from /claude /openai /lfm commands)
    global _backend_override
    if _backend_override:
        llm = get_llm(_backend_override)
    elif llm is None:
        llm = get_llm()

    # ---------------------------------------------------------------------------
    # Slash Command Handling — processed BEFORE the LLM
    # ---------------------------------------------------------------------------
    # Slash commands are handled directly in Python, not sent to the LLM.
    # This ensures they work instantly, don't cost API credits, and work
    # identically in CLI and WhatsApp. The LLM never sees slash commands.

    cmd = user_input.strip().lower()

    # --- /clear variants: clear chat history and/or task queue ---
    # Note: /reset is handled separately in cli() — it starts a new session.
    # /clear deletes ALL session files (CLI + WhatsApp history).
    # If you only want a fresh conversation without losing WhatsApp history, use /new in CLI.
    if cmd in ["/clear", "/clear-history"]:
        try:
            session_dir = Path(__file__).parent / "sessions"
            if session_dir.exists():
                for f in session_dir.glob("*.jsonl"):
                    f.unlink()
            return "🧹 Chat history cleared. (Tasks untouched — use `/clear tasks` to clear those.)"
        except Exception as e:
            return f"❌ Error clearing history: {e}"

    if cmd == "/clear tasks":
        tasks_file = WORKSPACE / "tasks.json"
        tasks_file.write_text(json.dumps({"tasks": []}, indent=2))
        return "🧹 Task queue cleared."

    if cmd == "/clear memory":
        # Clear local memory
        MEMORY_FILE.write_text(json.dumps({"facts": []}, indent=2))
        # Also clear MCP knowledge graph if available
        if MCP_ENABLED and _mcp_client is not None:
            try:
                execute_mcp_tool("mcp_memory_delete_all_nodes", {})
            except Exception:
                pass
        return "🧹 Memory cleared."

    if cmd == "/clear all":
        # Clear chat history
        try:
            session_dir = Path(__file__).parent / "sessions"
            if session_dir.exists():
                for f in session_dir.glob("*.jsonl"):
                    f.unlink()
        except Exception:
            pass
        # Clear task queue
        tasks_file = WORKSPACE / "tasks.json"
        tasks_file.write_text(json.dumps({"tasks": []}, indent=2))
        # Clear local memory
        MEMORY_FILE.write_text(json.dumps({"facts": []}, indent=2))
        # Also clear MCP knowledge graph if available
        if MCP_ENABLED and _mcp_client is not None:
            try:
                execute_mcp_tool("mcp_memory_delete_all_nodes", {})
            except Exception:
                pass
        return "🧹 All cleared — chat history, task queue, and memory."

    # --- /tools: list all available tools (built-in + MCP) ---
    if cmd == "/tools":
        tools = get_all_tools()
        tool_list = ["Available tools:"]
        for t in tools:
            prefix = "[MCP] " if t["name"].startswith("mcp_") else ""
            tool_list.append(f"  {prefix}{t['name']}: {t['description'][:60]}...")
        return "\n".join(tool_list)

    # --- /help (short) ---
    if cmd == "/help":
        from capabilities import DEPTH, TIER_LABEL
        return f"""🐺 **Obedient Beast — Your Personal AI Assistant**

**What can I do?**
Just talk to me like a person. I can:
• Run commands on your computer ("check disk space", "list my files")
• Read, write, and edit files ("create a shopping list", "update my notes")
• Take screenshots and control your mouse/keyboard
• Remember things for later ("remind me to call the dentist")
• Search the web (when connected to Brave Search)
• Fetch data from websites and APIs
• **Run code in a sandbox** — I write it, run it, you see results:
  `run_python` → text output + images sent to you automatically
  `run_html` → page opens in browser + screenshot sent via WhatsApp
• Work on tasks by myself in the background

**Two brain modes:**
• **Cloud** (Claude or OpenAI) — powerful, fast, uses the internet
• **Local** (runs on your machine) — private, your data never leaves

Currently: **{TIER_LABEL}** — depth {DEPTH} (chains up to {DEPTH} steps per request)

**Commands:**
`/status` — current brain, tasks, heartbeat info
`/tasks` — list all tasks
`/done 3` — mark task #3 complete
`/drop 3` — delete task #3
`/claude` — switch to Cloud (Claude)
`/openai` — switch to Cloud (OpenAI)
`/lfm` — switch to Local brain
`/depth 3` — set how many steps I can chain (current: {DEPTH})
`/model` — list/switch local models (hot-swap, no restart)
`/heartbeat on|off` — toggle background task processing
`/boot` — run workspace/BOOT.md startup routine on demand
`/clear` — clear chat history (`/clear tasks`, `/clear memory`, `/clear all`)
`/tools` — list all my abilities
`/sandbox` — list recent sandbox runs (Python scripts, HTML pages)
`/skills` — installable MCP plug-in skills
`/more` — detailed guide with examples
`/new` — start fresh conversation (CLI only)
`/quit` — exit (CLI only)

**✨ New powers:**
• **Loop detection** — I bail automatically if I get stuck repeating a tool
• **BOOT.md** — drop `workspace/BOOT.md` and I run it once a day on launch (`/boot install` to start from the example, `/boot` to rerun now)
• **spawn_agent** — ask me to "spawn a sub-agent to research X" and I'll run it in an isolated session
• **Smarter memory** — BM25 + temporal decay search, atomic facts, dedup, categories
• **LLM fallback** — set `LLM_FALLBACK=claude,openai` in .env and I retry on failure

**Tips:**
• Say "remind me to..." or "later, do..." and I'll queue it up!
• In shared WhatsApp groups: start your message with **@beast** to summon me for just that message"""

    # --- /more (detailed) ---
    if cmd == "/more":
        from capabilities import DEPTH
        return f"""🐺 **Obedient Beast — Full Guide**

**What am I?**
I'm an AI assistant that lives on your computer. You talk to me (here in the terminal or via WhatsApp), I understand what you want, and I use tools to get it done. Think of me like a smart intern who can use your computer.

**Two Brain Modes:**

  🌐 **Cloud** — Claude or OpenAI (over the internet)
     Powerful, fast, great at complex multi-step tasks.
     Best for: research, multi-file editing, hard questions.
     Switch: `/claude` or `/openai`

  🏠 **Local** — A model running on your machine (e.g. Qwen3.5-122B)
     Your data never leaves your computer. Totally private.
     Strong local models can chain multiple steps, just like cloud.
     Switch: `/lfm` — swap models with `/model`

**Depth (how many steps I can chain):**
  When you ask me something complex, I may need multiple steps —
  search the web, then fetch a page, then summarize it.
  "Depth" controls how many steps I can chain per request.
  • Cloud default: 10 steps — Local default: 5 steps
  • Change it anytime: `/depth 3` (fewer steps = faster, simpler)
  • Current depth: {DEPTH}

**My 19 Built-in Abilities:**
  Files — read, write, edit files, list folders
  Terminal — run any command on your computer
  Screen — take screenshots, move the mouse, click, type
  Memory — I remember things across conversations (BM25 + recency)
  Web — fetch web pages and API data
  Tasks — I keep a to-do list and can work on it by myself
  Sub-agents — spawn isolated sub-sessions for side quests

**Giving Me Tasks for Later:**
  Just say things naturally:
  • "remind me to check disk space"
  • "add a task to organize my downloads"
  • "later, review the log files"
  • "every day at 9am, check for new git commits" (recurring)
  I'll add it to my to-do list. If the heartbeat is on, I'll
  work on it automatically in the background.

**The Heartbeat (my autopilot):**
  When turned on, I wake up every few minutes and work on
  pending tasks by myself — no need to ask me.
  • `/heartbeat on` — turn on autopilot
  • `/heartbeat off` — pause autopilot
  • `/heartbeat` — check if it's on or off

**🚀 BOOT.md — Your Daily Startup Routine:**
  Drop a file at `workspace/BOOT.md` and I run it once per day
  the first time the CLI launches. Great for standing orders.
  • `/boot install` — copy BOOT.md.example to BOOT.md (quickest way to start)
  • `/boot` — rerun the BOOT routine on demand
  Example BOOT.md content:
    1. Use the shell tool to check disk space.
    2. Recall memories tagged with today's projects.
    3. Show me overdue tasks from the queue.

**🧠 Sub-Agents (spawn_agent tool):**
  Ask me to branch off for side quests without polluting our chat:
  • "spawn a sub-agent to research three MLX quantization approaches and report back"
  • "use spawn_agent to summarize the contents of ~/Downloads"
  The sub-agent runs in its own session with a fresh context window
  and a capped depth — I only see its final answer.

**🔁 Loop Detection:**
  If I get stuck repeating the same tool call 3 times in a row
  (or the same error twice), I bail out automatically and tell you
  what I was stuck on. No more burned depth budgets.

**💾 Smarter Memory:**
  • `recall_memory` uses BM25 + temporal decay — recent + relevant wins
  • Auto-save extracts atomic facts, dedupes by fingerprint
  • Each fact is tagged: preference / project / people / decision / conversation
  • `/clear memory` wipes everything if you want a fresh brain

**🛟 LLM Fallback Chain:**
  Set `LLM_FALLBACK=claude,openai` in .env — if my primary backend
  errors out mid-request, I retry with the next one using a
  text-only history so there are no format mismatches.

**📦 Sandbox — Run Generated Code:**
  Ask me to write and run code. I handle the full cycle:
  write → execute → return results.

  **What you see (CLI vs WhatsApp):**
    Scenario                      | Beast sees? | WhatsApp user sees?
    run_python → text output      | ✅ stdout   | ✅ pasted in reply
    run_python → creates .png     | ✅ file list | ✅ image auto-sent
    run_html → HTML page          | ✅ file path | ✅ auto-screenshot sent
    run_html → on CLI             | ✅ file path | ✅ opens in browser

  **Python:** "Write a script that plots a sine wave"
    → `run_python` creates sandbox/py_<ts>/, runs the script,
      returns stdout. If a .png is created, it's auto-sent to WhatsApp.

  **HTML/JS:** "Make a tic-tac-toe game" or "bouncing ball animation"
    → `run_html` writes to sandbox/html_<ts>/, opens in browser,
      AND auto-screenshots the page so WhatsApp users see it too.

  **Tips:**
    • `/sandbox` — list recent runs with their output files
    • Scripts run with Beast's Python, so installed packages work
    • HTML pages open in your default browser AND get screenshotted
    • All sandbox files persist in `workspace/Generated Code/`

**Using Me in Shared Group Chats:**
  I normally stay quiet in group chats with other people.
  To summon me for one message, start it with **@beast**:
  • "@beast what's the weather" → I respond in the group
  • Only YOU (the phone owner) can summon me this way

**Extra Skills (MCP Servers):**
  I can learn new abilities by connecting to MCP servers.
  Think of them as plug-in skills. Type `/skills` to see
  what's available (web search, GitHub, browser automation, etc.)
  All skills load in both Cloud and Local mode.

**All Commands:**
  `/help` — short help
  `/status` — current brain, tasks, heartbeat info
  `/tasks` — see all tasks
  `/done 3` — mark task #3 complete
  `/drop 3` — delete task #3
  `/clear` — clear history (`/clear tasks`, `/clear memory`, `/clear all`)
  `/tools` — list all my abilities
  `/sandbox` — list recent sandbox runs
  `/skills` — list installable MCP skills
  `/claude` — switch to Cloud (Claude)
  `/openai` — switch to Cloud (OpenAI)
  `/lfm` — switch to Local brain
  `/depth 5` — set tool-chain depth (steps per request)
  `/model` — list local models / `/model Qwen3` to hot-swap
  `/heartbeat on|off` — toggle autopilot
  `/boot` — run workspace/BOOT.md now
  `/boot install` — create BOOT.md from the example template
  `/image [path]` — attach an image to the next message
  `/new` — start fresh conversation (CLI only)
  `/quit` — exit (CLI only)"""

    # --- /skills: MCP server catalog organized by tier ---
    if cmd == "/skills":
        return """🐺 **MCP Server Catalog**

**Essential Tier** (always loaded):
  • filesystem — `npx -y @modelcontextprotocol/server-filesystem /Users/jonathanrothberg`
    File search/move beyond built-in tools
  • memory — `npx -y @modelcontextprotocol/server-memory`
    Persistent knowledge graph
  • time — `npx -y @modelcontextprotocol/server-time`
    Time/timezone queries (1-2 tools, trivial for any LLM)
  • fetch — `npx -y @modelcontextprotocol/server-fetch`
    HTTP fetching, works on LAN too

**Extended Tier** (always loaded):
  • sqlite — `npx -y @modelcontextprotocol/server-sqlite`
    Local database queries
  • git — `npx -y @modelcontextprotocol/server-git`
    Git operations (commit, diff, branch)
  • sequential-thinking — `npx -y @modelcontextprotocol/server-sequential-thinking`
    Step-by-step complex reasoning
  • playwright — `npx -y @anthropic/mcp-server-playwright`
    Full browser automation

**Cloud Tier** (needs API keys, works from both Cloud and Local brains):
  • brave-search — `npx -y @modelcontextprotocol/server-brave-search`
    Web search (needs BRAVE_API_KEY)
  • github — `npx -y @modelcontextprotocol/server-github`
    GitHub operations (needs GITHUB_TOKEN)
  • slack — `npx -y @anthropic/mcp-server-slack`
    Slack messaging (needs SLACK_TOKEN)

**Install any server:** Ask me "install the <name> MCP server" or use the install_mcp_server tool directly.
**Current config:** See `config/mcp_servers.json`"""

    # --- /claude, /openai, /lfm — switch backend (works from WhatsApp AND CLI) ---
    if cmd in ["/claude", "/openai", "/lfm"]:
        new_backend = cmd.lstrip("/")
        _backend_override = new_backend
        os.environ["LLM_BACKEND_TEST"] = new_backend
        # Reload capabilities for the new backend
        import importlib
        import capabilities
        importlib.reload(capabilities)
        from capabilities import TIER_LABEL as new_tier, DEPTH as new_depth
        return f"🔄 Switched to **{new_tier}** ({new_backend}). Depth: {new_depth} steps."

    # --- /model — list local models or switch the active model ---
    if cmd == "/model" or cmd.startswith("/model "):
        import urllib.request
        import urllib.error
        from llm import LFM_URL_LOCAL, LFM_URL_REMOTE, LFM_URL
        # Try to reach the local model server
        urls_to_try = [LFM_URL_LOCAL, LFM_URL_REMOTE]
        if LFM_URL not in urls_to_try:
            urls_to_try.insert(0, LFM_URL)
        server_url = None
        for url in urls_to_try:
            try:
                urllib.request.urlopen(url, timeout=3)
                server_url = url
                break
            except Exception:
                continue
        if not server_url:
            return "❌ Local model server not reachable. Start it with:\n  `python lfm_thinking.py --model latest --server`"

        arg = cmd[len("/model"):].strip()

        if arg and arg.lower() not in ("list", "ls", "available"):
            # Switch to the specified model
            try:
                data = json.dumps({"model": arg}).encode()
                req = urllib.request.Request(
                    f"{server_url}/v1/models/switch",
                    data=data,
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=60) as resp:
                    result = json.loads(resp.read().decode())
                if result.get("status") == "already_loaded":
                    return f"Already loaded: **{result['model']}**"
                return f"🔄 Switched to: **{result.get('model', '?')}** ({result.get('type', '?')})"
            except urllib.error.HTTPError as e:
                body = e.read().decode() if e.fp else ""
                return f"❌ Switch failed: {body}"
            except Exception as e:
                return f"❌ Switch failed: {e}"
        else:
            # List available models
            try:
                with urllib.request.urlopen(f"{server_url}/v1/models/available", timeout=10) as resp:
                    result = json.loads(resp.read().decode())
                current = result.get("current", "?")
                models = result.get("models", [])
                lines = [f"🧠 **Current model:** {current}", "", "**Available models** (`/model <name>` to switch):"]
                for m in models:
                    marker = " ← active" if m.get("active") else ""
                    lines.append(f"  {m['key']}. {m['description']}{marker}")
                lines.append("")
                lines.append("Examples: `/model latest`, `/model GLM`, `/model 3`")
                return "\n".join(lines)
            except Exception as e:
                return f"❌ Could not list models: {e}"

    # --- /tasks — list all tasks with status ---
    if cmd == "/tasks":
        tasks_file = WORKSPACE / "tasks.json"
        if not tasks_file.exists():
            return "📋 No tasks yet. Say 'remind me to...' to add one."
        data = json.loads(tasks_file.read_text())
        tasks = data.get("tasks", [])
        if not tasks:
            return "📋 Task queue is empty. Say 'remind me to...' to add one."
        lines = ["📋 **All Tasks:**"]
        for t in tasks:
            icon = {"pending": "⏳", "done": "✅", "failed": "❌", "in_progress": "🔄"}.get(t.get("status"), "❓")
            lines.append(f"  {icon} #{t.get('id','?')} [{t.get('priority','?')}] {t.get('description','')[:60]} — {t.get('status','?')}")
        return "\n".join(lines)

    # --- /done <id> — mark a task as done ---
    if cmd.startswith("/done "):
        try:
            task_id = int(cmd.split()[1])
        except (IndexError, ValueError):
            return "Usage: `/done <task_id>` — e.g. `/done 3`"
        tasks_file = WORKSPACE / "tasks.json"
        if not tasks_file.exists():
            return "❌ No tasks file found."
        data = json.loads(tasks_file.read_text())
        for t in data["tasks"]:
            if t.get("id") == task_id:
                t["status"] = "done"
                t["updated_at"] = datetime.now().isoformat()
                tasks_file.write_text(json.dumps(data, indent=2))
                return f"✅ Task #{task_id} marked as done."
        return f"❌ Task #{task_id} not found."

    # --- /drop <id> — remove a task entirely ---
    if cmd.startswith("/drop "):
        try:
            task_id = int(cmd.split()[1])
        except (IndexError, ValueError):
            return "Usage: `/drop <task_id>` — e.g. `/drop 3`"
        tasks_file = WORKSPACE / "tasks.json"
        if not tasks_file.exists():
            return "❌ No tasks file found."
        data = json.loads(tasks_file.read_text())
        original_len = len(data["tasks"])
        data["tasks"] = [t for t in data["tasks"] if t.get("id") != task_id]
        if len(data["tasks"]) == original_len:
            return f"❌ Task #{task_id} not found."
        tasks_file.write_text(json.dumps(data, indent=2))
        return f"🗑 Task #{task_id} removed."

    # --- /boot — run workspace/BOOT.md startup routine on demand ---
    # /boot           → run BOOT.md now
    # /boot install   → copy BOOT.md.example → BOOT.md (quickest way to start)
    if cmd == "/boot install":
        boot_file = WORKSPACE / "BOOT.md"
        example = WORKSPACE / "BOOT.md.example"
        if boot_file.exists():
            return f"🚀 {boot_file} already exists — edit it directly or delete it first."
        if not example.exists():
            return f"❌ Example file not found at {example}. Nothing to install."
        try:
            boot_file.write_text(example.read_text())
            return (
                f"🚀 Installed BOOT.md from the example template.\n"
                f"   Edit it at: {boot_file}\n"
                f"   Run it now with `/boot`, or it'll auto-run on your next CLI launch."
            )
        except Exception as e:
            return f"❌ Failed to install BOOT.md: {e}"

    if cmd == "/boot":
        boot_file = WORKSPACE / "BOOT.md"
        if not boot_file.exists():
            example = WORKSPACE / "BOOT.md.example"
            if example.exists():
                return (
                    f"🚀 No workspace/BOOT.md found.\n"
                    f"   Quickest start: type `/boot install` to create one from the example template.\n"
                    f"   Or copy it manually: cp {example} {boot_file}"
                )
            return "🚀 No workspace/BOOT.md found and no example template available."
        result = _run_boot_script(session_id, llm=llm, force=True)
        return result or "🚀 BOOT.md is empty."

    # --- /sandbox — list recent sandbox outputs ---
    # --- /sandbox log — show the activity log (what was run, what happened) ---
    if cmd == "/sandbox log":
        if not SANDBOX_LOG.exists():
            return "📦 No sandbox activity yet. Ask me to run a Python script or create an HTML page!"
        try:
            log_text = SANDBOX_LOG.read_text()
            # Show the last 3000 chars (most recent runs)
            if len(log_text) > 3000:
                log_text = "...(older entries trimmed)...\n" + log_text[-3000:]
            return f"📦 **Sandbox Activity Log:**\n```\n{log_text}\n```"
        except Exception as e:
            return f"❌ Error reading sandbox log: {e}"

    if cmd == "/sandbox":
        sandbox_dir = WORKSPACE / "Generated Code"
        if not sandbox_dir.exists() or not any(sandbox_dir.iterdir()):
            return "📦 No sandbox runs yet. Ask me to write and run a Python script or create an HTML page!"
        entries = sorted(sandbox_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)[:15]
        lines = ["📦 **Recent sandbox runs:**"]
        for entry in entries:
            if not entry.is_dir():
                continue
            kind = "🐍 Python" if entry.name.startswith("py_") else "🌐 HTML" if entry.name.startswith("html_") else "📁"
            files = [f.name for f in entry.iterdir()]
            lines.append(f"  {kind} `{entry.name}/` — {', '.join(files[:5])}")
        lines.append(f"\nAll files in: {sandbox_dir}")
        lines.append("Tip: `/sandbox log` for detailed activity log")
        return "\n".join(lines)

    # --- /heartbeat — control background task processing ---
    if cmd == "/heartbeat" or cmd == "/heartbeat status":
        control_file = WORKSPACE / "heartbeat_control.json"
        enabled = True  # default
        if control_file.exists():
            try:
                enabled = json.loads(control_file.read_text()).get("enabled", True)
            except Exception:
                pass
        state = "🟢 ON" if enabled else "🔴 OFF"
        return f"🫀 Heartbeat is {state}.\nUse `/heartbeat on` or `/heartbeat off` to change.\nRun `python heartbeat.py` in a terminal to start the background processor."

    if cmd == "/heartbeat on":
        control_file = WORKSPACE / "heartbeat_control.json"
        control_file.write_text(json.dumps({"enabled": True}, indent=2))
        return "🟢 Heartbeat enabled. Background processor will pick up tasks."

    if cmd == "/heartbeat off":
        control_file = WORKSPACE / "heartbeat_control.json"
        control_file.write_text(json.dumps({"enabled": False}, indent=2))
        return "🔴 Heartbeat disabled. Background processor will pause."

    # --- /depth — view or set tool-chain depth ---
    if cmd == "/depth" or cmd.startswith("/depth "):
        from capabilities import DEPTH, set_depth
        arg = cmd[len("/depth"):].strip()
        if arg:
            try:
                new_depth = int(arg)
                set_depth(new_depth)
                from capabilities import DEPTH as updated
                return f"Depth set to **{updated}** — I'll chain up to {updated} tool steps per request."
            except ValueError:
                return f"Usage: `/depth 5` (a number). Current depth: {DEPTH}"
        else:
            return f"Current depth: **{DEPTH}** steps per request.\nChange it: `/depth 3` (1-20)"

    # --- /status — overview of everything ---
    if cmd == "/status":
        from capabilities import TIER_LABEL, DEPTH, HEARTBEAT_INTERVAL_SEC
        tasks_file = WORKSPACE / "tasks.json"
        control_file = WORKSPACE / "heartbeat_control.json"
        hb_enabled = True
        if control_file.exists():
            try:
                hb_enabled = json.loads(control_file.read_text()).get("enabled", True)
            except Exception:
                pass
        hb_state = "🟢 ON" if hb_enabled else "🔴 OFF"
        status_lines = [
            f"🐺 **Beast Status**",
            f"  Brain: {TIER_LABEL} ({_backend_override or os.getenv('LLM_BACKEND_TEST') or os.getenv('LLM_BACKEND', 'lfm')})",
            f"  Depth: {DEPTH} steps per request",
            f"  Heartbeat: {hb_state} (every {HEARTBEAT_INTERVAL_SEC // 60} min)",
        ]
        if tasks_file.exists():
            try:
                data = json.loads(tasks_file.read_text())
                tasks = data.get("tasks", [])
                pending = [t for t in tasks if t.get("status") == "pending"]
                done = [t for t in tasks if t.get("status") == "done"]
                failed = [t for t in tasks if t.get("status") == "failed"]
                status_lines.append(f"\n📋 **Task Queue** ({len(tasks)} total)")
                status_lines.append(f"  ⏳ Pending: {len(pending)}  ✅ Done: {len(done)}  ❌ Failed: {len(failed)}")
                if pending:
                    status_lines.append("  **Pending tasks:**")
                    for t in pending:
                        status_lines.append(f"    #{t.get('id', '?')} [{t.get('priority', '?')}] {t.get('description', 'No description')[:50]}")
            except Exception:
                status_lines.append("  Task queue: error reading tasks.json")
        else:
            status_lines.append("  Task queue: empty")
        return "\n".join(status_lines)

    # ---------------------------------------------------------------------------
    # Agent Loop — LLM ↔ Tool execution cycle
    # ---------------------------------------------------------------------------

    # Load history and add user message
    history = load_session(session_id)

    # --- Build user message (text-only or multimodal with image) ---
    def _build_user_message(text_content: str) -> dict:
        """Build a user message, adding image content if image_path is set."""
        if not image_path or not Path(image_path).exists():
            return {"role": "user", "content": text_content}

        # Base64-encode the image
        img_data = base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")
        media_type = "image/jpeg"  # WhatsApp images are always JPEG

        if llm.backend == "claude":
            # Claude format: content array with image + text blocks
            return {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": img_data}},
                    {"type": "text", "text": text_content}
                ]
            }
        else:
            # OpenAI / lfm format: content array with image_url + text
            return {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{img_data}"}},
                    {"type": "text", "text": text_content}
                ]
            }

    # Auto-recall memories at session start (new sessions only).
    # If this is a fresh session with no prior messages, inject recent
    # memories as context so Beast remembers user preferences.
    if not history:
        memory_context = _get_startup_memory_context()
        if memory_context:
            # Add memory context as a system-like user message the LLM can see
            memory_msg = _build_user_message(
                f"[AUTO-RECALLED MEMORIES]\n{memory_context}\n[END MEMORIES]\n\nUser's actual message: {user_input}"
            )
            history.append(memory_msg)
            save_message(session_id, memory_msg)
        else:
            user_msg = _build_user_message(user_input)
            history.append(user_msg)
            save_message(session_id, user_msg)
    else:
        user_msg = _build_user_message(user_input)
        history.append(user_msg)
        save_message(session_id, user_msg)

    # ---------------------------------------------------------------------------
    # Depth — how many tool-call steps the model can chain per request.
    # Controlled by /depth command or capabilities.py defaults.
    # Cloud default: 10, Local default: 5. User can change at runtime.
    # ---------------------------------------------------------------------------
    from capabilities import DEPTH

    all_tools = get_all_tools()  # Built-in (18) + MCP tools

    # --- Context cap: prevent blowing the LLM context window ---
    # Long sessions with many tool calls grow unbounded. This trims old messages
    # to keep the most recent ones within a safe token budget. The full history
    # is preserved in the session JSONL file — only the working context is capped.
    # When trimming, a mechanical summary of dropped messages is inserted as the
    # first message to preserve key context (topics discussed, tools used).
    MAX_HISTORY_CHARS = 80000  # ~20k tokens, safe for most models
    SUMMARY_BUDGET = 1000     # Reserve space for the summary message

    def _msg_text_size(msg: dict) -> int:
        """Estimate message size excluding base64 image data (which inflates size
        but doesn't accumulate across turns — only the current message has it)."""
        content = msg.get("content")
        if isinstance(content, list):
            size = 0
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") in ("image", "image_url"):
                        size += 200  # Count image block as ~200 chars (metadata only)
                    else:
                        size += len(json.dumps(block))
                else:
                    size += len(str(block))
            return size
        return len(json.dumps(msg))

    history_size = sum(_msg_text_size(m) for m in history)
    if history_size > MAX_HISTORY_CHARS:
        trimmed = []
        budget = MAX_HISTORY_CHARS - SUMMARY_BUDGET
        for msg in reversed(history):
            msg_size = _msg_text_size(msg)
            if budget - msg_size < 0 and trimmed:
                break
            trimmed.append(msg)
            budget -= msg_size
        kept = list(reversed(trimmed))
        # Build summary of dropped messages
        dropped_count = len(history) - len(kept)
        dropped = history[:dropped_count]
        summary_text = _summarize_dropped_context(dropped)
        summary_msg = {
            "role": "user",
            "content": f"[EARLIER CONTEXT SUMMARY]\n{summary_text}\n({dropped_count} messages trimmed)\n[END SUMMARY]"
        }
        # Replace existing summary if first message is one, otherwise prepend
        if kept and isinstance(kept[0].get("content"), str) and kept[0]["content"].startswith("[EARLIER CONTEXT SUMMARY]"):
            kept[0] = summary_msg
        else:
            kept.insert(0, summary_msg)
        history = kept
        print(f"[Beast] Context trimmed to {len(history)} messages ({dropped_count} summarized)", file=sys.stderr)

    # When an image is attached, skip tools on the first turn so the VLM
    # focuses on describing the image instead of getting confused by 40+ tools.
    # Tools are available on all subsequent turns if the model needs them.
    _image_first_turn = image_path is not None

    # --- Loop detection state ---
    # Track the last few (tool_name, args_fingerprint) signatures. If the same
    # signature repeats >=3 turns in a row, we bail out to prevent the agent
    # from burning the entire depth budget on a stuck tool call.
    _recent_sigs: list[str] = []
    _last_error_result: str = ""
    _repeat_error_count: int = 0

    for turn in range(DEPTH):
        # Determine tools for this turn
        turn_tools = None if _image_first_turn else all_tools
        _image_first_turn = False  # Only skip tools on the very first turn

        # Call LLM with fallback chain — try primary backend, then fallbacks.
        # On fallback, use text-only history to avoid format incompatibilities.
        # If all backends fail and an image is present, retry without the image
        # (the model may not support vision).
        response = None
        errors = []
        backends_to_try = [None] + LLM_FALLBACK  # None = current/primary
        for fallback_backend in backends_to_try:
            try:
                if fallback_backend is None:
                    response = llm.chat(history, tools=turn_tools, system=SYSTEM_PROMPT)
                else:
                    fallback_llm = get_llm(fallback_backend)
                    print(f"[Beast] Falling back to {fallback_backend}...", file=sys.stderr)
                    text_history = _text_only_history(history)
                    response = fallback_llm.chat(text_history, tools=turn_tools, system=SYSTEM_PROMPT)
                    llm = fallback_llm  # Switch for remaining turns
                break  # Success
            except Exception as e:
                label = fallback_backend or llm.backend
                print(f"[Beast] LLM error ({label}): {e}", file=sys.stderr)
                errors.append(f"{label}: {e}")
        # If all backends failed and history contains images, retry without images.
        # This handles non-VLM models that choke on multimodal content.
        if response is None and image_path:
            print("[Beast] Retrying without image (model may not support vision)...", file=sys.stderr)
            history = _strip_images_from_history(history)
            try:
                response = llm.chat(history, tools=all_tools, system=SYSTEM_PROMPT)
            except Exception as e:
                errors.append(f"{llm.backend} (no-image retry): {e}")
        if response is None:
            return f"All LLM backends failed: {'; '.join(errors)}"

        if response.tool_calls:
            # --- Build ONE assistant message with ALL tool calls ---
            # Both Claude and OpenAI expect a single assistant message containing
            # all tool calls, not separate messages per call. The old code created
            # one assistant message per tool call, which produced malformed history.
            if llm.backend == "claude":
                assistant_msg = {"role": "assistant", "content": []}
                if response.text:
                    assistant_msg["content"].append({"type": "text", "text": response.text})
                for tc in response.tool_calls:
                    assistant_msg["content"].append(
                        {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.args}
                    )
            else:
                # OpenAI / local format: all tool_calls in one array
                assistant_msg = {
                    "role": "assistant",
                    "content": response.text or None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": json.dumps(tc.args)}
                        }
                        for tc in response.tool_calls
                    ]
                }

            history.append(assistant_msg)
            save_message(session_id, assistant_msg)

            # Execute each tool and collect results
            tool_results = []
            for tool_call in response.tool_calls:
                # Log to both stderr (CLI) and stdout (pm2 logs)
                args_preview = str(tool_call.args)[:200]
                print(f"  🔧 [{tool_call.name}] {args_preview}", file=sys.stderr)
                print(f"  🔧 [{tool_call.name}] {args_preview}")
                result = execute_tool(tool_call.name, tool_call.args)
                result_preview = str(result)[:200].replace('\n', ' ')
                print(f"  ✅ [{tool_call.name}] → {result_preview}", file=sys.stderr)
                print(f"  ✅ [{tool_call.name}] → {result_preview}")
                tool_results.append((tool_call, result))

            # --- Loop detection: bail if the agent is spinning ---
            # Signature = tool name + sorted args (stable hash of the call).
            # If the same signature repeats 3 turns in a row, OR the same
            # error result repeats 2 turns in a row, we stop early and
            # return a clear message instead of burning the depth budget.
            sig_parts = []
            err_results = []
            for tc, res in tool_results:
                try:
                    arg_blob = json.dumps(tc.args, sort_keys=True, default=str)
                except Exception:
                    arg_blob = str(tc.args)
                sig_parts.append(f"{tc.name}|{arg_blob}")
                if isinstance(res, str) and res.lower().startswith(("error", "http error", "url error")):
                    err_results.append(res[:200])
            turn_sig = "||".join(sig_parts)
            _recent_sigs.append(turn_sig)
            if len(_recent_sigs) > 5:
                _recent_sigs = _recent_sigs[-5:]
            if len(_recent_sigs) >= 3 and _recent_sigs[-1] == _recent_sigs[-2] == _recent_sigs[-3]:
                loop_msg = (
                    f"🔁 Stopping — I repeated the same tool call 3 times. "
                    f"Last call: `{tool_results[0][0].name}`. "
                    f"Last result: {str(tool_results[0][1])[:200]}"
                )
                final_msg = {"role": "assistant", "content": loop_msg}
                history.append(final_msg)
                save_message(session_id, final_msg)
                return loop_msg
            if err_results:
                joined_err = "||".join(err_results)
                if joined_err == _last_error_result:
                    _repeat_error_count += 1
                else:
                    _last_error_result = joined_err
                    _repeat_error_count = 1
                if _repeat_error_count >= 2:
                    loop_msg = (
                        f"🔁 Stopping — same error twice in a row: {err_results[0][:200]}"
                    )
                    final_msg = {"role": "assistant", "content": loop_msg}
                    history.append(final_msg)
                    save_message(session_id, final_msg)
                    return loop_msg
            else:
                _repeat_error_count = 0
                _last_error_result = ""

            # Add tool results in the correct format
            if llm.backend == "claude":
                # Claude: ALL results in one user message (API requires this
                # when the assistant message has multiple tool_use blocks)
                tool_result_msg = {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": tc.id, "content": res}
                        for tc, res in tool_results
                    ]
                }
                history.append(tool_result_msg)
                save_message(session_id, tool_result_msg)
            else:
                # OpenAI / local: separate "tool" role messages
                for tool_call, result in tool_results:
                    tool_result_msg = {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    }
                    history.append(tool_result_msg)
                    save_message(session_id, tool_result_msg)

        else:
            # No tool calls — LLM returned a text response, we're done
            final_msg = {"role": "assistant", "content": response.text}
            history.append(final_msg)
            save_message(session_id, final_msg)
            # Auto-save key facts to memory (both MCP and local JSON)
            _try_memory_save(session_id, user_input, response.text)
            return response.text

    return "(Max turns reached - stopping)"


def _run_boot_script(session_id: str, llm=None, force: bool = False) -> str | None:
    """
    Run workspace/BOOT.md once per day on CLI launch (or on demand via /boot).

    BOOT.md is a user-defined startup routine — any markdown text in that file
    is fed to the agent as a synthetic user message prefixed with [BOOT], so
    the LLM executes whatever tools the user wants on launch. A sentinel file
    workspace/.boot_done_YYYYMMDD.flag prevents re-running on /new within the
    same day. Pass force=True from /boot to bypass the sentinel.

    Returns the agent's response, or None if BOOT.md is missing / already ran.
    """
    boot_file = WORKSPACE / "BOOT.md"
    if not boot_file.exists():
        return None
    today = datetime.now().strftime("%Y%m%d")
    sentinel = WORKSPACE / f".boot_done_{today}.flag"
    if sentinel.exists() and not force:
        return None
    try:
        boot_text = boot_file.read_text().strip()
    except Exception as e:
        return f"[BOOT] Failed to read BOOT.md: {e}"
    if not boot_text:
        return None
    # Clean up older sentinels from previous days
    try:
        for f in WORKSPACE.glob(".boot_done_*.flag"):
            if f.name != sentinel.name:
                f.unlink()
    except Exception:
        pass
    sentinel.write_text(datetime.now().isoformat())
    boot_msg = f"[BOOT] Execute this startup routine:\n\n{boot_text}"
    return run(boot_msg, session_id=session_id, llm=llm)


def _get_startup_memory_context() -> str:
    """
    Get recent memories for auto-recall at session start.
    Returns a string of recent facts, or empty string if no memories.
    Tries MCP memory first, then local JSON fallback.
    """
    # Try MCP memory first
    if MCP_ENABLED and _mcp_client is not None:
        try:
            result = execute_mcp_tool("mcp_memory_read_graph", {})
            if result and not result.startswith("Error"):
                # Truncate to avoid overwhelming the context
                return result[:2000]
        except Exception:
            pass

    # Fallback: local memory — return last 10 facts
    data = _load_local_memory()
    facts = data.get("facts", [])
    if facts:
        recent = facts[-10:]
        return "\n".join(f["text"] for f in recent)
    return ""


# ---------------------------------------------------------------------------
# CLI Interface
# ---------------------------------------------------------------------------

def cli():
    """Interactive CLI mode — the main entry point for terminal use."""
    from llm import BACKEND
    from capabilities import TIER_LABEL, DEPTH

    all_tools = get_all_tools()
    builtin_count = len(TOOLS)
    mcp_count = len(all_tools) - builtin_count

    print("=" * 60)
    print("🐺 Obedient Beast - AI Assistant")
    print(f"   Brain: {TIER_LABEL} ({BACKEND})")
    print(f"   Depth: {DEPTH} steps per request")
    print(f"   Tools: {builtin_count} built-in" + (f" + {mcp_count} MCP" if mcp_count > 0 else ""))
    if MCP_ENABLED:
        print(f"   MCP: enabled")
    print("=" * 60)
    print("Commands: /help, /more, /status, /depth, /new, /clear, /quit, /tools, /image, /boot, /claude, /openai, /lfm, /model")
    print("=" * 60 + "\n")

    session_id = f"cli_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    llm = get_llm()
    pending_image = None  # Set by /image command, used on next message

    # --- BOOT.md: run the user's startup routine (once per day on launch) ---
    boot_result = _run_boot_script(session_id, llm=llm, force=False)
    if boot_result:
        print(f"🚀 BOOT.md executed:\n{boot_result}\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["/quit", "/exit", "quit", "exit"]:
                print("Goodbye!")
                break

            if user_input.lower() in ["/new", "/reset"]:
                session_id = f"cli_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                pending_image = None
                print("(Session reset)\n")
                continue

            if user_input.lower() == "/tools":
                tools = get_all_tools()
                print("\nAvailable tools:")
                for t in tools:
                    prefix = "[MCP] " if t["name"].startswith("mcp_") else ""
                    print(f"  {prefix}{t['name']}: {t['description'][:60]}...")
                print()
                continue

            # /image [path] — attach an image to the next message
            # No path: opens a file dialog. With path: uses that file.
            if user_input.lower().startswith("/image"):
                parts = user_input.split(maxsplit=1)
                if len(parts) >= 2:
                    # Path provided on command line
                    img_path = Path(parts[1].strip()).expanduser()
                else:
                    # No path — open file picker dialog
                    try:
                        import tkinter as tk
                        from tkinter import filedialog
                        root = tk.Tk()
                        root.withdraw()
                        root.attributes("-topmost", True)
                        selected = filedialog.askopenfilename(
                            title="Select an image",
                            filetypes=[
                                ("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.webp *.tiff"),
                                ("PNG", "*.png"),
                                ("JPEG", "*.jpg *.jpeg"),
                                ("All files", "*.*"),
                            ]
                        )
                        root.destroy()
                        if not selected:
                            print("(No image selected)")
                            print()
                            continue
                        img_path = Path(selected)
                    except Exception as e:
                        print(f"File dialog failed: {e}")
                        print("Usage: /image <path>")
                        print()
                        continue
                if not img_path.exists():
                    print(f"File not found: {img_path}")
                    print()
                    continue
                pending_image = str(img_path)
                print(f"(Image attached: {img_path.name} — type your question now)")
                continue

            # Send message, with image if one is pending
            print("Beast: ", end="", flush=True)
            response = run(user_input, session_id, llm, image_path=pending_image)
            pending_image = None  # Clear after use
            print(response)
            print()

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except EOFError:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    cli()
