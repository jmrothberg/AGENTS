#!/usr/bin/env python3
"""
Obedient Beast - CLI + Agent Loop + Tools
==========================================
A minimal agentic assistant with tool calling, autonomous task queue,
persistent memory, and tiered capabilities (FULL for Claude/OpenAI,
LITE for local models via "lfm" backend — legacy name, works with any local model).

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
4. Auto-save key facts to memory (MCP + local JSON fallback)

Tool Count: 18 built-in + N MCP tools (loaded dynamically)

Usage:
    python beast.py                     # Interactive CLI mode
    ./start.sh                          # 4 Terminal windows (server, WhatsApp, heartbeat, CLI)

Slash commands (work from CLI and WhatsApp):
    /help, /more, /status, /tasks, /done <id>, /drop <id>,
    /claude, /openai, /lfm, /heartbeat on|off,
    /clear, /clear tasks, /clear memory, /clear all, /tools, /skills
"""

import os
import sys
import json
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

MCP_ENABLED = os.getenv("MCP_ENABLED", "false").lower() == "true"
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

# Local memory file — JSON fallback when MCP memory server is unavailable.
# Stores facts as simple {"facts": [{"text": "...", "timestamp": "..."}]}
# Capped at 200 entries (FIFO) to prevent unbounded growth.
MEMORY_FILE = WORKSPACE / "memory.json"

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

def load_system_prompt() -> str:
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
    prompt += base_prompt
    return prompt


SYSTEM_PROMPT = load_system_prompt()


# ---------------------------------------------------------------------------
# Local Memory Helpers — JSON fallback for when MCP memory is unavailable
# ---------------------------------------------------------------------------
# These functions provide a simple local memory store so Beast can remember
# facts even without the MCP memory server running. Facts are stored in
# workspace/memory.json as a flat list with timestamps.
# The MCP memory server (knowledge graph) is richer, but this ensures
# memory always works, even offline or in LITE mode.

def _load_local_memory() -> dict:
    """Load local memory from workspace/memory.json. Returns {"facts": [...]}."""
    if not MEMORY_FILE.exists():
        return {"facts": []}
    try:
        return json.loads(MEMORY_FILE.read_text())
    except (json.JSONDecodeError, IOError):
        return {"facts": []}


def _save_local_memory_fact(fact: str):
    """
    Save a single fact to local memory. Caps at 200 facts (FIFO).
    Called by _try_memory_save() after every conversation turn.
    """
    data = _load_local_memory()
    data["facts"].append({
        "text": fact,
        "timestamp": datetime.now().isoformat()
    })
    # Cap at 200 facts — remove oldest when full (FIFO)
    if len(data["facts"]) > 200:
        data["facts"] = data["facts"][-200:]
    MEMORY_FILE.write_text(json.dumps(data, indent=2))


def _search_local_memory(query: str) -> str:
    """
    Search local memory by keyword matching. Returns matching facts as text.
    Used as fallback when MCP memory server is unavailable.
    """
    data = _load_local_memory()
    query_lower = query.lower()
    matches = [
        f["text"] for f in data["facts"]
        if query_lower in f["text"].lower()
    ]
    if matches:
        return "\n".join(matches[-10:])  # Return last 10 matches
    return f"No local memories found for '{query}'."


def _try_memory_save(session_id: str, user_input: str, response_text: str):
    """
    Auto-save key facts to memory at end of a conversation turn.
    Saves to BOTH MCP memory (if available) AND local JSON fallback.
    This ensures facts persist regardless of MCP state.
    Respects the capability tier (full vs minimal detail).
    """
    # Always save to local memory fallback (works offline, no MCP needed)
    from capabilities import MEMORY_DETAIL
    if MEMORY_DETAIL == "minimal":
        fact = f"Session {session_id}: user asked about '{user_input[:80]}'"
    else:
        fact = f"Session {session_id}: user asked '{user_input[:120]}', beast responded with '{response_text[:200]}'"
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
        "description": "Add, update, or complete a task in the autonomous task queue. Beast and users can queue work for later.",
        "params": {
            "description": "What the task is (required for new tasks)",
            "priority": "low, medium, or high (default: medium)",
            "status": "pending, done, or failed (default: pending). Use 'done'/'failed' to close a task.",
            "task_id": "Optional: ID of existing task to update its status"
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
]


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
                data["tasks"].append(new_task)
                tasks_file.write_text(json.dumps(data, indent=2))
                return f"Task #{new_task['id']} added: {new_task['description']} [{new_task['priority']}]"

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
                req = urllib.request.Request(url, method=method)
                req.add_header("User-Agent", "ObedientBeast/1.0")
                with urllib.request.urlopen(req, timeout=30) as resp:
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
# Agent Loop
# ---------------------------------------------------------------------------

def run(user_input: str, session_id: str = "default", llm=None) -> str:
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

    Loop prevention (SINGLE_TOOL_MODE / tools_used):
    - Local LLMs tend to loop on tool calls instead of summarizing.
    - In LITE mode (local LLMs), after the first tool call, we stop sending
      tools to the LLM. This forces it to generate a text response using
      the tool results it already has.
    - Cloud LLMs (Claude/OpenAI) handle multi-tool properly and always see all tools.
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
        # Clear local memory (workspace/memory.json)
        cleared = []
        if MEMORY_FILE.exists():
            MEMORY_FILE.write_text(json.dumps({"facts": []}, indent=2))
            cleared.append("local memory")
        else:
            cleared.append("local memory (was already empty)")
        # Note: MCP knowledge graph must be cleared manually via the MCP memory server.
        # We can't easily wipe it from here, but we tell the user.
        return "🧹 " + " and ".join(cleared) + " cleared.\n💡 MCP knowledge graph (if enabled) is separate — ask me to 'forget everything in memory' to clear that too."

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
        return "🧹 All cleared — chat history, task queue, and local memory.\n💡 MCP knowledge graph (if enabled) is separate — ask me to 'forget everything in memory' to clear that too."

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
        return """🐺 **Obedient Beast — Your Personal AI Assistant**

**What can I do?**
Just talk to me like a person. I can:
• Run commands on your computer ("check disk space", "list my files")
• Read, write, and edit files ("create a shopping list", "update my notes")
• Take screenshots and control your mouse/keyboard
• Remember things for later ("remind me to call the dentist")
• Search the web (when connected to Brave Search)
• Fetch data from websites and APIs
• Work on tasks by myself in the background

**Two brain modes:**
• **FULL** (Claude or OpenAI) — chains multiple actions, thinks deeply
• **LITE** (local model) — one thing at a time, simpler but totally private

**All Commands:**
`/status` — current brain, pending tasks, heartbeat info
`/tasks` — list all tasks
`/done 3` — mark task #3 complete
`/drop 3` — delete task #3
`/claude` — switch to Claude brain (FULL)
`/openai` — switch to OpenAI brain (FULL)
`/lfm` — switch to local brain (LITE)
`/heartbeat on|off` — toggle background task processing
`/clear` — clear chat history
`/clear tasks` — clear task list
`/clear memory` — clear saved memories (local)
`/clear all` — clear everything (history + tasks + memory)
`/tools` — list all my abilities
`/skills` — installable MCP plug-in skills
`/more` — detailed guide with examples
`/new` — start fresh conversation (CLI only)
`/quit` — exit (CLI only)

**Tips:**
• Say "remind me to..." or "later, do..." and I'll queue it up!
• In shared WhatsApp groups: start your message with **@beast** to summon me for just that message"""

    # --- /more (detailed) ---
    if cmd == "/more":
        return """🐺 **Obedient Beast — Full Guide**

**What am I?**
I'm an AI assistant that lives on your computer. You talk to me (here in the terminal or via WhatsApp), I understand what you want, and I use tools to get it done. Think of me like a smart intern who can use your computer.

**Two Brain Modes (important!):**
Your AI brain determines how smart and capable I am:

  🧠 **FULL mode** — Using Claude or OpenAI (cloud AI)
     I can do complex multi-step tasks, chain 10 actions together,
     remember rich details, and use all my extra skills.
     Best for: complex requests, research, multi-file editing.

  🧠 **LITE mode** — Using a local model running on your machine
     I do one thing at a time and keep it simple.
     Your data never leaves your computer. Totally private.
     Best for: quick tasks, privacy-sensitive work.

  Switch anytime: `/claude` `/openai` `/lfm`

**My 18 Built-in Abilities:**
  Files — read, write, edit files, list folders
  Terminal — run any command on your computer
  Screen — take screenshots, move the mouse, click, type
  Memory — I remember things across conversations
  Web — fetch web pages and API data
  Tasks — I keep a to-do list and can work on it by myself

**Giving Me Tasks for Later:**
  Just say things naturally:
  • "remind me to check disk space"
  • "add a task to organize my downloads"
  • "later, review the log files"
  I'll add it to my to-do list. If the heartbeat is on, I'll
  work on it automatically in the background.

**The Heartbeat (my autopilot):**
  When turned on, I wake up every few minutes and work on
  pending tasks by myself — no need to ask me.
  • `/heartbeat on` — turn on autopilot
  • `/heartbeat off` — pause autopilot
  • `/heartbeat` — check if it's on or off

**Using Me in Shared Group Chats:**
  I normally stay quiet in group chats with other people.
  To summon me for one message, start it with **@beast**:
  • "@beast what's the weather" → I respond in the group
  • Only YOU (the phone owner) can summon me this way

**Extra Skills (MCP Servers):**
  I can learn new abilities by connecting to MCP servers.
  Think of them as plug-in skills. Type `/skills` to see
  what's available (web search, GitHub, browser automation, etc.)
  All skills load in both LITE and FULL mode — I can use web search
  and other cloud tools even with a local brain.

**All Commands:**
  `/help` — short help
  `/status` — current brain, tasks, heartbeat info
  `/tasks` — see all tasks
  `/done 3` — mark task #3 complete
  `/drop 3` — delete task #3
  `/clear` — clear chat history
  `/clear tasks` — clear task list
  `/clear memory` — clear saved memories (local)
  `/clear all` — clear everything (history + tasks + memory)
  `/tools` — list all my abilities
  `/skills` — list installable MCP skills
  `/claude` — switch to Claude brain (FULL)
  `/openai` — switch to OpenAI brain (FULL)
  `/lfm` — switch to local brain (LITE)
  `/heartbeat on|off` — toggle autopilot
  `/new` — start fresh conversation (CLI only)
  `/quit` — exit (CLI only)"""

    # --- /skills: MCP server catalog organized by tier ---
    if cmd == "/skills":
        return """🐺 **MCP Server Catalog**

**Essential Tier** (local LLM friendly, loaded by default):
  • filesystem — `npx -y @modelcontextprotocol/server-filesystem /Users/jonathanrothberg`
    File search/move beyond built-in tools
  • memory — `npx -y @modelcontextprotocol/server-memory`
    Persistent knowledge graph
  • time — `npx -y @modelcontextprotocol/server-time`
    Time/timezone queries (1-2 tools, trivial for any LLM)
  • fetch — `npx -y @modelcontextprotocol/server-fetch`
    HTTP fetching, works on LAN too

**Extended Tier** (loaded in FULL mode, local LLM can use with guidance):
  • sqlite — `npx -y @modelcontextprotocol/server-sqlite`
    Local database queries
  • git — `npx -y @modelcontextprotocol/server-git`
    Git operations (commit, diff, branch)
  • sequential-thinking — `npx -y @modelcontextprotocol/server-sequential-thinking`
    Step-by-step complex reasoning
  • playwright — `npx -y @anthropic/mcp-server-playwright`
    Full browser automation

**Cloud-only Tier** (needs Claude/OpenAI + API keys):
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
        # Reload capabilities for the new backend tier
        import importlib
        import capabilities
        importlib.reload(capabilities)
        from capabilities import TIER_LABEL as new_tier, MAX_TOOL_TURNS as new_max
        return f"🔄 Switched to **{new_backend}** backend. Tier: {new_tier} (max {new_max} tool turns)"

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

    # --- /status — overview of everything ---
    if cmd == "/status":
        from capabilities import TIER_LABEL, MAX_TOOL_TURNS, HEARTBEAT_INTERVAL_SEC
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
            f"  Backend: {_backend_override or os.getenv('LLM_BACKEND_TEST') or os.getenv('LLM_BACKEND', 'lfm')}",
            f"  Tier: {TIER_LABEL}",
            f"  Max tool turns: {MAX_TOOL_TURNS}",
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

    # Auto-recall memories at session start (new sessions only).
    # If this is a fresh session with no prior messages, inject recent
    # memories as context so Beast remembers user preferences.
    if not history:
        memory_context = _get_startup_memory_context()
        if memory_context:
            # Add memory context as a system-like user message the LLM can see
            memory_msg = {
                "role": "user",
                "content": f"[AUTO-RECALLED MEMORIES]\n{memory_context}\n[END MEMORIES]\n\nUser's actual message: {user_input}"
            }
            history.append(memory_msg)
            save_message(session_id, memory_msg)
        else:
            user_msg = {"role": "user", "content": user_input}
            history.append(user_msg)
            save_message(session_id, user_msg)
    else:
        user_msg = {"role": "user", "content": user_input}
        history.append(user_msg)
        save_message(session_id, user_msg)

    # ---------------------------------------------------------------------------
    # Capability-tiered settings (from capabilities.py)
    # ---------------------------------------------------------------------------
    # MAX_TOOL_TURNS: how many LLM↔tool round-trips before forcing a text response.
    #   FULL (Claude/OpenAI): 10 turns — enough for complex multi-step tasks.
    #   LITE (local LLM): 2 turns — prevents infinite loops.
    # SINGLE_TOOL_MODE: after first tool use, stop sending tools to the LLM.
    #   Only active in LITE mode. Forces the LLM to summarize instead of looping.
    from capabilities import MAX_TOOL_TURNS, SINGLE_TOOL_MODE

    max_turns = MAX_TOOL_TURNS
    all_tools = get_all_tools()  # Built-in (18) + MCP tools
    tools_used = False  # Track if we've already used a tool (for loop prevention)

    for turn in range(max_turns):
        # Determine which tools to send based on backend and capability tier
        if SINGLE_TOOL_MODE and tools_used:
            # LITE mode (local LFM): Don't send tools after first use.
            # This forces the model to summarize the tool result instead of
            # making another tool call (local LLMs tend to loop otherwise).
            current_tools = None
        else:
            # FULL mode (Claude/OpenAI): Always send all tools.
            current_tools = all_tools

        # Call LLM — sends conversation history + system prompt + tools
        response = llm.chat(history, tools=current_tools, system=SYSTEM_PROMPT)

        if response.tool_calls:
            # LLM wants to use tools — execute each one
            for tool_call in response.tool_calls:
                print(f"  [Tool: {tool_call.name}({tool_call.args})]", file=sys.stderr)
                result = execute_tool(tool_call.name, tool_call.args)

                # ---------------------------------------------------------------------------
                # Claude vs OpenAI message format differences:
                # ---------------------------------------------------------------------------
                # Claude uses content blocks: [{"type": "tool_use", ...}] for tool calls
                #   and {"role": "user", content: [{"type": "tool_result", ...}]} for results.
                # OpenAI uses: {"role": "assistant", tool_calls: [...]} for tool calls
                #   and {"role": "tool", tool_call_id: "...", content: "..."} for results.
                # We build the correct format based on which backend is active.

                if llm.backend == "claude":
                    # Claude format: tool_use blocks in assistant content array
                    assistant_msg = {
                        "role": "assistant",
                        "content": [
                            {"type": "tool_use", "id": tool_call.id, "name": tool_call.name, "input": tool_call.args}
                        ]
                    }
                    if response.text:
                        assistant_msg["content"].insert(0, {"type": "text", "text": response.text})
                else:
                    # OpenAI format: tool_calls array on assistant message
                    assistant_msg = {
                        "role": "assistant",
                        "content": response.text or None,
                        "tool_calls": [{
                            "id": tool_call.id,
                            "type": "function",
                            "function": {"name": tool_call.name, "arguments": json.dumps(tool_call.args)}
                        }]
                    }

                history.append(assistant_msg)
                save_message(session_id, assistant_msg)

                # Add tool result in the correct format
                if llm.backend == "claude":
                    # Claude: tool results go in a "user" message with tool_result content
                    tool_result_msg = {
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": tool_call.id, "content": result}]
                    }
                else:
                    # OpenAI: tool results are "tool" role messages
                    tool_result_msg = {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    }

                history.append(tool_result_msg)
                save_message(session_id, tool_result_msg)

            # Mark that tools have been used (for SINGLE_TOOL_MODE loop prevention)
            tools_used = True

        else:
            # No tool calls — LLM returned a text response, we're done
            final_msg = {"role": "assistant", "content": response.text}
            history.append(final_msg)
            save_message(session_id, final_msg)
            # Auto-save key facts to memory (both MCP and local JSON)
            _try_memory_save(session_id, user_input, response.text)
            return response.text

    return "(Max turns reached - stopping)"


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
    from capabilities import TIER_LABEL, MAX_TOOL_TURNS

    all_tools = get_all_tools()
    builtin_count = len(TOOLS)
    mcp_count = len(all_tools) - builtin_count

    print("=" * 60)
    print("🐺 Obedient Beast - AI Assistant")
    print(f"   Backend: {BACKEND}")
    print(f"   Tier: {TIER_LABEL} (max {MAX_TOOL_TURNS} tool turns)")
    print(f"   Tools: {builtin_count} built-in" + (f" + {mcp_count} MCP" if mcp_count > 0 else ""))
    if MCP_ENABLED:
        print(f"   MCP: enabled")
    print("=" * 60)
    print("Commands: /help, /more, /status, /skills, /new, /clear, /quit, /tools, /claude, /openai, /lfm")
    print("=" * 60 + "\n")

    session_id = f"cli_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    llm = get_llm()

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

            print("Beast: ", end="", flush=True)
            response = run(user_input, session_id, llm)
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
