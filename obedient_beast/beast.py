#!/usr/bin/env python3
"""
Obedient Beast - CLI + Agent Loop + Tools
==========================================
A minimal agentic assistant with tool calling.

Usage:
    python beast.py              # Interactive CLI mode
    LLM_BACKEND=openai python beast.py  # Use OpenAI instead of Claude
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
# MCP tools are loaded dynamically if MCP servers are configured and running

MCP_ENABLED = os.getenv("MCP_ENABLED", "false").lower() == "true"
_mcp_client = None

def get_mcp_tools() -> list[dict]:
    """Get MCP tools if MCP is enabled and servers are running."""
    global _mcp_client
    if not MCP_ENABLED:
        return []
    try:
        from mcp_client import get_mcp_client, init_mcp
        if _mcp_client is None:
            _mcp_client = init_mcp()
        return _mcp_client.get_tools_for_llm()
    except Exception as e:
        print(f"[MCP] Not available: {e}", file=sys.stderr)
        return []

def execute_mcp_tool(name: str, args: dict) -> str:
    """Execute an MCP tool."""
    global _mcp_client
    if _mcp_client is None:
        return "Error: MCP not initialized"
    try:
        from mcp_client import execute_mcp_tool as mcp_exec
        return mcp_exec(name, args)
    except Exception as e:
        return f"Error executing MCP tool: {e}"

def get_all_tools() -> list[dict]:
    """Get all available tools: built-in + MCP."""
    return TOOLS + get_mcp_tools()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WORKSPACE = Path(__file__).parent / "workspace"
SESSIONS_DIR = Path(__file__).parent / "sessions"
SESSIONS_DIR.mkdir(exist_ok=True)

# Pending image to send with next response (for WhatsApp image sending)
_pending_image: str = None

# Backend override — set via /claude /openai /lfm commands (persists across calls)
_backend_override: str = None

def set_pending_image(path: str):
    """Set an image to be sent with the next response."""
    global _pending_image
    _pending_image = path

def get_and_clear_pending_image() -> str:
    """Get and clear the pending image path."""
    global _pending_image
    path = _pending_image
    _pending_image = None
    return path

# Load personality from SOUL.md and AGENTS.md if they exist
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


def _try_memory_save(session_id: str, user_input: str, response_text: str):
    """
    Auto-save key facts to memory MCP at end of a conversation turn.
    Only runs if MCP is enabled and memory server is available.
    Respects the capability tier (full vs minimal detail).
    """
    if not MCP_ENABLED or _mcp_client is None:
        return
    try:
        from capabilities import MEMORY_DETAIL
        # Build a concise fact to store
        if MEMORY_DETAIL == "minimal":
            # Just store that a conversation happened and any key facts
            fact = f"Session {session_id}: user asked about '{user_input[:80]}'"
        else:
            # Store richer context
            fact = f"Session {session_id}: user asked '{user_input[:120]}', beast responded with '{response_text[:200]}'"
        
        # Try to create an entity in the memory knowledge graph
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
# Tools Definition
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "shell",
        "description": "Execute a shell command and return the output. Use for running programs, checking system status, etc.",
        "params": {"command": "The shell command to execute"}
    },
    {
        "name": "read_file",
        "description": "Read and return the contents of a file.",
        "params": {"path": "Path to the file to read"}
    },
    {
        "name": "write_file",
        "description": "Write content to a file. Creates the file if it doesn't exist.",
        "params": {"path": "Path to the file", "content": "Content to write"}
    },
    {
        "name": "list_dir",
        "description": "List files and directories in a path.",
        "params": {"path": "Directory path to list"}
    },
    {
        "name": "edit_file",
        "description": "Replace text in a file. Finds 'old_text' and replaces with 'new_text'.",
        "params": {"path": "Path to the file", "old_text": "Text to find", "new_text": "Text to replace with"}
    },
    # ---------------------------------------------------------------------------
    # Computer Control Tools (Phase 1)
    # ---------------------------------------------------------------------------
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
]


def execute_tool(name: str, args: dict) -> str:
    """Execute a tool and return the result as a string."""
    try:
        if name == "shell":
            result = subprocess.run(
                args["command"],
                shell=True,
                capture_output=True,
                text=True,
                timeout=60
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
            path.parent.mkdir(parents=True, exist_ok=True)
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
            path = Path(args["path"]).expanduser()
            if not path.exists():
                return f"Error: File not found: {path}"
            content = path.read_text()
            if args["old_text"] not in content:
                return f"Error: Text not found in file"
            new_content = content.replace(args["old_text"], args["new_text"], 1)
            path.write_text(new_content)
            return f"Successfully replaced text in {path}"
        
        # ---------------------------------------------------------------------------
        # Computer Control Tools (Phase 1)
        # ---------------------------------------------------------------------------
        elif name == "screenshot":
            import mss
            from datetime import datetime as dt
            filename = args.get("filename") or f"screenshot_{dt.now().strftime('%Y%m%d_%H%M%S')}.png"
            screenshot_dir = WORKSPACE / "screenshots"
            screenshot_dir.mkdir(exist_ok=True)
            filepath = screenshot_dir / filename
            with mss.mss() as sct:
                sct.shot(output=str(filepath))
            # Set as pending image so it gets sent with WhatsApp response
            set_pending_image(str(filepath))
            return f"Screenshot saved to {filepath}. Will send via WhatsApp."
        
        elif name == "mouse_click":
            import pyautogui
            x = int(args["x"])
            y = int(args["y"])
            button = args.get("button", "left")
            pyautogui.click(x, y, button=button)
            return f"Clicked {button} button at ({x}, {y})"
        
        elif name == "mouse_move":
            import pyautogui
            x = int(args["x"])
            y = int(args["y"])
            pyautogui.moveTo(x, y)
            return f"Moved mouse to ({x}, {y})"
        
        elif name == "keyboard_type":
            import pyautogui
            text = args["text"]
            pyautogui.write(text)
            return f"Typed {len(text)} characters"
        
        elif name == "keyboard_hotkey":
            import pyautogui
            keys = args["keys"].split("+")
            pyautogui.hotkey(*keys)
            return f"Pressed hotkey: {args['keys']}"
        
        elif name == "get_screen_size":
            import pyautogui
            width, height = pyautogui.size()
            return f"Screen size: {width}x{height} pixels"
        
        elif name == "get_mouse_position":
            import pyautogui
            x, y = pyautogui.position()
            return f"Mouse position: ({x}, {y})"
        
        # ---------------------------------------------------------------------------
        # Self-Upgrade Tools (Phase 3) - Beast can modify its own capabilities!
        # ---------------------------------------------------------------------------
        elif name == "install_mcp_server":
            config_file = Path(__file__).parent / "config" / "mcp_servers.json"
            config_file.parent.mkdir(exist_ok=True)
            
            # Load existing config
            if config_file.exists():
                config = json.loads(config_file.read_text())
            else:
                config = {"servers": {}}
            
            # Add new server
            server_name = args["name"]
            config["servers"][server_name] = {
                "enabled": True,
                "command": args["command"],
                "description": args["description"],
                "local": True
            }
            
            # Save config
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
                desc = info.get("description", "No description")
                result += f"  [{status}] {name}: {desc}\n"
            
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
        
        # ---------------------------------------------------------------------------
        # Autonomous Agent Tools (Phase 4 - Clawdbot-inspired)
        # ---------------------------------------------------------------------------
        elif name == "add_task":
            tasks_file = WORKSPACE / "tasks.json"
            if tasks_file.exists():
                data = json.loads(tasks_file.read_text())
            else:
                data = {"tasks": []}
            
            task_id = args.get("task_id")
            if task_id:
                # Update existing task status
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
            # Use the memory MCP server if available, otherwise check session files
            if MCP_ENABLED and _mcp_client is not None:
                try:
                    # Call the memory MCP server to search for entities
                    query = args.get("query", "")
                    result = execute_mcp_tool("mcp_memory_search_nodes", {"query": query})
                    if result and not result.startswith("Error"):
                        return f"Memory recall for '{query}':\n{result}"
                    # Fallback: try to read all entities
                    result = execute_mcp_tool("mcp_memory_read_graph", {})
                    if result and not result.startswith("Error"):
                        return f"Full memory graph:\n{result}"
                    return f"No memories found for '{query}'. Memory MCP may not have data yet."
                except Exception as e:
                    return f"Memory recall error: {e}. Memory MCP may not be running."
            else:
                return "Memory MCP not available. Enable MCP_ENABLED=true and configure the memory server."
        
        # ---------------------------------------------------------------------------
        # MCP Tools (Phase 2) - handled by prefix
        # ---------------------------------------------------------------------------
        elif name.startswith("mcp_"):
            return execute_mcp_tool(name, args)
        
        else:
            return f"Error: Unknown tool: {name}"
    
    except Exception as e:
        return f"Error executing {name}: {str(e)}"


# ---------------------------------------------------------------------------
# Session Management
# ---------------------------------------------------------------------------

def get_session_path(session_id: str) -> Path:
    """Get path to session file."""
    return SESSIONS_DIR / f"{session_id}.jsonl"


def load_session(session_id: str) -> list:
    """Load conversation history from session file."""
    path = get_session_path(session_id)
    if not path.exists():
        return []
    
    messages = []
    for line in path.read_text().strip().split("\n"):
        if line:
            messages.append(json.loads(line))
    return messages


def save_message(session_id: str, message: dict):
    """Append a message to the session file."""
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
    """
    # Apply backend override if set (from /claude /openai /lfm commands)
    global _backend_override
    if _backend_override:
        llm = get_llm(_backend_override)
    elif llm is None:
        llm = get_llm()

    # Handle slash commands before LLM processing
    if user_input.strip().lower() in ["/clear", "/clear-history", "/reset"]:
        try:
            # Clear all session files - use path relative to beast.py location (works on any machine)
            session_dir = Path(__file__).parent / "sessions"
            if session_dir.exists():
                for f in session_dir.glob("*.jsonl"):
                    f.unlink()
            return "🧠 Memory cleared! All conversations have been forgotten."
        except Exception as e:
            return f"❌ Error clearing history: {e}"

    if user_input.strip().lower() == "/tools":
        tools = get_all_tools()
        tool_list = ["Available tools:"]
        for t in tools:
            prefix = "[MCP] " if t["name"].startswith("mcp_") else ""
            tool_list.append(f"  {prefix}{t['name']}: {t['description'][:60]}...")
        return "\n".join(tool_list)

    if user_input.strip().lower() == "/help":
        return """🐺 **Beast Commands:**
• `/status` - Task queue & current backend
• `/claude` `/openai` `/lfm` - Switch backend
• `/clear` - Clear history
• `/tools` - List tools
• `/more` - Full help with examples

**Quick tips:** Just talk to me! Say "remind me to..." to queue a task."""

    if user_input.strip().lower() == "/more":
        return """🐺 **Obedient Beast — Full Help**

**Commands:**
• `/help` - Quick help
• `/status` - Show task queue, current backend & tier
• `/claude` - Switch to Claude (FULL tier, 10 tool calls)
• `/openai` - Switch to OpenAI (FULL tier, 10 tool calls)
• `/lfm` - Switch to local LFM (LITE tier, 2 tool calls)
• `/clear` - Clear all conversation history
• `/tools` - List all available tools

**Task Queue — say things like:**
• "remind me to check disk space"
• "add a task to organize my downloads"
• "later, review the log files"
Beast queues these for autonomous processing.

**Tools — just ask:**
• "take a screenshot" → screenshot tool
• "list files in ~/Documents" → list_dir tool
• "run ls -la" → shell tool
• "what do you remember about X?" → recall_memory tool

**Autonomous mode (separate terminal):**
Run `python heartbeat.py` — Beast processes queued tasks on a timer.

**Tiers:** Claude/OpenAI = FULL (10 tool calls, rich memory). LFM = LITE (2 tool calls, minimal memory). Auto-switches with /claude /lfm."""

    # /claude, /openai, /lfm - switch backend (works from WhatsApp AND CLI)
    if user_input.strip().lower() in ["/claude", "/openai", "/lfm"]:
        global _backend_override
        new_backend = user_input.strip().lower().lstrip("/")
        _backend_override = new_backend
        os.environ["LLM_BACKEND_TEST"] = new_backend
        # Reload capabilities for the new backend tier
        import importlib
        import capabilities
        importlib.reload(capabilities)
        from capabilities import TIER_LABEL as new_tier, MAX_TOOL_TURNS as new_max
        return f"🔄 Switched to **{new_backend}** backend. Tier: {new_tier} (max {new_max} tool turns)"

    # /status - Show task queue and capability tier (works from CLI and WhatsApp)
    if user_input.strip().lower() == "/status":
        from capabilities import TIER_LABEL, MAX_TOOL_TURNS, HEARTBEAT_INTERVAL_SEC
        tasks_file = WORKSPACE / "tasks.json"
        status_lines = [f"🐺 **Beast Status**", f"  Tier: {TIER_LABEL}", f"  Max tool turns: {MAX_TOOL_TURNS}", f"  Heartbeat: every {HEARTBEAT_INTERVAL_SEC // 60} min"]
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
            status_lines.append("  Task queue: empty (no tasks.json)")
        return "\n".join(status_lines)

    # Load history and add user message
    history = load_session(session_id)
    user_msg = {"role": "user", "content": user_input}
    history.append(user_msg)
    save_message(session_id, user_msg)
    
    # ---------------------------------------------------------------------------
    # Capability-tiered settings (Phase 4 - from capabilities.py)
    # Replaces hardcoded max_turns=10 and LFM_SINGLE_TOOL_MODE=True
    # Claude/OpenAI get full power, local LFM gets restricted mode
    # ---------------------------------------------------------------------------
    from capabilities import MAX_TOOL_TURNS, SINGLE_TOOL_MODE
    
    max_turns = MAX_TOOL_TURNS
    all_tools = get_all_tools()  # Built-in + MCP tools
    tools_used = False  # Track if we've already used a tool
    
    for turn in range(max_turns):
        # Determine which tools to send based on backend and capability tier
        if SINGLE_TOOL_MODE and tools_used:
            # LITE mode (local LFM): Don't send tools after first use
            # This forces the model to summarize the result instead of looping
            current_tools = None
        else:
            # FULL mode (Claude/OpenAI): Always send all tools
            current_tools = all_tools
        
        # Call LLM
        response = llm.chat(history, tools=current_tools, system=SYSTEM_PROMPT)
        
        if response.tool_calls:
            # Execute tools and add results to history
            for tool_call in response.tool_calls:
                print(f"  [Tool: {tool_call.name}({tool_call.args})]", file=sys.stderr)
                result = execute_tool(tool_call.name, tool_call.args)
                
                # Add assistant message with tool call
                assistant_msg = {
                    "role": "assistant",
                    "content": response.text or None,
                    "tool_calls": [{
                        "id": tool_call.id,
                        "type": "function",
                        "function": {"name": tool_call.name, "arguments": json.dumps(tool_call.args)}
                    }]
                }
                
                # For Claude, use different format
                if llm.backend == "claude":
                    assistant_msg = {
                        "role": "assistant",
                        "content": [
                            {"type": "tool_use", "id": tool_call.id, "name": tool_call.name, "input": tool_call.args}
                        ]
                    }
                    if response.text:
                        assistant_msg["content"].insert(0, {"type": "text", "text": response.text})
                
                history.append(assistant_msg)
                save_message(session_id, assistant_msg)
                
                # Add tool result
                if llm.backend == "claude":
                    tool_result_msg = {
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": tool_call.id, "content": result}]
                    }
                else:
                    tool_result_msg = {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    }
                
                history.append(tool_result_msg)
                save_message(session_id, tool_result_msg)
            
            # Mark that tools have been used (for LFM loop prevention)
            tools_used = True
        
        else:
            # No tool calls - we're done
            final_msg = {"role": "assistant", "content": response.text}
            history.append(final_msg)
            save_message(session_id, final_msg)
            # Auto-save key facts to memory MCP (Phase 4 - best-effort)
            _try_memory_save(session_id, user_input, response.text)
            return response.text
    
    return "(Max turns reached - stopping)"


# ---------------------------------------------------------------------------
# CLI Interface
# ---------------------------------------------------------------------------

def cli():
    """Interactive CLI mode."""
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
    print("Commands: /help, /more, /status, /new, /clear, /quit, /tools, /claude, /openai, /lfm")
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
