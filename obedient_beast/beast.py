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
# Configuration
# ---------------------------------------------------------------------------

WORKSPACE = Path(__file__).parent / "workspace"
SESSIONS_DIR = Path(__file__).parent / "sessions"
SESSIONS_DIR.mkdir(exist_ok=True)

# Load personality from SOUL.md if it exists
def load_system_prompt() -> str:
    soul_file = WORKSPACE / "SOUL.md"
    base_prompt = """You are Obedient Beast, a helpful AI assistant that can execute commands and manage files.
You have access to tools to help the user. Use them when needed.
Be concise and helpful. When executing commands, explain what you're doing."""
    
    if soul_file.exists():
        return soul_file.read_text() + "\n\n" + base_prompt
    return base_prompt

SYSTEM_PROMPT = load_system_prompt()

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
    if llm is None:
        llm = get_llm()
    
    # Load history and add user message
    history = load_session(session_id)
    user_msg = {"role": "user", "content": user_input}
    history.append(user_msg)
    save_message(session_id, user_msg)
    
    max_turns = 10
    for turn in range(max_turns):
        # Call LLM
        response = llm.chat(history, tools=TOOLS, system=SYSTEM_PROMPT)
        
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
        
        else:
            # No tool calls - we're done
            final_msg = {"role": "assistant", "content": response.text}
            history.append(final_msg)
            save_message(session_id, final_msg)
            return response.text
    
    return "(Max turns reached - stopping)"


# ---------------------------------------------------------------------------
# CLI Interface
# ---------------------------------------------------------------------------

def cli():
    """Interactive CLI mode."""
    from llm import BACKEND
    
    print("=" * 60)
    print("🐺 Obedient Beast - AI Assistant")
    print(f"   Backend: {BACKEND}")
    print("=" * 60)
    print("Type your message. Commands: /new (reset), /quit (exit)")
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
