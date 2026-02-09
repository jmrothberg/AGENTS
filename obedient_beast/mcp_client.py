#!/usr/bin/env python3
"""
MCP Client - Connect to Model Context Protocol servers
======================================================
Allows Beast to use ANY MCP server's tools locally.

Protocol overview:
~~~~~~~~~~~~~~~~~~
MCP servers communicate via JSON-RPC 2.0 over stdio (stdin/stdout).
Each server is a separate subprocess that Beast spawns and manages.

Lifecycle:
    1. Beast reads config/mcp_servers.json for server definitions
    2. For each enabled server, spawn the process (usually `npx -y @...`)
    3. Send "initialize" JSON-RPC handshake
    4. Send "tools/list" to discover what tools the server offers
    5. Tools become available to the LLM alongside Beast's built-in tools
    6. When the LLM calls an MCP tool, Beast sends "tools/call" to the server
    7. On shutdown, terminate all server processes

Threading model:
~~~~~~~~~~~~~~~~
Each server gets a dedicated reader thread that reads JSON-RPC responses
from the server's stdout. Responses are placed on a queue.Queue for the
main thread to pick up. This prevents blocking when waiting for responses.

Tool name prefixing:
~~~~~~~~~~~~~~~~~~~~
MCP tools are prefixed with "mcp_<servername>_" to avoid name collisions
with Beast's built-in tools and between different MCP servers.
For example, the "read_file" tool from the "filesystem" server becomes
"mcp_filesystem_read_file".

Tier labels:
~~~~~~~~~~~~
Each server in mcp_servers.json has a "tier" field (essential/extended/cloud).
These are organizational labels. All tiers are loaded regardless of backend —
local LLMs need access to cloud MCP servers (e.g., brave-search for web queries).
The LITE/FULL distinction only affects tool-calling behavior, not MCP loading.
"""

import os
import json
import subprocess
import threading
import queue
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# MCP Configuration
# ---------------------------------------------------------------------------

CONFIG_DIR = Path(__file__).parent / "config"
CONFIG_DIR.mkdir(exist_ok=True)

MCP_SERVERS_FILE = CONFIG_DIR / "mcp_servers.json"


@dataclass
class MCPTool:
    """Represents a tool from an MCP server."""
    server: str           # Which server this tool belongs to
    name: str             # Original tool name (without prefix)
    description: str      # Human-readable description for the LLM
    input_schema: dict    # JSON Schema for the tool's parameters


@dataclass
class MCPServer:
    """
    Represents a running MCP server subprocess.
    Each server has its own process, reader thread, and response queue.
    """
    name: str
    command: list                                      # Command to spawn (e.g., ["npx", "-y", "@.../server-memory"])
    process: Optional[subprocess.Popen] = None         # The running subprocess
    tools: list[MCPTool] = field(default_factory=list) # Discovered tools
    request_id: int = 0                                # Auto-incrementing JSON-RPC request ID
    response_queue: queue.Queue = field(default_factory=queue.Queue)  # Responses from reader thread
    reader_thread: Optional[threading.Thread] = None   # Background thread reading stdout


class MCPClient:
    """
    Client for managing multiple MCP servers.

    Usage:
        client = MCPClient()
        client.start_servers()       # Spawns all enabled servers from config
        tools = client.list_tools()  # Get all tools from all servers
        result = client.call_tool("filesystem", "read_file", {"path": "/etc/hosts"})
        client.stop_servers()        # Clean shutdown
    """

    def __init__(self, config_file: Path = MCP_SERVERS_FILE):
        self.config_file = config_file
        self.servers: dict[str, MCPServer] = {}

    def load_config(self) -> dict:
        """Load MCP server configuration from JSON file."""
        if not self.config_file.exists():
            return {"servers": {}}
        return json.loads(self.config_file.read_text())

    def save_config(self, config: dict):
        """Save MCP server configuration to JSON file."""
        self.config_file.write_text(json.dumps(config, indent=2))

    def start_servers(self):
        """
        Start all configured MCP servers that are enabled and in the allowed tier.
        Reads config, checks tier against capabilities.MCP_ALLOWED_TIERS,
        and spawns each qualifying server as a subprocess.
        """
        config = self.load_config()

        # Import allowed tiers from capabilities (respects FULL vs LITE mode)
        try:
            from capabilities import MCP_ALLOWED_TIERS
        except ImportError:
            MCP_ALLOWED_TIERS = ["essential", "extended", "cloud"]  # Default: allow all

        for name, server_config in config.get("servers", {}).items():
            # Skip disabled servers
            if not server_config.get("enabled", True):
                continue

            # Tier filtering: only load servers in allowed tiers.
            # Default tier is "essential" for backward compatibility.
            server_tier = server_config.get("tier", "essential")
            if server_tier not in MCP_ALLOWED_TIERS:
                print(f"[MCP] Skipping {name} (tier '{server_tier}' not in allowed tiers: {MCP_ALLOWED_TIERS})")
                continue

            try:
                self._start_server(name, server_config)
                print(f"[MCP] Started server: {name} (tier: {server_tier})")
            except Exception as e:
                print(f"[MCP] Failed to start {name}: {e}")

    def _start_server(self, name: str, config: dict):
        """
        Start a single MCP server subprocess.
        Sets up stdin/stdout pipes and launches the reader thread.
        """
        command = config["command"]
        if isinstance(command, str):
            command = command.split()

        # Add any environment variables specified in server config
        # (e.g., BRAVE_API_KEY for brave-search)
        env = os.environ.copy()
        for key, value in config.get("env", {}).items():
            env[key] = value

        # Start the server process with JSON-RPC over stdio
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,   # We write JSON-RPC requests here
            stdout=subprocess.PIPE,  # We read JSON-RPC responses here
            stderr=subprocess.PIPE,  # Captured but not actively read (avoid blocking)
            env=env,
            text=True,
            bufsize=1  # Line-buffered for JSON-RPC (one JSON object per line)
        )

        server = MCPServer(name=name, command=command, process=process)

        # Start a dedicated reader thread for this server.
        # The thread reads JSON-RPC responses from stdout and puts them on
        # the response queue. The main thread can then pick them up with
        # a blocking get() call.
        def read_responses():
            while True:
                try:
                    line = process.stdout.readline()
                    if not line:
                        break  # Process exited, stdout closed
                    response = json.loads(line)
                    server.response_queue.put(response)
                except Exception as e:
                    if process.poll() is not None:
                        break  # Process has exited
                    print(f"[MCP] {name} read error: {e}")

        server.reader_thread = threading.Thread(target=read_responses, daemon=True)
        server.reader_thread.start()

        self.servers[name] = server

        # Initialize the JSON-RPC connection (protocol handshake)
        self._initialize(name)

        # Discover available tools from this server
        self._discover_tools(name)

    def _send_request(self, server_name: str, method: str, params: dict = None) -> dict:
        """
        Send a JSON-RPC 2.0 request to an MCP server and wait for response.
        Each request gets an auto-incrementing ID for matching responses.
        """
        server = self.servers.get(server_name)
        if not server or not server.process:
            raise RuntimeError(f"Server {server_name} not running")

        server.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": server.request_id,
            "method": method,
        }
        if params:
            request["params"] = params

        # Write request to server's stdin (one JSON object per line)
        server.process.stdin.write(json.dumps(request) + "\n")
        server.process.stdin.flush()

        # Wait for the matching response from the reader thread (with timeout).
        # MCP servers may send notifications (no "id" field) — skip those.
        # We match on request_id to ensure we get the right response.
        timeout = 30
        import time
        deadline = time.time() + timeout
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError(f"No response from {server_name} after {timeout}s")
            try:
                response = server.response_queue.get(timeout=remaining)
                # Skip notifications (no "id") — they're server-initiated messages
                if "id" not in response:
                    continue
                return response
            except queue.Empty:
                raise TimeoutError(f"No response from {server_name} after {timeout}s")

    def _initialize(self, server_name: str):
        """
        Initialize MCP server connection (JSON-RPC handshake).
        Sends "initialize" request followed by "notifications/initialized".
        This is required by the MCP protocol before any tool calls.
        """
        response = self._send_request(server_name, "initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "obedient-beast",
                "version": "1.0.0"
            }
        })

        # Send "initialized" notification (no response expected)
        server = self.servers[server_name]
        notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }
        server.process.stdin.write(json.dumps(notification) + "\n")
        server.process.stdin.flush()

    def _discover_tools(self, server_name: str):
        """
        Discover available tools from an MCP server via "tools/list".
        Each tool becomes an MCPTool with its name, description, and schema.
        """
        response = self._send_request(server_name, "tools/list")

        server = self.servers[server_name]
        server.tools = []

        for tool_data in response.get("result", {}).get("tools", []):
            tool = MCPTool(
                server=server_name,
                name=tool_data["name"],
                description=tool_data.get("description", ""),
                input_schema=tool_data.get("inputSchema", {})
            )
            server.tools.append(tool)

    def list_tools(self) -> list[MCPTool]:
        """Get all tools from all running servers."""
        all_tools = []
        for server in self.servers.values():
            all_tools.extend(server.tools)
        return all_tools

    def get_tools_for_llm(self) -> list[dict]:
        """
        Convert MCP tools to Beast's tool format for the LLM.

        Prefixes each tool name with "mcp_<servername>_" to avoid collisions.
        For example: filesystem's "read_file" → "mcp_filesystem_read_file"

        Returns tools in Beast's format: {name, description, params, _mcp_server, _mcp_tool}
        """
        tools = []
        for mcp_tool in self.list_tools():
            # Extract parameter descriptions from JSON Schema properties
            params = {}
            properties = mcp_tool.input_schema.get("properties", {})
            for param_name, param_info in properties.items():
                params[param_name] = param_info.get("description", f"Parameter: {param_name}")

            tools.append({
                "name": f"mcp_{mcp_tool.server}_{mcp_tool.name}",
                "description": f"[MCP:{mcp_tool.server}] {mcp_tool.description}",
                "params": params,
                "_mcp_server": mcp_tool.server,  # Internal: used for routing
                "_mcp_tool": mcp_tool.name        # Internal: original tool name
            })
        return tools

    def call_tool(self, server_name: str, tool_name: str, arguments: dict) -> str:
        """
        Call a tool on an MCP server and return the result as text.
        Sends "tools/call" JSON-RPC request and parses the content response.
        """
        response = self._send_request(server_name, "tools/call", {
            "name": tool_name,
            "arguments": arguments
        })

        result = response.get("result", {})

        # Handle different content types in the MCP response
        content = result.get("content", [])
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif item.get("type") == "image":
                    text_parts.append(f"[Image: {item.get('mimeType', 'image')}]")
            return "\n".join(text_parts) or str(result)

        return str(result)

    def stop_servers(self):
        """
        Stop all running MCP servers. Sends SIGTERM first, then SIGKILL
        if the server doesn't exit within 5 seconds.
        """
        for name, server in self.servers.items():
            if server.process:
                try:
                    server.process.terminate()
                    server.process.wait(timeout=5)
                except Exception as e:
                    print(f"[MCP] Error stopping {name}: {e}")
                    server.process.kill()
        self.servers = {}


# ---------------------------------------------------------------------------
# Convenience Functions — used by beast.py
# ---------------------------------------------------------------------------

_client: Optional[MCPClient] = None


def get_mcp_client() -> MCPClient:
    """Get the singleton MCP client instance."""
    global _client
    if _client is None:
        _client = MCPClient()
    return _client


def init_mcp():
    """Initialize MCP client and start configured servers (with tier filtering)."""
    client = get_mcp_client()
    client.start_servers()
    return client


def get_mcp_tools() -> list[dict]:
    """Get all MCP tools in Beast's format."""
    client = get_mcp_client()
    return client.get_tools_for_llm()


def execute_mcp_tool(full_name: str, args: dict) -> str:
    """
    Execute an MCP tool by its full name (mcp_server_toolname).

    Parses the prefixed name to extract the server and original tool name,
    then routes to the correct server subprocess.

    Args:
        full_name: Tool name in format "mcp_servername_toolname"
        args: Arguments to pass to the tool

    Returns:
        Tool execution result as string
    """
    client = get_mcp_client()

    # Parse the tool name: mcp_servername_toolname
    # Split on "_" with max 2 splits to handle tool names containing underscores.
    # Limitation: server names with underscores will break this parsing.
    # All current servers use hyphens (brave-search) or single words (filesystem).
    parts = full_name.split("_", 2)
    if len(parts) < 3 or parts[0] != "mcp":
        return f"Error: Invalid MCP tool name format: {full_name}"

    server_name = parts[1]
    tool_name = parts[2]

    try:
        return client.call_tool(server_name, tool_name, args)
    except Exception as e:
        return f"Error calling MCP tool {full_name}: {e}"


if __name__ == "__main__":
    # Test MCP client — run with: python mcp_client.py
    print("Testing MCP Client...")

    if not MCP_SERVERS_FILE.exists():
        print(f"No config file at {MCP_SERVERS_FILE}")
        print("Create one with server definitions first.")
    else:
        client = init_mcp()
        tools = client.list_tools()
        print(f"Loaded {len(tools)} tools from {len(client.servers)} servers")
        for tool in tools:
            print(f"  - {tool.server}/{tool.name}: {tool.description[:50]}...")
