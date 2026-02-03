#!/usr/bin/env python3
"""
MCP Client - Connect to Model Context Protocol servers
======================================================
Allows Beast to use ANY MCP server's tools locally.

MCP servers communicate via JSON-RPC over stdio.
This client spawns servers as subprocesses and manages communication.
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
    server: str
    name: str
    description: str
    input_schema: dict


@dataclass
class MCPServer:
    """Represents a running MCP server."""
    name: str
    command: list
    process: Optional[subprocess.Popen] = None
    tools: list[MCPTool] = field(default_factory=list)
    request_id: int = 0
    response_queue: queue.Queue = field(default_factory=queue.Queue)
    reader_thread: Optional[threading.Thread] = None


class MCPClient:
    """
    Client for managing multiple MCP servers.
    
    Usage:
        client = MCPClient()
        client.start_servers()
        tools = client.list_tools()  # Get all tools from all servers
        result = client.call_tool("filesystem", "read_file", {"path": "/etc/hosts"})
    """
    
    def __init__(self, config_file: Path = MCP_SERVERS_FILE):
        self.config_file = config_file
        self.servers: dict[str, MCPServer] = {}
        
    def load_config(self) -> dict:
        """Load MCP server configuration."""
        if not self.config_file.exists():
            return {"servers": {}}
        return json.loads(self.config_file.read_text())
    
    def save_config(self, config: dict):
        """Save MCP server configuration."""
        self.config_file.write_text(json.dumps(config, indent=2))
    
    def start_servers(self):
        """Start all configured MCP servers."""
        config = self.load_config()
        
        for name, server_config in config.get("servers", {}).items():
            if not server_config.get("enabled", True):
                continue
            try:
                self._start_server(name, server_config)
                print(f"[MCP] Started server: {name}")
            except Exception as e:
                print(f"[MCP] Failed to start {name}: {e}")
    
    def _start_server(self, name: str, config: dict):
        """Start a single MCP server."""
        command = config["command"]
        if isinstance(command, str):
            command = command.split()
        
        # Add any environment variables
        env = os.environ.copy()
        for key, value in config.get("env", {}).items():
            env[key] = value
        
        # Start the server process
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            bufsize=1
        )
        
        server = MCPServer(name=name, command=command, process=process)
        
        # Start reader thread to handle responses
        def read_responses():
            while True:
                try:
                    line = process.stdout.readline()
                    if not line:
                        break
                    response = json.loads(line)
                    server.response_queue.put(response)
                except Exception as e:
                    if process.poll() is not None:
                        break
                    print(f"[MCP] {name} read error: {e}")
        
        server.reader_thread = threading.Thread(target=read_responses, daemon=True)
        server.reader_thread.start()
        
        self.servers[name] = server
        
        # Initialize the connection
        self._initialize(name)
        
        # Get available tools
        self._discover_tools(name)
    
    def _send_request(self, server_name: str, method: str, params: dict = None) -> dict:
        """Send a JSON-RPC request to an MCP server and wait for response."""
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
        
        # Send request
        server.process.stdin.write(json.dumps(request) + "\n")
        server.process.stdin.flush()
        
        # Wait for response with matching ID
        timeout = 30
        try:
            response = server.response_queue.get(timeout=timeout)
            return response
        except queue.Empty:
            raise TimeoutError(f"No response from {server_name} after {timeout}s")
    
    def _initialize(self, server_name: str):
        """Initialize MCP server connection."""
        response = self._send_request(server_name, "initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "obedient-beast",
                "version": "1.0.0"
            }
        })
        
        # Send initialized notification
        server = self.servers[server_name]
        notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }
        server.process.stdin.write(json.dumps(notification) + "\n")
        server.process.stdin.flush()
    
    def _discover_tools(self, server_name: str):
        """Discover available tools from an MCP server."""
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
        Returns tools in the format Beast expects.
        """
        tools = []
        for mcp_tool in self.list_tools():
            # Extract parameter descriptions from JSON Schema
            params = {}
            properties = mcp_tool.input_schema.get("properties", {})
            for param_name, param_info in properties.items():
                params[param_name] = param_info.get("description", f"Parameter: {param_name}")
            
            tools.append({
                "name": f"mcp_{mcp_tool.server}_{mcp_tool.name}",
                "description": f"[MCP:{mcp_tool.server}] {mcp_tool.description}",
                "params": params,
                "_mcp_server": mcp_tool.server,
                "_mcp_tool": mcp_tool.name
            })
        return tools
    
    def call_tool(self, server_name: str, tool_name: str, arguments: dict) -> str:
        """Call a tool on an MCP server and return the result."""
        response = self._send_request(server_name, "tools/call", {
            "name": tool_name,
            "arguments": arguments
        })
        
        result = response.get("result", {})
        
        # Handle different content types
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
        """Stop all running MCP servers."""
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
# Convenience Functions
# ---------------------------------------------------------------------------

_client: Optional[MCPClient] = None


def get_mcp_client() -> MCPClient:
    """Get the singleton MCP client instance."""
    global _client
    if _client is None:
        _client = MCPClient()
    return _client


def init_mcp():
    """Initialize MCP client and start configured servers."""
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
    
    Args:
        full_name: Tool name in format "mcp_servername_toolname"
        args: Arguments to pass to the tool
    
    Returns:
        Tool execution result as string
    """
    client = get_mcp_client()
    
    # Parse the tool name: mcp_servername_toolname
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
    # Test MCP client
    print("Testing MCP Client...")
    
    # Check if config exists
    if not MCP_SERVERS_FILE.exists():
        print(f"No config file at {MCP_SERVERS_FILE}")
        print("Create one with server definitions first.")
    else:
        client = init_mcp()
        tools = client.list_tools()
        print(f"Loaded {len(tools)} tools from {len(client.servers)} servers")
        for tool in tools:
            print(f"  - {tool.server}/{tool.name}: {tool.description[:50]}...")
