# Obedient Beast

A minimal agentic assistant with tool calling, autonomous task queue, persistent memory, and multiple interfaces (CLI, WhatsApp, HTTP).

## Quick Start

```bash
./setup.sh              # Install Python + Node dependencies
python beast.py          # Interactive CLI
./start.sh               # Full stack: server + WhatsApp + heartbeat + CLI
```

## Architecture

```
User (CLI / WhatsApp / HTTP)
        │
        ▼
    beast.run()           ← Agent loop: LLM ↔ Tools, repeats until done
        │
   ┌────┼────────────────┐
   │    │                │
   ▼    ▼                ▼
llm.py  Built-in Tools   mcp_client.py
(3 backends)  (18 tools)   (MCP servers)
```

**Files:**
- `beast.py` — Agent loop, 18 built-in tools, CLI interface, slash commands
- `llm.py` — Unified LLM client (Claude, OpenAI, local models)
- `server.py` — Flask HTTP server for WhatsApp bridge
- `heartbeat.py` — Autonomous task scheduler (processes task queue on a timer)
- `mcp_client.py` — MCP server connection manager
- `capabilities.py` — Cloud vs Local tier settings
- `whatsapp/bridge.js` — WhatsApp connector (Baileys library)

## Memory System

Beast has a two-tier memory architecture:

### 1. Local Memory (`workspace/memory.json`) — Primary, Persistent

A simple flat list of facts with timestamps. This is the reliable backbone:
- Survives restarts — stored as plain JSON on disk
- Keyword search via the `recall_memory` tool
- Capped at 200 entries (oldest removed first)
- No external dependencies

### 2. MCP Knowledge Graph — Optional, Richer, Ephemeral

Provided by `@modelcontextprotocol/server-memory`, an MCP server that runs as a subprocess.

**What it does:** Stores facts as a knowledge graph — entities (nodes) connected by typed relationships (edges). For example:
- Entity "Jonathan" --[prefers]--> "dark mode"
- Entity "Jonathan" --[works_on]--> "Obedient Beast"
- Entity "Obedient Beast" --[uses]--> "Claude API"

**Why it's useful:** Flat memory can only keyword-match individual facts. The knowledge graph *connects* facts together, enabling richer recall. When the LLM asks "what do I know about Jonathan?", it traverses edges and gets back all connected entities — preferences, projects, relationships — in one query. This makes Beast's memory more human-like: it can associate related facts rather than treating each one in isolation.

**How it works:** MCP (Model Context Protocol) servers are external tool providers that communicate with Beast via JSON-RPC 2.0 over stdio. The memory server exposes tools like `create_entities`, `create_relations`, `search_nodes`, and `delete_all_nodes`. Beast calls these just like any other tool. See `config/mcp_servers.json` for the full list of MCP servers.

**Caveat:** The knowledge graph is ephemeral — it runs in-process as an `npx` subprocess and holds state in memory only. When Beast restarts, the graph starts empty. This is acceptable because key facts are *also* saved to local memory (`memory.json`) on every conversation turn, so nothing critical is lost across restarts.

**Clearing memory:** `/clear memory` and `/clear all` wipe both the local JSON file and the MCP knowledge graph.

## LLM Backends

Three backends configured via `LLM_BACKEND` in `.env`:

| Backend | Model | Use Case |
|---------|-------|----------|
| `claude` | Claude Sonnet | Cloud, best tool calling |
| `openai` | GPT-4o | Cloud, alternative |
| `lfm` | Any local model | Local, served by `lfm_thinking.py` |

**Fallback chain:** Set `LLM_FALLBACK=claude,openai` in `.env` — if the primary backend fails, Beast automatically tries fallbacks in order.

## Slash Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/status` | Current backend, model, session info |
| `/image [path]` | Attach image (opens file dialog if no path) |
| `/claude`, `/openai`, `/lfm` | Switch LLM backend |
| `/depth N` | Set max tool-call steps per request |
| `/clear` | Clear chat history |
| `/clear memory` | Clear all memory (local + MCP graph) |
| `/clear all` | Clear everything (chat + tasks + memory) |
| `/tasks` | Show task queue |
| `/tools` | List available tools |

## MCP (Model Context Protocol)

MCP servers are external tool providers that give Beast additional capabilities. They're configured in `config/mcp_servers.json` and loaded at startup when `MCP_ENABLED=true` (default).

**Included servers:**
- **filesystem** — Read/write/search files
- **memory** — Knowledge graph (see Memory System above)
- **brave-search** — Web search via Brave API
- **git**, **sqlite**, **playwright** — Available but disabled by default

Beast can self-upgrade: the `install_mcp_server` tool lets it add new MCP servers at runtime.

## Configuration

Copy `.env.example` to `.env` in the `obedient_beast/` directory. Key settings:

```bash
LLM_BACKEND=claude          # Primary LLM backend
LLM_FALLBACK=claude,openai  # Fallback chain if primary fails
MCP_ENABLED=true            # Load MCP tool servers
ALLOWED_NUMBERS=+1234567890 # WhatsApp access control
```
