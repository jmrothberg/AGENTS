# Obedient Beast

A minimal agentic assistant with tool calling, autonomous task queue, persistent memory, and multiple interfaces (CLI, WhatsApp, HTTP).

## For LLMs landing in this repo (60-second onboarding)

- **Entry point:** `beast.py` → `cli()` (line ~1624). HTTP entry: `server.py`. Heartbeat entry: `heartbeat.py`.
- **Agent loop:** `beast.run()` — load history → call LLM → execute tools → repeat up to `DEPTH` turns → save → return text.
- **Tools live in** `beast.py`: the `TOOLS` list defines schema; `execute_tool()` implements them. To add a tool, append a dict to `TOOLS` *and* add an `elif name == "foo":` branch in `execute_tool()`.
- **LLM abstraction:** `llm.py` exposes `get_llm(backend)` returning a client with `.chat(history, tools, system)`. Supports `claude`, `openai`, `lfm` (any OpenAI-compatible local server).
- **Memory:** `workspace/memory.json` (flat facts, BM25+decay search) is primary. MCP knowledge graph is optional + ephemeral. Auto-save happens after every turn via `_try_memory_save()`.
- **Slash commands:** handled inside `run()` before the LLM sees input — `if cmd == "/foo": return ...`. They cost 0 tokens.
- **Sessions:** `sessions/<id>.jsonl`, one JSON message per line. CLI sessions are `cli_YYYYMMDD_HHMMSS`; WhatsApp sessions are `wa_<phone>`.
- **Depth:** `capabilities.DEPTH` caps tool-chain length per request (cloud=10, local=5, `/depth N` at runtime).

## Running Beast

Beast has **5 processes**. Three run headless under pm2, two run interactively in terminal windows (because they need you to pick a model or type at them).

| Process | Runner | Why |
|---|---|---|
| `beast-server` (server.py) | **pm2** | Flask HTTP endpoint for WhatsApp — headless |
| `beast-heartbeat` (heartbeat.py) | **pm2** | Background task scheduler — headless |
| `whatsapp-bridge` (bridge.js) | **pm2** | Baileys WhatsApp forwarder — headless |
| `lfm_thinking.py` | **terminal** | Interactive model picker + model load output |
| `beast.py` (CLI) | **terminal** | Interactive — you type, it responds |

**One command starts everything:**
```bash
cd /Users/jonathanrothberg/Agents/obedient_beast && ./start.sh
```
`start.sh` boots the 3 pm2 services and opens terminal windows for `lfm_thinking.py` and `beast.py`.

### Running pieces manually

All commands use the full venv path so you can copy-paste from any directory:

```bash
# Interactive — local LLM model picker (Terminal 1)
cd /Users/jonathanrothberg/Agents && \
  /Users/jonathanrothberg/Agents/.venv/bin/python3 lfm_thinking.py

# Interactive — Beast CLI (Terminal 2)
cd /Users/jonathanrothberg/Agents/obedient_beast && \
  /Users/jonathanrothberg/Agents/.venv/bin/python3 beast.py

# pm2 background services (one-time registration)
pm2 start /Users/jonathanrothberg/Agents/obedient_beast/server.py \
    --name beast-server \
    --interpreter /Users/jonathanrothberg/Agents/.venv/bin/python3 \
    --cwd /Users/jonathanrothberg/Agents/obedient_beast

pm2 start /Users/jonathanrothberg/Agents/obedient_beast/heartbeat.py \
    --name beast-heartbeat \
    --interpreter /Users/jonathanrothberg/Agents/.venv/bin/python3 \
    --cwd /Users/jonathanrothberg/Agents/obedient_beast

pm2 start /Users/jonathanrothberg/Agents/obedient_beast/whatsapp/bridge.js \
    --name whatsapp-bridge \
    --cwd /Users/jonathanrothberg/Agents/obedient_beast/whatsapp

pm2 save   # persist across reboots
```

### Restart after a code change

```bash
# Python code changes — restart the pm2 services that load beast.py
pm2 restart beast-server beast-heartbeat

# Interactive processes — just exit and relaunch in their terminal:
# Ctrl-C in the beast.py terminal, then rerun the Terminal 2 command above.
# lfm_thinking.py only needs a restart if YOU changed lfm_thinking.py itself.

# WhatsApp bridge — only restart if whatsapp/bridge.js changed
pm2 restart whatsapp-bridge
```

### Day-to-day pm2 commands

```bash
pm2 status                     # see all 3 background services
pm2 logs                       # tail every service live
pm2 logs beast-server          # tail one service
pm2 logs --err --lines 30 --nostream   # last 30 error lines, no tail
```

First-time install: `./setup.sh` (installs Python + Node deps). More pm2 detail + troubleshooting in `../PM2_SETUP.md`.

## What's New — 5 More OpenClaw Power Features

| Feature | What it does |
|---|---|
| **Loop detection** | Agent bails automatically if the same tool call repeats 3 times or the same error twice — no more burned depth budgets |
| **`BOOT.md` startup** | Drop a `workspace/BOOT.md` file and Beast runs it once per day on launch (and on `/boot`) — your personal daily-driver routine |
| **`spawn_agent` tool** | Run subtasks in isolated sessions with fresh context windows — parent only sees the final answer |
| **BM25 memory search** | `recall_memory` now uses BM25 + temporal decay instead of substring match — recent + relevant facts win |
| **Smarter memory auto-save** | Extracts atomic facts from responses, dedupes by fingerprint, tags each fact with a category (`preference`/`project`/`people`/`decision`/`conversation`) |

Previous round (already shipped): LLM fallback chain, scheduled/recurring tasks, context trimming with summary, startup memory recall, `/image` vision input.

## Sandbox — `run_python` and `run_html`

Beast can generate a program and immediately run it. Two tools handle the full cycle: write → execute → return results.

| Scenario | Beast sees? | CLI user sees? | WhatsApp user sees? |
|---|---|---|---|
| `run_python` → text output | ✅ stdout/stderr | ✅ printed in reply | ✅ sent in message |
| `run_python` → creates `.png` plot | ✅ file listed | ✅ file path shown | ✅ **image auto-sent** |
| `run_html` → HTML page | ✅ file path | ✅ opens in browser | ✅ **auto-screenshot sent** |

- **Python scripts** run in `workspace/sandbox/py_<timestamp>/` with Beast's own venv (numpy, matplotlib, etc. available). Timeout: 30s default, 120s max.
- **HTML pages** are saved to `workspace/sandbox/html_<timestamp>/`, opened in the default browser, AND auto-screenshotted via Playwright so WhatsApp users see a rendered preview.
- `/sandbox` lists recent runs with their output files.

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
(3 backends)  (19 tools)   (MCP servers)
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
| `/boot` | Run `workspace/BOOT.md` startup routine on demand |

## BOOT.md — Your Startup Routine

Drop a markdown file at `workspace/BOOT.md` and Beast will execute it once per day the first time CLI launches. Use it for daily checkups, context priming, or any standing orders.

```markdown
# BOOT — runs once per day
1. Check disk space with the shell tool.
2. Pull any git repos under ~/code that have remote changes.
3. Recall anything in memory about today's planned work and summarize it.
```

Run on demand with `/boot`. A copy-pastable starter is at `workspace/BOOT.md.example`.

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
