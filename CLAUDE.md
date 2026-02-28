# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A multi-component AI agent system with two main parts:
1. **Local LLM inference servers** (`lfm_thinking.py` for macOS/MLX, `linux_thinking.py` for Linux/transformers) — serve any local model as an OpenAI-compatible API
2. **Obedient Beast** (`obedient_beast/`) — an AI agent with CLI, WhatsApp, and autonomous task processing

## Running the System

```bash
# Local LLM server (macOS)
python lfm_thinking.py                          # Interactive model selection
python lfm_thinking.py --model latest --server  # Headless/pm2 mode

# Beast agent CLI
cd obedient_beast && python beast.py

# Full stack (server + WhatsApp bridge + heartbeat + CLI)
cd obedient_beast && ./start.sh

# Test the local LLM server
python test_client.py
```

Setup: `cd obedient_beast && ./setup.sh` (or `./setup.sh --no-node` to skip Node.js/WhatsApp)

## Architecture

### Data Flow
```
User (CLI/WhatsApp/HTTP) → beast.run() agent loop → LLM (Cloud or Local) → Tool execution → Loop back
```

### Key Files

**Root level — Local LLM Servers:**
- `lfm_thinking.py` (~1400 lines) — macOS/MLX model server with dynamic model discovery, OpenAI-compatible API, vision/video support
- `linux_thinking.py` (~900 lines) — Linux/transformers equivalent
- `test_client.py` — Streaming test client

**`obedient_beast/` — Agent System:**
- `beast.py` (~1400 lines) — Main agent loop, 18 built-in tools, CLI interface, slash commands, local memory management. This is the core file.
- `llm.py` (~400 lines) — Unified LLM client supporting 3 backends (Claude, OpenAI, local/lfm) with dual tool-calling format (native + text parsing)
- `server.py` (~190 lines) — Flask HTTP server, only needed for WhatsApp bridge
- `heartbeat.py` (~260 lines) — Autonomous task scheduler, processes `workspace/tasks.json` on a timer
- `mcp_client.py` (~430 lines) — MCP server connection manager with 3-tier system (essential/extended/cloud)
- `capabilities.py` (~100 lines) — Cloud vs Local settings, depth control (tool-chain steps per request)
- `whatsapp/bridge.js` (~270 lines) — Node.js WhatsApp connector using Baileys library
- `config/mcp_servers.json` — MCP server catalog (11 servers across 3 tiers)
- `workspace/SOUL.md` — Agent personality and system prompt
- `workspace/AGENTS.md` — Task reasoning templates

### Three-Backend LLM Architecture (llm.py)

`llm.py` provides a unified interface across Claude (Anthropic SDK), OpenAI, and local models (lfm). The local backend communicates with `lfm_thinking.py`/`linux_thinking.py` via OpenAI-compatible HTTP API. Tool calling uses dual format: native API tool calling + text-based fallback parsing for models that don't support native tools.

### Tool System

18 built-in tools defined in `beast.py` (file ops, shell, computer control, MCP management, memory, tasks, fetch_url). MCP tools are dynamically loaded from `config/mcp_servers.json` and prefixed with `mcp_servername_` to avoid collisions. Tool definitions use Anthropic format (input_schema) internally; `llm.py` translates to OpenAI format (function) when needed.

### Configuration

Environment config via `.env` in `obedient_beast/` (see `.env.example` at root). Key vars:
- `LLM_BACKEND` — `lfm`, `openai`, or `claude`
- `LLM_BACKEND_TEST` — Override backend for a single session without modifying `.env`
- `MCP_ENABLED` — Enable/disable MCP tool servers
- `ALLOWED_NUMBERS` / `ALLOWED_GROUPS` — WhatsApp access control

### Session & State

- Conversations saved as JSONL in `obedient_beast/sessions/`
- Task queue in `workspace/tasks.json`
- Local memory in `workspace/memory.json` (auto-capped at 200 entries)
- MCP knowledge graph for persistent memory (external)

## Development Notes

- The "LFM" naming throughout (`lfm_thinking.py`, `LFM_URL`, `/lfm` command) is a legacy artifact from when the project used LiquidAI's LFM models. It now means "local model" and works with any compatible model.
- Model directories: macOS scans `/Users/jonathanrothberg/MLX_Models/`, Linux scans configurable `MODEL_SEARCH_PATHS`.
- Python venv lives at `.venv/` in the repo root. Use `.venv/bin/python3` for pm2 processes.
- Node.js is only required for WhatsApp bridge and MCP servers (npx commands).
- The `/depth N` slash command controls how many tool-call steps the agent can chain per request (default: 10 cloud, 5 local).
