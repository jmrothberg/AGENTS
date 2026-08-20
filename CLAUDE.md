# CLAUDE.md

Guidance for LLMs working in this repository. Human setup lives in [README.md](README.md).

## What this repo is

1. **Local LLM servers** — `lfm_thinking.py` (macOS/MLX), `linux_thinking.py` (Linux/transformers). OpenAI-compatible API on `:8000`.
2. **Obedient Beast** — `obedient_beast/beast.py` agent loop with CLI + WhatsApp. No other chat channels.

**lfm = local model** (legacy LiquidAI name). Works with Qwen, GLM, Llama, etc. Default run: Qwen3.8-27B via `LLM_BACKEND=lfm`.

## Run

```bash
# from repo root
python lfm_thinking.py --model Qwen3.8-27B-mxfp8 --server
cd obedient_beast && python beast.py          # CLI
cd obedient_beast && ./start.sh cli           # model + CLI
cd obedient_beast && ./start.sh               # WhatsApp stack
```

Setup: `obedient_beast/setup.sh` (or `--no-node`). Canonical `.env` is **repo root**.

## Data flow

```
User (CLI/WhatsApp/HTTP) → beast.run() → llm.py (claude|openai|lfm) → execute_tool() → loop
```

## Key files

**Root:** `lfm_thinking.py`, `linux_thinking.py`, `test_client.py`, `.env.example`

**`obedient_beast/`:**
- `beast.py` — loop, 30 built-in tools, slash commands, memory
- `llm.py` — three backends; local dual-parses native `tool_calls` + ```tool_call``` / `<tool_call>`
- `capabilities.py` — depth (cloud 10 / local 8), tool groups
- `server.py` — Flask `/message` for WhatsApp
- `heartbeat.py` — `workspace/tasks.json`; one-shots marked done in code
- `mcp_client.py` — MCP stdio; tools named `mcp_<server>_<tool>`
- `workspace/SOUL.md` / `AGENTS.md` — system prompt layers (skills index added in code; tool schemas go via the API tools array, not a prompt dump)

## Tools

Handlers stay in `execute_tool()`. What the model *sees* is filtered by `BEAST_TOOL_GROUPS` (local default `core,browser,art`; cloud `all`; `/tools all` restores everything). MCP tools pass through if `MCP_ENABLED=true`.

## Config

`load_beast_env()` reads repo-root `.env`, then `obedient_beast/.env`, then cwd.

- `LLM_BACKEND` / `LLM_BACKEND_TEST` — `lfm` | `openai` | `claude`
- `LFM_URL` — `http://localhost:8000` (no `/v1`)
- `QWEN_REASONING_EFFORT`, `QWEN_ENABLE_THINKING`, `QWEN_PRESERVE_THINKING`
- `BRAVE_API_KEY` from env (not from `mcp_servers.json`)

Sessions: `obedient_beast/sessions/*.jsonl`. Memory: `workspace/memory.json` (atomic facts only).

## Conventions

- Do not add new chat channels. WhatsApp is enough.
- Do not rename `lfm_thinking.py` / `/lfm`.
- Keep changes surgical. Prefer existing functions over new modules.
- `/depth N` caps tool-chain steps (1–20).
