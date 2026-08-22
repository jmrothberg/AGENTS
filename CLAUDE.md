# CLAUDE.md

Guidance for LLMs working in this repository. Human setup lives in [README.md](README.md).

## What this repo is

1. **Local LLM servers** — `lfm_thinking.py` (macOS/MLX), `linux_thinking.py` (Linux/transformers). OpenAI-compatible API on `:8000`.
2. **Obedient Beast** — `obedient_beast/beast.py` agent loop with CLI + WhatsApp. No other chat channels.

**lfm = local model** (legacy LiquidAI name). Works with Qwen, GLM, Llama, etc. Default backend: `LLM_BACKEND=lfm`. Default weights folder: `LFM_MODEL` (today `Qwen3.8-27B-mxfp8`). Do not hardcode a model name in the agent loop.

## Run

```bash
# from repo root
python lfm_thinking.py --model Qwen3.8-27B-mxfp8 --server   # macOS brain
# python linux_thinking.py --model latest --server          # Linux twin
cd obedient_beast && python beast.py          # CLI (needs a brain)
cd obedient_beast && ./start.sh cli           # macOS brain + CLI
cd obedient_beast && ./start.sh phone         # brain + mailbox + WhatsApp
cd obedient_beast && ./start.sh               # 5-window full stack
```

`start.sh` launches `lfm_thinking.py` on macOS and `linux_thinking.py` on Linux.

## Data flow

```
User (CLI/WhatsApp/HTTP) → beast.run() → llm.py (claude|openai|lfm) → execute_tool() → loop
```

Local extra hop: `llm.py` POST `/v1/chat/completions` → brain `:8000`. `local_harness.py` is imported by the brain and by `llm.py` — **not a process**. Human diagram: [README.md](README.md) Architecture.

## Key files

**Root:** `lfm_thinking.py`, `linux_thinking.py`, `local_harness.py`, `test_client.py`, `flux_art.py`, `zimage_art_linux.py`, `.env.example`

**`obedient_beast/`:**
- `beast.py` — loop, 30 built-in tools, slash commands, memory
- `llm.py` — three backends; local dual-parses native `tool_calls` + ```tool_call``` / `<tool_call>`
- `capabilities.py` — depth (cloud 10 / local 8), tool groups
- `server.py` — Flask `/message` for WhatsApp (`BEAST_PORT`, default 5001)
- `heartbeat.py` — `workspace/tasks.json`; one-shots marked done in code
- `mcp_client.py` — MCP stdio; tools named `mcp_<server>_<tool>`
- `skills_loader.py` — `workspace/skills/<name>/SKILL.md`
- `browser_tools.py` — Playwright; profile in `workspace/browser_profile/`
- `workspace/SOUL.md` / `AGENTS.md` — system prompt layers (skills index added in code; tool schemas go via the API tools array, not a prompt dump)

## Tools

Handlers stay in `execute_tool()`. What the model *sees* is filtered by `BEAST_TOOL_GROUPS` (local default `core,browser,art`; cloud `all`; `/tools all` restores everything). MCP tools pass through if MCP is enabled.

`/skills` is the MCP catalog. Markdown runbooks are `list_skills` / `use_skill`.

## Config

`load_beast_env()` reads repo-root `.env`, then `obedient_beast/.env`, then cwd.

- `LLM_BACKEND` / `LLM_BACKEND_TEST` — `lfm` | `openai` | `claude`
- `LFM_URL` — `http://localhost:8000` (no `/v1`); `LFM_URL_REMOTE` is a fallback
- `LFM_MODEL` — `start.sh` folder substring (default `Qwen3.8-27B-mxfp8`)
- `QWEN_REASONING_EFFORT` (`low`/`medium`/`xhigh`), `QWEN_ENABLE_THINKING`, `QWEN_PRESERVE_THINKING` — packed into `chat_template_kwargs`; other models ignore unknown fields
- `MCP_ENABLED` — defaults **false** if unset (matches `.env.example`). `start.sh phone` and `you` also force `false`. Handshake timeout: `MCP_RPC_TIMEOUT` (20s).
- `BRAVE_API_KEY` from env (not from `mcp_servers.json`)

Sessions: `obedient_beast/sessions/*.jsonl`. Memory: `workspace/memory.json` (atomic facts only).

## Conventions

- Do not add new chat channels. WhatsApp is enough.
- Do not rename `lfm_thinking.py` / `/lfm`.
- Keep the default local model as an env/config value (`LFM_MODEL`), not a branch in the loop.
- Keep changes surgical. Prefer existing functions over new modules.
- `/depth N` caps tool-chain steps (1–20).
