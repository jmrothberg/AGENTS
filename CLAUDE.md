# CLAUDE.md

LLMs editing this repo. Humans: [README.md](README.md).

## Repo

1. **Brain** — `lfm_thinking.py` (macOS/MLX) or `linux_thinking.py` (Linux). API `:8000`.
2. **Obedient Beast** — `obedient_beast/beast.py` (CLI + WhatsApp only).

`lfm` = local model. Default backend `LLM_BACKEND=lfm`. Default weights `LFM_MODEL` (today `Qwen3.8-27B-mxfp8`). Do not hardcode a model name in the agent loop.

## Run

```bash
# repo root
python lfm_thinking.py --model Qwen3.8-27B-mxfp8 --server
cd obedient_beast && ./start.sh cli      # brain + You:
cd obedient_beast && ./start.sh phone    # brain + mailbox + WhatsApp
cd obedient_beast && ./start.sh          # full stack
cd obedient_beast && ./start.sh stop
```

`start.sh` picks `lfm_thinking.py` on macOS, `linux_thinking.py` on Linux.

## Flow

```
CLI/WhatsApp → beast.run() → llm.py → execute_tool() → loop
WhatsApp: bridge.js → server.py :5001 → run()
Local:    llm.py POST → brain :8000
```

`local_harness.py` is imported by the brain and `llm.py` — not a process.

## Key files

Root: `lfm_thinking.py`, `linux_thinking.py`, `local_harness.py`, `test_client.py`, `.env.example`

`obedient_beast/`: `beast.py`, `llm.py`, `capabilities.py`, `server.py`, `heartbeat.py`, `mcp_client.py`, `skills_loader.py`, `browser_tools.py`, `whatsapp/bridge.js`, `workspace/SOUL.md`

## Config

`load_beast_env()`: repo-root `.env`, then `obedient_beast/.env`, then cwd.

- `LFM_MODEL` — `start.sh` folder substring (default `Qwen3.8-27B-mxfp8`)
- `LFM_URL` — `http://localhost:8000` (no `/v1`)
- `QWEN_*` — chat_template_kwargs; other models ignore unknown fields
- `MCP_ENABLED` — default **false**; `phone`/`you` also force false
- WhatsApp: `ALLOWED_NUMBERS`, `ALLOWED_GROUPS`, `RESPOND_TO_OTHERS`

## Conventions

- No new chat channels.
- Do not rename `lfm_thinking.py` / `/lfm`.
- Keep default model in env (`LFM_MODEL`), not branched in the loop.
- Surgical changes; prefer existing functions.
- `/depth N` caps tool steps (1–20).
