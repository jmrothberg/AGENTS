# Obedient Beast

Personal agent: CLI + WhatsApp, 30 built-in tools, heartbeat tasks, local memory.

**Setup and how to start:** see the [repo-root README](../README.md). Canonical `.env` is the **repo root**. `lfm` = local model (Qwen by default).

```bash
./setup.sh          # once
./start.sh phone    # 3 windows: brain, mailbox, WhatsApp — no local client
./start.sh you      # 4th window only: local client (beast.py, You:) — safe if phone is already up
./start.sh cli      # brain + local client, no WhatsApp
./start.sh stop     # stop all
```

**Local client** = `beast.py` (type at `You:`). **Not** `lfm_thinking.py` (brain) and **not** `test_client.py` (raw model ping). Full window map: [repo-root README](../README.md).

## For LLMs (60 seconds)

- **Loop:** `beast.run()` — history → LLM → tools (several per turn, sequential) → repeat up to `DEPTH` (cloud 10, local 8).
- **Tools:** `TOOLS` + `execute_tool()` in `beast.py`. Groups filter what the model is offered (`/tools all` restores every handler).
- **LLM:** `llm.py` — `claude` | `openai` | `lfm`.
- **Memory:** `workspace/memory.json` — atomic facts only (BM25 + decay). MCP graph is optional and ephemeral.
- **Sessions:** `sessions/<id>.jsonl` — CLI `cli_*`, WhatsApp `wa_<phone>`.
- **Slash commands** are handled in `run()` before the LLM.

## Slash commands

| Command | Description |
|---------|-------------|
| `/help` `/more` | Help |
| `/status` | Backend, depth, heartbeat |
| `/tools` | List active tools |
| `/tools all` | Offer every built-in group |
| `/tools core,browser,desktop` | Set groups for this process |
| `/claude` `/openai` `/lfm` | Switch backend |
| `/depth N` | Tool-chain steps (1–20) |
| `/model [name]` | List or hot-swap local models |
| `/tasks` `/done N` `/drop N` | Task queue |
| `/heartbeat on\|off` | Autopilot |
| `/boot` `/boot install` | Daily `workspace/BOOT.md` |
| `/sandbox` `/sandbox log` | Generated code runs |
| `/skills` | MCP catalog / skill runbooks |
| `/clear` `/clear tasks` `/clear memory` `/clear all` | Wipe state |
| `/image [path]` | Attach image (CLI) |
| `/new` `/quit` | CLI only |

## Processes

Full stack is five processes: local model server (`lfm_thinking.py`), `server.py`, WhatsApp `bridge.js`, `heartbeat.py`, `beast.py` CLI. `./start.sh pm2` backgrounds server/WhatsApp/heartbeat.

After Python changes: restart the processes that import the file. `pm2 restart beast-server beast-heartbeat` plus relaunch the model server and CLI terminals.

## Heartbeat

`heartbeat.py` runs pending items in `workspace/tasks.json`. Recurring tasks (`cron` / `repeat_seconds`) reschedule. **One-shots are marked `done` in code** after `run()` returns — the model does not need to call `add_task(status=done)`.

## Memory

- **Local:** `workspace/memory.json`, cap 200, BM25 + recency. Auto-save stores atomic facts only (skips chain-of-thought dumps).
- **MCP graph:** optional, dies with the npx process. `/clear memory` wipes both.

## MCP

Enabled with `MCP_ENABLED=true`. Catalog: `config/mcp_servers.json`. Put `BRAVE_API_KEY` in repo-root `.env`, not in the JSON. Tools are named `mcp_<server>_<tool>`.

## WhatsApp group control (OWNER)

| Command | Effect |
|---------|--------|
| `!openbeast` | Anyone in that group can `@beast` |
| `!closebeast` | Revoke |
| `!listbeast` | List open groups |

Open groups persist in `workspace/open_chats.json`. `ALLOWED_NUMBERS` / `ALLOWED_GROUPS` still apply.

## BOOT.md

`workspace/BOOT.md` runs once per day on CLI launch (`/boot` to rerun). Template: `workspace/BOOT.md.example`.

## Art and sandbox

- `generate_art` — local FLUX on macOS (`../flux_art.py`), auto-sends on WhatsApp.
- `run_python` / `run_html` — `workspace/Generated Code/`. `/sandbox` lists runs.
