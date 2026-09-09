# Obedient Beast

CLI + WhatsApp agent: tools, heartbeat, local memory.

**How to start:** [repo-root README](../README.md). Repo-root `.env` is canonical. `lfm` = any model on `:8000` (`LFM_MODEL`, default Qwen3.8-27B-class).

```bash
./setup.sh          # once
./start.sh cli      # brain + You:
./start.sh phone    # brain + mailbox + WhatsApp
./start.sh you      # You: only (if phone already up)
./start.sh          # full 5-window stack
./start.sh stop
./start.sh status
```

`phone` / `you` force `LFM_URL=http://localhost:8000` and `MCP_ENABLED=false`.

**Local client** = `beast.py` (`You:`). Not the brain (`lfm_thinking.py`), not `test_client.py`, not `local_harness.py`.

## For LLMs

- Loop: `beast.run()` — history → LLM → tools (sequential) → up to `DEPTH`
- Tools: `execute_tool()`; groups via `/tools`
- LLM: `llm.py` — `claude` | `openai` | `lfm`
- Memory: `workspace/memory.json`; sessions `sessions/*.jsonl` (`cli_*`, `wa_<phone>`)
- WhatsApp path: `bridge.js` → `server.py` `/message` → `run()` — watch `[In]`/`[Blocked]`/`[Out]` in Beast Server

## Slash commands

| Command | Description |
|---------|-------------|
| `/help` `/more` `/status` | Help / status |
| `/tools` `/tools all` | Tool groups |
| `/claude` `/openai` `/lfm` | Backend |
| `/depth N` | Tool steps (1–20) |
| `/model [name]` | List / hot-swap local model |
| `/tasks` `/done N` `/drop N` | Task queue |
| `/heartbeat on\|off` | Autopilot |
| `/skills` | MCP catalog |
| `/clear` `/new` `/quit` | State / CLI |

Markdown skills: `workspace/skills/<name>/SKILL.md` → `list_skills` / `use_skill`.

## WhatsApp (OWNER)

| Command | Effect |
|---------|--------|
| `!openbeast` | Anyone in that group can `@beast` |
| `!closebeast` | Revoke |
| `!listbeast` | List open groups |

`ALLOWED_NUMBERS` / `ALLOWED_GROUPS` / `RESPOND_TO_OTHERS` still apply. Open groups: `workspace/open_chats.json`.

## Other

- Heartbeat: `workspace/tasks.json`; one-shots marked done in code after run
- MCP: `MCP_ENABLED=true`; catalog `config/mcp_servers.json`
- Art: FLUX on macOS, Z-Image on Linux; PNGs in repo-root `generated_art/`
- BOOT: `workspace/BOOT.md` once per day on CLI launch
