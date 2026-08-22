# Future Upgrades — OpenClaw-Inspired Features

## Shipped

### Round 1
- **LLM fallback chain** — `LLM_FALLBACK=claude,openai`
- **Scheduled / recurring tasks** — `scheduled_at`, `repeat_seconds`, cron
- **Context trimming with summary**, startup memory recall, `/image` vision

### Round 2
- **Loop detection**, `BOOT.md`, `spawn_agent`, BM25+decay memory, atomic fact save

### Qwen3.8 local-agent pass
- Multi-tool turns, tool groups, Qwen thinking knobs as `chat_template_kwargs`, local-first setup, heartbeat auto-complete one-shots

### Local harness (model-agnostic, measured on 27B)
Shared code: `local_harness.py`. Default weights remain **`LFM_MODEL`** (today Qwen3.8-27B-class) — do not branch the agent loop on a model name.

- Native `apply_chat_template` roles + `tools=`; flatten only if there is no template. No last-6 cut, no "summarize after tool".
- One parser (`parse_tool_calls` / `clean_tool_calls_from_text` including orphan tags) used by both servers and `llm.py`.
- Family presets by folder substring (Qwen3.8 / Qwen / Gemma / GLM / default). `QWEN_*` env names stay as aliases.
- Server `max_tokens` default `LFM_MAX_TOKENS` (16384). `start.sh` launches `linux_thinking.py` on Linux. `LFM_URL_REMOTE` empty unless set.
- OpenAI tool schemas: int/bool where obvious; `Optional:` params not in `required`. `MCP_ENABLED` defaults **false**; handshake timeout `MCP_RPC_TIMEOUT` (20s).
- Evals: `python test_client.py --eval-parse` (offline) and `--eval` (live against whatever is on `:8000`).

## Optional later

- **Streaming / chunked WhatsApp** — send text as it generates
- **Vector memory** — embeddings on top of BM25
- **Dead flags** — `SEQUENTIAL_THINKING_ENABLED` is defined and never read
- **Expand tool groups on demand** — keep desktop/MCP-mgmt off a 27B until asked

## Won't add

WhatsApp is the messaging channel. **Do not add** iMessage, extra chat bridges, voice notes, or webhook ingestion. Skills registry and parallel `spawn_agents` stay out unless explicitly requested.
