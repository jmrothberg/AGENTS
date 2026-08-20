# Future Upgrades — OpenClaw-Inspired Features

## Shipped

### Round 1
- **LLM fallback chain** — `LLM_FALLBACK=claude,openai`
- **Scheduled / recurring tasks** — `scheduled_at`, `repeat_seconds`, cron
- **Context trimming with summary**, startup memory recall, `/image` vision

### Round 2
- **Loop detection**, `BOOT.md`, `spawn_agent`, BM25+decay memory, atomic fact save

### Qwen3.8 local-agent pass
- Multi-tool turns, tool groups, Qwen thinking knobs, local-first setup, heartbeat auto-complete one-shots

## Optional later

- **Streaming / chunked WhatsApp** — send text as it generates
- **Vector memory** — embeddings on top of BM25

## Won't add

WhatsApp is the messaging channel. **Do not add** iMessage, extra chat bridges, voice notes, or webhook ingestion. Skills registry and parallel `spawn_agents` stay out unless explicitly requested.
