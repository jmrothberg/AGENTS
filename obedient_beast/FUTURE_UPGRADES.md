# Future Upgrades — OpenClaw-Inspired Features

## ✅ Shipped

### Round 1 (commit `654a019`)
- **LLM fallback chain** — `LLM_FALLBACK=claude,openai` env var, retries with text-only history on failure
- **Scheduled / recurring tasks** — `scheduled_at` and `repeat_seconds` on `add_task`
- **Context trimming with summary** — `_summarize_dropped_context` keeps topics + tool names when history overflows
- **Startup memory auto-recall** — injects recent facts into new sessions
- **Image input** — `/image` command, multimodal message building, graceful fallback for non-VLM models

### Round 2 (this commit)
- **Loop detection** — agent bails when the same tool call repeats 3x or the same error repeats 2x
- **`workspace/BOOT.md` startup script** — daily standing-orders file, `/boot` to rerun on demand
- **`spawn_agent` tool** — sub-agent runs in isolated session with reduced depth, returns only the final answer
- **BM25 + temporal-decay memory search** — `_search_local_memory` replaced with hybrid lexical + recency scorer (no new deps)
- **Smarter memory auto-save** — atomic fact extraction, fingerprint dedup, category tagging (preference / project / people / decision / conversation)

## 🚧 Still on the backlog

### Streaming / Chunked WhatsApp Responses
Break long responses into chunks and send them as they're generated, rather than waiting for the full response. Reduces perceived latency for complex multi-tool tasks.

### Voice Messages (TTS + STT)
- **Input:** Receive WhatsApp voice notes, transcribe via Whisper or local STT
- **Output:** Convert Beast responses to voice via TTS, send as WhatsApp audio messages

### Vector Memory Search
Upgrade from BM25 to embedding-based semantic search. Requires a local embedding model (sentence-transformers adds ~200MB). Complements current lexical scoring for fuzzy recall.

### Parallel Sub-Agent Spawning
Current `spawn_agent` runs sub-tasks sequentially. Add a `spawn_agents` (plural) that uses `concurrent.futures` or asyncio to run subtasks truly in parallel and merge results.

### Webhook Ingestion
Accept webhooks from external services (GitHub, email, calendar) and convert them to Beast tasks. Requires extending `server.py` with new routes.

### Skills Registry
Catalog of installable tool packages — `/install weather-skill` adds a new tool + handler at runtime. Overlaps with the existing MCP install pipeline; would layer structured metadata on top.

### Browser Snapshot Interaction
Instead of raw HTML, take a screenshot of the page, annotate interactive elements with numbered refs, let the LLM say "click 3". Playwright MCP already handles the heavy lifting — this would be a thin convenience wrapper.

### DM Pairing Security
When an unknown sender messages Beast on WhatsApp: send a verification challenge (PIN, owner approval), temporary vs permanent access grants, audit log.
