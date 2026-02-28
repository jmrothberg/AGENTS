# Future Upgrades — OpenClaw-Inspired Features

Ideas for future development, inspired by comparing with OpenClaw's architecture. These are documented here for reference — not yet implemented.

## Streaming / Chunked WhatsApp Responses
Break long responses into chunks and send them as they're generated, rather than waiting for the full response. Reduces perceived latency for complex multi-tool tasks.

## Voice Messages (TTS + STT)
- **Input:** Receive WhatsApp voice notes, transcribe via Whisper or local STT
- **Output:** Convert Beast responses to voice via TTS, send as WhatsApp audio messages
- Useful for hands-free interaction (driving, cooking, etc.)

## BOOT.md Startup Script
A `workspace/BOOT.md` file that Beast reads and executes on startup. Lets users define initialization tasks (check disk space, pull repos, verify services) that run automatically.

## Loop Detection
Detect when the agent is stuck in a loop (repeated failures, same tool calls, no progress). After N identical failures, break out and report to user instead of burning through the depth budget.

## Better Memory Search
Replace linear JSON scan with hybrid search:
- **Vector search:** Embed facts with a local model, cosine similarity for semantic recall
- **BM25:** Keyword-based scoring for exact matches
- **Temporal decay:** Recent memories ranked higher than old ones

## Smarter Memory Auto-Save
Current: saves raw conversation snippets. Better approach:
- Distill conversations into atomic facts before saving
- Deduplicate against existing memories
- Expire stale facts automatically
- Categorize memories (preferences, project context, people, decisions)

## Sub-Agent Spawning
Spawn parallel background agents for independent subtasks. Example: "research these 3 topics" → 3 parallel Beast instances, results merged. Requires process management and result aggregation.

## Webhook Ingestion
Accept webhooks from external services and convert them to Beast tasks:
- **GitHub:** PR reviews, issue assignments, CI failures
- **Email:** Forward emails to Beast for processing
- **Calendar:** Upcoming meeting prep, reminders
- **Custom:** Generic JSON webhook → task queue

## Skills Registry
Discover and install tool packages dynamically:
- A catalog of tool "skills" (weather, email, calendar, code review, etc.)
- `/install skill-name` to add new capabilities at runtime
- Skills are self-contained: tool definitions + handler code + dependencies

## Browser Snapshot Interaction
Instead of raw HTML/text from web pages:
- Take a screenshot of the page
- Annotate interactive elements with numbered references
- User/Beast can say "click 3" or "fill field 7 with X"
- Bridges the gap between text-only and full browser automation

## DM Pairing Security
When an unknown sender messages Beast on WhatsApp:
- Send a verification challenge (PIN, code word, or OWNER approval)
- Temporary vs permanent access grants
- Audit log of who accessed Beast and when
