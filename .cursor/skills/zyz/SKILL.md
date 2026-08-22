---
name: zyz
description: Reset Obedient Beast chat history for WhatsApp and CLI without sending session logs to the phone. Use when the user mentions WhatsApp dumping history, context trimmed, long Beast replies, /clear, or starting a fresh conversation.
disable-model-invocation: true
---

# Reset Beast history (WhatsApp + CLI)

## What the user is seeing

`[Beast] Context trimmed to N messages` is a **server log** (`stderr` on `server.py`). It is **not** sent to WhatsApp.

Beast still **loads** `sessions/wa_OWNER.jsonl` (and other `sessions/*.jsonl`) into the LLM. A long file makes replies slow. The model may also echo old context into the WhatsApp reply. Wiping the session file fixes that.

## How to reset (tell the user this)

From **WhatsApp** (same as CLI — handled in `beast.run()` before the LLM):

| Command | Effect |
|---------|--------|
| `/clear` | Deletes **all** `obedient_beast/sessions/*.jsonl` (CLI + every WhatsApp sender). Tasks and `workspace/memory.json` stay. |
| `/clear tasks` | Empty task queue only |
| `/clear memory` | Empty local memory facts |
| `/clear all` | History + tasks + memory |

Reply after `/clear` is a one-liner (`Chat history cleared`). It does **not** dump history.

CLI-only: `/new` starts a new CLI session id; it does **not** delete WhatsApp `wa_*.jsonl`. For WhatsApp, use `/clear`.

No Beast Server restart needed. Next message is a new empty history.

## Do not

- Do not tell the user to restart WhatsApp or re-scan QR to reset chat.
- Do not paste `sessions/*.jsonl` contents into a WhatsApp reply.
- Do not treat "Context trimmed" as a message that went to the phone.

## If they still get a huge WhatsApp reply after `/clear`

The running `server.py` must have processed `/clear`. If they only restarted CLI, WhatsApp still uses the old process and old file. Send `/clear` **from the phone**, or delete `obedient_beast/sessions/wa_OWNER.jsonl` and send a new WhatsApp message.
