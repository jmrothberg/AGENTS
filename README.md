# Agents Monorepo — Local LLM Servers + Obedient Beast

**Author:** Jonathan M Rothberg

There are **four pieces**. Only one of them is a chat window.

| Piece | File in this repo | Terminal title | What it is |
|-------|-------------------|----------------|------------|
| **Brain** | `lfm_thinking.py` | LFM Thinking | Loads Qwen on port **8000**. Wait until it says the API is ready. **Do not type chat here.** |
| **Mailbox** | `obedient_beast/server.py` | Beast Server | WhatsApp texts land on port **5001**, then this process asks the brain. |
| **Phone link** | `obedient_beast/whatsapp/bridge.js` | WhatsApp Bridge | Connects your phone. Scan a QR only if it appears. |
| **Local client** | `obedient_beast/beast.py` | Local client — You: | **This is where you type.** Prompt is `You:`. Same Beast as WhatsApp (tools, memory, `/lfm` `/claude` `/openai`). |

WhatsApp does **not** use the local-client window. The local client does **not** use the mailbox or the bridge. Both need the brain.

`test_client.py` (repo root) is a raw ping of `:8000` with **no** Beast tools. It is not the local client.

---

## Commands (from `/Users/jonathanrothberg/Agents`)

First time only: `./obedient_beast/setup.sh`

| You want | Run | What opens |
|----------|-----|------------|
| Phone working on this Mac | `./obedient_beast/start.sh phone` | **3 windows:** brain, mailbox, WhatsApp. **No local client.** |
| Type in Terminal **and** the 3 are already up | `./obedient_beast/start.sh you` | **1 window:** `beast.py` only. Will not start a second brain. |
| Type in Terminal, no WhatsApp | `./obedient_beast/start.sh cli` | Brain + local client. |
| Stop them | `./obedient_beast/start.sh stop` | Kills brain, mailbox, WhatsApp, local client, heartbeat. |

Typical night: `start.sh phone`, wait for the brain, then `start.sh you`, then type at `You:` **or** text WhatsApp.

`phone` / `you` force `LFM_URL=http://localhost:8000` and `MCP_ENABLED=false` so this Mac is used and MCP does not hang.

Leave those windows open. Do not text or type until the brain says it is ready.

---

## Repo layout (where the local client lives)

```
Agents/
├── README.md                      ← you are here
├── lfm_thinking.py                ← BRAIN (not a chat window)
├── linux_thinking.py              ← Linux brain
├── test_client.py                 ← raw :8000 ping — NOT the local client
├── flux_art.py / zimage_art_linux.py
├── .env.example
└── obedient_beast/
    ├── beast.py                   ← LOCAL CLIENT  (python beast.py → You:)
    ├── server.py                  ← mailbox (:5001)
    ├── whatsapp/bridge.js         ← phone link
    ├── start.sh                   ← phone | you | cli | stop
    ├── llm.py                     ← Claude / OpenAI / local brain
    └── workspace/SOUL.md
```

---

Two projects: local model servers, and Obedient Beast (CLI + WhatsApp). **lfm** means local model (legacy LiquidAI name). Default weights: **Qwen3.8-27B-mxfp8** in `/Users/jonathanrothberg/MLX_Models/`. Canonical `.env` is **repo root**. Switch brains in a chat with `/lfm`, `/claude`, `/openai`.

`./obedient_beast/start.sh` (no argument) is the old five-window stack (adds heartbeat + CLI). `./obedient_beast/start.sh pm2` backgrounds mailbox/WhatsApp/heartbeat.

`LFM_MODEL` (env or `.env`) picks weights for `start.sh`. Missing folder → interactive picker.

```bash
python lfm_thinking.py --model Qwen3.8-27B-mxfp8 --server   # brain on :8000
python linux_thinking.py --model latest --server            # Linux twin
```

---

## Beast in brief

`beast.run()` loads history, calls the LLM with tools, executes tool calls (several per turn, sequentially), and repeats up to `DEPTH` (cloud 10, local 8, `/depth N`).

**30 built-in tools** stay in `execute_tool()`. A local 27B is only *offered* a subset by default (`core,browser,art` plus enabled MCP). `/tools all` or `BEAST_TOOL_GROUPS=…` restores everything. Cloud backends get all groups unless you override.

Qwen3.8 knobs (ignored by Claude/OpenAI): `QWEN_REASONING_EFFORT=medium`, `QWEN_ENABLE_THINKING`, `QWEN_PRESERVE_THINKING`.

Slash commands, sessions, MCP, and heartbeat details: [obedient_beast/README.md](obedient_beast/README.md).

---

## Adding a skill

Drop `obedient_beast/workspace/skills/<name>/SKILL.md`. Beast injects name + description into the prompt; `use_skill` loads the full body. No restart.

---

## Scheduling work with cron

`add_task` supports `scheduled_at`, `repeat_seconds`, and 5-field `cron`. `heartbeat.py` marks one-shot tasks done after it runs them; recurring tasks reschedule.

```
0 9 * * 1-5     # weekday 9am
*/15 * * * *    # every 15 minutes
```

---

## Persistent browser

`browser_*` tools use Playwright with `workspace/browser_profile/`. `pip install playwright && python -m playwright install chromium`. Headless: `BEAST_BROWSER_HEADLESS=true`.

---

## Local FLUX / Z-Image (optional)

macOS Apple Silicon: `flux_art.py` via **mflux** + local FLUX.2-klein weights (`~/FLUX.2-klein-4B-mflux-4bit` or `FLUX_ART_MODEL`). Beast `generate_art` on macOS uses that path. On Linux, Beast `generate_art` uses **`zimage_art_linux.py`** (Z-Image-Turbo + diffusers), not MLX FLUX. Standalone `flux_art_linux.py` is CLI-only.

```bash
pip install mflux
python flux_art.py "a watercolor fox"
# Linux Beast art: pip install -r requirements-zimage-linux.txt
```

---

## Environment variables

Repo-root `.env`. Important knobs:

| Variable | Purpose |
|----------|---------|
| `LLM_BACKEND` | `lfm` (local), `openai`, or `claude` |
| `LFM_URL` | Local server, default `http://localhost:8000` (no `/v1`) |
| `LFM_MODEL` | Folder substring for `start.sh` (default `Qwen3.8-27B-mxfp8`) |
| `QWEN_REASONING_EFFORT` | `low` / `medium` / `xhigh` |
| `BEAST_TOOL_GROUPS` | `core,browser,art` or `all` |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | Cloud only |
| `MCP_ENABLED` | Load MCP servers |
| `BRAVE_API_KEY` | Brave Search MCP (not a key in JSON) |
| `ALLOWED_NUMBERS` / `ALLOWED_GROUPS` | WhatsApp |
| `BEAST_PORT` | HTTP for the WhatsApp bridge (default 5001) |
| `NOTIFICATION_CHAT_ID` | WhatsApp target for heartbeat notifications |
| `FLUX_ART_MODEL` | Optional path to FLUX.2-klein weights (macOS `generate_art`) |
| `ZIMAGE_ART_MODEL` / `DIFFUSION_MODELS_DIR` | Linux `generate_art` (Z-Image-Turbo) |

---

## Troubleshooting

- **No `You:` prompt** → the local client is not running. `start.sh phone` does not open it. Run `./obedient_beast/start.sh you`.
- **Local LLM timing out** → `lfm_thinking.py --server` running, `LFM_URL=http://localhost:8000`.
- **`.env` not picked up** → file must be at repo root; Beast loads it even when cwd is `obedient_beast/`.
- **Too few / too many tools** → `/tools` to list, `/tools all` or `/tools core,browser,desktop`.
- **`browser_goto` missing Playwright** → `pip install playwright && python -m playwright install chromium`.
- **Heartbeat tasks stuck pending** → one-shots are marked done in code after the run; confirm `heartbeat.py` is running (`/heartbeat`).
- **FLUX / mlx dylib** → `pip install --upgrade --force-reinstall mlx mlx-metal`.
- **Beast `generate_art` on Linux** → torch + `pip install -r requirements-zimage-linux.txt`; set `ZIMAGE_ART_MODEL` or `DIFFUSION_MODELS_DIR`.

For code orientation (LLMs): [CLAUDE.md](CLAUDE.md).
