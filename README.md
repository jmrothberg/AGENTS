# Agents Monorepo — Local LLM Servers + Obedient Beast

**Author:** Jonathan M Rothberg

## Next time: this Mac + WhatsApp (the three windows)

```bash
cd /Users/jonathanrothberg/Agents
./obedient_beast/start.sh phone
```

Leave all three windows open. Do not close them while you want WhatsApp to work.

| Window | What it does |
|--------|----------------|
| **LFM Thinking** (the brain) | Loads Qwen on this Mac (port 8000). Wait until it says the API is ready — a few minutes. If this window dies, WhatsApp has nobody to talk to. |
| **Beast Server** (the mailbox) | Listens on port 5001. Your phone’s texts land here, then this process asks the brain. Started with `LFM_URL=http://localhost:8000` (this Mac, not a remote box) and `MCP_ENABLED=false` (skips the MCP hang). |
| **WhatsApp Bridge** (the phone link) | Connects WhatsApp to the mailbox. Scan the QR if it appears (WhatsApp → Settings → Linked Devices → Link a device). If you already linked recently, it just says connected. |

Text your own WhatsApp only after the brain window has finished loading. Stop everything with `./obedient_beast/start.sh stop`.

---

Two projects in one repo:

1. **Local LLM servers** (`lfm_thinking.py` on macOS/MLX, `linux_thinking.py` on Linux/transformers) — discover models on disk and serve an OpenAI-compatible API.
2. **Obedient Beast** (`obedient_beast/`) — a personal agent with CLI, WhatsApp, tools, heartbeat tasks, and memory.

**lfm = local model.** The name is leftover from LiquidAI. `LLM_BACKEND=lfm`, `LFM_URL`, and `/lfm` work with Qwen (default), GLM, Llama, or anything the local server loads. Claude and OpenAI stay one `/claude` or `/openai` (or `.env`) switch away.

Default local weights: **Qwen3.8-27B-mxfp8** under `/Users/jonathanrothberg/MLX_Models/`.

---

## Quick start

```bash
cd Agents
./obedient_beast/setup.sh          # venv + both requirements.txt + repo-root .env
# ./obedient_beast/setup.sh --no-node   # skip WhatsApp / Node.js

# This Mac + WhatsApp (3 windows: brain, mailbox, phone link)
./obedient_beast/start.sh phone

# Chat only (local model + CLI)
./obedient_beast/start.sh cli

# Full stack (WhatsApp + heartbeat + CLI)
./obedient_beast/start.sh
```

Canonical config is **repo-root `.env`** (created by setup, template: [`.env.example`](.env.example)). `LLM_BACKEND=lfm` and `LFM_URL=http://localhost:8000` (no `/v1` suffix).

Switch brains without restarting setup: `/lfm`, `/claude`, `/openai`, or `/model Qwen3.8`.

---

## Repo layout

```
Agents/
├── README.md                 ← you are here (human setup)
├── CLAUDE.md                 ← short onboarding for LLMs
├── lfm_thinking.py           ← macOS/MLX local LLM server
├── linux_thinking.py         ← Linux/transformers local LLM server
├── flux_art.py / flux_art_linux.py
├── zimage_art_linux.py       ← Beast generate_art on Linux (Z-Image-Turbo)
├── .env.example              ← copy to .env at this root
├── requirements.txt          ← server deps (mlx-lm / fastapi / …)
└── obedient_beast/           ← the agent (see its README for slash commands)
    ├── beast.py              ← agent loop + 30 built-in tools
    ├── llm.py                ← Claude / OpenAI / local client
    ├── setup.sh / start.sh
    └── workspace/SOUL.md     ← personality
```

---

## Two ways to run Beast

| Command | What starts |
|---------|-------------|
| `./obedient_beast/start.sh phone` | This Mac’s Qwen + WhatsApp (3 windows; localhost, no MCP) |
| `./obedient_beast/start.sh cli` | Local model server + CLI (no WhatsApp) |
| `./obedient_beast/start.sh` | Full stack: model, HTTP server, WhatsApp, heartbeat, CLI |
| `./obedient_beast/start.sh pm2` | Same as full, with server/WhatsApp/heartbeat under pm2 |

`LFM_MODEL` (env or `.env`, default `Qwen3.8-27B-mxfp8`) picks weights. If that folder is missing, the interactive picker opens.

Local models live in `/Users/jonathanrothberg/MLX_Models/` (macOS) or `MODEL_SEARCH_PATHS` (Linux).

```bash
python lfm_thinking.py --model Qwen3.8-27B-mxfp8 --server   # API on :8000
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

- **Local LLM timing out** → `lfm_thinking.py --server` running, `LFM_URL=http://localhost:8000`.
- **`.env` not picked up** → file must be at repo root; Beast loads it even when cwd is `obedient_beast/`.
- **Too few / too many tools** → `/tools` to list, `/tools all` or `/tools core,browser,desktop`.
- **`browser_goto` missing Playwright** → `pip install playwright && python -m playwright install chromium`.
- **Heartbeat tasks stuck pending** → one-shots are marked done in code after the run; confirm `heartbeat.py` is running (`/heartbeat`).
- **FLUX / mlx dylib** → `pip install --upgrade --force-reinstall mlx mlx-metal`.
- **Beast `generate_art` on Linux** → torch + `pip install -r requirements-zimage-linux.txt`; set `ZIMAGE_ART_MODEL` or `DIFFUSION_MODELS_DIR`.

For code orientation (LLMs): [CLAUDE.md](CLAUDE.md).
