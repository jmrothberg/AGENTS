# Agents Monorepo — Local LLM Servers + Obedient Beast

**Author:** Jonathan M Rothberg

Two projects in one repo, designed to work together:

1. **Local LLM servers** (`lfm_thinking.py` for macOS/MLX, `linux_thinking.py` for Linux/transformers) — dynamically discover any model on disk and serve it as an OpenAI-compatible API.
2. **Obedient Beast** (`obedient_beast/`) — a small, powerful personal AI agent with CLI, WhatsApp, and HTTP front-ends, tool calling, autonomous task scheduling, persistent memory, a skills registry, and persistent browser control.
3. **Local FLUX art** (`flux_art.py`, macOS Apple Silicon only) — text-to-image with **FLUX.2-klein** via the **mflux** package and your own downloaded weights; runs on Metal, no image API keys.

The LLM servers are optional — Beast also talks to Claude and OpenAI out of the box. But running both together gives you a fully local, self-hosted assistant.

---

## Table of contents

1. [Quick start](#quick-start)
2. [Repo layout](#repo-layout)
3. [Local LLM servers](#local-llm-servers)
4. [Local FLUX image generation (macOS)](#local-flux-image-generation-macos)
5. [Obedient Beast](#obedient-beast)
6. [Adding a skill](#adding-a-skill)
7. [Scheduling work with cron](#scheduling-work-with-cron)
8. [Persistent browser control](#persistent-browser-control)
9. [Environment variables](#environment-variables)
10. [For LLMs reading this repo](#for-llms-reading-this-repo)
11. [Troubleshooting](#troubleshooting)

---

## Quick start

```bash
# 1. Clone and install
git clone <this repo>
cd Agents
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r obedient_beast/requirements.txt

# 2. (Optional) Start a local LLM server — macOS
python lfm_thinking.py --model latest --server

# 2b. (Optional) Local FLUX text-to-image — macOS Apple Silicon only (see section below)
# pip install mflux
# python flux_art.py "a watercolor fox"

# 3. Configure Beast — pick a brain
cp obedient_beast/.env.example obedient_beast/.env
# Edit .env: set LLM_BACKEND=claude|openai|lfm and your API keys.

# 4. Talk to Beast
cd obedient_beast
python beast.py                 # CLI
./start.sh                      # full stack: HTTP server + WhatsApp + heartbeat + CLI
```

That's it. Beast starts with 24 built-in tools, an auto-generated tools manifest in its system prompt, and whatever skills you drop into `workspace/skills/`.

---

## Repo layout

```
Agents/
├── README.md                 ← you are here
├── lfm_thinking.py           ← macOS/MLX local LLM server
├── linux_thinking.py         ← Linux/transformers local LLM server
├── flux_art.py               ← local FLUX.2-klein text-to-image (macOS / mflux)
├── generated_art/            ← default output folder for flux_art.py PNGs (created on use)
├── test_client.py            ← streaming test client for the servers
├── requirements.txt          ← top-level (server) Python deps
└── obedient_beast/           ← the agent
    ├── beast.py              ← agent loop, built-in tools, slash commands
    ├── llm.py                ← unified client for Claude / OpenAI / local
    ├── server.py             ← Flask HTTP bridge (for WhatsApp)
    ├── heartbeat.py          ← autonomous task scheduler (interval + cron)
    ├── mcp_client.py         ← MCP server connection manager
    ├── capabilities.py       ← FULL vs LITE tier settings
    ├── skills_loader.py      ← SKILL.md runbook registry
    ├── cron_schedule.py      ← zero-dependency 5-field cron parser
    ├── browser_tools.py      ← Playwright persistent-profile browser
    ├── config/mcp_servers.json  ← MCP server catalog
    ├── workspace/
    │   ├── SOUL.md           ← personality + capabilities prompt
    │   ├── AGENTS.md         ← standing goals + reasoning templates
    │   ├── skills/           ← drop SKILL.md runbooks here
    │   ├── browser_profile/  ← Playwright cookies/logins (created on first use)
    │   ├── tasks.json        ← task queue state
    │   └── memory.json       ← persistent local facts (capped at 200)
    └── whatsapp/bridge.js    ← WhatsApp connector (Baileys)
```

---

## Local LLM servers

Both scripts scan your model directories at startup — any folder with a `config.json` becomes a selectable model. No code changes needed to add a new model, just copy the folder in.

| Platform | Script | Framework | Default model dir |
|----------|--------|-----------|-------------------|
| macOS (Apple Silicon) | `lfm_thinking.py` | MLX | `/Users/jonathanrothberg/MLX_Models/` |
| Linux (NVIDIA) | `linux_thinking.py` | transformers / PyTorch | configurable via `MODEL_SEARCH_PATHS` |

Common commands:

```bash
python lfm_thinking.py                            # interactive model picker + chat
python lfm_thinking.py --model latest             # interactive with most recent model
python lfm_thinking.py --model latest --server    # headless OpenAI-compatible API on :8000
python test_client.py                             # quick streaming sanity test
```

Features: dynamic model discovery, automatic text/vision detection, streaming, image and video analysis for VL models, optional TTS, OpenAI-compatible `/v1/chat/completions`.

> The `lfm_` and `LFM_URL` naming is a legacy artifact from the project's LiquidAI days. The scripts now work with any compatible model.

---

## Local FLUX image generation (macOS)

On **Apple Silicon**, you can generate images **entirely on-device** with `flux_art.py`: no Stability/Replicate/OpenAI image API keys. It uses **mflux** + **MLX** and a **local folder of FLUX.2-klein weights** (the pre-quantized mflux 4-bit bundle is the intended format).

**1. Install Python deps** (same venv as the rest of the repo is fine):

```bash
pip install mflux
```

On macOS, `mlx` / `mlx-metal` supply the Metal backend; if `import mlx.core` fails with `Library not loaded: libmlx.dylib`, reinstall them:

```bash
pip install --upgrade --force-reinstall mlx mlx-metal
```

**2. Download weights** into a directory that contains `config.json`, `transformer/`, `vae/`, `text_encoder/`, and `tokenizer/` (for example the Hugging Face **FLUX.2-klein-4B mflux-4bit** artifact). By default the script looks for:

`~/FLUX.2-klein-4B-mflux-4bit`

Override with **`--model /path/to/weights`** or the **`FLUX_ART_MODEL`** environment variable.

**3. Run:**

```bash
python flux_art.py "a red balloon over a city at dusk"
python flux_art.py --width 1024 --height 768 --seed 42 "sunset"
python flux_art.py --model ~/my-models/FLUX.2-klein-4B-mflux-4bit "a sketch of a bridge"
python flux_art.py   # interactive prompts; type quit to exit
```

Images are written under **`generated_art/`** in the repo root (or use `--output` for a filename). Use **`python flux_art.py --help`** for all flags.

**Note:** Use at least **2 inference steps** (the default is 4). Single-step runs can error inside the scheduler.

---

## Obedient Beast

Beast is a small agent that runs an LLM-tool loop:

```
User (CLI / WhatsApp / HTTP)
        │
        ▼
    beast.run(input)   ← loads history, picks tools, loops until done
        │
   ┌────┼────────────────────────────────┐
   │    │                │                │
   ▼    ▼                ▼                ▼
llm.py  Built-in Tools   Skills          MCP servers
(3      (24 tools)       (SKILL.md       (filesystem,
 back-                    runbooks)       memory, git,
 ends)                                    playwright, …)
```

### Why Beast is interesting

- **One file, one loop.** `beast.py` is ~1,800 lines of readable Python. No framework, no hidden magic.
- **Three brains.** Swap Claude, OpenAI, or a local model with one env var (`LLM_BACKEND=claude|openai|lfm`) or a slash command (`/claude`, `/openai`, `/lfm`).
- **Depth control.** `/depth N` sets how many tool-call steps the model may chain per request. Cloud defaults to 10, local to 5.
- **Layered system prompt.** Every turn, Beast composes its prompt from `SOUL.md` (personality) + `AGENTS.md` (standing goals) + an auto-generated **skills index** + an auto-generated **tools manifest** + a safety fallback.
- **Persistent memory.** Facts go to `workspace/memory.json` (always on), and optionally into an MCP knowledge-graph server for richer recall.
- **Autonomous task queue.** `add_task` writes to `workspace/tasks.json`; `heartbeat.py` picks up pending tasks on a timer and runs them in a fresh session.
- **Self-upgrade.** Beast can `install_mcp_server` to add new tool packs at runtime.
- **OpenClaw-inspired additions** (see below): skills registry, cron schedules, persistent browser, TOOLS.md composition, `use_skill` / `list_skills` tools.

### Built-in tool categories

| Category | Tools |
|----------|-------|
| Files & shell | `shell`, `read_file`, `write_file`, `edit_file`, `list_dir` |
| Desktop control | `screenshot`, `mouse_click`, `mouse_move`, `keyboard_type`, `keyboard_hotkey`, `get_screen_size`, `get_mouse_position` |
| Self-upgrade | `install_mcp_server`, `list_mcp_servers`, `enable_mcp_server` |
| Autonomy | `add_task` (supports `scheduled_at`, `repeat_seconds`, `cron`), `recall_memory` |
| Network | `fetch_url` |
| Skills | `list_skills`, `use_skill` |
| Persistent browser | `browser_goto`, `browser_read`, `browser_click`, `browser_type`, `browser_screenshot`, `browser_close` |

### Slash commands

`/help`, `/more`, `/status`, `/tasks`, `/done <id>`, `/drop <id>`, `/claude`, `/openai`, `/lfm`, `/depth <n>`, `/model`, `/heartbeat on|off`, `/clear`, `/clear tasks`, `/clear memory`, `/clear all`, `/tools`, `/skills`.

Slash commands are handled before the LLM sees the input.

---

## Adding a skill

Skills are markdown runbooks that live in `obedient_beast/workspace/skills/<name>/SKILL.md`. Each one is a high-level *recipe* composed from Beast's low-level tools. Adding a skill = writing a markdown file. No code, no restart (the skills loader rescans on every `list_skills` / `use_skill` call).

Minimal example:

```markdown
---
name: weekly-review
description: Summarize last week's activity from my journal and queue TODOs.
triggers: weekly review, Sunday planning
---

# Weekly Review Skill

1. Use `shell` to `ls -t ~/journal/*.md | head -7` to find this week's entries.
2. `read_file` each one and extract any line starting with "TODO".
3. `add_task` each TODO with priority=medium.
4. Write a one-page summary to `workspace/reports/<date>.md`.
5. Report what you did in one paragraph.
```

At startup Beast injects every skill's name + description into the system prompt under `## Available Skills`. When the user says "do my weekly review", the LLM sees the match, calls `use_skill(name="weekly-review")` to load the full body, and follows the instructions step by step.

A seed `research` skill ships with the repo — read it as a template.

---

## Scheduling work with cron

`add_task` supports three scheduling modes:

| Field | Behavior |
|-------|----------|
| _(none)_ | run immediately when the heartbeat next wakes |
| `scheduled_at` | one-shot, fires after the given ISO timestamp |
| `repeat_seconds` | simple interval recurrence |
| `cron` | 5-field cron expression (`minute hour dom month dow`) |

Cron examples (evaluated by `cron_schedule.py`, zero deps):

```
0 9 * * 1-5     # every weekday at 9am
*/15 * * * *    # every 15 minutes
0 0 1 * *       # midnight on the 1st of every month
30 8 * * 1      # 8:30am every Monday
```

The heartbeat loop (`heartbeat.py`) walks the task queue every `HEARTBEAT_INTERVAL_SEC` seconds, runs any task whose `next_run_at` has passed, and for recurring tasks computes the next fire time using the cron parser.

---

## Persistent browser control

`browser_*` tools drive a real headful Chromium context whose cookies, localStorage, and extensions live on disk in `workspace/browser_profile/`. Log into a site once in the visible window and Beast reuses the session forever — the same trick OpenClaw uses with its managed browser.

```bash
pip install playwright
python -m playwright install chromium
```

Then from a Beast conversation:

```
> open GitHub and tell me my notification count
```

The LLM calls `browser_goto(url="https://github.com/notifications")`, then `browser_read()`, and answers from the page text. To run headless on a server, set `BEAST_BROWSER_HEADLESS=true`.

**Security note:** This is full browser access with your real cookies. Treat Beast like any other tool that can act as you online.

---

## Environment variables

`.env` lives in `obedient_beast/`. The most important knobs:

| Variable | Purpose |
|----------|---------|
| `LLM_BACKEND` | `claude`, `openai`, or `lfm` |
| `LLM_BACKEND_TEST` | one-session override of `LLM_BACKEND` |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | cloud credentials |
| `LFM_URL` | local LLM server URL (default `http://localhost:8000/v1`) |
| `MCP_ENABLED` | `true`/`false` — load MCP tool servers at startup |
| `BEAST_BROWSER_HEADLESS` | run Playwright headless (default `false`) |
| `ALLOWED_NUMBERS` / `ALLOWED_GROUPS` | WhatsApp access control |
| `NOTIFICATION_CHAT_ID` | WhatsApp target for autonomous task notifications |
| `HEARTBEAT_INTERVAL_SEC` | seconds between heartbeat cycles |
| `FLUX_ART_MODEL` | (optional) path to FLUX.2-klein mflux-4bit weights for `flux_art.py` |

---

## For LLMs reading this repo

If you are an LLM reading this to understand the codebase, here is the minimal model you need:

- **Entry point for the agent is `obedient_beast/beast.py`.** The main loop is `run(user_input, session_id, llm)`. It loads history, calls the LLM with `SYSTEM_PROMPT` + tools, executes any tool calls via `execute_tool(name, args)`, and loops until the LLM returns a plain text reply.
- **`SYSTEM_PROMPT` is composed by `load_system_prompt()`**, which concatenates `workspace/SOUL.md`, `workspace/AGENTS.md`, the skills index from `skills_loader.get_skills_index()`, and an auto-generated tools manifest from `_render_tools_manifest()`. Never hard-code tool lists — they come from the `TOOLS` list in `beast.py` and are rendered at startup.
- **Tools are dicts** with `name`, `description`, `params`. Dispatch lives in `execute_tool()` as a long `elif` chain. To add a tool, append to `TOOLS` and add a branch in `execute_tool`.
- **Skills are markdown files**, not code. To add a skill, write `workspace/skills/<name>/SKILL.md` with optional frontmatter (`name`, `description`, `triggers`) and a step-by-step body. The agent discovers it on the next `list_skills` or `use_skill` call.
- **Tasks are JSON records** in `workspace/tasks.json`. Fields: `id`, `description`, `priority`, `status`, `scheduled_at?`, `repeat_seconds?`, `cron?`, `next_run_at?`. `heartbeat.py` owns recurrence logic via `_reset_recurring_task()`.
- **Three LLM backends** are unified in `llm.py`. The local backend speaks to `lfm_thinking.py` or `linux_thinking.py` over OpenAI-compatible HTTP. Tool calling falls back to a text-parsing format for models without native tool support.
- **MCP servers** are optional tool packs spawned as subprocesses at startup. Their tools appear with names prefixed `mcp_<servername>_`. See `mcp_client.py` and `config/mcp_servers.json`.
- **Do not add new frameworks.** The project's value is that it's small, flat Python with readable inline commentary. When adding features, prefer a new focused module (like `cron_schedule.py`, `skills_loader.py`, `browser_tools.py`) and wire it into `beast.py` with a few lines.

---

## Troubleshooting

- **`/skills` shows nothing** → the `list_skills` tool reads `workspace/skills/`. Drop a `SKILL.md` in there and call it again.
- **Cron task never fires** → check `heartbeat.py --status` to confirm the task has a `next_run_at` in the future, and make sure the heartbeat is actually running (`/heartbeat` from the CLI).
- **`browser_goto` returns "Playwright is not installed"** → `pip install playwright && python -m playwright install chromium`.
- **Local LLM timing out** → make sure `lfm_thinking.py --server` is running and `LFM_URL` in `.env` points to the right port.
- **`flux_art.py` / `import mlx` fails with `libmlx.dylib` missing** → the Metal wheel did not unpack fully; run `pip install --upgrade --force-reinstall mlx mlx-metal` in the same venv.
- **`flux_art.py` says model folder not found** → download the mflux 4-bit bundle and point `--model` or `FLUX_ART_MODEL` at that directory (default: `~/FLUX.2-klein-4B-mflux-4bit`).
- **MCP server won't load** → run with `MCP_ENABLED=true` and check logs; Beast keeps running on MCP failures, so missing MCP tools just reduce capability, they don't crash the agent.

---

Beast stays small on purpose. Every file is meant to be readable end-to-end in one sitting. When in doubt, read the source — inline comments explain the "why" alongside the "what".
