# Local LLM Inference Tools

**Written by Jonathan M Rothberg**

Local inference scripts that **dynamically discover and serve any model** in your model directories. Supports both:
- **macOS** (Apple Silicon) using MLX for optimized inference
- **Ubuntu/Linux** (NVIDIA GPUs) using transformers/PyTorch

> **Note on naming:** The filename `lfm_thinking.py` and the `LFM_URL` env var are legacy names from when this project started with LiquidAI's LFM-2.5 models. The scripts now support **any compatible model** — just drop model folders into your models directory and they appear in the selection menu automatically. The "LFM" naming is kept for backward compatibility with existing configs.

## Features

- **Dynamic Model Discovery**: Scans your model directories at startup — any model folder with a `config.json` is detected and offered in the selection menu
- **Automatic Type Detection**: Detects text vs. vision models from `config.json` (looks for "vl"/"vision"/"image" in `model_type`) or the presence of `processor_config.json`
- **Interactive Mode**: Chat directly in the terminal
- **Server Mode**: OpenAI-compatible API server accessible on your local network
- **Streaming**: Real-time token streaming (macOS/MLX text models)
- **Vision**: Image and video analysis (VL models)
- **TTS**: Optional text-to-speech for responses

## How Model Selection Works

Both scripts scan their respective model directories at startup:

| Platform | Script | Framework | Model Directory |
|----------|--------|-----------|-----------------|
| macOS | `lfm_thinking.py` | MLX | `/Users/jonathanrothberg/MLX_Models/` |
| Linux | `linux_thinking.py` | transformers/PyTorch | Configurable via `MODEL_SEARCH_PATHS` (default: `/home/jonathan/Models_Transformer/`) |

**To add a new model:** Download or copy the model folder into the appropriate directory. Next time you run the script, it appears in the numbered menu. No code changes needed.

The scripts detect model type automatically:
- **Text models**: Default. Used for chat, reasoning, tool calling.
- **Vision models**: Detected when `config.json` contains "vl", "vision", or "image" in `model_type`, or when `processor_config.json` exists. Enables image/video analysis.

## Scripts

### `lfm_thinking.py`
macOS (MLX) — Interactive chat OR OpenAI-compatible server. Scans `MLX_Models/` directory for all available models. (Legacy name — works with any MLX model, not just LFM.)

### `linux_thinking.py`
Linux (transformers/CUDA) — Interactive chat OR OpenAI-compatible server. Scans `MODEL_SEARCH_PATHS` directories for all available models. Drop-in replacement for `lfm_thinking.py` on Linux.

### `test_client.py`
Streaming test client for the server mode.

## Usage

```bash
# macOS (Apple Silicon)
python lfm_thinking.py

# Linux (NVIDIA GPU)
python linux_thinking.py
```

### Step 1: Select Model
The script scans your model directories and presents a numbered menu of all available models. For example:
```
==================================================
Model Selection
==================================================
  1. LFM2.5-1.2B-Thinking (Text)
  2. LFM2.5-VL-1.6B (Vision-Language) [VL]
  3. GLM-4.7-Flash (Text)
  4. MiniMax-M2.1-REAP-50 (Text)
==================================================
Select model (1/2/3/4):
```
Your list will differ based on which models you have downloaded.

### Step 2: Select Mode
- **1** = Interactive chat (local terminal)
- **2** = Server mode (OpenAI-compatible API)

### Media Options (VL Model, Interactive Mode)
- `i` = Image (opens file dialog)
- `v` = Video analysis
- `n` = Text only

---

## Server Mode (OpenAI-Compatible API)

Run the model as an API server accessible from any device on your local network.

### Starting the Server

```bash
python lfm_thinking.py        # macOS
python linux_thinking.py       # Linux
# Select a model from the numbered menu
# Select mode 2 (Server)
# Accept default port 8000 or enter custom
```

The server will display:
```
🚀 OpenAI-Compatible Server Starting
============================================================
Model: LFM2.5-1.2B-Thinking
Type:  text
============================================================
Access URLs:
  Local:   http://localhost:8000
  Network: http://192.168.x.x:8000
============================================================
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/v1/models` | GET | List available models |
| `/v1/chat/completions` | POST | Chat completions (OpenAI format) |

### Example: curl

```bash
curl http://192.168.x.x:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "LFM2.5-1.2B-Thinking", "messages": [{"role": "user", "content": "Hello!"}]}'
```

### Example: Python OpenAI Client

```python
from openai import OpenAI

client = OpenAI(base_url="http://192.168.x.x:8000/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="LFM2.5-1.2B-Thinking",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

### Example: Streaming with Python

```python
from openai import OpenAI

client = OpenAI(base_url="http://192.168.x.x:8000/v1", api_key="not-needed")
stream = client.chat.completions.create(
    model="LFM2.5-1.2B-Thinking",
    messages=[{"role": "user", "content": "Write a poem"}],
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### Test Client

Use the included test client for interactive streaming chat:

```bash
python test_client.py              # localhost:8000
python test_client.py 192.168.x.x  # custom host
python test_client.py 192.168.x.x 8080  # custom host and port
```

Type `quit` to exit.

### Platform Support

| Feature | macOS (MLX) | Linux (Transformers) |
|---------|-------------|---------------------|
| Interactive Mode | ✅ | ✅ |
| Server Mode | ✅ | ✅ |
| Streaming | ✅ (token-by-token) | ✅ (single-chunk SSE) |

**Note**: macOS/MLX streams tokens in real-time. Linux/transformers generates the full response first and sends it as a single SSE chunk (compatible with all OpenAI clients).

---

## Video Analysis

### How It Works

1. Select a video file via file dialog
2. Choose sampling interval (default: 2 seconds)
3. Script extracts frames at the specified interval
4. Each frame is sent to the VL model with your prompt
5. Descriptions are printed with timestamps
6. Option to save results to a timestamped text file

### Supported Video Formats

OpenCV (cv2) handles video decoding. Supported formats depend on your system's codecs:

| Format | Extension | Notes |
|--------|-----------|-------|
| MP4 | `.mp4` | H.264/H.265 codec, most common |
| AVI | `.avi` | Legacy format, widely supported |
| MOV | `.mov` | QuickTime format |
| MKV | `.mkv` | Matroska container |
| WebM | `.webm` | VP8/VP9 codec |

### Frame Sampling

The script does **not** process every frame. Instead:

- Calculates `frame_interval = FPS × interval_seconds`
- Reads frames sequentially but only analyzes every Nth frame
- Example: 30 FPS video with 2s interval = analyze every 60th frame

This allows fast processing of long videos while capturing scene changes.

### Save Output

After video analysis, type `y` when prompted to save results:

```
Save results? (y/n): y
Saved to: drone_test_analysis_20260201_143052.txt
```

Output filename format: `{video_name}_analysis_{YYYYMMDD_HHMMSS}.txt`

## Requirements

```bash
pip install -r requirements.txt
```

### Core Dependencies (All Platforms)
- `pillow` - Image processing
- `opencv-python` - Video processing
- `pyttsx3` - Text-to-speech (optional)

### Server Mode Dependencies
- `fastapi` - Web framework for API server
- `uvicorn` - ASGI server
- `pydantic` - Data validation

### macOS (Apple Silicon)
- `mlx-lm` - MLX Language Models (text)
- `mlx-vlm` - MLX Vision Language Models
- `torchvision` - Required by mlx_vlm processor

### Ubuntu/Linux
- `transformers` (5.0+)
- `torch` (with CUDA support)
- `accelerate` (model loading)

## Hardware

### macOS (Apple Silicon)
Tested on Mac Studio with M-series chips. Uses MLX framework for optimized inference on Apple Neural Engine.

### Ubuntu/Linux (NVIDIA GPUs)
Tested on NVIDIA Blackwell GPUs (DGX Spark). The scripts include Blackwell-specific optimizations:
- `CUDA_DEVICE_MAX_CONNECTIONS=1`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

## Local Model Directories

Models are discovered automatically from these directories. Drop any compatible model folder in and it appears in the menu.

### macOS
```
/Users/jonathanrothberg/MLX_Models/
```
Each subfolder (e.g., `LFM2.5-VL-1.6B-MLX-8bit/`, `GLM-4.7-Flash/`) is one model.

### Ubuntu/Linux
```
/home/jonathan/Models_Transformer/
```
Each subfolder (e.g., `LFM2.5-1.2B-Thinking/`, `Qwen3-32B/`) is one model. Additional search paths can be added by editing `MODEL_SEARCH_PATHS` at the top of `linux_thinking.py`.

---

# Obedient Beast - Personal AI Agent

**An AI assistant that lives on your computer and does things for you.**

Think of Beast like a smart intern sitting at your computer. You tell it what you want (by typing in the terminal or sending a WhatsApp message), and it figures out how to do it — reading files, running commands, taking screenshots, searching the web, whatever is needed.

### What can it do?

- **Use your computer** — run terminal commands, read/write/edit files, take screenshots, control the mouse and keyboard
- **Remember things** — it has persistent memory across conversations ("remember that my server IP is 10.0.1.5")
- **Work on its own** — give it tasks for later ("remind me to check disk space") and it works on them automatically in the background
- **Talk via WhatsApp** — message it from your phone anytime, anywhere
- **Learn new skills** — plug in MCP servers to add abilities like web search, GitHub access, browser automation, and more
- **Switch brains on the fly** — use a powerful cloud AI (Claude/OpenAI) for complex work, or a local model running on your own machine for privacy

### Two Brain Modes

Beast has two power levels. You pick which AI "brain" it uses:

| | **FULL mode** (Claude or OpenAI) | **LITE mode** (local model on your machine) |
|---|---|---|
| **Best for** | Complex tasks, research, multi-step work | Quick tasks, total privacy |
| **How smart** | Can chain 10 actions together, thinks deeply | Does one thing at a time, keeps it simple |
| **Privacy** | Messages go to cloud AI | Everything stays on your machine |
| **Cost** | Uses API credits | Free (your hardware does the work) |
| **Extra skills** | All MCP servers available | Only the simple MCP servers load |
| **Switch to it** | `/claude` or `/openai` | `/lfm` |

You can switch between modes anytime, even mid-conversation. Your local model server (the `lfm_thinking.py` or `linux_thinking.py` script from above) must be running in a separate terminal for LITE mode to work.

---

## How the Pieces Fit Together

There are 5 programs. Here is what each one does:

| Program | Language | What It Does | Required? |
|---------|----------|-------------|-----------|
| `linux_thinking.py` or `lfm_thinking.py` | Python | Scans model dirs and serves any local model as an OpenAI-compatible API on port 8000 | **Yes** (when using local LLM) |
| `beast.py` | Python | The agent itself. CLI chat, tools, memory, MCP | **Yes** |
| `server.py` | Python | HTTP API that receives WhatsApp messages and passes them to Beast | Only for WhatsApp |
| `node whatsapp/bridge.js` | Node.js | Connects to your WhatsApp account via Baileys | Only for WhatsApp |
| `heartbeat.py` | Python | Wakes up on a timer and works on queued tasks automatically | Optional |

### Data Flow

```
YOU (terminal)  -->  beast.py  -->  LLM server (port 8000)  -->  response
YOU (WhatsApp)  -->  bridge.js  -->  server.py  -->  beast.py  -->  LLM server  -->  response
```

The LLM server (`linux_thinking.py` on Linux, `lfm_thinking.py` on macOS) is **always separate** -- you start it in its own terminal first, pick which model to serve, then start Beast.

If you use Claude or OpenAI as backend instead of local LLM, you do NOT need the LLM server at all -- Beast talks directly to the cloud API.

---

## Quick Start

### CLI Only (simplest -- no WhatsApp, no Node.js)

Open **2 terminals**:

```bash
# Terminal 1: Start LLM server (skip if using Claude/OpenAI)
cd ~/AGENTS
python linux_thinking.py

# Terminal 2: Start Beast agent
cd ~/AGENTS/obedient_beast
python beast.py
```

That's it. You're chatting with your local AI agent.

### CLI + WhatsApp (full setup)

Open **4 terminals**:

```bash
# Terminal 1: Start LLM server (skip if using Claude/OpenAI)
cd ~/AGENTS
python linux_thinking.py

# Terminal 2: Start HTTP server (receives WhatsApp messages)
cd ~/AGENTS/obedient_beast
python server.py

# Terminal 3: Start WhatsApp bridge (requires Node.js)
cd ~/AGENTS/obedient_beast/whatsapp
node bridge.js

# Terminal 4: Start Beast CLI (optional -- you can also just use WhatsApp)
cd ~/AGENTS/obedient_beast
python beast.py
```

First time: the WhatsApp bridge prints a QR code. Scan it with your phone (WhatsApp > Settings > Linked Devices).

### Using `./start.sh` (convenience script)

`./start.sh` opens terminals 2-4 automatically (server, bridge, heartbeat, CLI). You still need to start the LLM server yourself in a separate terminal first.

```bash
# Terminal 1: Start LLM server manually
cd ~/AGENTS
python linux_thinking.py

# Then in obedient_beast/:
cd ~/AGENTS/obedient_beast
./start.sh          # Opens 4 windows: server.py, bridge.js, heartbeat.py, beast.py
./start.sh stop     # Stop all 4
./start.sh status   # Check what's running
./start.sh cli      # Start just the CLI (no WhatsApp)
```

---

## What Requires Node.js?

**Only the WhatsApp bridge** (`node whatsapp/bridge.js`) and **MCP external tools** (`npx` commands) require Node.js.

### What is Node.js?

Node.js is a **JavaScript runtime** — it runs JavaScript code outside of web browsers, similar to how `python` runs Python code:

| Command | What it does |
|---------|--------------|
| `python beast.py` | Runs the Python file `beast.py` |
| `node bridge.js` | Runs the JavaScript file `bridge.js` |

**Why JavaScript for WhatsApp?** The best WhatsApp Web libraries (like Baileys) are written in JavaScript. Rather than rewrite them in Python, we use Node.js to run the existing JavaScript library and have it forward messages to the Python server.

**The bridge is simple:** It connects to WhatsApp, receives your messages, sends them to `server.py` via HTTP, and relays the AI response back to WhatsApp.

Everything else is pure Python:
- `linux_thinking.py` / `lfm_thinking.py` -- Python
- `beast.py` -- Python
- `server.py` -- Python (Flask)
- `heartbeat.py` -- Python

Install Node.js only if you want WhatsApp or MCP tools:
```bash
# Ubuntu/Debian
sudo apt install nodejs npm

# macOS
brew install node
```

---

## Setup

```bash
cd obedient_beast
./setup.sh              # Full setup (Python + Node.js)
./setup.sh --no-node    # Python only (skip WhatsApp/MCP)
```

The setup script creates a Python virtual environment, installs dependencies, and creates a `.env` template.

---

## Architecture

```
                    ┌──────────────────────┐
                    │    User Interface     │
                    │  (CLI or WhatsApp)    │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │     beast.run()       │  ◄── Agent loop: LLM ↔ Tools
                    │  (beast.py)           │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
    ┌─────────▼──────┐ ┌──────▼──────┐ ┌───────▼────────┐
    │   llm.py       │ │ Built-in    │ │  mcp_client.py │
    │  (3 backends)  │ │ Tools (18)  │ │  (MCP servers) │
    │ Claude/OpenAI/ │ │             │ │  with 3-tier   │
    │ Local (lfm)    │ │             │ │  filtering     │
    └────────────────┘ └──────┬──────┘ └───────┬────────┘
                              │                │
              ┌───────────────┼───────┐  ┌─────┴──────────────┐
              │               │       │  │  MCP Tier System    │
         ┌────┴───┐  ┌───────┴──┐    │  │                     │
         │File &  │  │Computer │    │  │ Essential: fs,mem   │
         │System  │  │Control  │    │  │ Extended: git,sql   │
         │+ Net   │  │         │    │  │ Cloud: brave,github │
         └────────┘  └────────-┘    │  └─────────────────────┘
         • shell     • screenshot   │
         • read_file • mouse_click  │  ┌─────────────────────┐
         • write_file• mouse_move   │  │   Memory System     │
         • edit_file • keyboard_*   │  │                     │
         • list_dir  • screen_info  │  │ MCP knowledge graph │
         • fetch_url                │  │ + local JSON backup │
         • add_task                 │  │ (workspace/memory)  │
         • recall_memory            │  └─────────────────────┘
         • install/list/enable_mcp  │
                                    │
                        ┌───────────┴──────────┐
                        │  capabilities.py     │
                        │  FULL vs LITE tiers  │
                        └──────────────────────┘
```

## Files (inside `obedient_beast/`)

| File | Lines | Description |
|------|-------|-------------|
| `beast.py` | ~1270 | Agent loop + 18 built-in tools + CLI + slash commands + local memory |
| `llm.py` | ~405 | Unified LLM client (Claude/OpenAI/local with dual tool-calling) |
| `server.py` | ~173 | Flask HTTP server -- **only needed for WhatsApp** |
| `capabilities.py` | ~109 | Tiered settings (FULL/LITE) + MCP tier filtering |
| `heartbeat.py` | ~263 | Autonomous task scheduler (processes task queue on a timer) |
| `mcp_client.py` | ~420 | MCP server connection manager with tier filtering |
| `config/mcp_servers.json` | ~95 | MCP server catalog with 11 servers across 3 tiers |
| `whatsapp/bridge.js` | ~269 | Baileys WhatsApp connector -- **requires Node.js** |
| `workspace/SOUL.md` | ~72 | Agent personality + MCP tier awareness + self-upgrade awareness |
| `workspace/AGENTS.md` | ~57 | Task guidance, reasoning templates, memory recall guidelines |
| `workspace/memory.json` | auto | Local memory fallback (JSON, auto-created) |
| `workspace/tasks.json` | auto | Task queue for autonomous work |
| `setup.sh` | ~250 | One-command setup |
| `start.sh` | ~200 | Opens 4 Terminal windows (server, WhatsApp, heartbeat, CLI) |

## Built-in Tools (18 total)

### File & System Tools
| Tool | Description |
|------|-------------|
| `shell` | Execute any terminal command (with configurable timeout, max 300s) |
| `read_file` | Read file contents |
| `write_file` | Create/write files (auto-creates parent dirs) |
| `edit_file` | Find & replace text in files |
| `list_dir` | List directory contents |

### Computer Control Tools
| Tool | Description |
|------|-------------|
| `screenshot` | Capture the screen (auto-sends via WhatsApp) |
| `mouse_click` | Click at x,y coordinates |
| `mouse_move` | Move cursor to x,y |
| `keyboard_type` | Type text |
| `keyboard_hotkey` | Press shortcuts (cmd+c, etc) |
| `get_screen_size` | Get screen dimensions |
| `get_mouse_position` | Get cursor position |

### Self-Upgrade Tools
| Tool | Description |
|------|-------------|
| `install_mcp_server` | Add new MCP server capabilities |
| `list_mcp_servers` | Show configured MCP servers with tier info |
| `enable_mcp_server` | Enable/disable an MCP server |

### Autonomous Agent Tools
| Tool | Description |
|------|-------------|
| `add_task` | Add/update/complete tasks in the queue |
| `recall_memory` | Search persistent memory (MCP + local JSON fallback) |

### Network Tools
| Tool | Description |
|------|-------------|
| `fetch_url` | HTTP GET/POST any URL (stdlib, no deps, 4000 char truncation) |

## MCP Server Catalog (3-Tier System)

Enable MCP by setting `MCP_ENABLED=true` in `.env`. Servers are launched via `npx` (requires Node.js).

**All MCP tiers load in both LITE and FULL mode.** Local LLMs need access to cloud tools like brave-search for web queries. The tier labels are organizational only. LITE mode limits tool-calling behavior (single-tool mode, 2 max turns) to prevent loops — it doesn't restrict which servers load.

### Essential Tier (always loaded)

| Server | Install Command | What It Does |
|--------|----------------|--------------|
| **filesystem** | `npx -y @modelcontextprotocol/server-filesystem /Users/you` | File search/move beyond built-ins |
| **memory** | `npx -y @modelcontextprotocol/server-memory` | Persistent knowledge graph |
| **time** | `npx -y @modelcontextprotocol/server-time` | Time/timezone queries (1-2 tools) |
| **fetch** | `npx -y @modelcontextprotocol/server-fetch` | HTTP fetching via MCP |

### Extended Tier (FULL mode only)

| Server | Install Command | What It Does |
|--------|----------------|--------------|
| **sqlite** | `npx -y @modelcontextprotocol/server-sqlite` | Local database queries |
| **git** | `npx -y @modelcontextprotocol/server-git` | Git operations (commit, diff, branch) |
| **sequential-thinking** | `npx -y @modelcontextprotocol/server-sequential-thinking` | Step-by-step complex reasoning |
| **playwright** | `npx -y @anthropic/mcp-server-playwright` | Full browser automation |

### Cloud-Only Tier (FULL mode + API keys)

| Server | Install Command | What It Needs |
|--------|----------------|---------------|
| **brave-search** | `npx -y @modelcontextprotocol/server-brave-search` | `BRAVE_API_KEY` |
| **github** | `npx -y @modelcontextprotocol/server-github` | `GITHUB_TOKEN` |
| **slack** | `npx -y @anthropic/mcp-server-slack` | `SLACK_TOKEN` |

Edit `config/mcp_servers.json` to enable/configure servers. Use `/skills` in Beast to see the full catalog with install commands.

## Configuration (.env)

```bash
# LLM Backend: "lfm" (local models — legacy name), "openai", or "claude"
# Default is "lfm" — tries localhost:8000 first, then LFM_URL from below
LLM_BACKEND=lfm

# Local model server URL (when LLM_BACKEND=lfm)
LFM_URL=http://192.168.1.100:8000

# API Keys
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# WhatsApp Security - Only respond to these numbers
ALLOWED_NUMBERS=+12025551234

# MCP (optional) - Enable external tool servers
MCP_ENABLED=false
```

## Testing Different LLM Backends

**Safe testing without modifying your main `.env`:**

```bash
# Test local models temporarily (backend name "lfm" is legacy)
LLM_BACKEND_TEST=lfm python3 beast.py

# Test OpenAI temporarily
LLM_BACKEND_TEST=openai python3 beast.py

# Test Claude (default)
python3 beast.py  # Uses LLM_BACKEND from .env
```

This overrides your `.env` setting just for that session -- your main config stays unchanged.

### Local Model Single-Tool Mode

Local LLMs (Qwen3, GLM-4, etc.) currently tend to loop on tool calls instead of summarizing results. Beast automatically limits local models (the "lfm" backend) to **one tool call per request**, then forces a text response. This is controlled by `capabilities.py`.

**When local LLMs improve**, edit `capabilities.py` and change `SINGLE_TOOL_MODE` and `MAX_TOOL_TURNS`.

Claude and OpenAI handle multi-tool calls properly and get the full tier automatically.

## Slash Commands (work in both WhatsApp and CLI)

| Command | What it does |
|---------|-------------|
| `/help` | Quick help with top commands |
| `/more` | Full detailed help with examples |
| `/status` | Show backend, tier, heartbeat state, task summary |
| `/tasks` | List all tasks with status |
| `/done 3` | Mark task #3 as done |
| `/drop 3` | Remove task #3 |
| `/claude` | Switch to Claude backend (FULL tier) |
| `/openai` | Switch to OpenAI backend (FULL tier) |
| `/lfm` | Switch to local LFM backend (LITE tier) |
| `/heartbeat on` | Enable background task processing |
| `/heartbeat off` | Pause background task processing |
| `/heartbeat` | Show heartbeat status |
| `/clear` | Clear chat history only |
| `/clear tasks` | Clear task queue only |
| `/clear memory` | Clear saved memories (local JSON) |
| `/clear all` | Clear everything (history + tasks + memory) |
| `/tools` | List all available tools |
| `/skills` | MCP server catalog with install commands (3 tiers) |

**CLI-only commands:**

| Command | What it does |
|---------|-------------|
| `/new` | Reset conversation (start fresh session) |
| `/quit` | Exit CLI |

All backend switching, task management, and heartbeat control works from **both WhatsApp and CLI**.

## Task Queue — How to Add Tasks

Beast has an autonomous task queue. There are **3 ways to add tasks**:

### 1. Ask Beast (WhatsApp or CLI)
Just say things like:
- "Remind me to check disk space later"
- "Add a task to organize my downloads"
- "Queue up: review the log files"

Beast recognizes phrases like "remind me", "later", "add a task" and uses the `add_task` tool automatically.

### 2. Edit tasks.json directly
Open `workspace/tasks.json` and add a task:
```json
{
  "tasks": [
    {
      "id": 1,
      "description": "check disk space",
      "priority": "medium",
      "status": "pending",
      "created_at": "2026-02-06T10:00:00"
    }
  ]
}
```

### 3. Via the heartbeat
When Beast runs autonomously, it can add follow-up tasks to its own queue.

### Check your queue
- CLI: type `/status`
- WhatsApp: send `/status`
- Terminal: `python heartbeat.py --status`

## Autonomous Heartbeat

Beast can work on tasks **by itself** on a timer, without you sending messages.

**The heartbeat starts automatically with `./start.sh`** (it gets its own Terminal window). You can also run it standalone:

```bash
python heartbeat.py              # Run heartbeat loop (Ctrl+C to stop)
python heartbeat.py --once       # Process one cycle and exit
python heartbeat.py --status     # Show task queue
```

**How it works:**
1. Heartbeat wakes up every N minutes (5 min on Claude, 10 min on local models)
2. Checks `workspace/tasks.json` for pending tasks
3. Picks the highest-priority task and feeds it to `beast.run()`
4. Beast processes it (using tools as needed) and marks it done/failed
5. Goes back to sleep

**Control from WhatsApp or CLI:**
- `/heartbeat on` — enable processing
- `/heartbeat off` — pause processing (heartbeat stays running but skips tasks)
- `/heartbeat` — check if it's on or off

## What Beast Stores (and How to Clear It)

Beast saves 4 types of data. Each can be cleared independently:

| What | Where it lives | What it holds | How to clear |
|------|---------------|---------------|-------------|
| **Chat history** | `sessions/*.jsonl` | Every conversation (CLI + WhatsApp) | `/clear` |
| **Task queue** | `workspace/tasks.json` | To-do list for the heartbeat | `/clear tasks` |
| **Local memory** | `workspace/memory.json` | Facts Beast remembers (up to 200, auto-capped) | `/clear memory` |
| **MCP knowledge graph** | External MCP server | Rich knowledge graph (if MCP memory enabled) | Ask Beast: "forget everything in memory" |

- `/clear all` wipes chat history + tasks + local memory in one command
- The MCP knowledge graph is separate because it runs as an external server — Beast can't directly delete its data with a slash command, but you can ask Beast to do it via the MCP tools
- `/new` (CLI only) starts a fresh conversation **without** deleting anything

## Capability Tiers

Beast automatically adjusts its power based on which LLM backend is active:

| Setting | Claude/OpenAI (FULL) | Local Models (LITE) |
|---------|---------------------|-------------------|
| Max tool calls per turn | 10 | 2 |
| Single-tool mode | Off | On (prevents loops) |
| Heartbeat interval | 5 min | 10 min |
| Tasks per heartbeat cycle | 3 | 1 |
| Memory detail saved | Full context | Key facts only |
| MCP tiers loaded | All (Essential+Extended+Cloud) | All (same — local LLM needs web search etc.) |

Tiers are set in `capabilities.py` and auto-detect from `LLM_BACKEND` in `.env`.

## WhatsApp Setup

1. Run `./setup.sh` (installs Node.js dependencies)
2. Run `./start.sh`
3. Scan the QR code with WhatsApp (Settings → Linked Devices)
4. Message yourself or create a solo group to chat with Beast

### WhatsApp Credentials & Backup

Your WhatsApp session is stored in two places:

| Location | Purpose |
|----------|---------|
| `obedient_beast/whatsapp/auth_info/` | Active credentials (gitignored) |
| `~/.beast_whatsapp_backup/` | Auto-backup (in your home folder) |

**Auto-backup**: After each successful connection, Beast automatically backs up your credentials to `~/.beast_whatsapp_backup/`. If `auth_info/` is deleted, it auto-restores on next startup.

**"Try again later" error?** WhatsApp rate-limits device linking. Wait 10-15 minutes.

### Moving Beast to Another Machine

To run Beast on a new computer without re-scanning the QR code:

**1. Copy these files from your current machine:**
```bash
# Your configuration (API keys, settings)
scp ~/.env user@new-machine:/path/to/AGENTS/.env

# WhatsApp credentials (pick one):
scp -r ~/.beast_whatsapp_backup user@new-machine:~/
# OR
scp -r obedient_beast/whatsapp/auth_info user@new-machine:/path/to/AGENTS/obedient_beast/whatsapp/
```

**2. On the new machine:**
```bash
git clone https://github.com/jmrothberg/AGENTS.git
cd AGENTS/obedient_beast
./setup.sh
./start.sh  # No QR scan needed - credentials auto-restore!
```

**Full paths on macOS:**
```
~/.env                              → /Users/YOUR_USERNAME/.env
~/.beast_whatsapp_backup/           → /Users/YOUR_USERNAME/.beast_whatsapp_backup/
obedient_beast/whatsapp/auth_info/  → Inside the AGENTS repo
```

### @beast in Group Chats

Beast normally stays quiet in shared group chats (groups with other people). To summon Beast for a single message, start it with `@beast`:

```
@beast check disk space
@beast what's the weather in NYC
```

- Only YOU (the phone owner) can use `@beast` — other group members can't trigger it
- Beast responds in the group chat, then goes quiet again
- Works in any group, even those not in `ALLOWED_GROUPS`

### WhatsApp Security

- Beast uses YOUR WhatsApp account (not a separate bot)
- Set `ALLOWED_NUMBERS` to restrict who can trigger responses
- Group messages only respond to the account owner (or via `@beast`)
- DMs check against the allowlist

## Recommended Models for Tool Calling

| Model | Size (4-bit) | Tool Calling |
|-------|-------------|--------------|
| Qwen3-32B-MLX-4bit | ~18GB | ✅ Native |
| Llama-3.3-70B-Instruct-4bit | ~40GB | ✅ Native |
| Qwen2.5-72B-Instruct-4bit | ~41GB | ✅ Native |
| GLM-4.7-Flash | ~4GB | ✅ Native |

## Why Beast > OpenClaw

| Feature | OpenClaw | Obedient Beast |
|---------|----------|----------------|
| MCP Support | ❌ Custom only | ✅ Native MCP with 3-tier catalog (11 servers) |
| Tool Ecosystem | Closed | Open (any MCP + 18 built-in) |
| Computer Control | Via plugins | Built-in (screenshot, mouse, keyboard) |
| Autonomous Tasks | Cron/webhooks | Heartbeat + task queue |
| Persistent Memory | Via plugins | Built-in (MCP graph + local JSON fallback) |
| Web/HTTP Fetching | Via plugins | Built-in `fetch_url` + MCP fetch server |
| Backend Switching | ❌ | `/claude` `/openai` `/lfm` live |
| Capability Tiers | ❌ | Auto (FULL/LITE by backend) |
| MCP Tier Organization | ❌ | 3-tier catalog (Essential/Extended/Cloud) |
| Auto Memory Recall | ❌ | ✅ At session start |
| Codebase Size | 150k+ lines | ~2.7k lines (heavily commented) |
| Local-first | ✅ | ✅ |
| Setup | Complex | One command |

---

## License

MIT
