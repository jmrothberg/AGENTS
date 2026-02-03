# LFM-2.5 Inference Tools

**Written by Jonathan M Rothberg**

Local inference scripts for LiquidAI's LFM-2.5 models. Supports both:
- **macOS** (Apple Silicon) using MLX for optimized inference
- **Ubuntu/Linux** (NVIDIA GPUs) using transformers/PyTorch

## Features

- **Interactive Mode**: Chat directly in the terminal
- **Server Mode**: OpenAI-compatible API server accessible on your local network
- **Streaming**: Real-time token streaming (macOS/MLX text models)
- **Vision**: Image and video analysis (VL models)
- **TTS**: Optional text-to-speech for responses

## Models

| Model | Size | Description |
|-------|------|-------------|
| LFM2.5-1.2B-Thinking | 1.2B params | Text-only reasoning model |
| LFM2.5-VL-1.6B | 1.6B params | Vision-Language model (images + video) |
| GLM-4.7-Flash | 4.7B params | Fast text-only model (macOS) |
| MiniMax-M2.1-REAP-50 | - | Text-only model (macOS) |

## Scripts

### `lfm_thinking.py`
Main script - Interactive chat OR OpenAI-compatible server.

### `test_client.py`
Streaming test client for the server mode.

## Usage

```bash
python lfm_thinking.py
```

### Step 1: Select Model
- **1** = LFM2.5-1.2B-Thinking (Text-only reasoning)
- **2** = LFM2.5-VL-1.6B (Vision-Language)
- **3** = GLM-4.7-Flash (macOS only)
- **4** = MiniMax-M2.1-REAP-50 (macOS only)

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
python lfm_thinking.py
# Select model (e.g., 1)
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
| Streaming | ✅ (text models) | ❌ (non-streaming) |

**Note**: Streaming is currently only implemented for MLX text models on macOS. Linux/transformers falls back to non-streaming responses.

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

## Local Model Paths

### macOS
```
/Users/jonathanrothberg/MLX_Models/LFM2.5-VL-1.6B-MLX-8bit
```

### Ubuntu/Linux
```
/home/jonathan/Models_Transformer/LFM2.5-1.2B-Thinking
/home/jonathan/Models_Transformer/LFM2.5-VL-1.6B
```

---

# Obedient Beast - Personal AI Agent

**A powerful, MCP-native AI agent with computer control and WhatsApp integration.**

Obedient Beast is a personal AI assistant designed to be **better than OpenClaw** with:
- **12 built-in tools** including screenshot, mouse, and keyboard control
- **MCP support** - connect ANY Model Context Protocol server
- **WhatsApp integration** - message your agent 24/7
- **Local-first** - your data stays on your machine
- **Multiple LLM backends** - LFM (local), OpenAI, or Claude

## One-Command Setup

```bash
cd obedient_beast
./setup.sh
```

That's it. The script:
- Creates Python virtual environment
- Installs all dependencies (Python + Node.js)
- Creates `.env` template
- Verifies everything works

Then just:
```bash
./start.sh   # Start the agent
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    OBEDIENT BEAST                            │
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   LLM Core  │  │   Memory    │  │      Channels       │  │
│  │(LFM/Claude) │  │ (Sessions)  │  │ (CLI / WhatsApp)    │  │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │
│         └────────────────┼───────────────────-┘              │
│                          │                                   │
│         ┌────────────────┴────────────────┐                  │
│         │          Tool Router            │                  │
│         └────────────────┬────────────────┘                  │
└──────────────────────────┼───────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   ┌────┴────┐       ┌─────┴─────┐      ┌─────┴─────┐
   │Built-in │       │ Computer  │      │    MCP    │
   │  Tools  │       │  Control  │      │  Servers  │
   └─────────┘       └───────────┘      └───────────┘
   • shell           • screenshot       • filesystem
   • read_file       • mouse_click      • git
   • write_file      • mouse_move       • memory
   • edit_file       • keyboard_type    • (any MCP)
   • list_dir        • keyboard_hotkey  
```

## Files

| File | Description |
|------|-------------|
| `setup.sh` | **One-command setup** - run this first |
| `start.sh` | Start/stop the agent |
| `beast.py` | Agent loop + 12 built-in tools |
| `llm.py` | Unified LLM client (LFM/OpenAI/Claude) |
| `mcp_client.py` | MCP server connection manager |
| `server.py` | Flask HTTP server for WhatsApp |
| `config/mcp_servers.json` | MCP server configuration |
| `whatsapp/bridge.js` | Baileys WhatsApp connector |
| `workspace/SOUL.md` | Agent personality |

## Built-in Tools (12 total)

### File & System Tools
| Tool | Description |
|------|-------------|
| `shell` | Execute any terminal command |
| `read_file` | Read file contents |
| `write_file` | Create/write files |
| `edit_file` | Find & replace text in files |
| `list_dir` | List directory contents |

### Computer Control Tools
| Tool | Description |
|------|-------------|
| `screenshot` | Capture the screen |
| `mouse_click` | Click at x,y coordinates |
| `mouse_move` | Move cursor to x,y |
| `keyboard_type` | Type text |
| `keyboard_hotkey` | Press shortcuts (cmd+c, etc) |
| `get_screen_size` | Get screen dimensions |
| `get_mouse_position` | Get cursor position |

## MCP Tools (Optional)

Enable external tool servers by setting `MCP_ENABLED=true` in `.env`.

| MCP Server | What it does | Local? |
|------------|--------------|--------|
| **filesystem** | Read/write files anywhere | ✅ |
| **git** | Commit, diff, branch, status | ✅ |
| **memory** | Persistent knowledge graph | ✅ |
| **sequential-thinking** | Complex reasoning | ✅ |
| **brave-search** | Web search | ❌ (API) |

Edit `config/mcp_servers.json` to enable/configure servers.

## Configuration (.env)

```bash
# LLM Backend: "lfm" (local), "openai", or "claude"
LLM_BACKEND=claude

# Your LFM Server (when LLM_BACKEND=lfm)
LFM_URL=http://192.168.1.100:8000

# API Keys
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# WhatsApp Security - Only respond to these numbers
ALLOWED_NUMBERS=+12025551234

# MCP (optional) - Enable external tool servers
MCP_ENABLED=false
```

## Commands

```bash
# Setup (first time only)
./setup.sh              # Full setup
./setup.sh --no-node    # Skip WhatsApp (Python only)
./setup.sh --check      # Verify requirements

# Running
./start.sh              # Start server + WhatsApp
./start.sh stop         # Stop everything
./start.sh status       # Check if running
./start.sh server       # Python server only
./start.sh whatsapp     # WhatsApp bridge only

# Management
./start.sh clear-history  # Clear all conversations
./clear_history.sh        # Quick history clear (standalone)

# CLI mode (for testing)
source ../.venv/bin/activate
python3 beast.py
```

## CLI Commands

In CLI mode, type:
- `/tools` - List all available tools
- `/new` - Reset conversation
- `/clear` - Clear all conversation history
- `/quit` - Exit

## WhatsApp Commands

Send these messages to Beast via WhatsApp:
- `/clear` - Clear all conversation history
- `/tools` - List all available tools

**Note**: Commands work the same in WhatsApp as CLI mode.

## WhatsApp Setup

1. Run `./setup.sh` (installs Node.js dependencies)
2. Run `./start.sh`
3. Scan the QR code with WhatsApp (Settings → Linked Devices)
4. Message yourself or create a solo group to chat with Beast

### WhatsApp Security

- Beast uses YOUR WhatsApp account (not a separate bot)
- Set `ALLOWED_NUMBERS` to restrict who can trigger responses
- Group messages only respond to the account owner
- DMs check against the allowlist

## Using with Your LFM Server

1. Start your LFM server (see LFM section above)
2. Set in `.env`:
   ```bash
   LLM_BACKEND=lfm
   LFM_URL=http://YOUR_SERVER_IP:8000
   ```
3. Run Beast - it will use your local model!

**Note**: For full tool-calling support, use a model that supports OpenAI-style function calling.

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
| MCP Support | ❌ Custom only | ✅ Native MCP |
| Tool Ecosystem | Closed | Open (any MCP) |
| Computer Control | Via plugins | Built-in |
| Codebase Size | 150k+ lines | <1k lines |
| Local-first | ✅ | ✅ |
| Setup | Complex | One command |

---

## License

MIT
