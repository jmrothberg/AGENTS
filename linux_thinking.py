"""
Linux Transformer Model Server
===============================
Written by Jonathan M Rothberg

Dynamically scans model directories and serves any local transformer model
via an OpenAI-compatible API. Drop-in replacement for lfm_thinking.py on Linux.

- Scans MODEL_SEARCH_PATHS at startup — any model folder with config.json is detected
- Same OpenAI API as lfm_thinking.py (same .env, same ports)
- Beast agent connects identically via LFM_URL (legacy name for local model URL)
- Hot-swap models via API: POST /v1/models/switch {"model": "latest"}

# TODO: Switch to vLLM for concurrent batched inference when it supports
#       DGX Spark / Blackwell / CUDA 13 natively (without containers).
#       vLLM would allow multiple Beast agents to query simultaneously
#       via continuous batching. For now, transformers works reliably.
#       To switch: pip install vllm, then uncomment vLLM sections below.

USAGE:
  python linux_thinking.py                          # Interactive model selection
  python linux_thinking.py --model latest --server   # Serve most recent model (pm2-friendly)
  python linux_thinking.py --model Qwen3 --server    # Serve model matching "Qwen3"
  python linux_thinking.py --model latest --server --port 9000
  python linux_thinking.py --list                    # List available models and exit

PM2 (process manager) — copy/paste to start as a managed background service:

  pm2 start /home/jonathan/Agents/linux_thinking.py \\
    --name linux-thinking --interpreter python3 \\
    --max-restarts 3 --restart-delay 10000 \\
    -- --model latest --server

  pm2 status              # check running processes
  pm2 logs linux-thinking # view stdout/stderr
  pm2 restart linux-thinking
  pm2 stop linux-thinking
  pm2 delete linux-thinking
  pm2 save                # persist across reboots (pair with: pm2 startup)
"""

import os
import time
import json
import uuid
import re
import argparse
import threading
import queue

# ============================================================================
# CONFIGURABLE PATHS - Adjust these for your machine
# ============================================================================
# Add/remove/reorder paths below. All existing paths are scanned for models.
# Uncomment placeholders or add your own paths on other machines.
MODEL_SEARCH_PATHS = [
    "/home/jonathan/Models_Transformer",   # Primary: local models
    # "/mnt/nas/models",                   # Placeholder: NAS / shared storage
    # "/opt/models",                       # Placeholder: system-wide install
    # "/data/huggingface/models",          # Placeholder: HF cache location
    # "/home/user/models",                 # Placeholder: another user's models
]

DEFAULT_HOST = "0.0.0.0"    # Listen on all interfaces (127.0.0.1 for local-only)
DEFAULT_PORT = 8000          # Same port as lfm_thinking.py for Beast compatibility

# --- vLLM settings (for future use, when vLLM supports Blackwell/CUDA 13) ---
# GPU_MEMORY_UTILIZATION = 0.90  # GPU memory fraction (0.0-1.0)
# MAX_NUM_SEQS = 32              # Max concurrent sequences vLLM batches
# ============================================================================

# Blackwell / CUDA GPU optimizations
os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# ============================================================================
# TTS Setup (Optional)
# ============================================================================
TTS_ENABLED = False
tts_engine = None
tts_queue = None

def init_tts():
    """Initialize TTS engine if available."""
    global tts_engine, tts_queue, TTS_ENABLED
    try:
        import pyttsx3
        tts_engine = pyttsx3.init()
        tts_engine.setProperty('rate', 175)
        tts_queue = queue.Queue()
        TTS_ENABLED = True
        tts_thread = threading.Thread(target=tts_worker, daemon=True)
        tts_thread.start()
        return True
    except Exception as e:
        print(f"TTS not available: {e}")
        print("Install with: pip install pyttsx3")
        return False

def tts_worker():
    """Background thread that speaks queued text."""
    while True:
        text = tts_queue.get()
        if text is None:
            break
        if text.strip():
            try:
                tts_engine.say(text)
                tts_engine.runAndWait()
            except:
                pass
        tts_queue.task_done()

def speak(text):
    """Queue text to be spoken (non-blocking)."""
    if TTS_ENABLED and tts_queue and text.strip():
        tts_queue.put(text)


# ============================================================================
# Model Scanning - Detects models and types from directory structure
# ============================================================================
def scan_models(models_dir):
    """
    Dynamically scan a models directory and detect model types.
    Returns dict: {"1": (path, type, description), ...}

    Model type detection:
    - Vision models: config.json has "vl"/"vision"/"image" in model_type,
      or has processor_config.json, or has image_token_id/vision_config
    - Text models: everything else
    """
    models = {}
    if not os.path.exists(models_dir):
        return models

    model_dirs = sorted([
        d for d in os.listdir(models_dir)
        if os.path.isdir(os.path.join(models_dir, d)) and not d.startswith('.')
    ])

    for idx, model_name in enumerate(model_dirs, 1):
        model_path = os.path.join(models_dir, model_name)
        config_path = os.path.join(model_path, "config.json")
        processor_path = os.path.join(model_path, "processor_config.json")

        # Detect model type from config
        model_type = "text"
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    mt = config.get("model_type", "").lower()
                    if any(x in mt for x in ["vl", "vision", "image"]):
                        model_type = "vision"
                    # Newer multimodal models have vision token IDs or vision_config
                    if any(k in config for k in ["image_token_id", "vision_start_token_id", "vision_config"]):
                        model_type = "vision"
            except:
                pass

        # Also check for processor_config.json (vision models have this)
        if os.path.exists(processor_path):
            model_type = "vision"

        # Validate: if config says vision but weights don't have vision_tower,
        # the model was converted text-only (e.g., quantized without vision).
        if model_type == "vision":
            index_path = os.path.join(model_path, "model.safetensors.index.json")
            if os.path.exists(index_path):
                try:
                    with open(index_path, 'r') as f:
                        weight_map = json.load(f).get("weight_map", {})
                    has_vision_weights = any("vision" in k.lower() for k in weight_map)
                    if not has_vision_weights:
                        model_type = "text"  # Config says vision, but weights are text-only
                except:
                    pass

        type_label = "(Vision-Language)" if model_type == "vision" else "(Text)"
        description = f"{model_name} {type_label}"

        models[str(idx)] = (model_path, model_type, description)

    return models


def scan_all_model_paths():
    """Scan all configured MODEL_SEARCH_PATHS and merge results."""
    all_models = {}
    idx = 1
    for search_path in MODEL_SEARCH_PATHS:
        if not os.path.exists(search_path):
            continue
        models = scan_models(search_path)
        for _key, value in sorted(models.items(), key=lambda x: int(x[0])):
            all_models[str(idx)] = value
            idx += 1
    return all_models


def detect_model_type(model_path):
    """Detect whether a model is text or vision from its config.json."""
    model_type = "text"
    config_path = os.path.join(model_path, "config.json")
    processor_path = os.path.join(model_path, "processor_config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                mt = config.get("model_type", "").lower()
                if any(x in mt for x in ["vl", "vision", "image"]):
                    model_type = "vision"
                if any(k in config for k in ["image_token_id", "vision_start_token_id", "vision_config"]):
                    model_type = "vision"
        except:
            pass
    if os.path.exists(processor_path):
        model_type = "vision"
    # Validate against actual weights
    if model_type == "vision":
        index_path = os.path.join(model_path, "model.safetensors.index.json")
        if os.path.exists(index_path):
            try:
                with open(index_path, 'r') as f:
                    weight_map = json.load(f).get("weight_map", {})
                if not any("vision" in k.lower() for k in weight_map):
                    model_type = "text"
            except:
                pass
    return model_type


# Scan models at startup
ALL_MODELS = scan_all_model_paths()


# ============================================================================
# CLI Arguments (enables headless/pm2 operation)
# ============================================================================
parser = argparse.ArgumentParser(
    description="Linux Transformer Model Server - OpenAI-compatible API for local models"
)
parser.add_argument("--model", type=str, default=None,
    help='Model to load: "latest" for most recent, or a name/substring to match (e.g. "Qwen3")')
parser.add_argument("--server", action="store_true",
    help="Start in server mode automatically (no interactive prompt)")
parser.add_argument("--port", type=int, default=DEFAULT_PORT,
    help=f"Server port (default: {DEFAULT_PORT})")
parser.add_argument("--host", type=str, default=DEFAULT_HOST,
    help=f"Server host (default: {DEFAULT_HOST})")
parser.add_argument("--interactive", action="store_true",
    help="Run in interactive chat mode instead of server")
parser.add_argument("--list", action="store_true",
    help="List available models and exit")
cli_args = parser.parse_args()


def resolve_model_choice(model_arg):
    """
    Resolve --model argument to a model key from ALL_MODELS.
    - "latest": pick the most recently modified model directory
    - number (e.g. "3"): pick by menu number
    - string: fuzzy match against model folder names (case-insensitive)
    Returns the model key (string number) or None if no match.
    """
    if not ALL_MODELS:
        print("Error: No models found in any search path:")
        for p in MODEL_SEARCH_PATHS:
            exists = "EXISTS" if os.path.exists(p) else "NOT FOUND"
            print(f"  {p} ({exists})")
        exit(1)

    # "latest" — pick the model directory with the most recent modification time
    if model_arg.lower() == "latest":
        newest_key = None
        newest_mtime = 0
        for key, (path, _, desc) in ALL_MODELS.items():
            mtime = os.path.getmtime(path)
            if mtime > newest_mtime:
                newest_mtime = mtime
                newest_key = key
        if newest_key:
            _, _, desc = ALL_MODELS[newest_key]
            print(f"Auto-selected latest model: {desc}")
        return newest_key

    # Direct menu number (e.g. "3")
    if model_arg in ALL_MODELS:
        return model_arg

    # Substring match against folder names (case-insensitive)
    model_arg_lower = model_arg.lower()
    matches = []
    for key, (path, _, desc) in ALL_MODELS.items():
        folder_name = os.path.basename(path).lower()
        if model_arg_lower in folder_name:
            matches.append(key)

    if len(matches) == 1:
        _, _, desc = ALL_MODELS[matches[0]]
        print(f"Auto-selected model: {desc}")
        return matches[0]
    elif len(matches) > 1:
        print(f"Multiple models match '{model_arg}':")
        for key in matches:
            _, _, desc = ALL_MODELS[key]
            print(f"  {key}. {desc}")
        print("Be more specific or use the menu number.")
        exit(1)
    else:
        print(f"No model matching '{model_arg}'. Available models:")
        for key, (_, _, desc) in ALL_MODELS.items():
            print(f"  {key}. {desc}")
        exit(1)


# Handle --list
if cli_args.list:
    from datetime import datetime
    print("=" * 50)
    print("Available Models")
    print("=" * 50)
    for key, (path, model_type, desc) in ALL_MODELS.items():
        mtime = os.path.getmtime(path)
        date_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
        vl_tag = " [VL]" if model_type == "vision" else ""
        print(f"  {key}. {desc}{vl_tag}  ({date_str})")
    print("=" * 50)
    exit(0)


# ============================================================================
# Helper function to clear model from memory
# ============================================================================
def clear_model_memory():
    """Clear model from GPU memory."""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except:
        pass


# ============================================================================
# Tool Calling Support for Local LLMs
# ============================================================================
# Local models don't have native function calling like Claude/OpenAI.
# Instead, we inject tool info into the prompt and parse tool calls from
# the model's text output using regex patterns.
#
# Supported formats:
#   1. ```tool_call\n{"name": "...", "arguments": {...}}\n```
#   2. ```tool\n{"name": "...", "arguments": {...}}\n```
#   3. <tool_call>{"name": "...", "arguments": {...}}</tool_call>
# ============================================================================

def format_tools_for_prompt(tools):
    """
    Format tools into a CONCISE prompt section for the model.

    We keep the tool list short to avoid overwhelming the model's context.
    Priority tools are listed first for better attention.
    """
    if not tools:
        return ""

    # Keep only the most useful tools to avoid overwhelming the model
    priority_tools = [
        "screenshot", "shell", "read_file", "write_file", "edit_file", "list_dir",
        "mouse_click", "keyboard_type", "mcp_brave-search_brave_web_search"
    ]

    tool_list = []
    for tool in tools:
        func = tool.get("function", tool)
        name = func.get("name", "unknown")
        # Include priority tools first, then limit others
        if name in priority_tools or len(tool_list) < 15:
            desc = func.get("description", "")[:50]
            params = func.get("parameters", {}).get("properties", {})
            param_names = ", ".join(params.keys())
            tool_list.append(f"- {name}({param_names}): {desc}")

    tool_text = "TOOLS: " + " | ".join([t.split(":")[0].strip("- ") for t in tool_list[:10]])
    return tool_text


def parse_tool_calls(text):
    """Parse tool calls from model output."""
    tool_calls = []

    # Pattern 1: ```tool_call\n{...}\n``` - use greedy match for nested braces
    pattern1 = r'```tool_call\s*\n?\s*(\{.*?\})\s*\n?```'
    matches = re.findall(pattern1, text, re.DOTALL)

    # Pattern 1b: ```tool\n{...}\n``` - GLM Flash uses this shorter variant
    pattern1b = r'```tool\s*\n?\s*(\{.*?\})\s*\n?```'
    matches += re.findall(pattern1b, text, re.DOTALL)

    # Pattern 2: <tool_call>{...}</tool_call> (Qwen3 native format)
    pattern2 = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
    matches += re.findall(pattern2, text, re.DOTALL)

    # Pattern 3: Extract JSON between ```tool_call and ``` more robustly
    pattern3 = r'```tool_call\s*\n([\s\S]*?)\n```'
    block_matches = re.findall(pattern3, text)

    # Pattern 3b: Also match ```tool variant for GLM Flash
    pattern3b = r'```tool\s*\n([\s\S]*?)\n```'
    block_matches += re.findall(pattern3b, text)

    # Try to parse all matches, deduplicate by name+args
    all_candidates = matches + block_matches
    seen = set()

    def add_tool_call(data):
        """Add tool call if not duplicate."""
        name = data.get("name")
        args = json.dumps(data.get("arguments", data.get("args", {})))
        key = f"{name}:{args}"
        if key not in seen:
            seen.add(key)
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {"name": name, "arguments": args}
            })

    for match in all_candidates:
        match = match.strip()
        try:
            data = json.loads(match)
            if "name" in data:
                add_tool_call(data)
        except json.JSONDecodeError:
            # Try to extract JSON with brace matching
            try:
                start = match.find('{')
                if start >= 0:
                    depth = 0
                    for i, c in enumerate(match[start:]):
                        if c == '{':
                            depth += 1
                        elif c == '}':
                            depth -= 1
                            if depth == 0:
                                data = json.loads(match[start:start + i + 1])
                                if "name" in data:
                                    add_tool_call(data)
                                break
            except:
                pass

    # Only return first tool call to prevent loops
    return tool_calls[:1]


def clean_tool_calls_from_text(text):
    """Remove tool call blocks and thinking tags from text."""
    # Remove ```tool_call blocks
    text = re.sub(r'```tool_call\s*\n[\s\S]*?\n```', '', text)
    # Remove ```tool blocks (GLM Flash variant)
    text = re.sub(r'```tool\s*\n[\s\S]*?\n```', '', text)
    # Remove <tool_call> blocks
    text = re.sub(r'<tool_call>[\s\S]*?</tool_call>', '', text)
    # Remove <think> blocks (Qwen thinking)
    text = re.sub(r'<think>[\s\S]*?</think>', '', text)
    return text.strip()


# ============================================================================
# Transformers Server Mode - OpenAI-Compatible API
# ============================================================================
# NOTE: For concurrent batched inference (multiple Beast agents at once),
# switch to vLLM when it supports your hardware:
#   pip install vllm
#   Then replace the transformers model loading + generate() calls below
#   with vLLM's AsyncLLMEngine. See commented sections marked "# VLLM:"
# ============================================================================
def run_server_mode(model, tokenizer, model_name, model_type,
                    host=DEFAULT_HOST, port=DEFAULT_PORT):
    """
    Run the model as an OpenAI-compatible API server using transformers.
    Supports hot-swapping models via POST /v1/models/switch.

    Requests are handled sequentially (one generate() at a time).
    For concurrent batching, switch to vLLM when it supports Blackwell/CUDA 13.
    """
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel as PydanticModel
    from typing import List, Optional, Union, AsyncGenerator
    import uvicorn
    import socket
    import asyncio
    import functools

    # --- VLLM: Uncomment these when vLLM supports your hardware ---
    # from vllm.engine.async_llm_engine import AsyncLLMEngine
    # from vllm.engine.arg_utils import AsyncEngineArgs
    # from vllm import SamplingParams

    # Mutable state so model can be hot-swapped via /v1/models/switch
    state = {
        "model": model,
        "tokenizer": tokenizer,
        "model_name": model_name,
        "model_type": model_type,
    }

    # ----------------------------------------------------------------
    # FastAPI app setup
    # ----------------------------------------------------------------
    app = FastAPI(
        title="Linux Transformer Server",
        description="OpenAI-compatible API for local models"
    )

    # Allow CORS for local network access
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request/Response models matching OpenAI format
    class ChatMessage(PydanticModel):
        role: str
        content: Union[str, List, None] = None
        tool_calls: Optional[List[dict]] = None

    class ChatCompletionRequest(PydanticModel):
        model: str = model_name
        messages: List[ChatMessage]
        temperature: Optional[float] = 0.7
        max_tokens: Optional[int] = 512
        stream: Optional[bool] = False
        tools: Optional[List[dict]] = None  # Tool definitions for function calling

    class ChatCompletionChoice(PydanticModel):
        index: int
        message: ChatMessage
        finish_reason: str

    class Usage(PydanticModel):
        prompt_tokens: int
        completion_tokens: int
        total_tokens: int

    class ChatCompletionResponse(PydanticModel):
        id: str
        object: str = "chat.completion"
        created: int
        model: str
        choices: List[ChatCompletionChoice]
        usage: Usage

    class ModelInfo(PydanticModel):
        id: str
        object: str = "model"
        created: int
        owned_by: str = "local"

    class ModelsResponse(PydanticModel):
        object: str = "list"
        data: List[ModelInfo]

    class SwitchRequest(PydanticModel):
        model: str  # "latest", menu number, or name substring

    # ----------------------------------------------------------------
    # Helper: Build user_message from Beast's multi-turn conversation
    # ----------------------------------------------------------------
    def build_user_message(messages, tools=None):
        """
        Flatten Beast's multi-turn conversation into a user_message string.

        Beast sends multi-turn conversations including:
          - user: Original request
          - assistant: Tool call (if any)
          - tool: Result from tool execution

        We flatten this into a text conversation the model can understand,
        since local models don't have native tool result handling.
        """
        system_message = ""
        conversation_parts = []
        image_path = None  # Extracted from image_url content blocks

        for msg in messages:
            content = msg.content
            # Handle content as string or list
            if isinstance(content, list):
                text_content = " ".join(
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in content
                    if isinstance(item, dict) and item.get("type") in ["text", "tool_result"]
                )
                # Extract image from image_url blocks (base64 data URI from Beast)
                if image_path is None:
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "image_url":
                            data_url = item.get("image_url", {}).get("url", "")
                            if data_url.startswith("data:"):
                                try:
                                    import base64, tempfile
                                    header, b64data = data_url.split(",", 1)
                                    img_bytes = base64.b64decode(b64data)
                                    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                                    tmp.write(img_bytes)
                                    tmp.close()
                                    image_path = tmp.name
                                    print(f"[DEBUG] Extracted image to {image_path} ({len(img_bytes)} bytes)")
                                except Exception as img_err:
                                    print(f"[DEBUG] Image extraction failed: {img_err}")
                                break  # One image is enough
            else:
                text_content = content or ""

            if msg.role == "system":
                system_message = text_content
            elif msg.role == "user":
                conversation_parts.append(f"User: {text_content}")
            elif msg.role == "assistant":
                if text_content:
                    conversation_parts.append(f"Assistant: {text_content}")
            elif msg.role == "tool":
                # Tool results - format clearly for the model
                conversation_parts.append(f"[Tool Result]: {text_content}")

        # Build the user message from conversation (last few turns)
        user_message = "\n".join(conversation_parts[-6:]) if conversation_parts else ""

        # If there's a tool result, add instruction to summarize
        if "[Tool Result]:" in user_message:
            user_message += "\n\nNow summarize this result for the user in a helpful way."

        # Add tool definitions to the prompt if provided
        if tools:
            tools_prompt = format_tools_for_prompt(tools)
            if tools_prompt:
                tool_instruction = (
                    f"\n\n{tools_prompt}\n"
                    "To use a tool, respond ONLY with: "
                    "```tool_call\n{\"name\": \"TOOL_NAME\", \"arguments\": {}}\n```\n"
                    "Example for web search: "
                    "```tool_call\n{\"name\": \"mcp_brave-search_brave_web_search\", "
                    "\"arguments\": {\"query\": \"intel stock price\"}}\n```\n"
                    "DO NOT explain. Just output the tool_call block."
                )
                user_message = user_message + tool_instruction

        return user_message, system_message, image_path

    def generate_response(user_message, system_message, temperature, max_tokens):
        """
        Generate a response using transformers model.generate().
        Runs synchronously on GPU - called via run_in_executor for async.
        """
        import torch
        # Build chat messages for template
        chat_messages = []
        if system_message:
            chat_messages.append({"role": "system", "content": system_message})
        chat_messages.append({"role": "user", "content": user_message})

        inputs = state["tokenizer"].apply_chat_template(
            chat_messages,
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True,
        )
        input_ids = inputs["input_ids"].to(state["model"].device)
        attention_mask = inputs["attention_mask"].to(state["model"].device)

        output = state["model"].generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_tokens,
        )

        response_text = state["tokenizer"].decode(
            output[0][input_ids.shape[-1]:], skip_special_tokens=True
        )
        prompt_tokens = input_ids.shape[-1]
        completion_tokens = output.shape[-1] - input_ids.shape[-1]
        return response_text, prompt_tokens, completion_tokens

    # ----------------------------------------------------------------
    # Endpoints
    # ----------------------------------------------------------------
    @app.get("/")
    async def root():
        """Health check endpoint."""
        return {
            "status": "ok",
            "model": state["model_name"],
            "type": state["model_type"],
            "engine": "transformers",
        }

    @app.get("/v1/models")
    async def list_models():
        """List available models (OpenAI-compatible)."""
        return ModelsResponse(
            data=[ModelInfo(id=state["model_name"], created=int(time.time()))]
        )

    @app.get("/v1/models/available")
    async def available_models():
        """List all models that can be loaded (from model directories)."""
        available = []
        for key, (path, mtype, desc) in ALL_MODELS.items():
            mtime = os.path.getmtime(path)
            available.append({
                "key": key,
                "name": os.path.basename(path),
                "description": desc,
                "type": mtype,
                "modified": mtime,
                "active": (os.path.basename(path) in state["model_name"]),
            })
        return {"models": available, "current": state["model_name"]}

    @app.post("/v1/models/switch")
    async def switch_model_endpoint(req: SwitchRequest):
        """
        Hot-swap the currently loaded model.
        POST {"model": "latest"} or {"model": "Qwen3"} or {"model": "3"}
        The old model is unloaded and the new one loaded in its place.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        new_key = resolve_model_choice(req.model)
        if new_key is None:
            raise HTTPException(status_code=404, detail=f"No model matching '{req.model}'")

        new_path, new_type, new_desc = ALL_MODELS[new_key]

        # Skip if already loaded
        if os.path.basename(new_path) in state["model_name"]:
            return {"status": "already_loaded", "model": state["model_name"]}

        print(f"\n{'='*60}")
        print(f"Switching model: {state['model_name']} -> {new_desc}")
        print(f"{'='*60}")

        # Unload current model
        del state["model"]
        if state["tokenizer"] is not None:
            del state["tokenizer"]
        clear_model_memory()

        # Load new model
        print(f"Loading {new_desc} (transformers)...")
        new_tokenizer = AutoTokenizer.from_pretrained(
            new_path, trust_remote_code=True, local_files_only=True
        )
        new_model = AutoModelForCausalLM.from_pretrained(
            new_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
        )

        state["model"] = new_model
        state["tokenizer"] = new_tokenizer
        state["model_name"] = new_desc
        state["model_type"] = new_type

        print(f"Model switched to: {new_desc} (device: {new_model.device}, dtype: {new_model.dtype})")
        return {"status": "switched", "model": new_desc, "type": new_type}

    # ----------------------------------------------------------------
    # Chat Completions Endpoint
    # ----------------------------------------------------------------
    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        """
        OpenAI-compatible chat completions endpoint.
        Works with any OpenAI client library.
        Supports tool calling when tools are provided.
        """
        try:
            user_message, system_message, image_path = build_user_message(
                request.messages, request.tools
            )

            if not user_message:
                raise HTTPException(status_code=400, detail="No user message found")

            # Clean up extracted image (not used by text models, but don't leak tmp files)
            if image_path:
                try:
                    os.unlink(image_path)
                except OSError:
                    pass

            max_tokens = request.max_tokens or 512
            temperature = request.temperature or 0.7

            print(f"[DEBUG] Tools received: {len(request.tools) if request.tools else 0}")

            # Run model.generate() in executor to not block the event loop
            loop = asyncio.get_event_loop()
            response_text, prompt_tokens, completion_tokens = await loop.run_in_executor(
                None,
                functools.partial(
                    generate_response,
                    user_message, system_message, temperature, max_tokens
                )
            )

            # Always clean thinking tags from response
            response_text = clean_tool_calls_from_text(response_text)

            print(f"[DEBUG] Full response:\n{response_text[:500] if response_text else 'empty'}")

            # ----------------------------------------------------------------
            # STREAMING MODE - send full response as SSE chunks
            # ----------------------------------------------------------------
            if request.stream:
                async def stream_response():
                    chat_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
                    created = int(time.time())
                    # Send the full response as a single content chunk
                    chunk = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": state["model_name"],
                        "choices": [{
                            "index": 0,
                            "delta": {"content": response_text},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    # Send final chunk with finish_reason
                    final_chunk = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": state["model_name"],
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }]
                    }
                    yield f"data: {json.dumps(final_chunk)}\n\n"
                    yield "data: [DONE]\n\n"

                return StreamingResponse(
                    stream_response(),
                    media_type="text/event-stream"
                )

            # ----------------------------------------------------------------
            # NON-STREAMING MODE
            # ----------------------------------------------------------------
            # Parse tool calls from response if tools were requested
            tool_calls = []
            if request.tools:
                tool_calls = parse_tool_calls(response_text)
                print(f"[DEBUG] Tool calls found: {len(tool_calls)}")
                if tool_calls:
                    print(f"[DEBUG] Parsed tool: {tool_calls[0]}")
                if tool_calls:
                    # Clean tool call syntax from the text
                    response_text = clean_tool_calls_from_text(response_text)

            # Build OpenAI-format response
            if tool_calls:
                # Response with tool calls
                return {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": state["model_name"],
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response_text if response_text else None,
                            "tool_calls": tool_calls
                        },
                        "finish_reason": "tool_calls"
                    }],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens
                    }
                }
            else:
                # Regular response without tool calls
                return ChatCompletionResponse(
                    id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
                    created=int(time.time()),
                    model=state["model_name"],
                    choices=[
                        ChatCompletionChoice(
                            index=0,
                            message=ChatMessage(
                                role="assistant", content=response_text
                            ),
                            finish_reason="stop"
                        )
                    ],
                    usage=Usage(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens
                    )
                )

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ----------------------------------------------------------------
    # Server startup banner
    # ----------------------------------------------------------------
    def get_local_ip():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "localhost"

    local_ip = get_local_ip()

    print("\n" + "=" * 60)
    print("OpenAI-Compatible Transformer Server Starting")
    print("=" * 60)
    print(f"Model:  {state['model_name']}")
    print(f"Type:   {state['model_type']}")
    print(f"Engine: transformers (sequential)")
    print("=" * 60)
    print("Access URLs:")
    print(f"  Local:   http://localhost:{port}")
    print(f"  Network: http://{local_ip}:{port}")
    print("=" * 60)
    print("API Endpoints:")
    print(f"  POST http://{local_ip}:{port}/v1/chat/completions")
    print(f"  GET  http://{local_ip}:{port}/v1/models")
    print(f"  GET  http://{local_ip}:{port}/v1/models/available")
    print(f"  POST http://{local_ip}:{port}/v1/models/switch")
    print("=" * 60)
    print(f"\nExample usage with curl:")
    print(f'  curl http://{local_ip}:{port}/v1/chat/completions \\')
    print(f'    -H "Content-Type: application/json" \\')
    print(f'    -d \'{{"model": "{state["model_name"]}", '
          f'"messages": [{{"role": "user", "content": "Hello!"}}]}}\'')
    print(f"\nSwitch model via API:")
    print(f'''  curl -X POST http://{local_ip}:{port}/v1/models/switch \\
    -H "Content-Type: application/json" \\
    -d '{{"model": "latest"}}'
''')
    print("=" * 60)
    print(f"Beast .env: LFM_URL=http://{local_ip}:{port}")
    print("=" * 60)
    print("Press Ctrl+C to stop the server")
    print("=" * 60 + "\n")

    # Run the server
    uvicorn.run(app, host=host, port=port, log_level="info")


# ============================================================================
# Interactive Mode - Quick local testing with transformers
# ============================================================================
def run_interactive_mode(model, tokenizer, model_name):
    """
    Simple interactive chat using transformers.
    For quick testing without starting the full server.
    Type 'model' or 'switch' to hot-swap models.
    """
    from transformers import TextStreamer

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    print(f"\n{model_name} Interactive Chat (transformers)")
    print("=" * 50)
    print("Enter prompts below. Type 'quit' to exit, 'model' to switch models.")
    print("=" * 50 + "\n")

    while True:
        try:
            user_input = input("Prompt: ").strip()

            if user_input.lower() in {"quit", "exit", "q"}:
                print("Goodbye!")
                return False  # Don't switch, just exit

            if user_input.lower() in {"model", "switch", "m"}:
                return True  # Signal to switch models

            if not user_input:
                print("Please enter a prompt...")
                continue

            # Apply chat template
            messages = [{"role": "user", "content": user_input}]
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                tokenize=True,
            )
            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs["attention_mask"].to(model.device)

            print("Assistant: ", end="", flush=True)
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=0.7,
                max_new_tokens=2048,
                streamer=streamer,
            )

            # TTS: speak the response
            if TTS_ENABLED:
                response_text = tokenizer.decode(
                    output[0][input_ids.shape[-1]:], skip_special_tokens=True
                )
                speak(response_text)

            print("\n" + "=" * 50)

        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'quit' to exit or continue chatting.")
            continue
        except Exception as e:
            print(f"\nError: {type(e).__name__}: {str(e)}")
            print("Try again or type 'quit' to exit.")
            continue


# ============================================================================
# Main Program Loop (allows switching models)
# ============================================================================
def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    switch_model = True
    cli_model_used = False

    while switch_model:
        switch_model = False

        # ----------------------------------------------------------------
        # Model Selection
        # ----------------------------------------------------------------
        if cli_args.model and not cli_model_used:
            cli_model_used = True
            choice = resolve_model_choice(cli_args.model)
        else:
            if not ALL_MODELS:
                print("No models found in any search path:")
                for p in MODEL_SEARCH_PATHS:
                    exists = "EXISTS" if os.path.exists(p) else "NOT FOUND"
                    print(f"  {p} ({exists})")
                print("\nDownload models or update MODEL_SEARCH_PATHS "
                      "at the top of this file.")
                return

            print("=" * 50)
            print("Model Selection")
            print("=" * 50)
            for key, (path, mtype, desc) in sorted(
                ALL_MODELS.items(), key=lambda x: int(x[0])
            ):
                vl_tag = " [VL]" if mtype == "vision" else ""
                print(f"  {key}. {desc}{vl_tag}")
            print("=" * 50)

            valid_choices = set(ALL_MODELS.keys())
            while True:
                choice = input(
                    f"Select model ({'/'.join(sorted(valid_choices, key=int))}): "
                ).strip()
                if choice in valid_choices:
                    break
                print(f"Please enter one of: "
                      f"{', '.join(sorted(valid_choices, key=int))}")

        selected_path, selected_type, selected_desc = ALL_MODELS[choice]

        # ----------------------------------------------------------------
        # Mode Selection
        # ----------------------------------------------------------------
        if cli_args.server:
            run_as_server = True
        elif cli_args.interactive:
            run_as_server = False
        elif cli_args.model and cli_model_used:
            # --model without --interactive defaults to server
            run_as_server = True
        else:
            print("\nMode Selection:")
            print("  1. Server mode (OpenAI-compatible API for Beast)")
            print("  2. Interactive chat (local terminal)")
            mode_choice = input("Select mode (1 or 2): ").strip()
            run_as_server = (mode_choice != "2")

        # Server port selection (only if server mode, interactive menu)
        server_port = cli_args.port
        if run_as_server and not cli_args.server and not cli_args.model:
            port_input = input(f"Server port (default {cli_args.port}): ").strip()
            if port_input.isdigit():
                server_port = int(port_input)

        # TTS option (only for interactive mode)
        if not run_as_server:
            tts_choice = input("Read output aloud? (y/n): ").strip().lower()
            if tts_choice in ('y', 'yes'):
                if init_tts():
                    print("TTS enabled - responses will be read aloud")
                else:
                    print("Continuing without TTS")

        # ----------------------------------------------------------------
        # Load Model
        # ----------------------------------------------------------------
        print(f"\nLoading {selected_desc} with transformers...")
        tokenizer = AutoTokenizer.from_pretrained(
            selected_path, trust_remote_code=True, local_files_only=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            selected_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
        )
        print(f"Model loaded on {model.device} (dtype: {model.dtype})")

        # ----------------------------------------------------------------
        # Run
        # ----------------------------------------------------------------
        if run_as_server:
            run_server_mode(
                model, tokenizer, selected_desc, selected_type,
                host=cli_args.host, port=server_port
            )
            break  # Exit after server stops
        else:
            switch_model = run_interactive_mode(model, tokenizer, selected_desc)

        # Clean up if switching models
        if switch_model:
            print("\nClearing model from memory...")
            del model
            del tokenizer
            clear_model_memory()


if __name__ == "__main__":
    main()
