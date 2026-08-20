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
import tkinter as tk
from tkinter import filedialog
from datetime import datetime

import cv2
from PIL import Image

# Hide tkinter root for file dialogs (interactive VL mode).
tk.Tk().withdraw()

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


def speak_sync(text):
    """Speak text and wait for completion (parity with lfm_thinking.py)."""
    if TTS_ENABLED and tts_engine and text.strip():
        try:
            tts_engine.say(text)
            tts_engine.runAndWait()
        except Exception:
            pass


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
    """Short format reminder. The OpenAI tools array is the catalog."""
    if not tools:
        return ""
    return "\n".join([
        "## Tools",
        "Call tools with JSON:",
        "```tool_call",
        '{"name": "tool_name", "arguments": {"param": "value"}}',
        "```",
        "Or Qwen native: <tool_call>{\"name\": \"...\", \"arguments\": {...}}</tool_call>",
        "",
        "RULES:",
        "- Answer in text unless you need to act.",
        "- You may call several independent tools in one turn.",
        "- For Python use run_python. For HTML use run_html. For images use generate_art.",
        "- Never retry a tool call that already succeeded. When the task is done, answer in text.",
    ])


def parse_tool_calls(text):
    """
    Parse tool calls with JSON repair (aligned with lfm_thinking.py server).
    """
    tool_calls = []

    def _extract_json(s, start=0):
        idx = s.find('{', start)
        if idx < 0:
            return None, -1
        depth = 0
        in_str = False
        escape = False
        for i in range(idx, len(s)):
            c = s[i]
            if escape:
                escape = False
                continue
            if c == '\\':
                escape = True
                continue
            if c == '"' and not escape:
                in_str = not in_str
                continue
            if in_str:
                continue
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    return s[idx:i + 1], i + 1
        return None, -1

    def _repair_json(blob):
        result = []
        in_str = False
        escape = False
        for c in blob:
            if escape:
                result.append(c)
                escape = False
                continue
            if c == '\\':
                result.append(c)
                escape = True
                continue
            if c == '"':
                in_str = not in_str
                result.append(c)
                continue
            if in_str:
                if c == '\n':
                    result.append('\\n')
                    continue
                if c == '\r':
                    result.append('\\r')
                    continue
                if c == '\t':
                    result.append('\\t')
                    continue
            result.append(c)
        return ''.join(result)

    seen = set()

    def try_add(data):
        name = data.get("name")
        if not name:
            return
        args = data.get("arguments", data.get("args", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        args_str = json.dumps(args)
        key = f"{name}:{args_str}"
        if key not in seen:
            seen.add(key)
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {"name": name, "arguments": args_str}
            })

    def _try_parse(blob, label=""):
        if not blob:
            return None
        try:
            return json.loads(blob)
        except json.JSONDecodeError as e:
            print(f"[DEBUG] {label} json.loads failed: {e}")
            repaired = _repair_json(blob)
            try:
                data = json.loads(repaired)
                print(f"[DEBUG] {label} json.loads succeeded after repair")
                return data
            except json.JSONDecodeError as e2:
                print(f"[DEBUG] {label} json.loads STILL failed after repair: {e2}")
                print(f"[DEBUG] {label} blob[:200] = {blob[:200]}")
                return None

    for m in re.finditer(r'```tool(?:_call)?\s*\n', text):
        blob, _ = _extract_json(text, m.end())
        print(f"[DEBUG] Pattern A: fence at {m.start()}, blob={'found '+str(len(blob))+' chars' if blob else 'None'}")
        data = _try_parse(blob, "Pattern A")
        if data and "name" in data:
            try_add(data)

    if not tool_calls:
        for m in re.finditer(r'<tool_call>\s*', text):
            blob, _ = _extract_json(text, m.end())
            data = _try_parse(blob, "Pattern B")
            if data and "name" in data:
                try_add(data)

    if not tool_calls:
        blob, _ = _extract_json(text)
        if blob and '"name"' in blob:
            data = _try_parse(blob, "Pattern C")
            if data and "name" in data:
                try_add(data)

    # Return all parsed calls — Beast executes them sequentially and has loop detection.
    return tool_calls


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
def run_server_mode(model, tokenizer, processor, model_name, model_type,
                    host=DEFAULT_HOST, port=DEFAULT_PORT):
    """
    Run the model as an OpenAI-compatible API server using transformers.
    Supports hot-swapping models via POST /v1/models/switch.
    Vision models use processor + AutoModelForImageTextToText; text uses tokenizer + CausalLM.

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
        "processor": processor,
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

        # Add tool definitions to the prompt if provided (lfm_thinking-style)
        if tools:
            tools_prompt = format_tools_for_prompt(tools)
            if tools_prompt:
                tool_instruction = (
                    f"\n\n{tools_prompt}\n"
                    "To use a tool, respond ONLY with: ```tool_call\n"
                    '{"name": "TOOL_NAME", "arguments": {}}\n```\n'
                    "Example for web search: ```tool_call\n"
                    '{"name": "mcp_brave-search_brave_web_search", '
                    '"arguments": {"query": "intel stock price"}}\n```\n'
                    "DO NOT explain. Just output the tool_call block."
                )
                user_message = user_message + tool_instruction

        return user_message, system_message, image_path

    def _text_generate_kwargs(temperature, max_tokens):
        """Gemma-friendly sampling (parity with lfm_thinking MLX defaults)."""
        kwargs = {"do_sample": True, "max_new_tokens": max_tokens}
        if "gemma" in state["model_name"].lower():
            kwargs["temperature"] = temperature if temperature is not None else 1.0
            kwargs["top_p"] = 0.95
            kwargs["top_k"] = 64
        else:
            kwargs["temperature"] = temperature if temperature is not None else 0.7
        return kwargs

    def generate_text_response(user_message, system_message, temperature, max_tokens):
        """Causal LM + tokenizer."""
        import torch
        tok = state["tokenizer"]
        mdl = state["model"]
        chat_messages = []
        if system_message:
            chat_messages.append({"role": "system", "content": system_message})
        chat_messages.append({"role": "user", "content": user_message})
        inputs = tok.apply_chat_template(
            chat_messages,
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True,
        )
        input_ids = inputs["input_ids"].to(mdl.device)
        attention_mask = inputs["attention_mask"].to(mdl.device)
        gkw = _text_generate_kwargs(temperature, max_tokens)
        output = mdl.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gkw,
        )
        response_text = tok.decode(
            output[0][input_ids.shape[-1]:], skip_special_tokens=True
        )
        prompt_tokens = int(input_ids.shape[-1])
        completion_tokens = int(output.shape[-1] - input_ids.shape[-1])
        return response_text, prompt_tokens, completion_tokens

    def generate_vision_response(
        user_message, system_message, image_path, temperature, max_tokens
    ):
        """VLM: processor + image file path (optional) + text prompt."""
        proc = state["processor"]
        mdl = state["model"]
        user_content = []
        if image_path:
            user_content.append({
                "type": "image",
                "image": Image.open(image_path).convert("RGB"),
            })
        user_content.append({"type": "text", "text": user_message})
        msgs = []
        if system_message:
            msgs.append({"role": "system", "content": system_message})
        msgs.append({"role": "user", "content": user_content})
        inputs = proc.apply_chat_template(
            msgs,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        ).to(mdl.device)
        if "gemma" in state["model_name"].lower():
            temp = temperature if temperature is not None else 1.0
            outputs = mdl.generate(
                **inputs,
                do_sample=True,
                temperature=temp,
                top_p=0.95,
                top_k=64,
                max_new_tokens=max_tokens,
            )
        else:
            outputs = mdl.generate(
                **inputs,
                do_sample=True,
                temperature=temperature if temperature is not None else 0.7,
                max_new_tokens=max_tokens,
            )
        response_text = proc.batch_decode(outputs, skip_special_tokens=True)[0]
        if "assistant" in response_text.lower():
            response_text = response_text.split("assistant")[-1].strip()
        try:
            prompt_tokens = int(inputs["input_ids"].shape[-1])
            completion_tokens = int(outputs.shape[-1] - prompt_tokens)
        except Exception:
            prompt_tokens = len(user_message.split())
            completion_tokens = (
                len(response_text.split()) if response_text else 0
            )
        return response_text, prompt_tokens, completion_tokens

    def generate_response(user_message, system_message, temperature, max_tokens, image_path=None):
        """Dispatch text vs vision (lfm_thinking parity for Beast + local API)."""
        if state["model_type"] == "vision" and state.get("processor") is not None:
            return generate_vision_response(
                user_message, system_message, image_path,
                temperature, max_tokens,
            )
        return generate_text_response(
            user_message, system_message, temperature, max_tokens,
        )

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
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoProcessor,
            AutoTokenizer,
        )

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
        if state.get("tokenizer") is not None:
            del state["tokenizer"]
            state["tokenizer"] = None
        if state.get("processor") is not None:
            del state["processor"]
            state["processor"] = None
        clear_model_memory()

        # Load new model (text vs vision — same split as lfm_thinking transformers path)
        print(f"Loading {new_desc} (transformers)...")
        if new_type == "vision":
            new_processor = AutoProcessor.from_pretrained(
                new_path, trust_remote_code=True, local_files_only=True
            )
            new_model = AutoModelForImageTextToText.from_pretrained(
                new_path,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True,
            )
            state["model"] = new_model
            state["processor"] = new_processor
            state["tokenizer"] = None
        else:
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
            state["processor"] = None

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

            max_tokens = request.max_tokens or 512
            if request.temperature is not None:
                temperature = request.temperature
            elif "gemma" in state["model_name"].lower():
                temperature = 1.0
            else:
                temperature = 0.7

            print(f"[DEBUG] Tools received: {len(request.tools) if request.tools else 0}")

            loop = asyncio.get_event_loop()

            def _unlink_temp_image():
                if image_path:
                    try:
                        os.unlink(image_path)
                    except OSError:
                        pass

            is_vision = (
                state["model_type"] == "vision"
                and state.get("processor") is not None
            )

            # ----------------------------------------------------------------
            # STREAMING — token SSE for text (TextIteratorStreamer); one chunk for VL
            # ----------------------------------------------------------------
            if request.stream:
                if is_vision:
                    async def stream_vision_single_chunk():
                        try:
                            rt, _pt, _ct = await loop.run_in_executor(
                                None,
                                functools.partial(
                                    generate_response,
                                    user_message,
                                    system_message,
                                    temperature,
                                    max_tokens,
                                    image_path,
                                ),
                            )
                            rt = clean_tool_calls_from_text(rt)
                            chat_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
                            created = int(time.time())
                            chunk = {
                                "id": chat_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": state["model_name"],
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": rt},
                                    "finish_reason": None,
                                }],
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"
                            final_chunk = {
                                "id": chat_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": state["model_name"],
                                "choices": [{
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "stop",
                                }],
                            }
                            yield f"data: {json.dumps(final_chunk)}\n\n"
                            yield "data: [DONE]\n\n"
                        finally:
                            _unlink_temp_image()

                    return StreamingResponse(
                        stream_vision_single_chunk(),
                        media_type="text/event-stream",
                    )

                if state.get("tokenizer") is None:
                    raise HTTPException(
                        status_code=400,
                        detail="Streaming requires a text tokenizer (vision-only model loaded)",
                    )

                async def stream_text_tokens():
                    from threading import Thread
                    from transformers import TextIteratorStreamer

                    try:
                        tok = state["tokenizer"]
                        mdl = state["model"]
                        chat_messages = []
                        if system_message:
                            chat_messages.append({
                                "role": "system",
                                "content": system_message,
                            })
                        chat_messages.append({
                            "role": "user",
                            "content": user_message,
                        })
                        inputs = tok.apply_chat_template(
                            chat_messages,
                            add_generation_prompt=True,
                            return_tensors="pt",
                            tokenize=True,
                        )
                        input_ids = inputs["input_ids"].to(mdl.device)
                        attention_mask = inputs["attention_mask"].to(mdl.device)
                        streamer = TextIteratorStreamer(
                            tok, skip_prompt=True, skip_special_tokens=True
                        )
                        gkw = _text_generate_kwargs(temperature, max_tokens)
                        gen_kw = {
                            "input_ids": input_ids,
                            "attention_mask": attention_mask,
                            "streamer": streamer,
                            **gkw,
                        }
                        producer = Thread(
                            target=lambda: mdl.generate(**gen_kw),
                            daemon=True,
                        )
                        producer.start()
                        chat_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
                        created = int(time.time())
                        it = iter(streamer)
                        _done = object()

                        def _next_piece():
                            try:
                                return next(it)
                            except StopIteration:
                                return _done

                        while True:
                            piece = await loop.run_in_executor(None, _next_piece)
                            if piece is _done:
                                break
                            if piece:
                                chunk = {
                                    "id": chat_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": state["model_name"],
                                    "choices": [{
                                        "index": 0,
                                        "delta": {"content": piece},
                                        "finish_reason": None,
                                    }],
                                }
                                yield f"data: {json.dumps(chunk)}\n\n"
                        producer.join()
                        final_chunk = {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": state["model_name"],
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop",
                            }],
                        }
                        yield f"data: {json.dumps(final_chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                    finally:
                        _unlink_temp_image()

                return StreamingResponse(
                    stream_text_tokens(),
                    media_type="text/event-stream",
                )

            # ----------------------------------------------------------------
            # NON-STREAMING
            # ----------------------------------------------------------------
            try:
                response_text, prompt_tokens, completion_tokens = (
                    await loop.run_in_executor(
                        None,
                        functools.partial(
                            generate_response,
                            user_message,
                            system_message,
                            temperature,
                            max_tokens,
                            image_path,
                        ),
                    )
                )
            finally:
                _unlink_temp_image()

            response_text = clean_tool_calls_from_text(response_text)

            print(
                f"[DEBUG] Full response:\n"
                f"{response_text[:500] if response_text else 'empty'}"
            )

            tool_calls = []
            if request.tools:
                tool_calls = parse_tool_calls(response_text)
                print(f"[DEBUG] Tool calls found: {len(tool_calls)}")
                if tool_calls:
                    print(f"[DEBUG] Parsed tool: {tool_calls[0]}")
                    response_text = clean_tool_calls_from_text(response_text)

            if tool_calls:
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
                            "tool_calls": tool_calls,
                        },
                        "finish_reason": "tool_calls",
                    }],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                }
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
                        finish_reason="stop",
                    )
                ],
                usage=Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                ),
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
# Interactive Mode — text or VL (lfm_thinking.py parity, transformers only)
# ============================================================================
def run_interactive_mode(model, tokenizer, processor, model_name, model_type):
    """
    Local terminal chat: text models use streaming TextStreamer; VL models
    get image / video / text-only menus (same UX as lfm_thinking non-MLX path).
    """
    from transformers import TextStreamer

    def _vl_generate(inputs_dict, max_new_tokens):
        """Gemma-friendly VLM generate (interactive)."""
        if "gemma" in model_name.lower():
            return model.generate(
                **inputs_dict,
                do_sample=True,
                temperature=1.0,
                top_p=0.95,
                top_k=64,
                max_new_tokens=max_new_tokens,
            )
        return model.generate(
            **inputs_dict,
            do_sample=True,
            temperature=0.7,
            max_new_tokens=max_new_tokens,
        )

    if model_type == "vision" and processor is not None:
        print(f"\n{model_name} Interactive Chat (Vision — transformers)")
        print("=" * 50)
        while True:
            try:
                media_choice = input(
                    "Media? [i]mage, [v]ideo, [n]one, [m]odel switch, or [q]uit: "
                ).strip().lower()

                if media_choice in {"q", "quit", "exit"}:
                    print("Goodbye!")
                    return False
                if media_choice in {"m", "model", "switch"}:
                    return True

                if media_choice in {"v", "video"}:
                    print("Opening file dialog for video...")
                    video_path = filedialog.askopenfilename(
                        title="Select a video",
                        filetypes=[
                            ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"),
                            ("MP4", "*.mp4"),
                            ("All files", "*.*"),
                        ],
                    )
                    if not video_path:
                        print("No video selected.")
                        continue
                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpened():
                        print("Error: Could not open video.")
                        continue
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    print(f"Video: {fps:.1f} FPS, {total_frames} frames")
                    interval_input = input(
                        "Analyze every N seconds (default=2): "
                    ).strip()
                    interval_seconds = float(interval_input) if interval_input else 2.0
                    frame_interval = max(1, int(fps * interval_seconds))
                    user_input = input("Prompt: ").strip()
                    if not user_input:
                        user_input = "Describe what you see in this frame."
                    video_results = []
                    video_name = os.path.basename(video_path)
                    frame_count = 0
                    scene_count = 0
                    start_time = time.time()
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        if frame_count % frame_interval == 0:
                            scene_count += 1
                            timestamp = frame_count / fps if fps else 0
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            pil_image = Image.fromarray(frame_rgb)
                            conversation = [{
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": pil_image},
                                    {"type": "text", "text": user_input},
                                ],
                            }]
                            inputs = processor.apply_chat_template(
                                conversation,
                                add_generation_prompt=True,
                                return_tensors="pt",
                                return_dict=True,
                                tokenize=True,
                            ).to(model.device)
                            outputs = _vl_generate(inputs, 128)
                            response = processor.batch_decode(
                                outputs, skip_special_tokens=True
                            )[0]
                            if "assistant" in response.lower():
                                response = response.split("assistant")[-1].strip()
                            print(f"\n[{timestamp:.1f}s] Scene {scene_count}:\n  {response}")
                            speak(response)
                            video_results.append({
                                "timestamp": timestamp,
                                "scene": scene_count,
                                "description": response,
                            })
                        frame_count += 1
                    cap.release()
                    elapsed = time.time() - start_time
                    print(f"\nAnalysis complete: {scene_count} scenes in {elapsed:.1f}s")
                    if scene_count > 0:
                        print(f"Average: {elapsed/scene_count:.2f}s per scene")
                    save_choice = input("Save results? (y/n): ").strip().lower()
                    if save_choice in {"y", "yes"} and video_results:
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        out_fn = f"{os.path.splitext(video_name)[0]}_analysis_{ts}.txt"
                        with open(out_fn, "w") as f:
                            f.write(f"Video Analysis: {video_name}\nPrompt: {user_input}\n\n")
                            for r in video_results:
                                f.write(f"[{r['timestamp']:.1f}s] Scene {r['scene']}:\n  {r['description']}\n\n")
                        print(f"Saved to: {out_fn}")

                elif media_choice in {"i", "image"}:
                    image_path = filedialog.askopenfilename(
                        title="Select an image",
                        filetypes=[
                            ("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.webp"),
                            ("All files", "*.*"),
                        ],
                    )
                    if not image_path:
                        print("No image selected.")
                        continue
                    user_input = input("Prompt: ").strip()
                    if not user_input:
                        user_input = "Describe what you see in this image."
                    try:
                        image = Image.open(image_path)
                        conversation = [{
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": user_input},
                            ],
                        }]
                    except Exception as img_err:
                        print(f"Error loading image: {img_err}")
                        continue
                    print("Assistant: ", end="", flush=True)
                    inputs = processor.apply_chat_template(
                        conversation,
                        add_generation_prompt=True,
                        return_tensors="pt",
                        return_dict=True,
                        tokenize=True,
                    ).to(model.device)
                    outputs = _vl_generate(inputs, 512)
                    response = processor.batch_decode(
                        outputs, skip_special_tokens=True
                    )[0]
                    if "assistant" in response.lower():
                        response = response.split("assistant")[-1].strip()
                    print(response)
                    speak(response)
                    print("\n" + "=" * 50)

                elif media_choice in {"n", "none", ""}:
                    user_input = input("Prompt: ").strip()
                    if not user_input:
                        print("Please enter a prompt...")
                        continue
                    conversation = [{
                        "role": "user",
                        "content": [{"type": "text", "text": user_input}],
                    }]
                    print("Assistant: ", end="", flush=True)
                    inputs = processor.apply_chat_template(
                        conversation,
                        add_generation_prompt=True,
                        return_tensors="pt",
                        return_dict=True,
                        tokenize=True,
                    ).to(model.device)
                    outputs = _vl_generate(inputs, 512)
                    response = processor.batch_decode(
                        outputs, skip_special_tokens=True
                    )[0]
                    if "assistant" in response.lower():
                        response = response.split("assistant")[-1].strip()
                    print(response)
                    speak(response)
                    print("\n" + "=" * 50)
                else:
                    print("Invalid choice. Use: i, v, n, m, or q")

            except KeyboardInterrupt:
                print("\n\nInterrupted.")
                continue
            except Exception as e:
                print(f"\nError: {type(e).__name__}: {e}")
                continue

    # --- Text-only model ---
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
                return False

            if user_input.lower() in {"model", "switch", "m"}:
                return True

            if not user_input:
                print("Please enter a prompt...")
                continue

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
            if "gemma" in model_name.lower():
                output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.95,
                    top_k=64,
                    max_new_tokens=2048,
                    streamer=streamer,
                )
            else:
                output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    do_sample=True,
                    temperature=0.7,
                    max_new_tokens=2048,
                    streamer=streamer,
                )

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
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoProcessor,
        AutoTokenizer,
    )

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
        # Load Model (text vs vision — lfm_thinking transformers split)
        # ----------------------------------------------------------------
        print(f"\nLoading {selected_desc} with transformers...")
        tokenizer = None
        processor = None
        if selected_type == "vision":
            processor = AutoProcessor.from_pretrained(
                selected_path, trust_remote_code=True, local_files_only=True
            )
            model = AutoModelForImageTextToText.from_pretrained(
                selected_path,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True,
            )
        else:
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
                model, tokenizer, processor, selected_desc, selected_type,
                host=cli_args.host, port=server_port
            )
            break  # Exit after server stops
        else:
            switch_model = run_interactive_mode(
                model, tokenizer, processor, selected_desc, selected_type
            )

        # Clean up if switching models
        if switch_model:
            print("\nClearing model from memory...")
            del model
            if tokenizer is not None:
                del tokenizer
            if processor is not None:
                del processor
            clear_model_memory()


if __name__ == "__main__":
    main()
