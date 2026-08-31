"""
Local Model Inference Script (Text & Vision)
=============================================
Written by Jonathan M Rothberg

Dynamically scans model directories and serves any compatible model locally.
- macOS: Uses MLX (mlx-lm for text, mlx-vlm for vision)
- Linux: Falls back to transformers/PyTorch

Model type (text vs vision) is auto-detected from config.json.
Filename is "lfm_thinking.py" for legacy reasons (started with LFM-2.5 models).

USAGE:
  python lfm_thinking.py                          # Interactive model selection
  python lfm_thinking.py --model latest --server   # Serve most recent model (pm2-friendly)
  python lfm_thinking.py --model Qwen3 --server    # Serve model matching "Qwen3"
  python lfm_thinking.py --model latest --server --port 9000
  python lfm_thinking.py --list                    # List available models and exit

PM2 (process manager) — copy/paste to start as a managed background service:

  pm2 start /Users/jonathanrothberg/Agents/lfm_thinking.py \
    --name lfm-thinking --interpreter python3 \
    --max-restarts 3 --restart-delay 10000 \
    -- --model Qwen3.8-27B-mxfp8 --server

  NOTE: --max-restarts 3 prevents infinite crash loops (large models can
  take a long time to load). --restart-delay 10000 gives 10s between retries.

  pm2 status              # check running processes
  pm2 logs lfm-thinking   # view stdout/stderr
  pm2 restart lfm-thinking
  pm2 stop lfm-thinking
  pm2 delete lfm-thinking # remove from pm2 entirely before re-adding
  pm2 save                # persist across reboots (pair with: pm2 startup)
"""

import platform
import os

def _lfm_verbose() -> bool:
    """LFM_VERBOSE=1 enables parse/prompt debug prints. Off by default."""
    return os.environ.get("LFM_VERBOSE", "").strip().lower() in ("1", "true", "yes", "on")


def _dbg(msg: str):
    if _lfm_verbose():
        print(msg)

# Platform-specific imports and paths
IS_MACOS = platform.system() == "Darwin"

if IS_MACOS:
    # Use MLX on macOS
    try:
        # MLX for vision models
        from mlx_vlm import load as vlm_load, generate as vlm_generate
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_config
        MLX_VLM_AVAILABLE = True
    except ImportError:
        print("mlx-vlm not available. Install with: pip install mlx-vlm")
        MLX_VLM_AVAILABLE = False
    
    try:
        # MLX for text-only models
        from mlx_lm import load as lm_load, generate as lm_generate
        MLX_LM_AVAILABLE = True
    except ImportError:
        print("mlx-lm not available. Install with: pip install mlx-lm")
        MLX_LM_AVAILABLE = False
    
    MLX_AVAILABLE = MLX_VLM_AVAILABLE or MLX_LM_AVAILABLE
else:
    # Use transformers on Ubuntu/Linux
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from transformers.image_utils import load_image
    import torch
    MLX_VLM_AVAILABLE = False
    MLX_LM_AVAILABLE = False

from PIL import Image
from datetime import datetime
import cv2
import time
import tkinter as tk
from tkinter import filedialog
import threading
import queue
import argparse
import json
import uuid
import re

from local_harness import (
    DEFAULT_MAX_TOKENS,
    parse_tool_calls,
    clean_tool_calls_from_text,
    split_thinking,
    format_tools_for_prompt,
    to_chat_dicts,
    extract_image_from_messages,
    flatten_as_user_text,
    render_chat,
    merge_template_kwargs,
    sampling_for_model,
)

# Hide tkinter root window
tk.Tk().withdraw()

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
        # Start TTS worker thread
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
    """Speak text and wait for completion."""
    if TTS_ENABLED and tts_engine and text.strip():
        try:
            tts_engine.say(text)
            tts_engine.runAndWait()
        except:
            pass

# Blackwell GPU optimizations (Linux only)
if not IS_MACOS:
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Model directories - all subfolders are scanned and offered in the selection menu
MLX_MODELS_DIR = "/Users/jonathanrothberg/MLX_Models"          # macOS MLX models

def scan_mlx_models(models_dir):
    """
    Dynamically scan the MLX models directory and detect model types.
    Returns dict: {"1": (path, type, description), ...}
    
    Model type detection:
    - Vision models: config.json contains "vl" or "vision" in model_type, 
      or has processor_config.json
    - Text models: everything else
    """
    models = {}
    if not os.path.exists(models_dir):
        return models
    
    # Get all subdirectories (each is a model)
    model_dirs = sorted([
        d for d in os.listdir(models_dir) 
        if os.path.isdir(os.path.join(models_dir, d)) and not d.startswith('.')
    ])
    
    for idx, model_name in enumerate(model_dirs, 1):
        model_path = os.path.join(models_dir, model_name)
        config_path = os.path.join(model_path, "config.json")
        processor_path = os.path.join(model_path, "processor_config.json")
        
        # Detect model type
        model_type = "text"  # default
        
        # Check config.json for vision signals
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    mt = config.get("model_type", "").lower()
                    # Vision models: "vl", "vision", "image" in model_type
                    if any(x in mt for x in ["vl", "vision", "image"]):
                        model_type = "vision"
                    # Newer multimodal models (Qwen3.5, qwen4_exp, etc.) have vision token IDs
                    # or vision_config even without "vl" in model_type
                    if any(k in config for k in ["image_token_id", "vision_start_token_id", "vision_config"]):
                        model_type = "vision"
                    if mt in ("qwen4_exp", "qwen4-exp"):
                        model_type = "vision"
            except:
                pass

        # Also check for processor_config.json (vision models have this)
        if os.path.exists(processor_path):
            model_type = "vision"

        # Final check: if config says vision but weights don't have vision_tower,
        # the model was converted text-only (e.g., Qwen3.5 quantized without vision).
        # Check the safetensors index for actual vision weights.
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
        
        # Create description from folder name
        type_label = "(Vision-Language)" if model_type == "vision" else "(Text)"
        description = f"{model_name} {type_label}"
        
        models[str(idx)] = (model_path, model_type, description)
    
    return models


def refresh_mlx_models():
    """Re-scan MLX_Models on disk. Import-time cache goes stale if folders are moved/deleted."""
    global MLX_MODELS
    MLX_MODELS = scan_mlx_models(MLX_MODELS_DIR)


def _canonicalize_qwen4_exp_rmsnorm(model):
    """Vontra Flash-Next MLX stores Qwen4ExpRMSNorm as ones-centered gamma (~1.0).

    mlx-vlm applies y*(1+w) assuming zero-centered weights. On Vontra that
    doubles every norm and the model emits junk tokens (oMLX issue #3181).
    Subtract 1 from Qwen4ExpRMSNorm weights when we detect that dialect.
    Leave Qwen4ExpRMSNormGated alone (already ones-centered, uses weight directly).
    """
    try:
        from mlx_vlm.models.qwen4_exp.language import Qwen4ExpRMSNorm
        import mlx.core as mx
    except ImportError:
        return
    named = getattr(model, "named_modules", None)
    if named is None:
        return
    norms = [mod for _n, mod in named() if isinstance(mod, Qwen4ExpRMSNorm)]
    if not norms:
        return
    try:
        mean = float(mx.mean(norms[0].weight.astype(mx.float32)).item())
    except Exception:
        return
    if mean < 0.5:
        return
    for mod in norms:
        mod.weight = mod.weight.astype(mx.float32) - 1.0
    mx.eval(*[m.weight for m in norms])
    print(
        f"[lfm] qwen4_exp RMSNorm: ones-centered checkpoint (mean={mean:.3f}); "
        f"subtracted 1 from {len(norms)} norms",
        flush=True,
    )


# Scan models dynamically at startup
MLX_MODELS = scan_mlx_models(MLX_MODELS_DIR)

# ============================================================================
# CLI Arguments (enables headless/pm2 operation)
# ============================================================================
parser = argparse.ArgumentParser(description="Local Model Inference Server")
parser.add_argument("--model", type=str, default=None,
    help='Model to load: "latest" for most recent, or a name/substring to match (e.g. "Qwen3")')
parser.add_argument("--server", action="store_true",
    help="Start in server mode automatically (no interactive prompt)")
parser.add_argument("--port", type=int, default=8000,
    help="Server port (default: 8000)")
parser.add_argument("--list", action="store_true",
    help="List available models and exit")
cli_args = parser.parse_args()

def resolve_model_choice(model_arg):
    """
    Resolve --model argument to a model key from MLX_MODELS.
    - "latest": pick the most recently modified model directory
    - number (e.g. "3"): pick by menu number
    - string: fuzzy match against model folder names (case-insensitive)
    Returns the model key (string number) or None if no match.
    """
    refresh_mlx_models()
    if not MLX_MODELS:
        print("Error: No models found in", MLX_MODELS_DIR)
        exit(1)

    # "latest" — pick the model directory with the most recent modification time
    if model_arg.lower() == "latest":
        newest_key = None
        newest_mtime = 0
        for key, (path, _, desc) in MLX_MODELS.items():
            mtime = os.path.getmtime(path)
            if mtime > newest_mtime:
                newest_mtime = mtime
                newest_key = key
        if newest_key:
            _, _, desc = MLX_MODELS[newest_key]
            print(f"Auto-selected latest model: {desc}")
        return newest_key

    # Direct menu number (e.g. "3")
    if model_arg in MLX_MODELS:
        return model_arg

    # Substring match against folder names (case-insensitive)
    model_arg_lower = model_arg.lower()
    matches = []
    for key, (path, _, desc) in MLX_MODELS.items():
        folder_name = os.path.basename(path).lower()
        if model_arg_lower in folder_name:
            matches.append(key)

    if len(matches) == 1:
        _, _, desc = MLX_MODELS[matches[0]]
        print(f"Auto-selected model: {desc}")
        return matches[0]
    elif len(matches) > 1:
        print(f"Multiple models match '{model_arg}':")
        for key in matches:
            _, _, desc = MLX_MODELS[key]
            print(f"  {key}. {desc}")
        print("Be more specific or use the menu number.")
        exit(1)
    else:
        print(f"No model matching '{model_arg}'. Available models:")
        for key, (_, _, desc) in MLX_MODELS.items():
            print(f"  {key}. {desc}")
        exit(1)

# Handle --list
if cli_args.list:
    refresh_mlx_models()
    print("=" * 50)
    print("Available Models")
    print("=" * 50)
    for key, (path, model_type, desc) in MLX_MODELS.items():
        mtime = os.path.getmtime(path)
        from datetime import datetime
        date_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
        vl_tag = " [VL]" if model_type == "vision" else ""
        print(f"  {key}. {desc}{vl_tag}  ({date_str})")
    print("=" * 50)
    exit(0)

# ============================================================================
# OpenAI-Compatible Server Mode
# ============================================================================
def run_server_mode(model, tokenizer, processor, model_name, model_type, host="0.0.0.0", port=8000):
    """
    Run the model as an OpenAI-compatible API server.
    Accessible on local network at http://<your-ip>:port/v1/chat/completions
    Supports hot-swapping models via POST /v1/models/switch
    """
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import StreamingResponse
        from pydantic import BaseModel
        from typing import List, Optional, Union, AsyncGenerator
        import uvicorn
        import socket
        import asyncio
    except ImportError:
        print("Server mode requires fastapi and uvicorn.")
        print("Install with: pip install fastapi uvicorn")
        return

    # Mutable state so model can be hot-swapped via /v1/models/switch
    state = {
        "model": model,
        "tokenizer": tokenizer,
        "processor": processor,
        "model_name": model_name,
        "model_type": model_type,
    }

    app = FastAPI(title="Local Model Server", description="OpenAI-compatible API for local models")
    
    # Allow CORS for local network access
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Request/Response models matching OpenAI format
    class ChatMessage(BaseModel):
        model_config = {"extra": "allow"}
        role: str
        content: Union[str, List, None] = None
        tool_calls: Optional[List[dict]] = None
        tool_call_id: Optional[str] = None
        name: Optional[str] = None
        reasoning_content: Optional[str] = None  # <think> extracted from visible text
    
    class ChatCompletionRequest(BaseModel):
        model: str = "auto"
        messages: List[ChatMessage]
        temperature: Optional[float] = 0.7
        max_tokens: Optional[int] = None  # default DEFAULT_MAX_TOKENS (LFM_MAX_TOKENS)
        stream: Optional[bool] = False
        tools: Optional[List[dict]] = None  # Tool definitions for function calling
        chat_template_kwargs: Optional[dict] = None  # family extras; unknown keys dropped
    
    class ChatCompletionChoice(BaseModel):
        index: int
        message: ChatMessage
        finish_reason: str
    
    class Usage(BaseModel):
        prompt_tokens: int
        completion_tokens: int
        total_tokens: int
    
    class ChatCompletionResponse(BaseModel):
        id: str
        object: str = "chat.completion"
        created: int
        model: str
        choices: List[ChatCompletionChoice]
        usage: Usage
    
    class ModelInfo(BaseModel):
        id: str
        object: str = "model"
        created: int
        owned_by: str = "local"
    
    class ModelsResponse(BaseModel):
        object: str = "list"
        data: List[ModelInfo]
    
    # ----------------------------------------------------------------
    # Tool Calling Support — parse/clean live in local_harness.py
    # Prefer native chat-template roles + tools=; flatten only if no template.
    # ----------------------------------------------------------------
    def apply_text_chat_template(messages, request=None, tools=None):
        """apply_chat_template with family kwargs and optional tools=."""
        extra = merge_template_kwargs(
            state["model_name"],
            getattr(request, "chat_template_kwargs", None) if request else None,
        )
        return render_chat(
            state["tokenizer"], messages, tools=tools, extra_kwargs=extra,
        )


    @app.get("/")
    async def root():
        """Health check endpoint."""
        return {"status": "ok", "model": state["model_name"], "type": state["model_type"]}

    @app.get("/v1/models")
    async def list_models():
        """List available models (OpenAI-compatible)."""
        return ModelsResponse(
            data=[ModelInfo(id=state["model_name"], created=int(time.time()))]
        )

    @app.get("/v1/models/available")
    async def available_models():
        """List all models that can be loaded (from MLX_Models directory)."""
        refresh_mlx_models()
        available = []
        for key, (path, mtype, desc) in MLX_MODELS.items():
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

    class SwitchRequest(BaseModel):
        model: str  # "latest", menu number, or name substring

    @app.post("/v1/models/switch")
    async def switch_model_endpoint(req: SwitchRequest):
        """
        Hot-swap the currently loaded model.
        POST {"model": "latest"} or {"model": "Qwen3"} or {"model": "3"}
        The old model is unloaded and the new one loaded in its place.
        """
        new_key = resolve_model_choice(req.model)
        if new_key is None:
            raise HTTPException(status_code=404, detail=f"No model matching '{req.model}'")

        new_path, new_type, new_desc = MLX_MODELS[new_key]

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
        if state["processor"] is not None:
            del state["processor"]
        clear_model_memory()

        # Load new model
        use_vl = (new_type == "vision")
        if use_vl and MLX_VLM_AVAILABLE:
            print(f"Loading {new_desc} (MLX Vision)...")
            new_model, new_processor = vlm_load(new_path)
            _canonicalize_qwen4_exp_rmsnorm(new_model)
            state["model"] = new_model
            state["processor"] = new_processor
            state["tokenizer"] = None
        elif not use_vl and MLX_LM_AVAILABLE:
            print(f"Loading {new_desc} (MLX Text)...")
            new_model, new_tokenizer = lm_load(new_path)
            state["model"] = new_model
            state["tokenizer"] = new_tokenizer
            state["processor"] = None
        else:
            raise HTTPException(status_code=500, detail="Required MLX library not available for this model type")

        state["model_name"] = new_desc
        state["model_type"] = new_type

        print(f"Model switched to: {new_desc}")
        return {"status": "switched", "model": new_desc, "type": new_type}

    # ----------------------------------------------------------------
    # Streaming generator for SSE (Server-Sent Events)
    # ----------------------------------------------------------------
    async def stream_mlx_text(prompt: str, max_tokens: int, sampler=None) -> AsyncGenerator[str, None]:
        """Stream tokens from MLX text model using mlx_lm.stream_generate. prompt is already templated."""
        from mlx_lm import stream_generate

        chat_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())

        stream_kwargs = {"max_tokens": max_tokens}
        if sampler is not None:
            stream_kwargs["sampler"] = sampler

        # Stream tokens using mlx_lm's stream_generate
        for response in stream_generate(
            state["model"], state["tokenizer"], prompt=prompt, **stream_kwargs
        ):
            # response.text contains the next text segment (delta)
            if response.text:
                chunk = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": state["model_name"],
                    "choices": [{
                        "index": 0,
                        "delta": {"content": response.text},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0)  # Allow other tasks to run

            # Check if generation is complete
            if response.finish_reason:
                break

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

    def _vlm_chat_generate(chat_dicts, max_tokens, image_path, request, samp):
        """mlx-vlm path for qwen4_exp / VLMs: native roles + family sampling.

        The old path flattened to "User: ..." and called generate at temperature 0,
        which makes Flash-Next emit a single junk token.
        """
        extra = merge_template_kwargs(
            state["model_name"], getattr(request, "chat_template_kwargs", None)
        )
        template_kw = {}
        if "enable_thinking" in extra:
            template_kw["enable_thinking"] = extra["enable_thinking"]
        if "reasoning_effort" in extra:
            template_kw["reasoning_effort"] = extra["reasoning_effort"]
        if request.tools:
            template_kw["tools"] = request.tools
        formatted_prompt = apply_chat_template(
            state["processor"],
            state["model"].config,
            chat_dicts,
            num_images=1 if image_path else 0,
            **template_kw,
        )
        gen_kwargs = {
            "max_tokens": max_tokens,
            "verbose": False,
            "temperature": samp.get("temperature", 0.7),
        }
        if samp.get("top_p") is not None:
            gen_kwargs["top_p"] = samp["top_p"]
        if samp.get("top_k") is not None:
            gen_kwargs["top_k"] = samp["top_k"]
        result = vlm_generate(
            state["model"], state["processor"], formatted_prompt,
            image=image_path,
            **gen_kwargs,
        )
        return getattr(result, "text", None) or ""

    async def stream_mlx_vision(chat_dicts, max_tokens, image_path, request, samp) -> AsyncGenerator[str, None]:
        """Stream tokens from MLX vision model (with optional image)."""
        # Vision model streaming is more complex, fall back to non-streaming
        # and send as single chunk
        chat_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())

        response_text = _vlm_chat_generate(
            chat_dicts, max_tokens, image_path, request, samp
        )
        # Clean up temp image file
        if image_path:
            try:
                os.unlink(image_path)
            except OSError:
                pass

        # Send as single chunk (vision model doesn't easily support token streaming)
        chunk = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": state["model_name"],
            "choices": [{
                "index": 0,
                "delta": {"content": response_text},
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        """
        OpenAI-compatible chat completions endpoint.
        Works with any OpenAI client library.
        Supports streaming when stream=true.
        Supports tool calling when tools are provided.
        """
        try:
            # Native roles for apply_chat_template. Flatten only if the tokenizer
            # has no template (or for vision, whose MLX helper still wants a string).
            chat_dicts = to_chat_dicts(request.messages)
            image_path = extract_image_from_messages(request.messages)
            if not any((d.get("content") or d.get("tool_calls")) for d in chat_dicts):
                raise HTTPException(status_code=400, detail="No user message found")

            max_tokens = request.max_tokens or DEFAULT_MAX_TOKENS
            samp = sampling_for_model(state["model_name"], request.temperature)
            _dbg(f"[DEBUG] Tools received: {len(request.tools) if request.tools else 0}")

            prompt = apply_text_chat_template(chat_dicts, request, tools=request.tools)
            user_message, system_message = flatten_as_user_text(chat_dicts)
            if prompt is None:
                # No chat template — flatten (full history, no "summarize after tool").
                if request.tools:
                    reminder = format_tools_for_prompt(request.tools)
                    if reminder:
                        user_message = user_message + "\n\n" + reminder
                prompt = (system_message + "\n\n" if system_message else "") + user_message

            def _mlx_sampler():
                from mlx_lm.sample_utils import make_sampler
                if samp.get("top_p") is not None:
                    return make_sampler(
                        samp["temperature"],
                        top_p=samp.get("top_p", 0.95),
                        top_k=samp.get("top_k", 64),
                    )
                if request.temperature is not None or samp.get("temperature") != 0.7:
                    return make_sampler(samp["temperature"])
                return None

            # ----------------------------------------------------------------
            # STREAMING MODE
            # ----------------------------------------------------------------
            if request.stream:
                if IS_MACOS and MLX_LM_AVAILABLE and state["model_type"] == "text":
                    return StreamingResponse(
                        stream_mlx_text(prompt, max_tokens, sampler=_mlx_sampler()),
                        media_type="text/event-stream"
                    )
                elif IS_MACOS and MLX_VLM_AVAILABLE and state["model_type"] == "vision":
                    return StreamingResponse(
                        stream_mlx_vision(chat_dicts, max_tokens, image_path, request, samp),
                        media_type="text/event-stream"
                    )
                else:
                    # Transformers streaming not implemented yet, fall through to non-streaming
                    pass

            # ----------------------------------------------------------------
            # NON-STREAMING MODE (original behavior)
            # ----------------------------------------------------------------
            # Generate response based on platform and model type
            if IS_MACOS and MLX_LM_AVAILABLE and state["model_type"] == "text":
                gen_kwargs = {"max_tokens": max_tokens, "verbose": False}
                sampler = _mlx_sampler()
                if sampler is not None:
                    gen_kwargs["sampler"] = sampler

                response_text = lm_generate(
                    state["model"], state["tokenizer"], prompt=prompt,
                    **gen_kwargs
                )

            elif IS_MACOS and MLX_VLM_AVAILABLE and state["model_type"] == "vision":
                # mlx-vlm: native messages + family sampling (qwen4_exp chat template).
                response_text = _vlm_chat_generate(
                    chat_dicts, max_tokens, image_path, request, samp
                )
                # Clean up temp image file
                if image_path:
                    try:
                        os.unlink(image_path)
                    except OSError:
                        pass

            else:
                # Transformers (Linux) — native messages when the template exists
                extra = merge_template_kwargs(
                    state["model_name"], getattr(request, "chat_template_kwargs", None)
                )
                inputs = render_chat(
                    state["tokenizer"], chat_dicts, tools=request.tools,
                    extra_kwargs=extra, tokenize=True, return_tensors="pt",
                )
                if inputs is None:
                    fallback = [{"role": "system", "content": system_message}] if system_message else []
                    fallback.append({"role": "user", "content": user_message})
                    inputs = state["tokenizer"].apply_chat_template(
                        fallback,
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
                    temperature=samp["temperature"],
                    max_new_tokens=max_tokens,
                )
                response_text = state["tokenizer"].decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)

            # mlx_lm may return None; mlx_vlm GenerationResult.text may be None — parsers need a str.
            if response_text is None:
                response_text = ""
            elif not isinstance(response_text, str):
                response_text = getattr(response_text, "text", None) or str(response_text)

            # Parse tools from the raw output, then strip think/tool syntax from user-visible text.
            tool_calls = parse_tool_calls(response_text) if request.tools else []
            visible, reasoning = split_thinking(response_text)
            response_text = clean_tool_calls_from_text(visible)
            _dbg(f"[DEBUG] Full response ({len(response_text or '')} chars visible, {len(reasoning)} reasoning)")
            _dbg(f"[DEBUG] Tool calls found: {len(tool_calls)}")

            msg_out = {
                "role": "assistant",
                "content": response_text if response_text else None,
            }
            if reasoning:
                msg_out["reasoning_content"] = reasoning
            finish = "stop"
            if tool_calls:
                msg_out["tool_calls"] = tool_calls
                finish = "tool_calls"
            prompt_words = len((prompt or user_message or "").split())
            return {
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": state["model_name"],
                "choices": [{
                    "index": 0,
                    "message": msg_out,
                    "finish_reason": finish
                }],
                "usage": {
                    "prompt_tokens": prompt_words,
                    "completion_tokens": len((response_text or "").split()),
                    "total_tokens": prompt_words + len((response_text or "").split())
                }
            }
            
        except HTTPException:
            raise  # Let HTTP errors (like 400 for image rejection) pass through
        except Exception as e:
            # One line so uvicorn logs show the real failure (otherwise 500 is opaque).
            print(f"[ERROR] /v1/chat/completions: {type(e).__name__}: {e}", flush=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    # Get local IP for display
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
    print("🚀 OpenAI-Compatible Server Starting")
    print("=" * 60)
    print(f"Model: {state['model_name']}")
    print(f"Type:  {state['model_type']}")
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
    print("\nExample usage with curl:")
    print(f'''  curl http://{local_ip}:{port}/v1/chat/completions \\
    -H "Content-Type: application/json" \\
    -d '{{"model": "{state['model_name']}", "messages": [{{"role": "user", "content": "Hello!"}}]}}'
''')
    print("Switch model via API:")
    print(f'''  curl -X POST http://{local_ip}:{port}/v1/models/switch \\
    -H "Content-Type: application/json" \\
    -d '{{"model": "latest"}}'
''')
    print("=" * 60)
    print("Press Ctrl+C to stop the server")
    print("=" * 60 + "\n")
    
    # Run the server
    uvicorn.run(app, host=host, port=port, log_level="info")

# ============================================================================
# Helper function to clear model from memory
# ============================================================================
def clear_model_memory():
    """Clear model from memory on both macOS (MLX) and Ubuntu (PyTorch)."""
    import gc
    gc.collect()
    
    if IS_MACOS:
        try:
            import mlx.core as mx
            mx.clear_cache()
        except:
            pass
    else:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except:
            pass

# ============================================================================
# Main Program Loop (allows switching models)
# ============================================================================
switch_model = True  # Start by selecting a model
cli_model_used = False  # Track if --model was used (first iteration only)

while switch_model:
    switch_model = False  # Reset flag

    # ============================================================================
    # Model Selection
    # ============================================================================
    # If --model was passed on CLI (first time only), skip the interactive menu
    if cli_args.model and not cli_model_used:
        cli_model_used = True
        choice = resolve_model_choice(cli_args.model)
    else:
        refresh_mlx_models()
        print("=" * 50)
        print("Model Selection")
        print("=" * 50)

        # Show all dynamically scanned models
        for key, (path, model_type, desc) in MLX_MODELS.items():
            vl_tag = " [VL]" if model_type == "vision" else ""
            print(f"{key}. {desc}{vl_tag}")

        print("=" * 50)

        valid_choices = set(MLX_MODELS.keys())
        while True:
            choice = input(f"Select model ({'/'.join(sorted(valid_choices))}): ").strip()
            if choice in valid_choices:
                break
            print(f"Please enter one of: {', '.join(sorted(valid_choices))}")

    selected_path, selected_type, selected_desc = MLX_MODELS[choice]
    use_vl_model = (selected_type == "vision")

    # ============================================================================
    # Mode Selection: Interactive or Server
    # ============================================================================
    # If --server was passed on CLI, skip the interactive prompt
    if cli_args.server:
        run_as_server = True
    else:
        print("\nMode Selection:")
        print("1. Interactive chat (local terminal)")
        print("2. Server mode (OpenAI-compatible API on network)")
        mode_choice = input("Select mode (1 or 2): ").strip()
        run_as_server = (mode_choice == "2")

    # Server port selection (only if server mode)
    server_port = cli_args.port
    if run_as_server and not cli_args.server:
        # Only ask interactively if --server wasn't used
        port_input = input(f"Server port (default {cli_args.port}): ").strip()
        if port_input.isdigit():
            server_port = int(port_input)

    # ============================================================================
    # TTS Option (only for interactive mode)
    # ============================================================================
    if not run_as_server:
        tts_choice = input("Read output aloud? (y/n): ").strip().lower()
        if tts_choice in ('y', 'yes'):
            if init_tts():
                print("TTS enabled - responses will be read aloud")
            else:
                print("Continuing without TTS")
    print("=" * 50)

    # ============================================================================
    # Load Selected Model
    # ============================================================================
    if use_vl_model and MLX_VLM_AVAILABLE:
        print(f"\nLoading {selected_desc} (MLX Vision)...")
        try:
            model, processor = vlm_load(selected_path)
            _canonicalize_qwen4_exp_rmsnorm(model)
        except Exception as e:
            print(f"mlx-vlm failed to load this VLM: {type(e).__name__}: {e}")
            print("Qwen3.8 VLMs need mlx-vlm>=0.6.8. Flash-Next (qwen4_exp) needs mlx-vlm git main.")
            print("Fix:  pip install -U git+https://github.com/Blaizzy/mlx-vlm.git")
            raise
        tokenizer = None  # VLM uses processor
        
        # If server mode, start server and skip interactive loop
        if run_as_server:
            run_server_mode(model, tokenizer, processor, selected_desc, "vision", port=server_port)
            switch_model = False  # Exit after server stops
            break
        
        print(f"\n{selected_desc} Interactive Chat")
        print("=" * 50)
        print("Type your question, then choose media type.")
        print("Type 'quit' to exit.")
        print("=" * 50 + "\n")
    elif not use_vl_model and MLX_LM_AVAILABLE:
        print(f"\nLoading {selected_desc} (MLX Text)...")
        model, tokenizer = lm_load(selected_path)
        processor = None  # Text model uses tokenizer
        
        # If server mode, start server and skip interactive loop
        if run_as_server:
            run_server_mode(model, tokenizer, processor, selected_desc, "text", port=server_port)
            switch_model = False  # Exit after server stops
            break
        
        print(f"\n{selected_desc} Interactive Chat")
        print("=" * 50)
        print("Enter prompts below. Type 'quit' to exit, 'model' to switch models.")
        print("=" * 50 + "\n")
    else:
        print(f"Error: Required MLX library not available for this model type.")
        print("Install with: pip install mlx-vlm mlx-lm")
        exit(1)

    # ============================================================================
    # Main Chat Loop
    # ============================================================================
    while True:
        try:
            if use_vl_model:
                # VL model: ask for media type FIRST
                media_choice = input("Media? [i]mage, [v]ideo, [n]one, [m]odel switch, or [q]uit: ").strip().lower()

                if media_choice in {"q", "quit", "exit"}:
                    print("Goodbye!")
                    break
                
                if media_choice in {"m", "model", "switch"}:
                    # Switch model - set flag and break to restart
                    switch_model = True
                    break

                # ----------------------------------------------------------------
                # VIDEO MODE: Extract frames at interval and describe each
                # ----------------------------------------------------------------
                if media_choice in {"v", "video"}:
                    print("Opening file dialog for video...")
                    video_path = filedialog.askopenfilename(
                        title="Select a video",
                        filetypes=[
                            ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"),
                            ("MP4", "*.mp4"),
                            ("AVI", "*.avi"),
                            ("All files", "*.*"),
                        ]
                    )

                    if not video_path:
                        print("No video selected.")
                        continue

                    print(f"Loading video: {video_path}")
                    cap = cv2.VideoCapture(video_path)

                    if not cap.isOpened():
                        print("Error: Could not open video.")
                        continue

                    # Get video metadata
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = total_frames / fps if fps > 0 else 0

                    print(f"Video: {fps:.1f} FPS, {total_frames} frames, {duration:.1f}s duration")

                    # Set frame sampling interval
                    interval_input = input("Analyze every N seconds (default=2): ").strip()
                    interval_seconds = float(interval_input) if interval_input else 2.0
                    frame_interval = int(fps * interval_seconds)

                    # Ask for prompt AFTER selecting video
                    user_input = input("Prompt: ").strip()
                    if not user_input:
                        user_input = "Describe what you see in this frame."

                    print(f"Sampling every {interval_seconds}s ({frame_interval} frames)")
                    print("=" * 50)
                    print("VIDEO SCENE DESCRIPTIONS:")
                    print("=" * 50)

                    # Store results for optional save
                    video_results = []
                    video_name = os.path.basename(video_path)
                    frame_count = 0
                    scene_count = 0
                    start_time = time.time()

                    # Process video frame by frame, analyze at intervals
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        if frame_count % frame_interval == 0:
                            scene_count += 1
                            timestamp = frame_count / fps

                            # Convert OpenCV BGR to RGB PIL Image
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            pil_image = Image.fromarray(frame_rgb)

                            # Build conversation with image
                            conversation = [
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "image", "image": pil_image},
                                        {"type": "text", "text": user_input},
                                    ],
                                },
                            ]

                            # Generate description
                            if IS_MACOS and MLX_AVAILABLE and use_vl_model:
                                # Use MLX API - save frame to temp file for MLX
                                import tempfile
                                temp_path = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False).name
                                pil_image.save(temp_path)
                                formatted_prompt = apply_chat_template(
                                    processor, model.config, user_input, num_images=1
                                )
                                result = vlm_generate(
                                    model, processor, formatted_prompt, image=temp_path,
                                    max_tokens=512, verbose=False
                                )
                                response = result.text  # GenerationResult has .text attribute
                                os.unlink(temp_path)  # Clean up temp file
                            else:
                                # Use transformers API
                                inputs = processor.apply_chat_template(
                                    conversation,
                                    add_generation_prompt=True,
                                    return_tensors="pt",
                                    return_dict=True,
                                    tokenize=True,
                                ).to(model.device)

                                outputs = model.generate(**inputs, max_new_tokens=128)
                                response = processor.batch_decode(outputs, skip_special_tokens=True)[0]

                                # Extract assistant response
                                if "assistant" in response.lower():
                                    response = response.split("assistant")[-1].strip()

                            print(f"\n[{timestamp:.1f}s] Scene {scene_count}:")
                            print(f"  {response}")
                            speak(response)  # TTS for each scene

                            # Store for save
                            video_results.append({
                                "timestamp": timestamp,
                                "scene": scene_count,
                                "description": response
                            })

                        frame_count += 1

                    cap.release()
                    elapsed = time.time() - start_time

                    print("\n" + "=" * 50)
                    print(f"Analysis complete: {scene_count} scenes in {elapsed:.1f}s")
                    print(f"Average: {elapsed/scene_count:.2f}s per scene")
                    print("=" * 50)

                    # Offer to save results
                    save_choice = input("Save results? (y/n): ").strip().lower()
                    if save_choice in {"y", "yes"}:
                        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                        video_basename = os.path.splitext(video_name)[0]
                        output_filename = f"{video_basename}_analysis_{timestamp_str}.txt"

                        with open(output_filename, "w") as f:
                            f.write(f"Video Analysis: {video_name}\n")
                            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                            f.write(f"Prompt: {user_input}\n")
                            f.write(f"Interval: {interval_seconds}s\n")
                            f.write(f"Total scenes: {scene_count}\n")
                            f.write(f"Processing time: {elapsed:.1f}s\n")
                            f.write("=" * 50 + "\n\n")

                            for result in video_results:
                                f.write(f"[{result['timestamp']:.1f}s] Scene {result['scene']}:\n")
                                f.write(f"  {result['description']}\n\n")

                        print(f"Saved to: {output_filename}")

                # ----------------------------------------------------------------
                # IMAGE MODE: Single image analysis
                # ----------------------------------------------------------------
                elif media_choice in {"i", "image"}:
                    print("Opening file dialog for image...")
                    image_path = filedialog.askopenfilename(
                        title="Select an image",
                        filetypes=[
                            ("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.webp"),
                            ("PNG", "*.png"),
                            ("JPEG", "*.jpg *.jpeg"),
                            ("All files", "*.*"),
                        ]
                    )

                    if not image_path:
                        print("No image selected.")
                        continue

                    # Ask for prompt AFTER selecting image
                    user_input = input("Prompt: ").strip()
                    if not user_input:
                        user_input = "Describe what you see in this image."

                    try:
                        print(f"Loading image: {image_path}")
                        image = Image.open(image_path)
                        conversation = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": image},
                                    {"type": "text", "text": user_input},
                                ],
                            },
                        ]
                    except Exception as img_err:
                        print(f"Error loading image: {img_err}")
                        continue

                    print("Assistant: ", end="", flush=True)

                    if IS_MACOS and MLX_AVAILABLE and use_vl_model:
                        # Use MLX API - format prompt with chat template first
                        formatted_prompt = apply_chat_template(
                            processor, model.config, user_input, num_images=1
                        )
                        result = vlm_generate(
                            model, processor, formatted_prompt, image=image_path,
                            max_tokens=512, verbose=False
                        )
                        response = result.text  # GenerationResult has .text attribute
                        print(response)
                    else:
                        # Use transformers API
                        inputs = processor.apply_chat_template(
                            conversation,
                            add_generation_prompt=True,
                            return_tensors="pt",
                            return_dict=True,
                            tokenize=True,
                        ).to(model.device)

                        outputs = model.generate(**inputs, max_new_tokens=512)
                        response = processor.batch_decode(outputs, skip_special_tokens=True)[0]

                        if "assistant" in response.lower():
                            response = response.split("assistant")[-1].strip()

                        print(response)
                    speak(response)  # TTS for image response
                    print("\n" + "=" * 50)

                # ----------------------------------------------------------------
                # TEXT ONLY MODE
                # ----------------------------------------------------------------
                elif media_choice in {"n", "none", ""}:
                    # Ask for prompt
                    user_input = input("Prompt: ").strip()
                    if not user_input:
                        print("Please enter a prompt...")
                        continue

                    conversation = [{"role": "user", "content": [{"type": "text", "text": user_input}]}]

                    print("Assistant: ", end="", flush=True)

                    if IS_MACOS and MLX_AVAILABLE and use_vl_model:
                        # Use MLX API for text-only with VL model
                        formatted_prompt = apply_chat_template(
                            processor, model.config, user_input, num_images=0
                        )
                        result = vlm_generate(
                            model, processor, formatted_prompt, image=None,
                            max_tokens=512, verbose=False
                        )
                        response = result.text  # GenerationResult has .text attribute
                        print(response)
                    else:
                        # Use transformers API
                        inputs = processor.apply_chat_template(
                            conversation,
                            add_generation_prompt=True,
                            return_tensors="pt",
                            return_dict=True,
                            tokenize=True,
                        ).to(model.device)

                        outputs = model.generate(**inputs, max_new_tokens=512)
                        response = processor.batch_decode(outputs, skip_special_tokens=True)[0]

                        if "assistant" in response.lower():
                            response = response.split("assistant")[-1].strip()

                        print(response)
                    speak(response)  # TTS for text response
                    print("\n" + "=" * 50)

                # ----------------------------------------------------------------
                # INVALID CHOICE
                # ----------------------------------------------------------------
                else:
                    print("Invalid choice. Use: i, v, n, or q")
                    continue

            # ====================================================================
            # Text-Only Model Mode
            # ====================================================================
            else:
                user_input = input("Prompt: ").strip()

                if user_input.lower() in {"quit", "exit", "q"}:
                    print("Goodbye!")
                    break
                
                if user_input.lower() in {"model", "switch", "m"}:
                    # Switch model - set flag and break to restart
                    switch_model = True
                    break

                if not user_input:
                    print("Please enter a prompt...")
                    continue

                print("Assistant: ", end="", flush=True)

                if IS_MACOS and MLX_LM_AVAILABLE:
                    # Use MLX-LM for text-only models on macOS
                    messages = [{"role": "user", "content": user_input}]
                    if tokenizer.chat_template is not None:
                        prompt = tokenizer.apply_chat_template(
                            messages, add_generation_prompt=True, return_dict=False,
                        )
                    else:
                        prompt = user_input
                    
                    # Gemma 4 recommended: temperature=1.0, top_p=0.95, top_k=64
                    interactive_kwargs = {"max_tokens": 15120, "verbose": True}
                    if "gemma" in model_name.lower():
                        from mlx_lm.sample_utils import make_sampler
                        interactive_kwargs["sampler"] = make_sampler(1.0, top_p=0.95, top_k=64)
                    response = lm_generate(model, tokenizer, prompt=prompt, **interactive_kwargs)
                    
                    # TTS: speak the response
                    if TTS_ENABLED:
                        speak(response)
                else:
                    # Use transformers on Ubuntu/Linux
                    messages = [{"role": "user", "content": user_input}]
                    inputs = tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        return_tensors="pt",
                        tokenize=True,
                    )

                    input_ids = inputs["input_ids"].to(model.device)
                    attention_mask = inputs["attention_mask"].to(model.device)

                    output = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        do_sample=True,
                        temperature=0.1,
                        top_k=50,
                        top_p=0.1,
                        repetition_penalty=1.05,
                        max_new_tokens=512,
                        streamer=streamer,
                    )

                    # TTS: decode and speak the response
                    if TTS_ENABLED:
                        response_text = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)
                        speak(response_text)

                print("\n" + "=" * 50)

        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'quit' to exit or continue chatting.")
            continue
        except Exception as e:
            print(f"\nError: {type(e).__name__}: {str(e)}")
            print("Try again or type 'quit' to exit.")
            continue

    # End of inner chat loop - check if switching models
    if switch_model:
        print("\nClearing model from memory...")
        # Delete model references
        del model
        if 'processor' in dir() and processor is not None:
            del processor
        if 'tokenizer' in dir() and tokenizer is not None:
            del tokenizer
        clear_model_memory()
        print("Memory cleared. Returning to model selection...\n")
