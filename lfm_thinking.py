"""
LFM-2.5 Inference Script (Text & Vision)
========================================
Written by Jonathan M Rothberg

Runs LiquidAI's LFM models locally:
- LFM2.5-1.2B-Thinking: Text-only reasoning model
- LFM2.5-VL-1.6B: Vision-Language model with image/video support

USAGE: python lfm_thinking.py
"""

import platform
import os

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

# Local model paths - platform specific
# macOS MLX models directory
MLX_MODELS_DIR = "/Users/jonathanrothberg/MLX_Models"
# Linux transformers models directory
LINUX_MODELS_DIR = "/home/jonathan/Models_Transformer"

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
        
        # Check config.json for model_type field
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    mt = config.get("model_type", "").lower()
                    # Vision models typically have "vl", "vision", "image" in model_type
                    if any(x in mt for x in ["vl", "vision", "image"]):
                        model_type = "vision"
            except:
                pass
        
        # Also check for processor_config.json (vision models have this)
        if os.path.exists(processor_path):
            model_type = "vision"
        
        # Create description from folder name
        type_label = "(Vision-Language)" if model_type == "vision" else "(Text)"
        description = f"{model_name} {type_label}"
        
        models[str(idx)] = (model_path, model_type, description)
    
    return models

def scan_linux_models(models_dir):
    """
    Dynamically scan the Linux transformers models directory.
    Returns dict: {"1": (path, type, description), ...}
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
        
        # Detect model type from config
        model_type = "text"
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    mt = config.get("model_type", "").lower()
                    if any(x in mt for x in ["vl", "vision", "image"]):
                        model_type = "vision"
            except:
                pass
        
        type_label = "(Vision-Language)" if model_type == "vision" else "(Text)"
        description = f"{model_name} {type_label}"
        
        models[str(idx)] = (model_path, model_type, description)
    
    return models

# Scan models dynamically at startup
if IS_MACOS:
    MLX_MODELS = scan_mlx_models(MLX_MODELS_DIR)
else:
    MLX_MODELS = scan_linux_models(LINUX_MODELS_DIR)

# Fallback paths for Linux (legacy support)
TEXT_MODEL_PATH = "/home/jonathan/Models_Transformer/LFM2.5-1.2B-Thinking"
VL_MODEL_PATH = "/home/jonathan/Models_Transformer/LFM2.5-VL-1.6B"

# ============================================================================
# OpenAI-Compatible Server Mode
# ============================================================================
def run_server_mode(model, tokenizer, processor, model_name, model_type, host="0.0.0.0", port=8000):
    """
    Run the model as an OpenAI-compatible API server.
    Accessible on local network at http://<your-ip>:port/v1/chat/completions
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
    
    app = FastAPI(title="LFM Local Server", description="OpenAI-compatible API for local LFM models")
    
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
        role: str
        content: Union[str, List, None] = None
        tool_calls: Optional[List[dict]] = None
    
    class ChatCompletionRequest(BaseModel):
        model: str = model_name
        messages: List[ChatMessage]
        temperature: Optional[float] = 0.7
        max_tokens: Optional[int] = 512
        stream: Optional[bool] = False
        tools: Optional[List[dict]] = None  # Tool definitions for function calling
    
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
    # Tool Calling Support
    # ----------------------------------------------------------------
    def format_tools_for_prompt(tools):
        """Format tools into a prompt section for the model."""
        if not tools:
            return ""
        
        tool_text = "\n\n## Available Tools\n"
        tool_text += "You can call tools by outputting a tool_call block like this:\n"
        tool_text += "```tool_call\n{\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}\n```\n\n"
        tool_text += "Available tools:\n"
        
        for tool in tools:
            func = tool.get("function", tool)
            name = func.get("name", "unknown")
            desc = func.get("description", "")
            params = func.get("parameters", {}).get("properties", {})
            param_names = ", ".join(params.keys())
            tool_text += f"- **{name}**({param_names}): {desc}\n"
        
        tool_text += "\nIf you need to use a tool, output ONLY the tool_call JSON block. Keep responses concise.\n"
        return tool_text

    def parse_tool_calls(text):
        """Parse tool calls from model output."""
        tool_calls = []
        
        # Pattern 1: ```tool_call\n{...}\n```
        pattern1 = r'```tool_call\s*\n?\s*(\{[^}]+\})\s*\n?```'
        matches = re.findall(pattern1, text, re.DOTALL)
        
        # Pattern 2: <tool_call>{...}</tool_call> (Qwen3 native format)
        pattern2 = r'<tool_call>\s*(\{[^}]+\})\s*</tool_call>'
        matches += re.findall(pattern2, text, re.DOTALL)
        
        # Pattern 3: Raw JSON {"name": "...", "arguments": {...}}
        pattern3 = r'\{"name":\s*"(\w+)",\s*"arguments":\s*(\{[^}]*\})\}'
        json_matches = re.findall(pattern3, text)
        
        for match in matches:
            try:
                data = json.loads(match)
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": data.get("name"),
                        "arguments": json.dumps(data.get("arguments", data.get("args", {})))
                    }
                })
            except json.JSONDecodeError:
                pass
        
        for name, args_str in json_matches:
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": args_str
                }
            })
        
        return tool_calls

    def clean_tool_calls_from_text(text):
        """Remove tool call blocks from text."""
        text = re.sub(r'```tool_call\s*\n?\s*\{[^}]+\}\s*\n?```', '', text, flags=re.DOTALL)
        text = re.sub(r'<tool_call>\s*\{[^}]+\}\s*</tool_call>', '', text, flags=re.DOTALL)
        text = re.sub(r'\{"name":\s*"\w+",\s*"arguments":\s*\{[^}]*\}\}', '', text)
        return text.strip()
    
    @app.get("/")
    async def root():
        """Health check endpoint."""
        return {"status": "ok", "model": model_name, "type": model_type}
    
    @app.get("/v1/models")
    async def list_models():
        """List available models (OpenAI-compatible)."""
        return ModelsResponse(
            data=[ModelInfo(id=model_name, created=int(time.time()))]
        )
    
    # ----------------------------------------------------------------
    # Streaming generator for SSE (Server-Sent Events)
    # ----------------------------------------------------------------
    async def stream_mlx_text(user_message: str, max_tokens: int) -> AsyncGenerator[str, None]:
        """Stream tokens from MLX text model using mlx_lm.stream_generate."""
        from mlx_lm import stream_generate
        
        chat_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())
        
        # Prepare prompt with chat template
        messages = [{"role": "user", "content": user_message}]
        if tokenizer.chat_template is not None:
            prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_dict=False,
            )
        else:
            prompt = user_message
        
        # Stream tokens using mlx_lm's stream_generate
        for response in stream_generate(
            model, tokenizer, prompt=prompt, max_tokens=max_tokens
        ):
            # response.text contains the next text segment (delta)
            if response.text:
                chunk = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
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
            "model": model_name,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"
    
    async def stream_mlx_vision(user_message: str, max_tokens: int) -> AsyncGenerator[str, None]:
        """Stream tokens from MLX vision model (text-only mode)."""
        # Vision model streaming is more complex, fall back to non-streaming
        # and send as single chunk
        chat_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())
        
        formatted_prompt = apply_chat_template(
            processor, model.config, user_message, num_images=0
        )
        result = vlm_generate(
            model, processor, formatted_prompt, image=None,
            max_tokens=max_tokens, verbose=False
        )
        response_text = result.text
        
        # Send as single chunk (vision model doesn't easily support token streaming)
        chunk = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
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
            # Extract the last user message for simple single-turn
            # (Multi-turn conversation would need more logic)
            user_message = ""
            system_message = ""
            for msg in request.messages:
                if msg.role == "user":
                    content = msg.content
                    # Handle content as string or list
                    if isinstance(content, list):
                        user_message = " ".join(
                            item.get("text", "") for item in content 
                            if isinstance(item, dict) and item.get("type") == "text"
                        )
                    else:
                        user_message = content or ""
                elif msg.role == "system":
                    system_message = msg.content if isinstance(msg.content, str) else ""
            
            if not user_message:
                raise HTTPException(status_code=400, detail="No user message found")
            
            # Add tool definitions to the prompt if provided
            tools_prompt = format_tools_for_prompt(request.tools) if request.tools else ""
            if tools_prompt:
                if system_message:
                    system_message = system_message + tools_prompt
                else:
                    user_message = tools_prompt + "\n\nUser request: " + user_message
            
            max_tokens = request.max_tokens or 512
            
            # ----------------------------------------------------------------
            # STREAMING MODE
            # ----------------------------------------------------------------
            if request.stream:
                if IS_MACOS and MLX_LM_AVAILABLE and model_type == "text":
                    return StreamingResponse(
                        stream_mlx_text(user_message, max_tokens),
                        media_type="text/event-stream"
                    )
                elif IS_MACOS and MLX_VLM_AVAILABLE and model_type == "vision":
                    return StreamingResponse(
                        stream_mlx_vision(user_message, max_tokens),
                        media_type="text/event-stream"
                    )
                else:
                    # Transformers streaming not implemented yet, fall through to non-streaming
                    pass
            
            # ----------------------------------------------------------------
            # NON-STREAMING MODE (original behavior)
            # ----------------------------------------------------------------
            # Generate response based on platform and model type
            if IS_MACOS and MLX_LM_AVAILABLE and model_type == "text":
                # MLX text model
                messages = [{"role": "user", "content": user_message}]
                if tokenizer.chat_template is not None:
                    prompt = tokenizer.apply_chat_template(
                        messages, add_generation_prompt=True, return_dict=False,
                    )
                else:
                    prompt = user_message
                
                response_text = lm_generate(
                    model, tokenizer, prompt=prompt, 
                    max_tokens=max_tokens, 
                    verbose=False
                )
                
            elif IS_MACOS and MLX_VLM_AVAILABLE and model_type == "vision":
                # MLX vision model (text-only mode)
                formatted_prompt = apply_chat_template(
                    processor, model.config, user_message, num_images=0
                )
                result = vlm_generate(
                    model, processor, formatted_prompt, image=None,
                    max_tokens=max_tokens, verbose=False
                )
                response_text = result.text
                
            else:
                # Transformers (Linux)
                messages = [{"role": "user", "content": user_message}]
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
                    temperature=request.temperature or 0.7,
                    max_new_tokens=max_tokens,
                )
                response_text = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)
            
            # Parse tool calls from response if tools were requested
            tool_calls = []
            if request.tools:
                tool_calls = parse_tool_calls(response_text)
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
                    "model": model_name,
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
                        "prompt_tokens": len(user_message.split()),
                        "completion_tokens": len(response_text.split()) if response_text else 0,
                        "total_tokens": len(user_message.split()) + (len(response_text.split()) if response_text else 0)
                    }
                }
            else:
                # Regular response without tool calls
                return ChatCompletionResponse(
                    id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
                    created=int(time.time()),
                    model=model_name,
                    choices=[
                        ChatCompletionChoice(
                            index=0,
                            message=ChatMessage(role="assistant", content=response_text),
                            finish_reason="stop"
                        )
                    ],
                    usage=Usage(
                        prompt_tokens=len(user_message.split()),
                        completion_tokens=len(response_text.split()),
                        total_tokens=len(user_message.split()) + len(response_text.split())
                    )
                )
            
        except Exception as e:
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
    print(f"Model: {model_name}")
    print(f"Type:  {model_type}")
    print("=" * 60)
    print("Access URLs:")
    print(f"  Local:   http://localhost:{port}")
    print(f"  Network: http://{local_ip}:{port}")
    print("=" * 60)
    print("API Endpoints:")
    print(f"  POST http://{local_ip}:{port}/v1/chat/completions")
    print(f"  GET  http://{local_ip}:{port}/v1/models")
    print("=" * 60)
    print("\nExample usage with curl:")
    print(f'''  curl http://{local_ip}:{port}/v1/chat/completions \\
    -H "Content-Type: application/json" \\
    -d '{{"model": "{model_name}", "messages": [{{"role": "user", "content": "Hello!"}}]}}'
''')
    print("Example usage with Python OpenAI client:")
    print(f'''  from openai import OpenAI
  client = OpenAI(base_url="http://{local_ip}:{port}/v1", api_key="not-needed")
  response = client.chat.completions.create(
      model="{model_name}",
      messages=[{{"role": "user", "content": "Hello!"}}]
  )
  print(response.choices[0].message.content)
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

while switch_model:
    switch_model = False  # Reset flag
    
    # ============================================================================
    # Model Selection
    # ============================================================================
    print("=" * 50)
    print("Model Selection")
    print("=" * 50)

    if IS_MACOS:
        # Show all MLX models on macOS
        for key, (path, model_type, desc) in MLX_MODELS.items():
            vl_tag = " [VL]" if model_type == "vision" else ""
            print(f"{key}. {desc}{vl_tag}")
    else:
        # Ubuntu: show original options
        print("1. LFM2.5-1.2B-Thinking (Text-only, reasoning)")
        print("2. LFM2.5-VL-1.6B (Vision-Language, images + video)")

    print("=" * 50)

    if IS_MACOS:
        valid_choices = set(MLX_MODELS.keys())
        while True:
            choice = input(f"Select model ({'/'.join(sorted(valid_choices))}): ").strip()
            if choice in valid_choices:
                break
            print(f"Please enter one of: {', '.join(sorted(valid_choices))}")
        
        selected_path, selected_type, selected_desc = MLX_MODELS[choice]
        use_vl_model = (selected_type == "vision")
    else:
        while True:
            choice = input("Select model (1 or 2): ").strip()
            if choice in {"1", "2"}:
                break
            print("Please enter 1 or 2")
        use_vl_model = (choice == "2")

    # ============================================================================
    # Mode Selection: Interactive or Server
    # ============================================================================
    print("\nMode Selection:")
    print("1. Interactive chat (local terminal)")
    print("2. Server mode (OpenAI-compatible API on network)")
    mode_choice = input("Select mode (1 or 2): ").strip()
    run_as_server = (mode_choice == "2")
    
    # Server port selection (only if server mode)
    server_port = 8000
    if run_as_server:
        port_input = input("Server port (default 8000): ").strip()
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
    if IS_MACOS and MLX_AVAILABLE:
        # macOS: Use MLX
        if use_vl_model and MLX_VLM_AVAILABLE:
            print(f"\nLoading {selected_desc} (MLX Vision)...")
            model, processor = vlm_load(selected_path)
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
    else:
        # Ubuntu/Linux: Use transformers
        if use_vl_model:
            print("\nLoading LFM2.5-VL-1.6B (Vision-Language)...")
            model = AutoModelForImageTextToText.from_pretrained(
                VL_MODEL_PATH,
                device_map="auto",
                dtype="bfloat16",
                trust_remote_code=True,
                local_files_only=True,
            )
            processor = AutoProcessor.from_pretrained(VL_MODEL_PATH, trust_remote_code=True, local_files_only=True)
            tokenizer = None
            
            # If server mode, start server and skip interactive loop
            if run_as_server:
                run_server_mode(model, tokenizer, processor, "LFM2.5-VL-1.6B", "vision", port=server_port)
                switch_model = False
                break
            
            print("\nLFM2.5-VL-1.6B Interactive Chat")
            print("=" * 50)
            print("Type your question, then choose media type.")
            print("Type 'quit' to exit.")
            print("=" * 50 + "\n")
        else:
            print("\nLoading LFM2.5-1.2B-Thinking (Text-only)...")
            model = AutoModelForCausalLM.from_pretrained(
                TEXT_MODEL_PATH,
                device_map="auto",
                dtype="bfloat16",
                trust_remote_code=True,
                local_files_only=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_PATH, trust_remote_code=True, local_files_only=True)
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            processor = None
            
            # If server mode, start server and skip interactive loop
            if run_as_server:
                run_server_mode(model, tokenizer, processor, "LFM2.5-1.2B-Thinking", "text", port=server_port)
                switch_model = False
                break
            
            print("\nLFM2.5-1.2B-Thinking Interactive Chat")
            print("=" * 50)
            print("Enter prompts below. Type 'quit' to exit, 'model' to switch models.")
            print("=" * 50 + "\n")

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
                    
                    response = lm_generate(model, tokenizer, prompt=prompt, max_tokens=15120, verbose=True)
                    
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
