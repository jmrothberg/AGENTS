"""
Linux Transformer Model Server
===============================
Written by Jonathan M Rothberg

Dynamically scans model directories and serves any local transformer model
via an OpenAI-compatible API. Drop-in replacement for lfm_thinking.py on Linux.

- Scans MODEL_SEARCH_PATHS at startup — any model folder with config.json is detected
- Same OpenAI API as lfm_thinking.py (same .env, same ports)
- Beast agent connects identically via LFM_URL (legacy name for local model URL)

# TODO: Switch to vLLM for concurrent batched inference when it supports
#       DGX Spark / Blackwell / CUDA 13 natively (without containers).
#       vLLM would allow multiple Beast agents to query simultaneously
#       via continuous batching. For now, transformers works reliably.
#       To switch: pip install vllm, then uncomment vLLM sections below.

USAGE:
  python linux_thinking.py                         # Interactive model selection
  python linux_thinking.py --port 8000             # Server on port 8000
  python linux_thinking.py --interactive           # Interactive chat mode
  python linux_thinking.py --model /path/to/model  # Skip model menu
"""

import os
import time
import json
import uuid
import re
import argparse

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
# Model Scanning - Detects models and types from directory structure
# ============================================================================
def scan_models(models_dir):
    """
    Dynamically scan a models directory and detect model types.
    Returns dict: {"1": (path, type, description), ...}

    Model type detection:
    - Vision models: config.json has "vl"/"vision"/"image" in model_type,
      or has processor_config.json
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
            except:
                pass

        # Also check for processor_config.json (vision models have this)
        if os.path.exists(processor_path):
            model_type = "vision"

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
        except:
            pass
    if os.path.exists(processor_path):
        model_type = "vision"
    return model_type


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
def run_server_mode(model_path, model_name, model_type,
                    host=DEFAULT_HOST, port=DEFAULT_PORT):
    """
    Run the model as an OpenAI-compatible API server using transformers.

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
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # --- VLLM: Uncomment these when vLLM supports your hardware ---
    # from vllm.engine.async_llm_engine import AsyncLLMEngine
    # from vllm.engine.arg_utils import AsyncEngineArgs
    # from vllm import SamplingParams

    # ----------------------------------------------------------------
    # Load model and tokenizer with transformers
    # ----------------------------------------------------------------
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )
    # torch_dtype="auto" lets transformers respect the model's quantization config (e.g. FP8)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    print(f"Model loaded on {model.device} (dtype: {model.dtype})")

    # --- VLLM: Replace the above with this when ready ---
    # engine_args = AsyncEngineArgs(
    #     model=model_path,
    #     trust_remote_code=True,
    #     gpu_memory_utilization=0.90,
    #     max_num_seqs=32,
    #     dtype="auto",
    #     enforce_eager=False,
    # )
    # engine = AsyncLLMEngine.from_engine_args(engine_args)

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

        for msg in messages:
            content = msg.content
            # Handle content as string or list
            if isinstance(content, list):
                text_content = " ".join(
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in content
                    if isinstance(item, dict) and item.get("type") in ["text", "tool_result"]
                )
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

        return user_message, system_message

    def generate_response(user_message, system_message, temperature, max_tokens):
        """
        Generate a response using transformers model.generate().
        Runs synchronously on GPU - called via run_in_executor for async.
        """
        # Build chat messages for template
        chat_messages = []
        if system_message:
            chat_messages.append({"role": "system", "content": system_message})
        chat_messages.append({"role": "user", "content": user_message})

        inputs = tokenizer.apply_chat_template(
            chat_messages,
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
            temperature=temperature,
            max_new_tokens=max_tokens,
        )

        response_text = tokenizer.decode(
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
            "model": model_name,
            "type": model_type,
            "engine": "transformers",
            # "engine": "vllm",  # VLLM: switch label when ready
        }

    @app.get("/v1/models")
    async def list_models():
        """List available models (OpenAI-compatible)."""
        return ModelsResponse(
            data=[ModelInfo(id=model_name, created=int(time.time()))]
        )

    # ----------------------------------------------------------------
    # Chat Completions Endpoint
    # ----------------------------------------------------------------
    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        """
        OpenAI-compatible chat completions endpoint.
        Works with any OpenAI client library.
        Supports tool calling when tools are provided.

        Note: Requests are sequential with transformers. Switch to vLLM
        for concurrent batching when it supports Blackwell/CUDA 13.
        """
        try:
            user_message, system_message = build_user_message(
                request.messages, request.tools
            )

            if not user_message:
                raise HTTPException(status_code=400, detail="No user message found")

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

            # --- VLLM: Replace the above with async vLLM generate ---
            # sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
            # request_id = f"req-{uuid.uuid4().hex[:8]}"
            # final_output = None
            # async for output in engine.generate(prompt, sampling_params, request_id):
            #     final_output = output
            # response_text = final_output.outputs[0].text
            # prompt_tokens = len(final_output.prompt_token_ids)
            # completion_tokens = len(final_output.outputs[0].token_ids)

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
                        "model": model_name,
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
                        "model": model_name,
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
                    model=model_name,
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
    print(f"Model:  {model_name}")
    print(f"Type:   {model_type}")
    print(f"Engine: transformers (sequential)")
    # print(f"Engine: vLLM (concurrent batching)")  # VLLM: switch when ready
    print("=" * 60)
    print("Access URLs:")
    print(f"  Local:   http://localhost:{port}")
    print(f"  Network: http://{local_ip}:{port}")
    print("=" * 60)
    print("API Endpoints:")
    print(f"  POST http://{local_ip}:{port}/v1/chat/completions")
    print(f"  GET  http://{local_ip}:{port}/v1/models")
    print("=" * 60)
    print(f"\nExample usage with curl:")
    print(f'  curl http://{local_ip}:{port}/v1/chat/completions \\')
    print(f'    -H "Content-Type: application/json" \\')
    print(f'    -d \'{{"model": "{model_name}", '
          f'"messages": [{{"role": "user", "content": "Hello!"}}]}}\'')
    print(f"\nExample usage with Python OpenAI client:")
    print(f'  from openai import OpenAI')
    print(f'  client = OpenAI(base_url="http://{local_ip}:{port}/v1", '
          f'api_key="not-needed")')
    print(f'  response = client.chat.completions.create(')
    print(f'      model="{model_name}",')
    print(f'      messages=[{{"role": "user", "content": "Hello!"}}]')
    print(f'  )')
    print(f'  print(response.choices[0].message.content)')
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
def run_interactive_mode(model_path, model_name):
    """
    Simple interactive chat using transformers.
    For quick testing without starting the full server.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

    # --- VLLM: Replace with these when ready ---
    # from vllm import LLM, SamplingParams

    print(f"\nLoading {model_name} with transformers...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # --- VLLM: Replace with this when ready ---
    # llm = LLM(model=model_path, trust_remote_code=True, dtype="auto")

    print(f"\n{model_name} Interactive Chat (transformers)")
    print("=" * 50)
    print("Enter prompts below. Type 'quit' to exit.")
    print("=" * 50 + "\n")

    while True:
        try:
            user_input = input("Prompt: ").strip()

            if user_input.lower() in {"quit", "exit", "q"}:
                print("Goodbye!")
                break

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
            model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=0.7,
                max_new_tokens=2048,
                streamer=streamer,
            )

            # --- VLLM: Replace with this when ready ---
            # prompt = tokenizer.apply_chat_template(
            #     messages, add_generation_prompt=True, tokenize=False
            # )
            # outputs = llm.generate([prompt], SamplingParams(temperature=0.7, max_tokens=2048))
            # print(outputs[0].outputs[0].text)

            print("\n" + "=" * 50)

        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'quit' to exit or continue chatting.")
            continue
        except Exception as e:
            print(f"\nError: {type(e).__name__}: {str(e)}")
            print("Try again or type 'quit' to exit.")
            continue


# ============================================================================
# Main Entry Point
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Linux Transformer Model Server - OpenAI-compatible API for local models"
    )
    parser.add_argument(
        "--model", type=str,
        help="Path to model directory (skip selection menu)"
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT,
        help=f"Server port (default: {DEFAULT_PORT})"
    )
    parser.add_argument(
        "--host", type=str, default=DEFAULT_HOST,
        help=f"Server host (default: {DEFAULT_HOST})"
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Run in interactive chat mode instead of server"
    )
    args = parser.parse_args()

    # ----------------------------------------------------------------
    # Model Selection
    # ----------------------------------------------------------------
    if args.model:
        # Direct model path specified via CLI
        model_path = args.model
        model_name = os.path.basename(model_path)
        model_type = detect_model_type(model_path)
    else:
        # Interactive model selection menu
        all_models = scan_all_model_paths()

        if not all_models:
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
            all_models.items(), key=lambda x: int(x[0])
        ):
            vl_tag = " [VL]" if mtype == "vision" else ""
            print(f"  {key}. {desc}{vl_tag}")
        print("=" * 50)

        valid_choices = set(all_models.keys())
        while True:
            choice = input(
                f"Select model ({'/'.join(sorted(valid_choices, key=int))}): "
            ).strip()
            if choice in valid_choices:
                break
            print(f"Please enter one of: "
                  f"{', '.join(sorted(valid_choices, key=int))}")

        model_path, model_type, model_name = all_models[choice]

    # ----------------------------------------------------------------
    # Mode Selection
    # ----------------------------------------------------------------
    if args.interactive:
        # CLI flag: interactive mode
        run_interactive_mode(model_path, model_name)
    elif args.model:
        # CLI --model without --interactive: default to server mode
        run_server_mode(
            model_path, model_name, model_type,
            host=args.host, port=args.port
        )
    else:
        # Interactive menu: ask user
        print("\nMode Selection:")
        print("  1. Server mode (OpenAI-compatible API for Beast)")
        print("  2. Interactive chat (local terminal)")
        mode_choice = input("Select mode (1 or 2): ").strip()

        if mode_choice == "2":
            run_interactive_mode(model_path, model_name)
        else:
            # Server mode (default)
            port_input = input(f"Server port (default {args.port}): ").strip()
            if port_input.isdigit():
                args.port = int(port_input)
            run_server_mode(
                model_path, model_name, model_type,
                host=args.host, port=args.port
            )


if __name__ == "__main__":
    main()
