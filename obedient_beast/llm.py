#!/usr/bin/env python3
"""
Unified LLM Client — Three-Backend Architecture
=================================================
Handles tool calling for all backends with a single consistent interface.

Three backends:
    1. "claude"  — Anthropic Claude API (cloud, FULL tier)
    2. "openai"  — OpenAI API (cloud, FULL tier)
    3. "lfm"     — Any local model served via lfm_thinking.py / linux_thinking.py
                    Name is legacy from LFM-2.5 models but works with any model
                    on the local server. LITE tier.

Why "lfm"?
    This project started with LiquidAI's LFM-2.5 models. The config name stuck.
    It now works with Qwen, GLM, Llama, or any model served on the local server.
    We keep the name for backward compatibility with existing .env files.

Tool calling differences between backends:
    - Claude: Uses Anthropic's native tool format (input_schema, tool_use blocks)
    - OpenAI: Uses OpenAI function calling (function objects, tool_calls array)
    - LFM:    Local models may or may not support native tool calling.
              We send tools in BOTH the prompt (as text) AND the API (as OpenAI format).
              Then we parse the response for tool calls in two ways:
              1. Native tool_calls in the API response (if the model supports it)
              2. Text-based parsing: ```tool\\n{...}\\n``` blocks or raw JSON patterns
              This dual approach works with any local model, whether it supports
              function calling natively or not.
"""

import os
import json
import sys
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from capabilities import load_beast_env

load_beast_env()

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from local_harness import (
    tools_to_openai,
    parse_tool_calls as harness_parse_tool_calls,
    clean_tool_calls_from_text,
    split_thinking,
)

load_beast_env()

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------
# LLM_BACKEND_TEST allows testing different backends without changing .env
# Priority: LLM_BACKEND_TEST > LLM_BACKEND > default "lfm" (local first)
BACKEND = os.getenv("LLM_BACKEND_TEST") or os.getenv("LLM_BACKEND", "lfm")  # "lfm", "openai", or "claude"

# Local model server URLs — Beast tries localhost first, then falls back to remote.
# This lets you run the model on a different machine (e.g., Linux GPU server).
LFM_URL = os.getenv("LFM_URL", "http://localhost:8000")
LFM_URL_LOCAL = "http://localhost:8000"  # Always try local first
# Empty unless set — do not assume a LAN IP.
LFM_URL_REMOTE = (os.getenv("LFM_URL_REMOTE") or "").strip()
# Off by default: a dead LAN box (Errno 60) used to stall every WhatsApp turn ~60s.
LFM_TRY_REMOTE = os.getenv("LFM_TRY_REMOTE", "false").lower() in ("1", "true", "yes", "on")
_lfm_dead_urls: set[str] = set()


def lfm_urls_to_try() -> list:
    """Localhost only, unless LFM_TRY_REMOTE=true. Never put a dead LAN IP first."""
    urls = [LFM_URL_LOCAL]
    if not LFM_TRY_REMOTE:
        return urls
    for u in (LFM_URL, LFM_URL_REMOTE):
        if not u or u in urls or u in _lfm_dead_urls:
            continue
        if "localhost" in u or "127.0.0.1" in u:
            continue
        urls.append(u)
    return urls

# Qwen3.8 thinking knobs — ignored by Claude/OpenAI. Defaults suit agents, not max chat.
def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")

QWEN_REASONING_EFFORT = os.getenv("QWEN_REASONING_EFFORT", "medium")
QWEN_ENABLE_THINKING = _env_bool("QWEN_ENABLE_THINKING", True)
QWEN_PRESERVE_THINKING = _env_bool("QWEN_PRESERVE_THINKING", True)
LFM_MAX_TOKENS = int(os.getenv("LFM_MAX_TOKENS", "16384"))

# Cloud API keys (only needed when using respective backends)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


@dataclass
class ToolCall:
    """Represents a tool call from the LLM. Same structure for all backends."""
    id: str       # Unique ID for matching tool results back to calls
    name: str     # Tool name (e.g., "shell", "mcp_memory_search_nodes")
    args: dict    # Arguments as a dict (e.g., {"command": "ls -la"})


@dataclass
class LLMResponse:
    """Unified response from any LLM backend."""
    text: str              # The text portion of the response (may be empty if only tool calls)
    tool_calls: list[ToolCall]  # Zero or more tool calls the LLM wants to make
    raw: dict              # Original response for debugging
    reasoning: str = ""    # Qwen <think> block, if any (not shown to the user)


class LLM:
    """
    Unified LLM client for local/OpenAI/Claude with tool calling.

    Usage:
        llm = LLM("claude")  # or "openai" or "lfm"
        response = llm.chat(messages, tools=tools, system=system_prompt)
        if response.tool_calls:
            # Execute tools, add results to messages, call again
        else:
            print(response.text)
    """

    def __init__(self, backend: str = None):
        self.backend = backend or BACKEND

    def chat(self, messages: list, tools: list = None, system: str = None) -> LLMResponse:
        """Send messages to LLM and get response with optional tool calls."""
        if self.backend == "claude":
            return self._claude(messages, tools, system)
        elif self.backend == "openai":
            return self._openai(messages, tools, system)
        else:  # lfm (any local model)
            return self._lfm(messages, tools, system)

    def _claude(self, messages: list, tools: list, system: str) -> LLMResponse:
        """
        Call Anthropic Claude API.

        Tool format: Claude uses its own tool schema with input_schema (JSON Schema).
        We convert Beast's simple params dict to {"type": "string"} for each param.
        All params are marked optional (required: []) for flexibility — Claude is
        smart enough to figure out which params are needed from the description.

        Model: claude-sonnet-4 — good balance of speed and capability.
        """
        import anthropic  # Lazy import — only needed when Claude backend is active

        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        # Convert Beast's tool format to Claude's format
        claude_tools = None
        if tools:
            claude_tools = [
                {
                    "name": t["name"],
                    "description": t["description"],
                    "input_schema": {
                        "type": "object",
                        "properties": {k: {"type": "string"} for k in t.get("params", {})},
                        "required": []  # All params optional — Claude handles this well
                    }
                }
                for t in tools
            ]

        print(f"[Claude] Tools: {len(claude_tools) if claude_tools else 0}", flush=True)

        # Build request kwargs
        kwargs = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 4096,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system  # Claude has a dedicated system param
        if claude_tools:
            kwargs["tools"] = claude_tools

        response = client.messages.create(**kwargs)
        print(f"[Claude] Response stop_reason: {response.stop_reason}", flush=True)

        # Parse response — Claude returns content blocks (text and/or tool_use)
        text = ""
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                text += block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    args=block.input
                ))

        return LLMResponse(text=text, tool_calls=tool_calls, raw=response.model_dump())

    def _openai(self, messages: list, tools: list, system: str) -> LLMResponse:
        """
        Call OpenAI API.

        Tool format: OpenAI uses "function" objects with JSON Schema parameters.
        Unlike Claude, OpenAI requires all params in the "required" array.
        System message is prepended to the messages array (OpenAI doesn't have
        a separate system param like Claude).
        """
        import openai  # Lazy import

        client = openai.OpenAI(api_key=OPENAI_API_KEY)

        # Prepend system message (OpenAI uses it as the first message)
        msgs = messages.copy()
        if system:
            msgs = [{"role": "system", "content": system}] + msgs

        # Convert Beast's tool format to OpenAI's function calling format
        openai_tools = None
        if tools:
            openai_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": {
                            "type": "object",
                            "properties": {k: {"type": "string"} for k in t.get("params", {})},
                            "required": list(t.get("params", {}).keys())
                        }
                    }
                }
                for t in tools
            ]

        kwargs = {
            "model": "gpt-4o",
            "messages": msgs,
        }
        if openai_tools:
            kwargs["tools"] = openai_tools

        response = client.chat.completions.create(**kwargs)

        # Parse response
        msg = response.choices[0].message
        text = msg.content or ""
        tool_calls = []

        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    args=json.loads(tc.function.arguments)
                ))

        return LLMResponse(text=text, tool_calls=tool_calls, raw=response.model_dump())

    def _lfm(self, messages: list, tools: list, system: str) -> LLMResponse:
        """
        Call local model server (OpenAI-compatible) with text-based tool calling.

        Method name "lfm" is legacy — works with any model on the local server.
        Tries localhost first, then falls back to remote server (configurable via env).

        Tool calling strategy (dual approach):
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Local models (Qwen, GLM, Llama, etc.) may or may not support native
        function calling via the OpenAI tools API. We handle both cases:

        1. PROMPT INJECTION: Tools are described in the system prompt as text,
           telling the model to output ```tool\\n{...}\\n``` blocks. This works
           with ANY model, even those without function calling support.

        2. API TOOLS: We also send tools in OpenAI format via the API, in case
           the model/server supports native function calling (mlx-lm does for
           some models).

        Response parsing:
        ~~~~~~~~~~~~~~~~~
        1. First check for native tool_calls in the API response
        2. If none, parse the text for tool calls using two regex patterns:
           a. ```tool\\n{JSON}\\n``` — fenced code block format
           b. {"name": "...", "args": {...}} — raw inline JSON format
        3. Clean parsed tool call JSON from the text response

        URL fallback chain:
        ~~~~~~~~~~~~~~~~~~~~
        Tries servers in order: LFM_URL (if custom) → localhost:8000 → LFM_URL_REMOTE.
        This lets you run the model on a remote GPU machine and auto-discover it.
        """
        import urllib.request
        import urllib.error

        # Short reminder only — the OpenAI tools array is the catalog.
        tool_prompt = ""
        if tools:
            # Short reminder only — the OpenAI tools array is the catalog.
            # Qwen3.8 may emit <tool_call> or ```tool_call```; both are parsed.
            tool_prompt = (
                "\n\nUse the tools from the API when you need to act. "
                "You may call several independent tools in one turn. "
                "Formats: ```tool_call\\n{\"name\": \"...\", \"arguments\": {...}}\\n``` "
                "or <tool_call>{\"name\": \"...\", \"arguments\": {...}}</tool_call>. "
                "Never retry a tool call that already succeeded. "
                "When the user's task is done, answer in plain text."
            )

        # Prepend system message with tool info injected
        msgs = messages.copy()
        full_system = (system or "") + tool_prompt
        if full_system:
            msgs = [{"role": "system", "content": full_system}] + msgs

        # Also convert tools to OpenAI format for native function calling support.
        # This is the "belt AND suspenders" approach — we send tools both in the
        # prompt text and in the API, covering models with and without native support.
        openai_tools = tools_to_openai(tools) if tools else None

        payload = {
            "model": "lfm",
            "messages": msgs,
            "max_tokens": LFM_MAX_TOKENS,
            # Tokenizer extras (QWEN_* aliases). Other models ignore unknown fields.
            "chat_template_kwargs": {
                "enable_thinking": QWEN_ENABLE_THINKING,
                "preserve_thinking": QWEN_PRESERVE_THINKING,
                "reasoning_effort": QWEN_REASONING_EFFORT,
            },
        }
        if openai_tools:
            payload["tools"] = openai_tools

        data = json.dumps(payload).encode('utf-8')

        urls_to_try = lfm_urls_to_try()

        result = None
        last_error = None
        for url in urls_to_try:
            is_local = "localhost" in url or "127.0.0.1" in url
            try:
                req = urllib.request.Request(
                    f"{url}/v1/chat/completions",
                    data=data,
                    headers={'Content-Type': 'application/json'}
                )
                # Remote connect must fail fast; local inference can take minutes
                with urllib.request.urlopen(req, timeout=300 if is_local else 2) as response:
                    result = json.loads(response.read().decode())
                    print(f"[LFM] Connected to {url}", flush=True)
                    break
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
                last_error = e
                if not is_local:
                    _lfm_dead_urls.add(url)
                # Keep logs quiet — a dead remote used to spam every tool turn
                if is_local:
                    print(f"[LFM] {url} unavailable: {e}", flush=True)
                continue

        if result is None:
            raise ConnectionError(f"All LFM servers unavailable. Last error: {last_error}")

        # Parse response
        msg = result["choices"][0]["message"]
        text = msg.get("content", "") or ""
        reasoning = msg.get("reasoning_content", "") or ""
        tool_calls = []

        # Strategy 1: Check for native tool_calls (if server/model supports it)
        if msg.get("tool_calls"):
            seen_ids = set()
            for tc in msg["tool_calls"]:
                tc_id = tc["id"]
                if tc_id in seen_ids:
                    continue  # Skip duplicate tool call from server
                seen_ids.add(tc_id)
                tool_calls.append(ToolCall(
                    id=tc_id,
                    name=tc["function"]["name"],
                    args=json.loads(tc["function"]["arguments"])
                ))
        else:
            # Strategy 2: text tool calls (```tool_call / <tool_call> / raw JSON)
            for tc in harness_parse_tool_calls(text):
                try:
                    args = json.loads(tc["function"]["arguments"])
                except (json.JSONDecodeError, TypeError, KeyError):
                    args = {}
                tool_calls.append(ToolCall(
                    id=tc["id"],
                    name=tc["function"]["name"],
                    args=args,
                ))
            if tool_calls:
                text = clean_tool_calls_from_text(text)

        # If the server didn't split thinking, pull <think> out of visible text.
        if not reasoning and "<think>" in text:
            text, reasoning = split_thinking(text)

        return LLMResponse(text=text, tool_calls=tool_calls, raw=result, reasoning=reasoning)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def get_llm(backend: str = None) -> LLM:
    """Get an LLM client for the specified backend (defaults to env config)."""
    return LLM(backend)


if __name__ == "__main__":
    # Quick test — run with: python llm.py
    llm = get_llm()
    print(f"Testing {llm.backend} backend...")
    response = llm.chat([{"role": "user", "content": "Say hello in 5 words or less."}])
    print(f"Response: {response.text}")
