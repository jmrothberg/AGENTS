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
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

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
LFM_URL_REMOTE = os.getenv("LFM_URL_REMOTE", "http://192.168.7.57:8000")  # Fallback

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
        import re
        import uuid

        # Build tool description for the prompt (text-based, works with any model)
        tool_prompt = ""
        if tools:
            tool_prompt = "\n\n## Available Tools\nYou can call tools by outputting JSON in this exact format:\n```tool\n{\"name\": \"tool_name\", \"args\": {\"param\": \"value\"}}\n```\n\nAvailable tools:\n"
            for t in tools:
                params = ", ".join(t.get("params", {}).keys())
                tool_prompt += f"- **{t['name']}**({params}): {t['description']}\n"
            tool_prompt += "\nIf you need to use a tool, output the tool call JSON. After the tool result, continue your response. Keep responses SHORT."

        # Prepend system message with tool info injected
        msgs = messages.copy()
        full_system = (system or "") + tool_prompt
        if full_system:
            msgs = [{"role": "system", "content": full_system}] + msgs

        # Also convert tools to OpenAI format for native function calling support.
        # This is the "belt AND suspenders" approach — we send tools both in the
        # prompt text and in the API, covering models with and without native support.
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
                            "properties": {k: {"type": "string", "description": v} for k, v in t.get("params", {}).items()},
                            "required": list(t.get("params", {}).keys())
                        }
                    }
                }
                for t in tools
            ]

        payload = {
            "model": "lfm",
            "messages": msgs,
            "max_tokens": 4096,
        }
        if openai_tools:
            payload["tools"] = openai_tools

        data = json.dumps(payload).encode('utf-8')

        # URL fallback chain: try local first, then remote
        urls_to_try = [LFM_URL_LOCAL, LFM_URL_REMOTE]
        if LFM_URL not in urls_to_try:
            urls_to_try.insert(0, LFM_URL)  # Custom URL gets highest priority

        result = None
        last_error = None
        for url in urls_to_try:
            try:
                req = urllib.request.Request(
                    f"{url}/v1/chat/completions",
                    data=data,
                    headers={'Content-Type': 'application/json'}
                )
                with urllib.request.urlopen(req, timeout=120) as response:
                    result = json.loads(response.read().decode())
                    print(f"[LFM] Connected to {url}", flush=True)
                    break  # Success, stop trying other URLs
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
                last_error = e
                print(f"[LFM] {url} unavailable: {e}", flush=True)
                continue  # Try next URL

        if result is None:
            raise ConnectionError(f"All LFM servers unavailable. Last error: {last_error}")

        # Parse response
        msg = result["choices"][0]["message"]
        text = msg.get("content", "") or ""
        tool_calls = []

        # Strategy 1: Check for native tool_calls (if server/model supports it)
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                tool_calls.append(ToolCall(
                    id=tc["id"],
                    name=tc["function"]["name"],
                    args=json.loads(tc["function"]["arguments"])
                ))
        else:
            # Strategy 2: Parse text-based tool calls from model output.
            # Pattern A: Fenced code blocks — ```tool\n{...}\n```
            # This is the format we asked for in the tool_prompt above.
            tool_pattern = r'```tool\s*\n?\s*(\{[^}]+\})\s*\n?```'
            matches = re.findall(tool_pattern, text, re.DOTALL)

            if not matches:
                # Pattern B: Raw inline JSON — {"name": "...", "args": {...}}
                # Some models output tool calls without the fenced block wrapper.
                json_pattern = r'\{"name":\s*"(\w+)",\s*"args":\s*(\{[^}]*\})\}'
                json_matches = re.findall(json_pattern, text)
                for name, args_str in json_matches:
                    try:
                        args = json.loads(args_str)
                        tool_calls.append(ToolCall(
                            id=f"call_{uuid.uuid4().hex[:8]}",
                            name=name,
                            args=args
                        ))
                    except json.JSONDecodeError:
                        pass
            else:
                # Parse fenced tool blocks
                for match in matches:
                    try:
                        tool_data = json.loads(match)
                        tool_calls.append(ToolCall(
                            id=f"call_{uuid.uuid4().hex[:8]}",
                            name=tool_data["name"],
                            args=tool_data.get("args", {})
                        ))
                    except json.JSONDecodeError:
                        pass

            # Clean tool call JSON from the text response so the user
            # only sees the natural language part, not the raw JSON.
            if tool_calls:
                text = re.sub(tool_pattern, '', text)
                text = re.sub(r'\{"name":\s*"\w+",\s*"args":\s*\{[^}]*\}\}', '', text)
                text = text.strip()

        return LLMResponse(text=text, tool_calls=tool_calls, raw=result)


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
