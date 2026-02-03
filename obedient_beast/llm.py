#!/usr/bin/env python3
"""
Unified LLM client supporting LFM (local), OpenAI, and Claude.
Handles tool calling for all backends.
"""

import os
import json
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# Config from environment
# LLM_BACKEND_TEST allows testing different backends without changing .env
BACKEND = os.getenv("LLM_BACKEND_TEST") or os.getenv("LLM_BACKEND", "claude")  # "lfm", "openai", or "claude"
LFM_URL = os.getenv("LFM_URL", "http://localhost:8000")
LFM_URL_LOCAL = "http://localhost:8000"  # Always try local first
LFM_URL_REMOTE = os.getenv("LFM_URL_REMOTE", "http://192.168.7.57:8000")  # Fallback
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


@dataclass
class ToolCall:
    """Represents a tool call from the LLM."""
    id: str
    name: str
    args: dict


@dataclass
class LLMResponse:
    """Unified response from any LLM backend."""
    text: str
    tool_calls: list[ToolCall]
    raw: dict  # Original response for debugging


class LLM:
    """Unified LLM client for LFM/OpenAI/Claude with tool calling."""
    
    def __init__(self, backend: str = None):
        self.backend = backend or BACKEND
        
    def chat(self, messages: list, tools: list = None, system: str = None) -> LLMResponse:
        """Send messages to LLM and get response with optional tool calls."""
        if self.backend == "claude":
            return self._claude(messages, tools, system)
        elif self.backend == "openai":
            return self._openai(messages, tools, system)
        else:  # lfm
            return self._lfm(messages, tools, system)
    
    def _claude(self, messages: list, tools: list, system: str) -> LLMResponse:
        """Call Anthropic Claude API."""
        import anthropic
        
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        
        # Convert tools to Claude format
        claude_tools = None
        if tools:
            claude_tools = [
                {
                    "name": t["name"],
                    "description": t["description"],
                    "input_schema": {
                        "type": "object",
                        "properties": {k: {"type": "string"} for k in t.get("params", {})},
                        "required": []  # Make params optional for flexibility
                    }
                }
                for t in tools
            ]
        
        print(f"[Claude] Tools: {len(claude_tools) if claude_tools else 0}", flush=True)
        
        # Build request
        kwargs = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 4096,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if claude_tools:
            kwargs["tools"] = claude_tools
        
        response = client.messages.create(**kwargs)
        print(f"[Claude] Response stop_reason: {response.stop_reason}", flush=True)
        
        # Parse response
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
        """Call OpenAI API."""
        import openai
        
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        # Prepend system message if provided
        msgs = messages.copy()
        if system:
            msgs = [{"role": "system", "content": system}] + msgs
        
        # Convert tools to OpenAI format
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
        """Call local LFM server (OpenAI-compatible) with text-based tool calling.
        Tries localhost first, then falls back to remote server."""
        import urllib.request
        import urllib.error
        import re
        import uuid
        
        # Build tool description for the prompt (MLX-LM doesn't support tools API)
        tool_prompt = ""
        if tools:
            tool_prompt = "\n\n## Available Tools\nYou can call tools by outputting JSON in this exact format:\n```tool\n{\"name\": \"tool_name\", \"args\": {\"param\": \"value\"}}\n```\n\nAvailable tools:\n"
            for t in tools:
                params = ", ".join(t.get("params", {}).keys())
                tool_prompt += f"- **{t['name']}**({params}): {t['description']}\n"
            tool_prompt += "\nIf you need to use a tool, output the tool call JSON. After the tool result, continue your response. Keep responses SHORT."
        
        # Prepend system message with tool info
        msgs = messages.copy()
        full_system = (system or "") + tool_prompt
        if full_system:
            msgs = [{"role": "system", "content": full_system}] + msgs
        
        # Convert tools to OpenAI format
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
        
        # Try local first, then remote
        urls_to_try = [LFM_URL_LOCAL, LFM_URL_REMOTE]
        # If LFM_URL is explicitly set and different, try it first
        if LFM_URL not in urls_to_try:
            urls_to_try.insert(0, LFM_URL)
        
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
                    break  # Success, exit loop
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
        
        # First check for native tool_calls (if server supports it)
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                tool_calls.append(ToolCall(
                    id=tc["id"],
                    name=tc["function"]["name"],
                    args=json.loads(tc["function"]["arguments"])
                ))
        else:
            # Parse text-based tool calls from model output
            # Look for ```tool\n{...}\n``` or just {"name": "...", "args": {...}}
            tool_pattern = r'```tool\s*\n?\s*(\{[^}]+\})\s*\n?```'
            matches = re.findall(tool_pattern, text, re.DOTALL)
            
            # Also try to find raw JSON tool calls
            if not matches:
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
            
            # Clean tool call JSON from the text response
            if tool_calls:
                text = re.sub(tool_pattern, '', text)
                text = re.sub(r'\{"name":\s*"\w+",\s*"args":\s*\{[^}]*\}\}', '', text)
                text = text.strip()
        
        return LLMResponse(text=text, tool_calls=tool_calls, raw=result)


# Convenience function
def get_llm(backend: str = None) -> LLM:
    """Get an LLM client for the specified backend."""
    return LLM(backend)


if __name__ == "__main__":
    # Quick test
    llm = get_llm()
    print(f"Testing {llm.backend} backend...")
    response = llm.chat([{"role": "user", "content": "Say hello in 5 words or less."}])
    print(f"Response: {response.text}")
