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
BACKEND = os.getenv("LLM_BACKEND", "claude")  # "lfm", "openai", or "claude"
LFM_URL = os.getenv("LFM_URL", "http://localhost:8000")
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
                        "required": list(t.get("params", {}).keys())
                    }
                }
                for t in tools
            ]
        
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
        """Call local LFM server (OpenAI-compatible)."""
        import urllib.request
        
        # Prepend system message if provided
        msgs = messages.copy()
        if system:
            msgs = [{"role": "system", "content": system}] + msgs
        
        # Convert tools to OpenAI format (LFM is OpenAI-compatible)
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
        
        payload = {
            "model": "lfm",
            "messages": msgs,
            "max_tokens": 4096,
        }
        if openai_tools:
            payload["tools"] = openai_tools
        
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            f"{LFM_URL}/v1/chat/completions",
            data=data,
            headers={'Content-Type': 'application/json'}
        )
        
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode())
        
        # Parse response (OpenAI format)
        msg = result["choices"][0]["message"]
        text = msg.get("content", "") or ""
        tool_calls = []
        
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                tool_calls.append(ToolCall(
                    id=tc["id"],
                    name=tc["function"]["name"],
                    args=json.loads(tc["function"]["arguments"])
                ))
        
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
