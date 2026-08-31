#!/usr/bin/env python3
"""
Shared local-model harness — used by lfm_thinking.py, linux_thinking.py, llm.py.
================================================================================
Parse tool calls, render chat templates, family presets, OpenAI tool schemas.

Default weights are whatever LFM_MODEL / the loaded folder is (today Qwen3.8-27B-class).
Presets match folder substrings; unknown families get a safe default.
"""

from __future__ import annotations

import json
import os
import re
import uuid

DEFAULT_MAX_TOKENS = int(os.getenv("LFM_MAX_TOKENS", "16384"))

# ---------------------------------------------------------------------------
# Family presets — first substring match on the loaded folder/name wins.
# QWEN_* env names are convenience aliases overlaid on chat_template_kwargs.
# ---------------------------------------------------------------------------
FAMILY_PRESETS = [
    {
        # qwen4_exp (Flash-Next). Must be before the generic qwen3.8 match.
        # Sampling from this checkpoint's generation_config.json.
        "match": ("flash-next", "qwen4_exp", "qwen4-exp"),
        "chat_template_kwargs": {
            "enable_thinking": True,
            "preserve_thinking": True,
            "reasoning_effort": "medium",
        },
        "reasoning_effort_values": ("low", "medium", "xhigh"),
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
    },
    {
        "match": ("qwen3.8", "qwen3_8", "qwen-3.8"),
        "chat_template_kwargs": {
            "enable_thinking": True,
            "preserve_thinking": True,
            "reasoning_effort": "medium",
        },
        "reasoning_effort_values": ("low", "medium", "xhigh"),
        "temperature": 0.7,
    },
    {
        "match": ("qwen3", "qwen"),
        "chat_template_kwargs": {"enable_thinking": True},
        "temperature": 0.7,
    },
    {
        "match": ("gemma",),
        "chat_template_kwargs": {},
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
    },
    {
        "match": ("glm",),
        "chat_template_kwargs": {},
        "temperature": 0.7,
    },
]

_SAFE_DEFAULT = {
    "match": (),
    "chat_template_kwargs": {},
    "temperature": 0.7,
}


def _env_bool(name: str, default: bool | None = None) -> bool | None:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def family_preset(model_name: str) -> dict:
    """Return the first matching family preset, or a safe default."""
    needle = (model_name or "").lower()
    for preset in FAMILY_PRESETS:
        if any(m in needle for m in preset["match"]):
            return preset
    return _SAFE_DEFAULT


def merge_template_kwargs(model_name: str, request_kwargs: dict | None = None) -> dict:
    """Preset kwargs, then QWEN_* aliases, then per-request overlay."""
    preset = family_preset(model_name)
    extra = dict(preset.get("chat_template_kwargs") or {})
    # QWEN_* aliases — templates that do not know these keys are retried without them.
    if os.getenv("QWEN_ENABLE_THINKING") is not None:
        extra["enable_thinking"] = _env_bool("QWEN_ENABLE_THINKING", True)
    if os.getenv("QWEN_PRESERVE_THINKING") is not None:
        extra["preserve_thinking"] = _env_bool("QWEN_PRESERVE_THINKING", True)
    effort = os.getenv("QWEN_REASONING_EFFORT")
    allowed = preset.get("reasoning_effort_values")
    if effort:
        effort = effort.strip().lower()
        if allowed and effort not in allowed:
            # Qwen3.8 jinja raises on "high"; keep the agent default.
            print(
                f"[harness] reasoning_effort={effort!r} not in {allowed}; using medium",
                flush=True,
            )
            effort = "medium" if "medium" in allowed else allowed[0]
        extra["reasoning_effort"] = effort
    elif allowed and "reasoning_effort" in extra:
        val = str(extra["reasoning_effort"]).lower()
        if val not in allowed:
            extra["reasoning_effort"] = "medium"
    if request_kwargs:
        extra.update({k: v for k, v in request_kwargs.items() if v is not None})
        if allowed and extra.get("reasoning_effort") not in (None, *allowed):
            extra["reasoning_effort"] = "medium"
    return extra


def sampling_for_model(model_name: str, request_temperature=None) -> dict:
    """Sampler knobs for mlx_lm / transformers. Request temperature wins."""
    preset = family_preset(model_name)
    out = {"temperature": preset.get("temperature", 0.7)}
    if preset.get("top_p") is not None:
        out["top_p"] = preset["top_p"]
    if preset.get("top_k") is not None:
        out["top_k"] = preset["top_k"]
    if request_temperature is not None:
        out["temperature"] = request_temperature
    return out


# ---------------------------------------------------------------------------
# Messages → chat-template dicts (native roles). Flatten only as fallback.
# ---------------------------------------------------------------------------
def content_to_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") in ("text", "tool_result"):
                    parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return " ".join(p for p in parts if p)
    return str(content)


def extract_image_from_messages(messages) -> str | None:
    """First data:image_url across the request → temp jpeg path, or None."""
    import base64
    import tempfile

    for msg in messages:
        raw = msg.model_dump() if hasattr(msg, "model_dump") else (
            msg.dict() if hasattr(msg, "dict") else msg
        )
        content = raw.get("content") if isinstance(raw, dict) else getattr(msg, "content", None)
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "image_url":
                continue
            data_url = (item.get("image_url") or {}).get("url", "")
            if not data_url.startswith("data:"):
                continue
            try:
                _header, b64data = data_url.split(",", 1)
                img_bytes = base64.b64decode(b64data)
                tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                tmp.write(img_bytes)
                tmp.close()
                print(f"[DEBUG] Extracted image to {tmp.name} ({len(img_bytes)} bytes)")
                return tmp.name
            except Exception as img_err:
                print(f"[DEBUG] Image extraction failed: {img_err}")
                return None
    return None


def to_chat_dicts(messages) -> list[dict]:
    """Pydantic/dict OpenAI messages → list of role/content dicts for apply_chat_template."""
    out = []
    for msg in messages:
        if hasattr(msg, "model_dump"):
            raw = msg.model_dump()
        elif hasattr(msg, "dict"):
            raw = msg.dict()
        else:
            raw = dict(msg)
        d = {
            "role": raw.get("role"),
            "content": content_to_text(raw.get("content")),
        }
        if raw.get("tool_calls"):
            d["tool_calls"] = raw["tool_calls"]
        if raw.get("tool_call_id"):
            d["tool_call_id"] = raw["tool_call_id"]
        if raw.get("name"):
            d["name"] = raw["name"]
        if raw.get("reasoning_content"):
            d["reasoning_content"] = raw["reasoning_content"]
        out.append(d)
    return out


def flatten_as_user_text(chat_dicts: list[dict]) -> tuple[str, str]:
    """Fallback when the tokenizer has no chat_template. No last-N cut, no 'summarize'."""
    system = ""
    parts = []
    for d in chat_dicts:
        role = d.get("role")
        content = d.get("content") or ""
        if role == "system":
            system = content
        elif role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            names = []
            for tc in d.get("tool_calls") or []:
                fn = tc.get("function") if isinstance(tc, dict) else None
                names.append((fn or {}).get("name") or tc.get("name") or "tool")
            used = f" [Used tools: {', '.join(names)}]" if names else ""
            text = (content + used).strip()
            if text:
                parts.append(f"Assistant: {text}")
        elif role == "tool":
            parts.append(f"[Tool Result]: {content}")
    return "\n".join(parts), system


def format_tools_for_prompt(tools) -> str:
    """Short reminder when the template cannot take tools=. API array is the catalog."""
    if not tools:
        return ""
    return "\n".join([
        "## Tools",
        "Call tools with JSON:",
        "```tool_call",
        '{"name": "tool_name", "arguments": {"param": "value"}}',
        "```",
        "Or: <tool_call>{\"name\": \"...\", \"arguments\": {...}}</tool_call>",
        "",
        "RULES:",
        "- Answer in text unless you need to act.",
        "- You may call several independent tools in one turn.",
        "- For Python use run_python. For HTML use run_html. For images use generate_art.",
        "- Never retry a tool call that already succeeded. When the task is done, answer in text.",
    ])


def render_chat(
    tokenizer,
    chat_dicts: list[dict],
    tools=None,
    extra_kwargs: dict | None = None,
    tokenize: bool = False,
    return_tensors=None,
):
    """
    apply_chat_template with tools= and family kwargs.
    Returns None if there is no template (caller should flatten).
    Retries without tools, then without extra kwargs, on TypeError / bad jinja kwargs.
    """
    if tokenizer is None or getattr(tokenizer, "chat_template", None) is None:
        return None
    extra = dict(extra_kwargs or {})
    base = dict(add_generation_prompt=True, return_dict=False, tokenize=tokenize)
    if return_tensors:
        base["return_tensors"] = return_tensors
    attempts = []
    if tools:
        attempts.append({**base, **extra, "tools": tools})
    if extra:
        attempts.append({**base, **extra})
    attempts.append(dict(base))

    last_err = None
    for kw in attempts:
        try:
            return tokenizer.apply_chat_template(chat_dicts, **kw)
        except TypeError as e:
            last_err = e
            continue
        except Exception as e:
            last_err = e
            err = str(e).lower()
            if "reasoning" in err or "unexpected" in err or "jinja" in err:
                continue
            raise
    if last_err:
        print(f"[harness] apply_chat_template fell back ({last_err})", flush=True)
    return None


# ---------------------------------------------------------------------------
# Tool JSON schema (Beast params dict → OpenAI function tools)
# ---------------------------------------------------------------------------
_INT_NAMES = {
    "timeout", "width", "height", "seed", "x", "y", "max_chars", "depth",
    "repeat_seconds",
}
_BOOL_NAMES = {"submit", "enabled", "open_browser", "full_page"}


def _param_schema(name: str, desc: str) -> dict:
    d = desc or ""
    n = name.lower()
    low = d.lower()
    if n in _INT_NAMES or "pixels" in low or "seconds" in low:
        t = "integer"
    elif n in _BOOL_NAMES or "true or false" in low or "true/false" in low:
        t = "boolean"
    else:
        t = "string"
    return {"type": t, "description": d}


def _is_optional_param(desc: str) -> bool:
    low = (desc or "").lower()
    return low.startswith("optional") or "optional:" in low or "(optional" in low


def tools_to_openai(tools: list | None) -> list | None:
    """Beast {name, description, params} or already-OpenAI tools → OpenAI function list."""
    if not tools:
        return None
    out = []
    for t in tools:
        if isinstance(t, dict) and t.get("type") == "function" and "function" in t:
            out.append(t)
            continue
        props = {}
        required = []
        for k, v in (t.get("params") or {}).items():
            props[k] = _param_schema(k, v)
            if not _is_optional_param(v):
                required.append(k)
        out.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": required,
                },
            },
        })
    return out


# ---------------------------------------------------------------------------
# Tool-call parse / clean / split thinking
# ---------------------------------------------------------------------------
def _extract_json(s: str, start: int = 0):
    idx = s.find("{", start)
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
        if c == "\\":
            escape = True
            continue
        if c == '"' and not escape:
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[idx : i + 1], i + 1
    return None, -1


def _repair_json(blob: str) -> str:
    result = []
    in_str = False
    escape = False
    for c in blob:
        if escape:
            result.append(c)
            escape = False
            continue
        if c == "\\":
            result.append(c)
            escape = True
            continue
        if c == '"':
            in_str = not in_str
            result.append(c)
            continue
        if in_str:
            if c == "\n":
                result.append("\\n")
                continue
            if c == "\r":
                result.append("\\r")
                continue
            if c == "\t":
                result.append("\\t")
                continue
        result.append(c)
    return "".join(result)


def parse_tool_calls(text) -> list:
    """Brace-depth parse of ```tool_call, JSON <tool_call>, Qwen XML <function=>, or raw JSON."""
    if text is None:
        return []
    if not isinstance(text, str):
        text = str(text)
    tool_calls = []
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
        if key in seen:
            return
        seen.add(key)
        tool_calls.append({
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {"name": name, "arguments": args_str},
        })

    def try_parse(blob):
        if not blob:
            return None
        for attempt in (blob, _repair_json(blob)):
            try:
                return json.loads(attempt)
            except json.JSONDecodeError:
                continue
        return None

    for m in re.finditer(r"```tool(?:_call)?\s*\n", text):
        blob, _ = _extract_json(text, m.end())
        data = try_parse(blob)
        if data and "name" in data:
            try_add(data)

    if not tool_calls:
        for m in re.finditer(r"<tool_call>\s*", text):
            blob, _ = _extract_json(text, m.end())
            data = try_parse(blob)
            if data and "name" in data:
                try_add(data)

    # Qwen3.8 Flash-Next / qwen4_exp chat template: XML, not JSON.
    # <tool_call><function=get_weather><parameter=location>\nMiami\n</parameter></function></tool_call>
    for m in re.finditer(
        r"<tool_call>\s*<function=([^>\s]+)\s*>([\s\S]*?)</function>\s*</tool_call>",
        text,
        flags=re.IGNORECASE,
    ):
        args = {}
        for pm in re.finditer(
            r"<parameter=([^>\s]+)\s*>([\s\S]*?)</parameter>",
            m.group(2),
            flags=re.IGNORECASE,
        ):
            raw = pm.group(2).strip()
            try:
                args[pm.group(1).strip()] = json.loads(raw)
            except Exception:
                args[pm.group(1).strip()] = raw
        try_add({"name": m.group(1).strip(), "arguments": args})

    if not tool_calls:
        blob, _ = _extract_json(text)
        if blob and '"name"' in blob:
            data = try_parse(blob)
            if data and "name" in data:
                try_add(data)

    return tool_calls


def clean_tool_calls_from_text(text) -> str:
    """Strip tool/think blocks and orphan tags from visible text."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r"```tool_call\s*\n[\s\S]*?\n```", "", text)
    text = re.sub(r"```tool\s*\n[\s\S]*?\n```", "", text)
    text = re.sub(r"<tool_call>[\s\S]*?</tool_call>", "", text)
    text = re.sub(r"<think>[\s\S]*?</think>", "", text)
    # Orphan/stray tags models emit without a matching pair
    text = re.sub(r"</think>", "", text)
    text = re.sub(r"</?tool_call>", "", text)
    text = re.sub(r"\[/?tool_call\]", "", text)
    return text.strip()


def split_thinking(text) -> tuple[str, str]:
    """Pull a <think>...</think> block out. Visible text is what the user sees."""
    if not text:
        return "", ""
    m = re.search(r"<think>([\s\S]*?)</think>", text)
    if not m:
        return text, ""
    visible = (text[: m.start()] + text[m.end() :]).strip()
    return visible, m.group(1).strip()
