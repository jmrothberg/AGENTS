# Anthropic Format Normalization — Bug Fixes to Apply

These fixes were discovered during testing of the Anthropic Messages API normalization work.
The full normalization (single canonical format) was too large to merge cleanly with the
Phase 4 features, but these targeted fixes are independently valuable and can be applied
to the current codebase.

---

## Fix 1: `format_tools_for_prompt()` can't read Anthropic tool format

**File:** `lfm_thinking.py` — inside `run_server_mode()`, the `format_tools_for_prompt()` function

**Problem:** The function only looks for `parameters.properties` (OpenAI format), but when
tools arrive in Anthropic format they use `input_schema.properties`. Result: the model sees
`shell()` with zero parameters instead of `shell(command)`. Tool calling silently breaks.

**Current code:**
```python
params = func.get("parameters", {}).get("properties", {})
```

**Fix — read both formats:**
```python
schema = func.get("parameters") or func.get("input_schema") or {}
props = schema.get("properties", {})
```

**Impact:** Critical. Without this, any tool sent in Anthropic format has empty parameter names.

---

## Fix 2: `clean_tool_calls_from_text()` doesn't handle orphan tags

**File:** `lfm_thinking.py` — inside `run_server_mode()`, the `clean_tool_calls_from_text()` function

**Problem:** Models emit stray `</think>`, `[/tool_call]`, `<tool_call>` tags without matching
opening/closing pairs. The current regex patterns only match complete pairs like
`<think>...</think>`. Orphan tags leak into the response text sent back to Beast.

**Current code only handles matched pairs:**
```python
def clean_tool_calls_from_text(text):
    text = re.sub(r'```tool_call\s*\n[\s\S]*?\n```', '', text)
    text = re.sub(r'<tool_call>[\s\S]*?</tool_call>', '', text)
    text = re.sub(r'<think>[\s\S]*?</think>', '', text)
    return text.strip()
```

**Fix — add orphan tag patterns:**
```python
def clean_tool_calls_from_text(text):
    # Remove matched pairs
    text = re.sub(r'```tool_call\s*\n[\s\S]*?\n```', '', text)
    text = re.sub(r'<tool_call>[\s\S]*?</tool_call>', '', text)
    text = re.sub(r'<think>[\s\S]*?</think>', '', text)
    # Remove orphan/stray tags that models sometimes emit
    text = re.sub(r'</think>', '', text)
    text = re.sub(r'</?tool_call>', '', text)
    text = re.sub(r'\[/?tool_call\]', '', text)
    return text.strip()
```

**Impact:** Medium. Without this, users see raw XML/bracket tags in responses.

---

## Fix 3: Text cleaning only runs when tool calls are found

**File:** `lfm_thinking.py` — in the `/v1/chat/completions` endpoint (and `/v1/messages` if added)

**Problem:** `clean_tool_calls_from_text()` is only called inside the `if tool_calls:` branch.
If the model outputs `</think>` tags but no parseable tool call, the tags leak through.

**Fix:** Always clean response text regardless of whether tools were found. Move the
`clean_tool_calls_from_text()` call outside the tool-call conditional:

```python
response_text = _generate_response(...)
response_text = clean_tool_calls_from_text(response_text)  # Always clean
tool_calls = parse_tool_calls(response_text)
if tool_calls:
    response_text = clean_tool_calls_from_text(response_text)  # Clean again after tool extraction
```

**Impact:** Medium. Prevents tag leakage in non-tool-call responses.

---

## Fix 4: `tool_use` block flattening confuses the model

**File:** `lfm_thinking.py` — in the conversation flattening logic (where history messages
are converted to plain text for local models)

**Problem:** When converting `tool_use` content blocks from conversation history into plain
text, the format `"Assistant called tool: shell({"command": "ls"})"` looks like an actual
tool invocation. The model tries to mimic this format instead of using proper tool_call blocks.

**Fix:** Use neutral format:
```python
# Instead of:
lines.append(f"Assistant called tool: {block['name']}({json.dumps(block.get('input', {}))})")
# Use:
lines.append(f"[Used tool {block['name']}]")
```

**Impact:** Medium. Reduces model confusion about tool-calling format.

---

## Fix 5: `LFM_SINGLE_TOOL_MODE` is too restrictive

**File:** `obedient_beast/beast.py`

**Problem:** The boolean `LFM_SINGLE_TOOL_MODE` limits local LLMs to exactly 1 tool call
per turn. If a tool returns empty output (e.g., API failure), the model can't retry.

**Fix:** Replace with configurable `LFM_MAX_TOOL_TURNS`:
```python
# At top of file:
LFM_MAX_TOOL_TURNS = int(os.getenv("LFM_MAX_TOOL_TURNS", "3"))

# In the agent loop, replace:
#   if LFM_SINGLE_TOOL_MODE and llm.backend == "lfm":
#       break
# With:
if llm.backend == "lfm" and LFM_MAX_TOOL_TURNS > 0 and tool_turns_used >= LFM_MAX_TOOL_TURNS:
    break
```

Set `LFM_MAX_TOOL_TURNS=1` for weak models, `3` (default) for capable ones, `0` for unlimited.

**Impact:** High. Allows local models to recover from transient tool failures.

---

## Fix 6: Tool prompt too terse for local models

**File:** `lfm_thinking.py` — `format_tools_for_prompt()` output

**Problem:** The one-liner format `TOOLS: shell() | read_file() | ...` gives the model
very little context about what each tool does or what parameters it needs.

**Fix:** Use multi-line format with descriptions:
```python
tool_lines.append(f"- **{name}**({param_names}): {desc}")
# ...
return "\n".join(tool_lines)
```

And inject as a proper section in the system prompt:
```
## Available Tools

- **shell**(command): Execute a shell command and return the output
- **read_file**(path): Read the contents of a file
...

To use a tool, respond with a JSON block:
```tool_call
{"name": "tool_name", "arguments": {"param": "value"}}
`` `
```

**Impact:** High. Local models need detailed tool descriptions to call tools correctly.

---

## Future: Full Anthropic Format Normalization

The full normalization plan (single canonical Anthropic format, eliminate dual-format
branching in beast.py, add /v1/messages endpoint to lfm_thinking.py) is documented in
the plan file and CLAUDE.md changelog from the previous session. It's a clean architectural
improvement but requires ~850 lines of changes across 3 core files. Best applied during
a quiet period when the codebase isn't actively gaining new features.

Key files from that work are preserved in git history at commit `aff2da0`:
```bash
git show aff2da0:obedient_beast/llm.py     # Normalized llm.py
git show aff2da0:obedient_beast/beast.py    # Normalized beast.py
git show aff2da0:lfm_thinking.py            # With /v1/messages endpoint
git show aff2da0:CLAUDE.md                  # Full changelog
```
