#!/usr/bin/env python3
"""
Streaming Test Client for the local model OpenAI-compatible server
===================================================================
Run this AFTER starting the server with: python lfm_thinking.py or linux_thinking.py (choose server mode)

Usage: python test_client.py [host] [port]
       python test_client.py --eval-parse     # offline parser tests (no server)
       python test_client.py --eval           # parser + live tasks against :8000
  Default: python test_client.py localhost 8000

Features:
  - Runs quick connectivity tests first
  - Interactive chat mode with STREAMING responses (tokens appear as generated)
  - No memory - each message is a fresh conversation
  - Type 'quit' to exit
"""

import sys
import json
import urllib.request
import urllib.error

# Default server address. python test_client.py [--eval|--eval-parse] [host] [port]
_args = [a for a in sys.argv[1:] if not a.startswith("--")]
HOST = _args[0] if _args else "localhost"
PORT = _args[1] if len(_args) > 1 else "8000"
BASE_URL = f"http://{HOST}:{PORT}"

def test_health():
    """Test the health check endpoint."""
    print("=" * 50)
    print("Test 1: Health Check (GET /)")
    print("=" * 50)
    try:
        req = urllib.request.Request(f"{BASE_URL}/")
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            print(f"✅ Server is running!")
            print(f"   Model: {data.get('model', 'unknown')}")
            print(f"   Type:  {data.get('type', 'unknown')}")
            return data.get('model', 'unknown')
    except urllib.error.URLError as e:
        print(f"❌ Failed to connect: {e}")
        print(f"   Make sure the server is running at {BASE_URL}")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_models():
    """Test the models listing endpoint."""
    print("\n" + "=" * 50)
    print("Test 2: List Models (GET /v1/models)")
    print("=" * 50)
    try:
        req = urllib.request.Request(f"{BASE_URL}/v1/models")
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            print(f"✅ Models endpoint working!")
            for model in data.get('data', []):
                print(f"   - {model.get('id', 'unknown')}")
            return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def send_chat_streaming(user_message, max_tokens=15120):
    """
    Send a chat message with streaming enabled.
    Prints tokens as they arrive (no timeout issues).
    """
    payload = {
        "model": "test",
        "messages": [
            {"role": "user", "content": user_message}
        ],
        "max_tokens": max_tokens,
        "stream": True  # Enable streaming
    }
    
    try:
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            f"{BASE_URL}/v1/chat/completions",
            data=data,
            headers={'Content-Type': 'application/json'}
        )
        
        # Open connection and read streaming response
        with urllib.request.urlopen(req) as response:
            # Read line by line for SSE format
            buffer = ""
            for chunk in iter(lambda: response.read(1).decode('utf-8'), ''):
                buffer += chunk
                
                # Process complete SSE messages (end with \n\n)
                while "\n\n" in buffer:
                    message, buffer = buffer.split("\n\n", 1)
                    
                    # Skip empty messages
                    if not message.strip():
                        continue
                    
                    # Parse SSE data line
                    for line in message.split("\n"):
                        if line.startswith("data: "):
                            data_str = line[6:]  # Remove "data: " prefix
                            
                            # Check for end signal
                            if data_str.strip() == "[DONE]":
                                return True
                            
                            try:
                                chunk_data = json.loads(data_str)
                                # Extract and print the delta content
                                if chunk_data.get("choices"):
                                    delta = chunk_data["choices"][0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        print(content, end="", flush=True)
                            except json.JSONDecodeError:
                                pass
        
        return True
        
    except urllib.error.HTTPError as e:
        try:
            error_body = e.read().decode()
            print(f"\nError {e.code}: {error_body}")
        except:
            print(f"\nError {e.code}: {e.reason}")
        return False
    except Exception as e:
        print(f"\nError: {e}")
        return False

def interactive_chat(model_name):
    """Interactive chat loop with streaming - each message is independent (no memory)."""
    print("\n" + "=" * 50)
    print(f"💬 Interactive Chat Mode (Streaming)")
    print(f"   Model: {model_name}")
    print(f"   Server: {BASE_URL}")
    print("=" * 50)
    print("Each message is a fresh conversation (no memory).")
    print("Responses stream in real-time as tokens are generated.")
    print("Type 'quit' to exit.")
    print("=" * 50 + "\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Check for quit command
            if user_input.lower() == "quit":
                print("Goodbye!")
                break
            
            print("Assistant: ", end="", flush=True)
            send_chat_streaming(user_input)
            print("\n")  # Newline after streaming completes
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except EOFError:
            print("\nGoodbye!")
            break

def eval_parse_offline():
    """Parser tests — no server. Follows whatever model is loaded later; no model name here."""
    from local_harness import parse_tool_calls, clean_tool_calls_from_text, tools_to_openai

    print("=" * 50)
    print("Eval: tool-call parse (offline)")
    print("=" * 50)
    cases = [
        ("fence", '```tool_call\n{"name": "shell", "arguments": {"command": "echo hi"}}\n```'),
        ("qwen", '<tool_call>{"name": "read_file", "arguments": {"path": "/tmp/x"}}</tool_call>'),
        ("nested", '```tool_call\n{"name": "write_file", "arguments": {"path": "a", "content": "{ok}"}}\n```'),
    ]
    ok = True
    for label, text in cases:
        parsed = parse_tool_calls(text)
        if not parsed or parsed[0]["function"]["name"] not in ("shell", "read_file", "write_file"):
            print(f"  ❌ {label}: {parsed}")
            ok = False
        else:
            print(f"  ✅ {label}: {parsed[0]['function']['name']}")
    cleaned = clean_tool_calls_from_text("hi</think> there <tool_call>")
    if "</think>" in cleaned or "<tool_call>" in cleaned:
        print(f"  ❌ orphan tags remain: {cleaned!r}")
        ok = False
    else:
        print("  ✅ orphan tags stripped")
    schema = tools_to_openai([
        {"name": "shell", "description": "run", "params": {
            "command": "The shell command to execute",
            "timeout": "Optional: max seconds to wait",
        }}
    ])
    props = schema[0]["function"]["parameters"]
    if props["properties"]["timeout"]["type"] != "integer" or "timeout" in props["required"]:
        print(f"  ❌ schema: {props}")
        ok = False
    else:
        print("  ✅ schema types + optional timeout")
    print("offline:", "PASS" if ok else "FAIL")
    return ok


def eval_live():
    """Live tasks against :8000 — uses whatever model the server loaded (LFM_MODEL)."""
    print("\n" + "=" * 50)
    print("Eval: live server (no-tool, one-tool, two-tool)")
    print("=" * 50)
    model = test_health()
    if not model:
        print("  ⏭  server not up — skip live eval")
        return True

    def complete(payload, timeout=120):
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{BASE_URL}/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())

    dummy = [{
        "type": "function",
        "function": {
            "name": "echo_tool",
            "description": "Echo a string back. Use only when asked to call a tool.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    }]
    dummy2 = dummy + [{
        "type": "function",
        "function": {
            "name": "ping_tool",
            "description": "Ping. Use together with echo_tool when asked for both.",
            "parameters": {"type": "object", "properties": {}},
        },
    }]
    ok = True
    try:
        r = complete({
            "model": "lfm",
            "messages": [{"role": "user", "content": "Reply with exactly the word OK and nothing else."}],
            "max_tokens": 64,
            "chat_template_kwargs": {"enable_thinking": False},
        })
        text = (r["choices"][0]["message"].get("content") or "").strip()
        print(f"  no-tool: {text[:80]!r}")
        if "OK" not in text.upper() and not r["choices"][0]["message"].get("tool_calls"):
            print("  ⚠ no-tool did not contain OK (model-dependent; not a hard fail)")
    except Exception as e:
        print(f"  ❌ no-tool: {e}")
        ok = False

    try:
        r = complete({
            "model": "lfm",
            "messages": [{"role": "user", "content": "Call echo_tool with text=hello. Do not answer in prose."}],
            "tools": dummy,
            "max_tokens": 512,
        })
        msg = r["choices"][0]["message"]
        tcs = msg.get("tool_calls") or []
        print(f"  one-tool: {len(tcs)} call(s) finish={r['choices'][0].get('finish_reason')}")
        if not tcs:
            print("  ⚠ no tool_calls (parser/template); check server logs")
        else:
            print(f"     {tcs[0].get('function', {}).get('name')}")
    except Exception as e:
        print(f"  ❌ one-tool: {e}")
        ok = False

    try:
        r = complete({
            "model": "lfm",
            "messages": [{"role": "user", "content": "Call echo_tool with text=a AND ping_tool. Both, one turn."}],
            "tools": dummy2,
            "max_tokens": 512,
        })
        tcs = r["choices"][0]["message"].get("tool_calls") or []
        print(f"  two-tool: {len(tcs)} call(s)")
    except Exception as e:
        print(f"  ❌ two-tool: {e}")
        ok = False

    # Empty tool result — model should continue, not be told to "summarize"
    try:
        r = complete({
            "model": "lfm",
            "messages": [
                {"role": "user", "content": "Call echo_tool with text=hi then tell me the result."},
                {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "call_test", "type": "function",
                    "function": {"name": "echo_tool", "arguments": "{\"text\":\"hi\"}"},
                }]},
                {"role": "tool", "tool_call_id": "call_test", "name": "echo_tool", "content": ""},
            ],
            "tools": dummy,
            "max_tokens": 512,
        })
        msg = r["choices"][0]["message"]
        print(f"  empty-tool-result: finish={r['choices'][0].get('finish_reason')} "
              f"tools={len(msg.get('tool_calls') or [])} text={(msg.get('content') or '')[:60]!r}")
    except Exception as e:
        print(f"  ❌ empty-tool-result: {e}")
        ok = False

    print("live:", "PASS" if ok else "FAIL")
    return ok


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("--eval", "--eval-parse"):
        live = sys.argv[1] == "--eval"
        ok = eval_parse_offline()
        if live:
            ok = eval_live() and ok
        sys.exit(0 if ok else 1)

    print(f"\n🔍 Testing local model server at {BASE_URL}\n")
    
    # Run connectivity tests
    model_name = test_health()
    
    if not model_name:
        print("\n" + "=" * 50)
        print("⚠️  Server not reachable. Start it first with:")
        print("   python lfm_thinking.py")
        print("   Then choose option 2 (Server mode)")
        print("=" * 50)
        return
    
    test_models()
    
    # Enter interactive chat mode with streaming
    interactive_chat(model_name)

if __name__ == "__main__":
    main()
