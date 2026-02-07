#!/usr/bin/env python3
"""
Streaming Test Client for the local model OpenAI-compatible server
===================================================================
Run this AFTER starting the server with: python lfm_thinking.py or linux_thinking.py (choose server mode)

Usage: python test_client.py [host] [port]
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

# Default server address
HOST = sys.argv[1] if len(sys.argv) > 1 else "localhost"
PORT = sys.argv[2] if len(sys.argv) > 2 else "8000"
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

def main():
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
