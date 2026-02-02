#!/usr/bin/env python3
"""
Obedient Beast - HTTP Server for WhatsApp Bridge
=================================================
Receives messages from bridge.js and returns responses.

Usage:
    python server.py              # Start server on port 5000
    PORT=8080 python server.py    # Use different port
"""

import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv

load_dotenv()

from beast import run
from llm import get_llm

app = Flask(__name__)

# Configuration
PORT = int(os.getenv("BEAST_PORT", "5001"))
ALLOWED_NUMBERS = os.getenv("ALLOWED_NUMBERS", "").split(",")

# Per-sender session history (keyed by phone number)
llm = get_llm()


def is_allowed(sender: str) -> bool:
    """Check if sender is in allowlist (empty list = allow all)."""
    # Special case: OWNER means the logged-in WhatsApp account owner
    if sender == "OWNER":
        return True
    
    if not ALLOWED_NUMBERS or ALLOWED_NUMBERS == [""]:
        return True
    # Normalize phone numbers (remove @s.whatsapp.net suffix)
    sender_clean = sender.split("@")[0]
    return any(sender_clean.endswith(num.replace("+", "")) for num in ALLOWED_NUMBERS)


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "backend": llm.backend})


@app.route("/message", methods=["POST"])
def message():
    """
    Handle incoming message from WhatsApp bridge.
    Expects JSON: {"text": "...", "sender": "..."}
    Returns JSON: {"response": "..."}
    """
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No JSON body"}), 400
    
    text = data.get("text", "").strip()
    sender = data.get("sender", "unknown")
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    # Check allowlist
    if not is_allowed(sender):
        print(f"[Blocked] {sender}: {text[:50]}...")
        return jsonify({"response": "Not authorized."}), 403
    
    print(f"[{sender}] {text[:100]}...")
    
    # Use sender as session ID for conversation continuity
    session_id = f"wa_{sender.split('@')[0]}"
    
    try:
        response = run(text, session_id, llm)
        print(f"[Response] {response[:100]}...")
        return jsonify({"response": response})
    
    except Exception as e:
        print(f"[Error] {e}")
        return jsonify({"response": f"Error: {str(e)}"}), 500


if __name__ == "__main__":
    from llm import BACKEND
    print("=" * 60)
    print("🐺 Obedient Beast Server")
    print(f"   Port: {PORT}")
    print(f"   Backend: {BACKEND}")
    print(f"   Allowlist: {ALLOWED_NUMBERS if ALLOWED_NUMBERS != [''] else '(all)'}")
    print("=" * 60)
    app.run(host="0.0.0.0", port=PORT, debug=False)
