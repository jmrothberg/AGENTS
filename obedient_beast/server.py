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

from beast import run, get_and_clear_pending_image
from llm import get_llm

app = Flask(__name__)

# Configuration
PORT = int(os.getenv("BEAST_PORT", "5001"))
ALLOWED_NUMBERS = [n.strip() for n in os.getenv("ALLOWED_NUMBERS", "").split(",") if n.strip()]
ALLOWED_GROUPS = [g.strip() for g in os.getenv("ALLOWED_GROUPS", "").split(",") if g.strip()]
RESPOND_TO_OTHERS = os.getenv("RESPOND_TO_OTHERS", "false").lower() == "true"

# Per-sender session history (keyed by phone number)
llm = get_llm()


def is_allowed(sender: str, chat_id: str = None) -> bool:
    """
    Check if sender is allowed to trigger Beast.
    
    Rules:
    1. OWNER (your own messages) - always allowed
    2. If ALLOWED_NUMBERS is empty - allow nobody except OWNER
    3. If sender's number is in ALLOWED_NUMBERS - allowed
    4. In groups: only respond to others if RESPOND_TO_OTHERS=true
    5. If ALLOWED_GROUPS is set, only respond in those groups
    """
    # Special case: OWNER means the logged-in WhatsApp account owner
    if sender == "OWNER":
        # Check group restrictions for owner too
        if chat_id and ALLOWED_GROUPS:
            group_id = chat_id.split("@")[0]
            if not any(group_id == g or group_id.endswith(g) for g in ALLOWED_GROUPS):
                print(f"[Blocked] Group {group_id} not in allowed list")
                return False
        return True
    
    # If no allowed numbers set, only OWNER can use Beast
    if not ALLOWED_NUMBERS:
        print(f"[Blocked] {sender} - no allowed numbers configured (OWNER only mode)")
        return False
    
    # Check if this is a group message from someone other than owner
    is_group = chat_id and chat_id.endswith("@g.us") if chat_id else False
    if is_group and not RESPOND_TO_OTHERS:
        print(f"[Blocked] {sender} - RESPOND_TO_OTHERS is false")
        return False
    
    # Check group restrictions
    if is_group and ALLOWED_GROUPS:
        group_id = chat_id.split("@")[0]
        if not any(group_id == g or group_id.endswith(g) for g in ALLOWED_GROUPS):
            print(f"[Blocked] Group {group_id} not in allowed list")
            return False
    
    # Normalize phone numbers (remove @s.whatsapp.net suffix)
    sender_clean = sender.split("@")[0]
    allowed = any(sender_clean.endswith(num.replace("+", "")) for num in ALLOWED_NUMBERS)
    if not allowed:
        print(f"[Blocked] {sender_clean} not in ALLOWED_NUMBERS")
    return allowed


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "backend": llm.backend})


@app.route("/message", methods=["POST"])
def message():
    """
    Handle incoming message from WhatsApp bridge.
    Expects JSON: {"text": "...", "sender": "...", "chat_id": "..."}
    Returns JSON: {"response": "..."}
    """
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No JSON body"}), 400
    
    text = data.get("text", "").strip()
    sender = data.get("sender", "unknown")
    chat_id = data.get("chat_id", "")
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    # Check allowlist with chat context
    if not is_allowed(sender, chat_id):
        return jsonify({"error": "Not authorized"}), 403
    
    print(f"[{sender}] {text[:100]}...")
    
    # Use sender as session ID for conversation continuity
    session_id = f"wa_{sender.split('@')[0]}"
    
    try:
        response = run(text, session_id, llm)
        print(f"[Response] {response[:100]}...")
        
        # Check if there's an image to send
        image_path = get_and_clear_pending_image()
        result = {"response": response}
        if image_path:
            result["image"] = image_path
            print(f"[Image] {image_path}")
        
        return jsonify(result)
    
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
