#!/usr/bin/env python3
"""
Obedient Beast - HTTP Server for WhatsApp Bridge
=================================================
Receives messages from bridge.js and returns responses via beast.run().

This server sits between the WhatsApp bridge (Node.js) and Beast (Python):
    WhatsApp → bridge.js → HTTP POST /message → server.py → beast.run() → response

Security model (allowlist):
~~~~~~~~~~~~~~~~~~~~~~~~~~~
- OWNER (your own WhatsApp messages) — always allowed
- ALLOWED_NUMBERS — comma-separated phone numbers that can trigger Beast
- ALLOWED_GROUPS — comma-separated group IDs (optional, restricts which groups)
- RESPOND_TO_OTHERS — if false (default), only respond to OWNER in groups
- @beast mention — OWNER can start any group message with "@beast" to summon
  Beast in that group for just that one message, bypassing ALLOWED_GROUPS

Session IDs:
~~~~~~~~~~~~
Each WhatsApp sender gets a unique session ID derived from their phone number:
    "wa_<phone>" (e.g., "wa_12025551234")
This gives each sender their own conversation history (stored in sessions/).

Pending image pattern:
~~~~~~~~~~~~~~~~~~~~~~
When Beast takes a screenshot (via the screenshot tool), it sets a pending
image path. After beast.run() returns, server.py checks for this image
and includes it in the JSON response so bridge.js can send it via WhatsApp.

Usage:
    python server.py              # Start server on port 5001
    PORT=8080 python server.py    # Use different port
"""

import os
import json
from flask import Flask, request, jsonify
from dotenv import load_dotenv

load_dotenv()

from beast import run, get_and_clear_pending_image
from llm import get_llm

app = Flask(__name__)

# Configuration from environment
PORT = int(os.getenv("BEAST_PORT", "5001"))

# Allowlist: comma-separated phone numbers (e.g., "+12025551234,+44...")
ALLOWED_NUMBERS = [n.strip() for n in os.getenv("ALLOWED_NUMBERS", "").split(",") if n.strip()]

# Optional: restrict to specific WhatsApp groups
ALLOWED_GROUPS = [g.strip() for g in os.getenv("ALLOWED_GROUPS", "").split(",") if g.strip()]

# Whether to respond to non-owner messages in groups
RESPOND_TO_OTHERS = os.getenv("RESPOND_TO_OTHERS", "false").lower() == "true"

# Single LLM instance shared across all requests
llm = get_llm()

# ---------------------------------------------------------------------------
# Dynamic @beast access: OWNER can open/close groups at runtime
# ---------------------------------------------------------------------------
# OWNER sends "!openbeast" in a group → anyone in that group can @beast
# OWNER sends "!closebeast" in a group → revoke access
# OWNER sends "!listbeast" anywhere → list open groups
# Persisted to workspace/open_chats.json so it survives restarts.
# ---------------------------------------------------------------------------
OPEN_CHATS_FILE = os.path.join(os.path.dirname(__file__), "workspace", "open_chats.json")


def load_open_chats() -> set:
    """Load the set of dynamically opened group chat IDs from disk."""
    try:
        with open(OPEN_CHATS_FILE, "r") as f:
            return set(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError):
        return set()


def save_open_chats(chats: set):
    """Persist the set of opened group chat IDs to disk."""
    os.makedirs(os.path.dirname(OPEN_CHATS_FILE), exist_ok=True)
    with open(OPEN_CHATS_FILE, "w") as f:
        json.dump(sorted(chats), f, indent=2)


OPEN_CHATS = load_open_chats()


def is_allowed(sender: str, chat_id: str = None) -> bool:
    """
    Check if sender is allowed to trigger Beast.

    Authorization rules (checked in order):
    1. OWNER (your own messages) — always allowed (subject to group restrictions)
    2. If ALLOWED_NUMBERS is empty — allow nobody except OWNER
    3. If sender's number is in ALLOWED_NUMBERS — allowed
    4. In groups: only respond to others if RESPOND_TO_OTHERS=true
    5. If ALLOWED_GROUPS is set, only respond in those groups
    """
    # Special case: OWNER means the logged-in WhatsApp account owner.
    # bridge.js sets sender="OWNER" when it detects the message is from
    # the same phone number that's running the WhatsApp session.
    if sender == "OWNER":
        # Even OWNER respects group restrictions if ALLOWED_GROUPS is set
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

    # Normalize phone numbers (remove @s.whatsapp.net suffix) and check allowlist
    sender_clean = sender.split("@")[0]
    allowed = any(sender_clean.endswith(num.replace("+", "")) for num in ALLOWED_NUMBERS)
    if not allowed:
        print(f"[Blocked] {sender_clean} not in ALLOWED_NUMBERS")
    return allowed


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint — used by bridge.js and monitoring."""
    return jsonify({"status": "ok", "backend": llm.backend})


@app.route("/message", methods=["POST"])
def message():
    """
    Handle incoming message from WhatsApp bridge.
    Expects JSON: {"text": "...", "sender": "...", "chat_id": "..."}
    Returns JSON: {"response": "...", "image": "path/to/image.png" (optional)}
    """
    data = request.get_json()

    if not data:
        return jsonify({"error": "No JSON body"}), 400

    text = data.get("text", "").strip()
    sender = data.get("sender", "unknown")
    chat_id = data.get("chat_id", "")
    image_path = data.get("image_path")  # Optional: image from WhatsApp

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # --- Magic commands: OWNER can open/close groups for @beast access ---
    text_lower = text.lower().strip()
    if sender == "OWNER":
        global OPEN_CHATS
        if text_lower == "!openbeast":
            if not chat_id or not chat_id.endswith("@g.us"):
                return jsonify({"response": "Send !openbeast from inside a group chat."}), 200
            OPEN_CHATS.add(chat_id)
            save_open_chats(OPEN_CHATS)
            print(f"[!openbeast] Opened group {chat_id}")
            return jsonify({"response": f"Beast is now listening for @beast in this group."}), 200
        elif text_lower == "!closebeast":
            if not chat_id or not chat_id.endswith("@g.us"):
                return jsonify({"response": "Send !closebeast from inside a group chat."}), 200
            OPEN_CHATS.discard(chat_id)
            save_open_chats(OPEN_CHATS)
            print(f"[!closebeast] Closed group {chat_id}")
            return jsonify({"response": f"Beast no longer listening in this group."}), 200
        elif text_lower == "!listbeast":
            if OPEN_CHATS:
                listing = "\n".join(f"  • {c}" for c in sorted(OPEN_CHATS))
                return jsonify({"response": f"Open groups:\n{listing}"}), 200
            else:
                return jsonify({"response": "No groups currently open."}), 200

    # --- @beast mention handling ---
    # 1. OWNER @beast → always works (bypasses group restrictions)
    # 2. Anyone @beast in an OPEN_CHATS group → allowed
    # 3. Normal allowlist rules apply otherwise
    beast_mention = False
    if text_lower.startswith("@beast"):
        beast_mention = True
        text = text[6:].strip()  # Strip "@beast" prefix
        if not text:
            return jsonify({"response": "Usage: @beast <your message>"}), 200

    # Check authorization
    if beast_mention and sender == "OWNER":
        print(f"[@beast] OWNER mention in {chat_id}")
    elif beast_mention and chat_id in OPEN_CHATS:
        print(f"[@beast] Open-group mention by {sender} in {chat_id}")
    elif not is_allowed(sender, chat_id):
        return jsonify({"error": "Not authorized"}), 403

    print(f"\n{'='*50}")
    print(f"[{sender}] {text[:200]}")
    if image_path:
        print(f"[Attached image] {image_path}")

    # Use sender phone number as session ID for conversation continuity.
    # This means each WhatsApp contact has their own persistent conversation.
    session_id = f"wa_{sender.split('@')[0]}"

    try:
        response = run(text, session_id, llm, image_path=image_path)
        # Show which tools were used (parse from stderr output of beast.run)
        print(f"[Response] ({len(response)} chars) {response[:200]}...")

        # Check if Beast generated an image (e.g., screenshot tool was used).
        # If so, include the path in the response for bridge.js to send.
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
    print(f"   Allowlist: {ALLOWED_NUMBERS if ALLOWED_NUMBERS else '(OWNER only — set ALLOWED_NUMBERS to add others)'}")
    if OPEN_CHATS:
        print(f"   Open groups (@beast): {len(OPEN_CHATS)} group(s)")
    print("=" * 60)
    app.run(host="0.0.0.0", port=PORT, debug=False)
