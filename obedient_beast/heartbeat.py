#!/usr/bin/env python3
"""
Heartbeat - Autonomous Task Scheduler for Beast
================================================
Runs as a background loop (or standalone script) that periodically
checks the task queue and processes pending tasks via beast.run().

Inspired by Clawdbot/OpenClaw's autonomous agent cron system.

Usage:
    python heartbeat.py              # Run heartbeat loop (foreground)
    python heartbeat.py --once       # Process one cycle and exit
    python heartbeat.py --status     # Show task queue status

The heartbeat respects capability tiers:
    - FULL mode (Claude/OpenAI): processes multiple tasks per cycle
    - LITE mode (local LFM): processes one task per cycle, longer interval
"""

import os
import sys
import json
import time
import signal
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Import Beast and capabilities
# ---------------------------------------------------------------------------
# Ensure we can import from the same directory
sys.path.insert(0, str(Path(__file__).parent))

from capabilities import (
    HEARTBEAT_INTERVAL_SEC,
    HEARTBEAT_TASKS_PER_CYCLE,
    TIER_LABEL,
)
from beast import run
from llm import get_llm

WORKSPACE = Path(__file__).parent / "workspace"
TASKS_FILE = WORKSPACE / "tasks.json"
HEARTBEAT_CONTROL_FILE = WORKSPACE / "heartbeat_control.json"

# Graceful shutdown flag
_shutdown = False


def _handle_signal(signum, frame):
    """Handle SIGINT/SIGTERM for graceful shutdown."""
    global _shutdown
    print("\n[Heartbeat] Shutting down gracefully...")
    _shutdown = True


def load_tasks() -> dict:
    """Load the task queue from tasks.json."""
    if not TASKS_FILE.exists():
        return {"tasks": []}
    try:
        return json.loads(TASKS_FILE.read_text())
    except (json.JSONDecodeError, IOError) as e:
        print(f"[Heartbeat] Error reading tasks: {e}", file=sys.stderr)
        return {"tasks": []}


def save_tasks(data: dict):
    """Save the task queue to tasks.json."""
    TASKS_FILE.write_text(json.dumps(data, indent=2))


def get_pending_tasks(data: dict) -> list:
    """Get pending tasks sorted by priority (high > medium > low)."""
    priority_order = {"high": 0, "medium": 1, "low": 2}
    pending = [t for t in data.get("tasks", []) if t.get("status") == "pending"]
    pending.sort(key=lambda t: priority_order.get(t.get("priority", "medium"), 1))
    return pending


def process_task(task: dict, llm) -> str:
    """
    Process a single task by calling beast.run() with the task description.
    Returns the response from Beast.
    """
    task_id = task.get("id", "?")
    description = task.get("description", "No description")
    
    # Build a prompt that tells Beast this is an autonomous task
    prompt = (
        f"[AUTONOMOUS TASK #{task_id}] {description}\n"
        f"This is an autonomous task from your task queue. "
        f"Complete it and report the result. "
        f"When done, use add_task with task_id={task_id} and status=done to mark it complete. "
        f"If it fails, use add_task with task_id={task_id} and status=failed."
    )
    
    session_id = "heartbeat_auto"
    
    print(f"[Heartbeat] Processing task #{task_id}: {description[:60]}...")
    
    try:
        response = run(prompt, session_id, llm)
        print(f"[Heartbeat] Task #{task_id} response: {response[:100]}...")
        return response
    except Exception as e:
        error_msg = f"Error processing task #{task_id}: {e}"
        print(f"[Heartbeat] {error_msg}", file=sys.stderr)
        # Mark task as failed
        data = load_tasks()
        for t in data["tasks"]:
            if t.get("id") == task_id:
                t["status"] = "failed"
                t["error"] = str(e)
                t["updated_at"] = datetime.now().isoformat()
        save_tasks(data)
        return error_msg


def is_heartbeat_enabled() -> bool:
    """Check if heartbeat is enabled via the control file (default: True)."""
    if not HEARTBEAT_CONTROL_FILE.exists():
        return True
    try:
        return json.loads(HEARTBEAT_CONTROL_FILE.read_text()).get("enabled", True)
    except Exception:
        return True


def run_cycle(llm) -> int:
    """
    Run one heartbeat cycle: check for pending tasks and process them.
    Respects heartbeat_control.json — if disabled, skips processing.
    Returns the number of tasks processed.
    """
    # Check control file — /heartbeat off from WhatsApp pauses processing
    if not is_heartbeat_enabled():
        return 0

    data = load_tasks()
    pending = get_pending_tasks(data)
    
    if not pending:
        return 0
    
    # Process up to HEARTBEAT_TASKS_PER_CYCLE tasks
    tasks_to_process = pending[:HEARTBEAT_TASKS_PER_CYCLE]
    processed = 0
    
    for task in tasks_to_process:
        if _shutdown:
            break
        process_task(task, llm)
        processed += 1
    
    return processed


def heartbeat_loop():
    """
    Main heartbeat loop. Runs until interrupted.
    Checks the task queue every HEARTBEAT_INTERVAL_SEC seconds.
    """
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    
    llm = get_llm()
    
    print("=" * 60)
    print("🫀 Beast Heartbeat - Autonomous Scheduler")
    print(f"   Tier: {TIER_LABEL}")
    print(f"   Interval: {HEARTBEAT_INTERVAL_SEC}s ({HEARTBEAT_INTERVAL_SEC // 60} min)")
    print(f"   Tasks/cycle: {HEARTBEAT_TASKS_PER_CYCLE}")
    print(f"   Task file: {TASKS_FILE}")
    print("=" * 60)
    print("[Heartbeat] Running... (Ctrl+C to stop)\n")
    
    while not _shutdown:
        try:
            if not is_heartbeat_enabled():
                print(f"[Heartbeat] Paused (use /heartbeat on from WhatsApp to resume). Sleeping {HEARTBEAT_INTERVAL_SEC}s...")
            else:
                processed = run_cycle(llm)
                if processed > 0:
                    print(f"[Heartbeat] Processed {processed} task(s).")
                else:
                    print(f"[Heartbeat] No pending tasks. Sleeping {HEARTBEAT_INTERVAL_SEC}s...")
            
            # Sleep in small increments so we can respond to shutdown quickly
            for _ in range(HEARTBEAT_INTERVAL_SEC):
                if _shutdown:
                    break
                time.sleep(1)
        
        except Exception as e:
            print(f"[Heartbeat] Cycle error: {e}", file=sys.stderr)
            # Sleep before retrying to avoid tight error loops
            time.sleep(30)
    
    print("[Heartbeat] Stopped.")


def show_status():
    """Print the current task queue status."""
    data = load_tasks()
    tasks = data.get("tasks", [])
    
    if not tasks:
        print("Task queue is empty.")
        return
    
    print(f"Task Queue ({len(tasks)} total):")
    print("-" * 50)
    for t in tasks:
        status_icon = {"pending": "⏳", "done": "✅", "failed": "❌"}.get(t.get("status"), "?")
        print(f"  {status_icon} #{t.get('id', '?')} [{t.get('priority', '?')}] {t.get('description', 'No description')[:50]}")
    
    pending = [t for t in tasks if t.get("status") == "pending"]
    print(f"\n{len(pending)} pending task(s).")


if __name__ == "__main__":
    if "--status" in sys.argv:
        show_status()
    elif "--once" in sys.argv:
        llm = get_llm()
        processed = run_cycle(llm)
        print(f"Processed {processed} task(s).")
    else:
        heartbeat_loop()
