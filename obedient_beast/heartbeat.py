#!/usr/bin/env python3
"""
Heartbeat - Autonomous Task Scheduler for Beast
================================================
Runs as a background loop (or standalone script) that periodically
checks the task queue and processes pending tasks via beast.run().

How it works:
~~~~~~~~~~~~~
1. Heartbeat wakes up every N minutes (5 min FULL, 10 min LITE)
2. Checks workspace/tasks.json for pending tasks
3. Picks the highest-priority task and feeds it to beast.run()
4. Beast processes it (using tools as needed) and marks it done/failed
5. Goes back to sleep

Graceful shutdown pattern:
~~~~~~~~~~~~~~~~~~~~~~~~~~
The heartbeat uses signal handlers (SIGINT/SIGTERM) to set a _shutdown flag.
Instead of sleeping for the full interval in one call, it sleeps in 1-second
increments and checks the flag each time. This means Ctrl+C stops the
heartbeat within ~1 second instead of waiting for the full 5/10-minute interval.

Integration with /heartbeat on|off:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The slash commands in beast.py write to workspace/heartbeat_control.json:
    {"enabled": true}  or  {"enabled": false}
The heartbeat checks this file at the start of each cycle. If disabled,
it skips processing but keeps running (checking again on the next cycle).
This lets you pause/resume via WhatsApp without restarting the process.

Inspired by Clawdbot/OpenClaw's autonomous agent cron system.

Usage:
    python heartbeat.py              # Run heartbeat loop (foreground)
    python heartbeat.py --once       # Process one cycle and exit
    python heartbeat.py --status     # Show task queue status
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
OUTBOX_DIR = WORKSPACE / "outbox"

# NOTIFICATION_CHAT_ID: WhatsApp number or group ID to send task completion alerts to.
# If not set, notifications are skipped (console-only output).
NOTIFICATION_CHAT_ID = os.getenv("NOTIFICATION_CHAT_ID", "")

# Graceful shutdown flag — set by signal handler, checked in sleep loop
_shutdown = False


def _handle_signal(signum, frame):
    """
    Handle SIGINT (Ctrl+C) and SIGTERM for graceful shutdown.
    Sets the _shutdown flag so the main loop exits on the next 1-second check.
    """
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
    """
    Get pending tasks that are ready to run, sorted by priority.
    Scheduling rules:
    - Tasks with scheduled_at: only ready if scheduled_at <= now
    - Tasks with next_run_at (recurring): only ready if next_run_at <= now
    - Tasks with neither: always ready (immediate)
    """
    priority_order = {"high": 0, "medium": 1, "low": 2}
    now = datetime.now().isoformat()
    ready = []
    for t in data.get("tasks", []):
        if t.get("status") != "pending":
            continue
        # Check scheduled_at (one-shot timed tasks)
        if t.get("scheduled_at") and t["scheduled_at"] > now:
            continue  # Not yet time
        # Check next_run_at (recurring tasks)
        if t.get("next_run_at") and t["next_run_at"] > now:
            continue  # Not yet time
        ready.append(t)
    ready.sort(key=lambda t: priority_order.get(t.get("priority", "medium"), 1))
    return ready


def _reset_recurring_task(task: dict):
    """
    Reset a recurring task for its next run.

    Supports two recurrence modes (cron wins if both are set):
    - `cron` (string, 5-field): uses cron_schedule.next_run to compute the
      next fire time matching the expression. This mirrors OpenClaw's cron
      tool and lets users say things like "every weekday at 9am" with a
      single expression.
    - `repeat_seconds` (int): simple interval recurrence, e.g. every 3600s.

    After rescheduling, status is reset to "pending" and `last_run_at` is
    stamped so the task queue can show recent activity.
    """
    cron_expr = task.get("cron")
    interval = task.get("repeat_seconds")
    if not cron_expr and not interval:
        return
    task["status"] = "pending"
    task["last_run_at"] = datetime.now().isoformat()
    if cron_expr:
        try:
            from cron_schedule import next_run as _cron_next_run
            task["next_run_at"] = _cron_next_run(cron_expr).isoformat()
            return
        except Exception as exc:
            print(
                f"[Heartbeat] Bad cron {cron_expr!r} on task #{task.get('id')}: {exc}. "
                "Falling back to repeat_seconds if set.",
                file=sys.stderr,
            )
    if interval:
        next_ts = datetime.now().timestamp() + int(interval)
        task["next_run_at"] = datetime.fromtimestamp(next_ts).isoformat()


def _write_notification(task: dict, result: str):
    """
    Write a notification to the outbox for WhatsApp delivery.
    If NOTIFICATION_CHAT_ID is not set, skip silently.
    bridge.js polls the outbox directory and sends + deletes each file.
    """
    if not NOTIFICATION_CHAT_ID:
        return
    try:
        OUTBOX_DIR.mkdir(parents=True, exist_ok=True)
        task_id = task.get("id", "?")
        status = task.get("status", "done")
        # Truncate result to avoid huge messages
        short_result = result[:500] if result else "(no output)"
        notification = {
            "chat_id": NOTIFICATION_CHAT_ID,
            "text": f"[Task #{task_id}] {status.upper()}: {task.get('description', '')[:100]}\n\n{short_result}",
            "timestamp": datetime.now().isoformat()
        }
        filename = f"task_{task_id}_{int(time.time())}.json"
        (OUTBOX_DIR / filename).write_text(json.dumps(notification, indent=2))
    except Exception as e:
        print(f"[Heartbeat] Notification write error: {e}", file=sys.stderr)


def process_task(task: dict, llm) -> str:
    """
    Process a single task by calling beast.run() with the task description.
    Beast will execute tools, mark the task done/failed, and return a response.
    Uses a dedicated "heartbeat_auto" session to avoid polluting user sessions.
    Recurring tasks (repeat_seconds) get rescheduled instead of marked done.
    """
    task_id = task.get("id", "?")
    description = task.get("description", "No description")
    # A task is recurring if it has EITHER a cron expression or a repeat
    # interval. Both paths flow through _reset_recurring_task().
    is_recurring = bool(task.get("repeat_seconds") or task.get("cron"))

    # Build a prompt that tells Beast this is an autonomous task
    if is_recurring:
        # Recurring tasks: don't tell Beast to mark it done (we handle rescheduling)
        prompt = (
            f"[AUTONOMOUS TASK #{task_id} - RECURRING] {description}\n"
            f"This is a recurring autonomous task. Complete it and report the result."
        )
    else:
        prompt = (
            f"[AUTONOMOUS TASK #{task_id}] {description}\n"
            f"This is an autonomous task from your task queue. "
            f"Complete it and report the result. "
            f"When done, use add_task with task_id={task_id} and status=done to mark it complete. "
            f"If it fails, use add_task with task_id={task_id} and status=failed."
        )

    session_id = f"heartbeat_task_{task_id}"

    print(f"[Heartbeat] Processing task #{task_id}: {description[:60]}...")

    try:
        response = run(prompt, session_id, llm)
        print(f"[Heartbeat] Task #{task_id} response: {response[:100]}...")
        # Reschedule recurring tasks
        if is_recurring:
            data = load_tasks()
            for t in data["tasks"]:
                if t.get("id") == task_id:
                    _reset_recurring_task(t)
            save_tasks(data)
            recurrence = (
                f"cron {task.get('cron')!r}" if task.get("cron")
                else f"every {task.get('repeat_seconds')}s"
            )
            print(f"[Heartbeat] Recurring task #{task_id} rescheduled ({recurrence})")
        _write_notification(task, response)
        return response
    except Exception as e:
        error_msg = f"Error processing task #{task_id}: {e}"
        print(f"[Heartbeat] {error_msg}", file=sys.stderr)
        # Mark task as failed directly (in case Beast couldn't do it)
        data = load_tasks()
        for t in data["tasks"]:
            if t.get("id") == task_id:
                if is_recurring:
                    # Recurring tasks reschedule on failure instead of failing permanently
                    _reset_recurring_task(t)
                    t["last_error"] = str(e)
                    print(f"[Heartbeat] Recurring task #{task_id} rescheduled after error")
                else:
                    t["status"] = "failed"
                    t["error"] = str(e)
                    t["updated_at"] = datetime.now().isoformat()
        save_tasks(data)
        _write_notification(task, error_msg)
        return error_msg


def is_heartbeat_enabled() -> bool:
    """
    Check if heartbeat is enabled via the control file.
    Default: True (enabled) if no control file exists.
    The /heartbeat on|off slash commands write to this file.
    """
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
    if not is_heartbeat_enabled():
        return 0

    data = load_tasks()
    pending = get_pending_tasks(data)

    if not pending:
        return 0

    # Process up to HEARTBEAT_TASKS_PER_CYCLE tasks per cycle
    tasks_to_process = pending[:HEARTBEAT_TASKS_PER_CYCLE]
    processed = 0

    for task in tasks_to_process:
        if _shutdown:
            break  # Respect shutdown flag between tasks
        process_task(task, llm)
        processed += 1

    return processed


def heartbeat_loop():
    """
    Main heartbeat loop. Runs until interrupted (Ctrl+C or SIGTERM).
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

            # Sleep in 1-second increments so we can respond to shutdown quickly.
            # Instead of time.sleep(300), we do 300 x time.sleep(1) and check
            # the _shutdown flag each iteration. This gives ~1s shutdown latency.
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
