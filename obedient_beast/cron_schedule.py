#!/usr/bin/env python3
"""
cron_schedule - Minimal 5-field cron parser (zero dependencies)
================================================================
Computes the next fire time for a standard cron expression without pulling in
`croniter` or any third-party library. Used by `heartbeat.py` to schedule
recurring tasks with natural cron syntax like `0 9 * * 1-5`
("9am every weekday").

Supported syntax (5 fields separated by whitespace):

    minute  hour  day-of-month  month  day-of-week
    0-59    0-23  1-31          1-12   0-6 (0 = Sunday)

Each field supports:
- `*`              — any value
- `5`              — exact value
- `1,3,5`          — a list
- `1-5`            — a range
- `*/15`           — every 15 units from the start
- `10-30/5`        — every 5 units from 10 to 30

This is deliberately small: one function (`next_run`) and one parser helper.
If you need the full cron grammar (last-day-of-month, `L`, `W`, named months)
install `croniter` and swap it in — the public signature is compatible.
"""

from __future__ import annotations

from datetime import datetime, timedelta


def _parse_field(field: str, low: int, high: int) -> set[int]:
    """Expand a single cron field into the set of integers it matches."""
    values: set[int] = set()
    for part in field.split(","):
        step = 1
        if "/" in part:
            part, step_str = part.split("/", 1)
            step = int(step_str)
        if part == "*":
            start, end = low, high
        elif "-" in part:
            start_str, end_str = part.split("-", 1)
            start, end = int(start_str), int(end_str)
        else:
            start = end = int(part)
        values.update(range(start, end + 1, step))
    # Clamp to the legal range; invalid values are silently dropped.
    return {v for v in values if low <= v <= high}


def _cron_dow(dt: datetime) -> int:
    """Return day-of-week in cron form (0 = Sunday, 6 = Saturday).

    Python's `datetime.weekday()` returns 0 = Monday, so we rotate it.
    """
    return (dt.weekday() + 1) % 7


def next_run(cron_expr: str, after: datetime | None = None) -> datetime:
    """Return the next datetime matching `cron_expr` strictly after `after`.

    Raises `ValueError` if the expression is malformed or has no match within
    the next 4 years (a safety cap, not a real limit — anything realistic will
    hit within the first year).
    """
    fields = cron_expr.strip().split()
    if len(fields) != 5:
        raise ValueError(
            f"Cron expression must have 5 fields (min hour dom month dow), got: {cron_expr!r}"
        )
    minutes = _parse_field(fields[0], 0, 59)
    hours = _parse_field(fields[1], 0, 23)
    doms = _parse_field(fields[2], 1, 31)
    months = _parse_field(fields[3], 1, 12)
    dows = _parse_field(fields[4], 0, 6)

    if not (minutes and hours and doms and months and dows):
        raise ValueError(f"Cron expression has an empty field: {cron_expr!r}")

    # Start from the next whole minute after `after`.
    start = (after or datetime.now()) + timedelta(minutes=1)
    dt = start.replace(second=0, microsecond=0)

    # Brute-force scan minute by minute. Capped at 4 years; realistic patterns
    # match within the first few days at most.
    for _ in range(4 * 366 * 24 * 60):
        if (
            dt.month in months
            and dt.day in doms
            and _cron_dow(dt) in dows
            and dt.hour in hours
            and dt.minute in minutes
        ):
            return dt
        dt += timedelta(minutes=1)
    raise ValueError(f"No match for cron {cron_expr!r} within 4 years")


if __name__ == "__main__":
    # Quick smoke test when run directly: `python cron_schedule.py`
    import sys

    expr = sys.argv[1] if len(sys.argv) > 1 else "*/5 * * * *"
    print(f"Next run for {expr!r}: {next_run(expr).isoformat()}")
