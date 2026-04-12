#!/usr/bin/env python3
"""
browser_tools - Persistent Chromium session via Playwright (OpenClaw-style)
===========================================================================
Gives Beast a dedicated, always-on Chromium profile it can drive with
CDP-level reliability. The profile lives in `workspace/browser_profile/`,
so logins, cookies, localStorage, and extensions persist across restarts —
just like OpenClaw's managed browser. Log into GitHub / Gmail / Slack once in
the visible window, and Beast can keep using the session forever.

Design notes
~~~~~~~~~~~~
- **Lazy import.** Playwright is only imported on first call, so Beast starts
  fine on machines without it installed. Missing Playwright yields a friendly
  error message rather than a crash.
- **Singleton.** A single `(playwright, context, page)` tuple is cached in
  module state. Every tool call reuses it, which keeps navigation fast and
  lets sessions build up naturally (cookies, history, auth tokens).
- **Headful by default.** We launch with `headless=False` because the whole
  point is letting the human log in once. Override with env var
  `BEAST_BROWSER_HEADLESS=true` for CI / servers.
- **No cross-tab magic.** Beast always drives the first page of the context.
  If the page navigates to a popup, call `browser_close` and re-open.

Public tool-facing functions: `browser_goto`, `browser_read`, `browser_click`,
`browser_type`, `browser_screenshot`, `browser_close`.

Install
~~~~~~~
    pip install playwright
    python -m playwright install chromium
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

BROWSER_DIR = Path(__file__).parent / "workspace" / "browser_profile"
SCREENSHOT_DIR = Path(__file__).parent / "workspace" / "screenshots"

# Module-level singleton: (playwright_handle, browser_context, page).
_browser: Optional[tuple] = None


class BrowserUnavailable(RuntimeError):
    """Raised when Playwright isn't installed. Turned into a string error by
    the tool dispatcher in beast.py."""


def _ensure_browser():
    """Launch the persistent browser context if it isn't already running.

    Returns the cached `(pw, ctx, page)` tuple. The first call can take a few
    seconds; subsequent calls are instant.
    """
    global _browser
    if _browser is not None:
        return _browser
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise BrowserUnavailable(
            "Playwright is not installed. Run `pip install playwright && "
            "python -m playwright install chromium` to enable browser tools."
        ) from exc

    BROWSER_DIR.mkdir(parents=True, exist_ok=True)
    headless = os.getenv("BEAST_BROWSER_HEADLESS", "false").lower() == "true"
    pw = sync_playwright().start()
    ctx = pw.chromium.launch_persistent_context(
        user_data_dir=str(BROWSER_DIR),
        headless=headless,
        viewport={"width": 1280, "height": 800},
        # Useful realism: match a modern Chrome user agent so sites don't gate
        # us behind "please upgrade your browser" walls.
        user_agent=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
    )
    page = ctx.pages[0] if ctx.pages else ctx.new_page()
    _browser = (pw, ctx, page)
    return _browser


def browser_goto(url: str) -> str:
    """Navigate to a URL and return the page title + current URL."""
    _, _, page = _ensure_browser()
    page.goto(url, wait_until="domcontentloaded", timeout=30000)
    return f"Opened {page.url}\nTitle: {page.title()}"


def browser_read(max_chars: int = 4000) -> str:
    """Return visible page text, truncated to `max_chars` for LLM context."""
    _, _, page = _ensure_browser()
    text = page.evaluate("() => document.body.innerText")
    if not text:
        return "(empty page)"
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n... [truncated, {len(text)} chars total]"
    return text


def browser_click(selector: str) -> str:
    """Click the first element matching `selector` (CSS or Playwright locator)."""
    _, _, page = _ensure_browser()
    page.click(selector, timeout=10000)
    return f"Clicked {selector!r}"


def browser_type(selector: str, text: str, submit: bool = False) -> str:
    """Fill an input matching `selector` with `text`, optionally pressing Enter."""
    _, _, page = _ensure_browser()
    page.fill(selector, text)
    if submit:
        page.press(selector, "Enter")
    suffix = " and submitted" if submit else ""
    return f"Typed {len(text)} chars into {selector!r}{suffix}"


def browser_screenshot(filename: Optional[str] = None, full_page: bool = True) -> str:
    """Save a screenshot of the current page and return its file path."""
    _, _, page = _ensure_browser()
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    filename = filename or f"browser_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    path = SCREENSHOT_DIR / filename
    page.screenshot(path=str(path), full_page=full_page)
    return str(path)


def browser_close() -> str:
    """Shut down the persistent browser context (cookies stay on disk)."""
    global _browser
    if _browser is None:
        return "Browser is not running."
    pw, ctx, _ = _browser
    try:
        ctx.close()
    finally:
        pw.stop()
        _browser = None
    return "Browser closed. Profile saved to workspace/browser_profile/."
