"""Capture Reddit (Postmill) auth state for WebArena tasks that require login.

Logs into the local forum with WebArena's reddit test account and writes a Playwright
storage_state JSON to ``docker/webarena/.auth/reddit_state.json`` — the file WebArenaSource loads
(via ``auth_dir``) so login-gated tasks run authenticated. Selectors mirror WebArena's own
auto_login. Creds and URL are env-overridable.

Run after the forum is up:  python docker/webarena/scripts/capture-auth.py
"""

from __future__ import annotations

import os
from pathlib import Path

REDDIT = os.environ.get("REDDIT", "http://localhost:9999")
USER = os.environ.get("REDDIT_USER", "MarvelsGrantMan136")
PASS = os.environ.get("REDDIT_PASS", "test1234")
AUTH_DIR = Path(os.environ.get("WEBARENA_AUTH_DIR") or (Path(__file__).resolve().parents[1] / ".auth"))


def main() -> None:
    from playwright.sync_api import sync_playwright

    AUTH_DIR.mkdir(parents=True, exist_ok=True)
    dest = AUTH_DIR / "reddit_state.json"
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context()
        page = ctx.new_page()
        page.goto(f"{REDDIT.rstrip('/')}/login")
        page.get_by_label("Username").fill(USER)
        page.get_by_label("Password").fill(PASS)
        page.get_by_role("button", name="Log in").click()
        page.wait_for_load_state("networkidle")
        # Sanity check: the logged-in navbar shows the username.
        if USER.lower() not in page.content().lower():
            print("WARNING: username not found on page after login — check creds/selectors")
        ctx.storage_state(path=str(dest))
        browser.close()
    print(f"saved -> {dest}")


if __name__ == "__main__":
    main()
