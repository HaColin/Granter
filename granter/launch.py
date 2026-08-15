"""One-command local start.

    python -m granter.launch

Does the three things that otherwise have to be remembered in order: installs
dependencies if any are missing, fetches opportunities if the corpus is empty,
and starts the server with the browser pointed at it.

Every step is skipped when it is already done, so a second run starts in
seconds. Nothing here fabricates data: if the ingest cannot reach a source it
says so and starts anyway, and the app reports having nothing to search rather
than showing something it did not retrieve.
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

from . import store

REQUIRED_PACKAGES = ("fastapi", "uvicorn", "jinja2", "httpx", "pydantic", "multipart")
DEFAULT_PORT = 8000
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def missing_packages() -> list[str]:
    return [name for name in REQUIRED_PACKAGES if importlib.util.find_spec(name) is None]


def install_dependencies() -> bool:
    requirements = PROJECT_ROOT / "requirements.txt"
    print("Installing dependencies (first run only)...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements)],
        cwd=PROJECT_ROOT,
    )
    if result.returncode != 0:
        print("\nCould not install dependencies. Check your internet connection.")
        return False
    return True


def corpus_is_empty() -> bool:
    return store.load().is_empty


def fetch_opportunities(limit: int = 2000) -> bool:
    """Populate the corpus. Returns False if nothing could be fetched."""
    print("Fetching funding opportunities (first run only, about a minute)...")
    result = subprocess.run(
        [
            sys.executable, "-m", "granter.ingest",
            "--source", "all", "--limit", str(limit), "--include-forecasted",
        ],
        cwd=PROJECT_ROOT,
    )
    return result.returncode == 0


def open_browser_when_ready(url: str, delay: float = 2.0) -> None:
    """Open the browser once the server has had a moment to bind its port."""
    def wait_then_open() -> None:
        time.sleep(delay)
        webbrowser.open(url)

    threading.Thread(target=wait_then_open, daemon=True).start()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Start Granter locally.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--skip-fetch", action="store_true", help="start even with an empty corpus")
    args = parser.parse_args(argv)

    if missing_packages() and not install_dependencies():
        return 1

    if corpus_is_empty() and not args.skip_fetch:
        if not fetch_opportunities():
            print("\nCould not fetch opportunities. Starting anyway — the app will say")
            print("it has nothing to search rather than showing anything it did not retrieve.")
    else:
        corpus = store.load()
        stamp = f", last checked {corpus.fetched_at.date()}" if corpus.fetched_at else ""
        print(f"{len(corpus):,} opportunities ready{stamp}.")

    from . import chat

    url = f"http://localhost:{args.port}"
    print(f"\nGranter is starting at {url}")
    print("  the form works with no setup" + (
        "; the chat assistant is enabled" if chat.is_available() else ""
    ))
    print("  press Ctrl+C to stop\n")

    if not args.no_browser:
        open_browser_when_ready(url)

    import uvicorn

    uvicorn.run("granter.app:app", host="127.0.0.1", port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
