"""Launcher tests: the decisions it makes before starting the server."""

from __future__ import annotations

from granter import launch
from granter.store import Corpus

from .fixtures import opportunity


def test_every_runtime_dependency_is_checked_for():
    """A missing package must be detected before the server tries to import it."""
    assert launch.missing_packages() == []  # this environment has them all
    assert "fastapi" in launch.REQUIRED_PACKAGES
    assert "multipart" in launch.REQUIRED_PACKAGES  # form parsing, easy to miss


def test_a_populated_corpus_is_not_refetched(monkeypatch):
    monkeypatch.setattr(launch.store, "load", lambda: Corpus([opportunity()]))
    assert launch.corpus_is_empty() is False


def test_an_empty_corpus_triggers_a_fetch(monkeypatch):
    monkeypatch.setattr(launch.store, "load", lambda: Corpus([]))
    assert launch.corpus_is_empty() is True


def test_a_failed_fetch_does_not_stop_the_app_starting(monkeypatch, capsys):
    """Better to start and say there is nothing to search than to refuse to run."""
    monkeypatch.setattr(launch.store, "load", lambda: Corpus([]))
    monkeypatch.setattr(launch, "missing_packages", lambda: [])
    monkeypatch.setattr(launch, "fetch_opportunities", lambda *a, **k: False)

    started = {}
    monkeypatch.setattr(launch, "open_browser_when_ready", lambda *a, **k: None)

    import uvicorn

    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: started.update(k))

    assert launch.main(["--no-browser"]) == 0
    assert started["port"] == launch.DEFAULT_PORT
    assert "nothing to search" in capsys.readouterr().out


def test_the_server_is_bound_to_localhost_only(monkeypatch):
    """A local launcher must not expose the app to the network."""
    monkeypatch.setattr(launch.store, "load", lambda: Corpus([opportunity()]))
    monkeypatch.setattr(launch, "missing_packages", lambda: [])
    captured = {}

    import uvicorn

    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: captured.update(k))
    launch.main(["--no-browser", "--port", "8123"])
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8123
