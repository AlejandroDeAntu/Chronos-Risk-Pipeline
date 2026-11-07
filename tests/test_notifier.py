"""Notificación de fallos: mensaje acotado y sin filtrar la URL."""

from __future__ import annotations

from typing import Any

import pytest
import requests

from src.notifications import notifier

URL = "https://hooks.slack.com/services/T000/B000/secret-token"


def test_webhook_payload_is_truncated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: dict[str, Any] = {}

    class _Ok:
        def raise_for_status(self) -> None:
            return None

    def _post(url: str, json: dict[str, str], timeout: int) -> _Ok:
        sent.update(json)
        return _Ok()

    monkeypatch.setenv("PIPELINE_ALERT_WEBHOOK_URL", URL)
    monkeypatch.setattr(notifier.requests, "post", _post)

    notifier.notify_failure("predict:^TEST", ValueError("x" * 1000))

    assert len(sent["text"]) < 400


def test_non_https_webhook_is_not_called(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _post(*args: object, **kwargs: object) -> None:
        raise AssertionError("no debía enviarse")

    monkeypatch.setenv("PIPELINE_ALERT_WEBHOOK_URL", "http://example.com/x")
    monkeypatch.setattr(notifier.requests, "post", _post)

    notifier.notify_failure("predict:^TEST", ValueError("boom"))


def test_webhook_failure_log_hides_url(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def _post(*args: object, **kwargs: object) -> None:
        raise requests.exceptions.ConnectionError(f"cannot reach {URL}")

    monkeypatch.setenv("PIPELINE_ALERT_WEBHOOK_URL", URL)
    monkeypatch.setattr(notifier.requests, "post", _post)

    notifier.notify_failure("predict:^TEST", ValueError("boom"))

    assert "secret-token" not in caplog.text
    assert "ConnectionError" in caplog.text
