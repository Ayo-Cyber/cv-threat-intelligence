"""Telegram delivery is paced and retried (16 Sep).

Every multi-camera demo run lost alerts to HTTP 429: bursts of alerts, two
uploads each, into a group chat that allows ~20 messages a minute. The
notifier now spaces calls per chat and retries a 429 after Telegram's own
retry_after — and in production it does so on its own thread.
"""
from __future__ import annotations

import io
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cvti.serving.alert_sink import MultiNotifier, TelegramNotifier, _build_one  # noqa: E402


def _event(tmp_path, n_frames=1):
    ev = tmp_path / "event"
    ev.mkdir(exist_ok=True)
    for i in range(n_frames):
        (ev / f"frame_{i:02d}.jpg").write_bytes(b"jpg")
    return {"priority": "high", "rule": "vehicle_entered", "camera_id": "Gate",
            "confidence": 0.99, "reason": "car crossed the line", "evidence_dir": str(ev)}


class _OK:
    status = 200


def _http429(retry_after=None):
    body = b'{"ok":false,"error_code":429,"parameters":{"retry_after":%d}}' % retry_after \
        if retry_after is not None else b""
    return urllib.error.HTTPError("https://api.telegram.org/x", 429, "Too Many Requests",
                                  {}, io.BytesIO(body))


def test_429_waits_retry_after_and_retries(tmp_path, monkeypatch):
    calls, sleeps = [], []
    answers = [_http429(retry_after=4), _OK()]

    def fake_urlopen(req, timeout=None):
        calls.append(req.full_url.rsplit("/", 1)[-1])
        a = answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return a

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    import cvti.serving.alert_sink as mod
    monkeypatch.setattr(mod.time, "sleep", lambda s: sleeps.append(s))
    n = TelegramNotifier("111:AAA", "1")
    n.notify(_event(tmp_path))
    assert calls == ["sendPhoto", "sendPhoto"]          # retried, delivered
    assert 4.0 in sleeps                                # waited what Telegram asked
    assert n.retried == 1


def test_429_without_retry_after_backs_off_then_gives_up_loudly(tmp_path, monkeypatch):
    calls = []

    def always_429(req, timeout=None):
        calls.append(1)
        raise _http429()

    monkeypatch.setattr(urllib.request, "urlopen", always_429)
    import cvti.serving.alert_sink as mod
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    n = TelegramNotifier("111:AAA", "1")
    n.notify(_event(tmp_path))                          # must not raise
    assert len(calls) == 1 + TelegramNotifier.MAX_RETRIES


def test_calls_to_one_chat_are_paced(tmp_path, monkeypatch):
    sleeps = []
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: _OK())
    import cvti.serving.alert_sink as mod
    monkeypatch.setattr(mod.time, "sleep", lambda s: sleeps.append(s))
    n = TelegramNotifier("111:AAA", "-5324877092")      # a group: 20/min
    assert n.min_gap == TelegramNotifier.GAP_GROUP
    n.notify(_event(tmp_path))
    n.notify(_event(tmp_path))                          # second call must wait
    assert sleeps and max(sleeps) > 2.5
    assert TelegramNotifier("111:AAA", "1883642843").min_gap == TelegramNotifier.GAP_PRIVATE


def test_background_delivery_does_not_block_and_flushes(tmp_path, monkeypatch):
    import threading
    calls = []
    sent = threading.Event()

    def fake_urlopen(req, timeout=None):
        calls.append(req.full_url.rsplit("/", 1)[-1])
        sent.set()
        return _OK()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    n = _build_one("telegram:111:AAA:1")
    assert isinstance(n, TelegramNotifier) and n._queue is not None
    n.notify(_event(tmp_path))
    assert sent.wait(5.0)
    n.close()
    assert calls == ["sendPhoto"]


def test_multi_notifier_closes_every_channel():
    class C:
        closed = False

        def notify(self, e):
            pass

        def close(self):
            self.closed = True

    a, b = C(), C()
    MultiNotifier([a, b]).close()
    assert a.closed and b.closed
