"""The notify spec must survive a REAL Telegram token (it contains a colon).

'telegram:<token>:<chat>' with token='8691681982:AAH...' was split with
split(':', 2), which handed half the token to chat_id — no message could
ever leave. Found the first time a real token was wired (11 Sep).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvti.serving.alert_sink import TelegramNotifier, _build_one


def test_a_real_token_with_colon_parses_whole():
    n = _build_one("telegram:8691681982:AAHtokenSecondHalf:1883642843")
    assert isinstance(n, TelegramNotifier)
    assert n.base.endswith("/bot8691681982:AAHtokenSecondHalf")
    assert n.chat_id == "1883642843"


def test_a_negative_group_chat_id_parses(  ):
    n = _build_one("telegram:111:AAA:-1002233445566")
    assert n.chat_id == "-1002233445566"


def test_the_evidence_clip_rides_along_as_a_video(tmp_path, monkeypatch):
    """'i need the videos landing on the group chat' (12 Sep): when the event
    dir holds a clip.mp4, the notifier sends it via sendVideo AFTER the photo
    album — and a missing/oversized clip degrades to photos-only, silently."""
    ev = tmp_path / "event"
    ev.mkdir()
    (ev / "frame_00.jpg").write_bytes(b"jpg")
    (ev / "clip.mp4").write_bytes(b"mp4bytes")
    calls = []

    def fake_urlopen(req, timeout=None):
        url = req if isinstance(req, str) else req.full_url
        calls.append(url.rsplit("/", 1)[-1])
        class R:  # noqa: D401 - minimal response stub
            status = 200
        return R()

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    n = TelegramNotifier("111:AAA", "-5324877092")
    n.notify({"priority": "high", "rule": "restricted_entrance", "camera_id": "Facility Entrance",
              "confidence": 0.99, "reason": "person entered the restricted zone",
              "evidence_dir": str(ev)})
    assert calls == ["sendPhoto", "sendVideo"]


def test_a_missing_clip_still_sends_photos(tmp_path, monkeypatch):
    ev = tmp_path / "event"
    ev.mkdir()
    (ev / "frame_00.jpg").write_bytes(b"jpg")
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append((req if isinstance(req, str) else req.full_url).rsplit("/", 1)[-1])
        class R:
            status = 200
        return R()

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    TelegramNotifier("111:AAA", "1").notify(
        {"priority": "low", "rule": "r", "camera_id": "c", "confidence": 0.5,
         "reason": "x", "evidence_dir": str(ev)})
    assert calls == ["sendPhoto"]
