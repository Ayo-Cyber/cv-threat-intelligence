"""W1.3-W1.6: the gateway carries the streams, and losing it costs nothing.

Four properties, one file:

  W1.3  a decoder whose source is the restream falls back to the camera's own
        URL when the gateway dies — coverage survives the upgrade failing.
  W1.4  /cameras/{id}/stream flips to `kind: webrtc` exactly while the engine
        advertises a live gateway, and carries the MJPEG fallback with it.
  W1.5  one upstream session per camera stays true, pinned at the wiring
        (frame_source for mapper/scanner/watches; gateway sources for ingest).
  W1.6  a camera with a `detect_source` sends its SUBSTREAM to detection, its
        mainstream to the wall, and one full-resolution frame to the evidence
        folder — never a blanket downscale.
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cvti.serving.go2rtc import Go2rtcGateway  # noqa: E402

CAMS = [
    {"id": "Front Door", "source": "rtsp://cam7/main",
     "detect_source": "rtsp://cam7/sub"},
    {"id": "Yard", "source": "rtsp://cam8/main"},
    {"id": "Demo Clip", "source": "clips/a.mp4"},
]


def _live(gw: Go2rtcGateway, rtsp=8554, api=1984) -> Go2rtcGateway:
    gw.rtsp_port, gw.api_port, gw.webrtc_port = rtsp, api, 8555
    proc = mock.MagicMock()
    proc.poll.return_value = None
    gw._proc = proc
    return gw


# ---------------------------------------------------------------------------
# W1.6 — the substream goes to detection, the mainstream stays for the wall
# ---------------------------------------------------------------------------

class SubstreamTests(unittest.TestCase):
    def test_detection_gets_the_substream_the_wall_keeps_the_main(self):
        gw = _live(Go2rtcGateway(CAMS, tempfile.mkdtemp()))
        self.assertEqual(gw.restream_url("Front Door"),
                         "rtsp://127.0.0.1:8554/front_door_sub")
        self.assertEqual(gw.wall_stream_name("Front Door"), "front_door")

    def test_a_camera_without_detect_source_is_unchanged(self):
        gw = _live(Go2rtcGateway(CAMS, tempfile.mkdtemp()))
        self.assertEqual(gw.restream_url("Yard"), "rtsp://127.0.0.1:8554/yard")

    def test_both_streams_ride_the_config(self):
        gw = Go2rtcGateway(CAMS, tempfile.mkdtemp())
        gw.api_port = gw.rtsp_port = gw.webrtc_port = 1
        streams = gw.build_config()["streams"]
        self.assertEqual(streams["front_door"], "rtsp://cam7/main")
        self.assertEqual(streams["front_door_sub"], "rtsp://cam7/sub")
        self.assertNotIn("demo_clip", streams)          # files stay local

    def test_a_detect_source_equal_to_the_main_is_ignored(self):
        """Misconfiguration, not a second stream — one decode, not two."""
        cams = [{"id": "C", "source": "rtsp://x/1", "detect_source": "rtsp://x/1"}]
        gw = Go2rtcGateway(cams, tempfile.mkdtemp())
        self.assertEqual(gw.detect_streams, {})

    def test_snapshot_answers_only_for_substream_cameras(self):
        """Everyone else's evidence is already full-resolution — a snapshot
        there is a wasted fetch on the alert path."""
        gw = _live(Go2rtcGateway(CAMS, tempfile.mkdtemp()))
        jpeg = b"\xff\xd8fakejpegbytes\xff\xd9"
        with mock.patch("cvti.serving.go2rtc.urllib.request.urlopen") as u:
            u.return_value.__enter__.return_value.read.return_value = jpeg
            self.assertEqual(gw.snapshot_jpeg("Front Door"), jpeg)
            self.assertIsNone(gw.snapshot_jpeg("Yard"))
        called_url = u.call_args[0][0]
        self.assertIn("frame.jpeg?src=front_door", called_url)   # the MAIN stream

    def test_snapshot_failure_is_none_never_a_raise(self):
        gw = _live(Go2rtcGateway(CAMS, tempfile.mkdtemp()))
        with mock.patch("cvti.serving.go2rtc.urllib.request.urlopen",
                        side_effect=OSError("down")):
            self.assertIsNone(gw.snapshot_jpeg("Front Door"))


# ---------------------------------------------------------------------------
# W1.6 — the sink attaches the full-resolution frame, best-effort
# ---------------------------------------------------------------------------

class FullFrameEvidenceTests(unittest.TestCase):
    def _sink_and_alert(self, tmp):
        import numpy as np
        from cvti.serving.alert_queue import QueuedAlert
        from cvti.serving.alert_sink import AlertSink
        from cvti.contracts import VerificationResult

        sink = AlertSink(tmp, notifier=mock.MagicMock())
        frame = np.zeros((36, 64, 3), dtype=np.uint8)
        alert = QueuedAlert(camera_id="Front Door", rule_name="test_rule",
                            priority="high", title="t", timestamp=time.time(),
                            payload={"candidate": None, "frames": [frame],
                                     "scene": {}, "enqueued_at": time.time()})
        result = VerificationResult(confirmed=True, confidence=0.9, reason="r",
                                    alert_priority="high", timestamp=time.time(),
                                    raw_response="x")
        return sink, alert, result

    def _evidence_dir(self, sink, alert, result) -> Path:
        sink.handle(alert, result)
        rows = sink._db.execute("SELECT evidence_dir FROM events").fetchall()
        self.assertEqual(len(rows), 1)
        return Path(rows[0][0])

    def test_the_full_frame_lands_beside_the_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            sink, alert, result = self._sink_and_alert(tmp)
            sink.full_frame_provider = lambda cam: b"\xff\xd8FULL\xff\xd9"
            ev = self._evidence_dir(sink, alert, result)
            self.assertEqual((ev / "evidence_full.jpg").read_bytes(),
                             b"\xff\xd8FULL\xff\xd9")

    def test_a_none_provider_answer_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            sink, alert, result = self._sink_and_alert(tmp)
            sink.full_frame_provider = lambda cam: None
            ev = self._evidence_dir(sink, alert, result)
            self.assertFalse((ev / "evidence_full.jpg").exists())

    def test_a_raising_provider_never_costs_the_alert(self):
        with tempfile.TemporaryDirectory() as tmp:
            sink, alert, result = self._sink_and_alert(tmp)
            def boom(cam):
                raise RuntimeError("gateway mid-crash")
            sink.full_frame_provider = boom
            ev = self._evidence_dir(sink, alert, result)   # alert persisted anyway
            self.assertTrue((ev / "event.json").exists())


# ---------------------------------------------------------------------------
# W1.3 — a dead gateway costs a reconnect, never coverage
# ---------------------------------------------------------------------------

class DecoderFallbackTests(unittest.TestCase):
    def _run_decoder(self, opens: list, **kw):
        """Drive StreamDecoder._loop against a scripted open_capture."""
        from cvti.serving import streams as st

        class DeadCap:
            def get(self, *_a): return 15.0
            def grab(self): return False
            def read(self): return False, None
            def release(self): pass
            def set(self, *_a): pass

        class LiveCap(DeadCap):
            def __init__(self): self.n = 0
            def grab(self): return True
            def read(self):
                import numpy as np
                self.n += 1
                return True, np.zeros((8, 8, 3), dtype=np.uint8)

        script = {"calls": []}
        def fake_open(source, **_kw):
            script["calls"].append(str(source))
            return LiveCap() if opens.pop(0) else DeadCap()

        d = st.StreamDecoder("cam", "rtsp://127.0.0.1:8554/cam",
                             target_fps=30.0, reconnect_backoff=0.01,
                             offline_grace_seconds=0.05, **kw)
        with mock.patch("cvti.serving.capture.open_capture", side_effect=fake_open):
            d.start()
            deadline = time.time() + 5.0
            while time.time() < deadline:
                if d.read_latest() is not None:     # a LiveCap frame arrived
                    break
                time.sleep(0.02)
            d.stop()
        return d, script["calls"]

    def test_two_failed_restream_opens_swap_to_the_cameras_own_url(self):
        d, calls = self._run_decoder(
            [False, False, False, True],           # restream dead, camera fine
            fallback_source="rtsp://cam7/main")
        self.assertTrue(d.fell_back)
        self.assertEqual(d.source, "rtsp://cam7/main")
        self.assertIn("rtsp://cam7/main", calls)
        self.assertTrue(d.link_status()["gateway_fallback"])

    def test_without_a_fallback_the_decoder_just_keeps_reconnecting(self):
        d, calls = self._run_decoder([False, False, False, True])
        self.assertFalse(d.fell_back)
        self.assertEqual({c for c in calls}, {"rtsp://127.0.0.1:8554/cam"})

    def test_the_swap_is_one_way(self):
        """Flapping between a half-alive gateway and the camera is worse than
        settling on the camera. Once direct, stay direct until restart."""
        d, calls = self._run_decoder(
            [False, False, True],
            fallback_source="rtsp://cam7/main")
        self.assertTrue(d.fell_back)
        self.assertNotIn("rtsp://127.0.0.1:8554/cam",
                         calls[calls.index("rtsp://cam7/main"):])


# ---------------------------------------------------------------------------
# W1.4 — the API flips on the descriptor file
# ---------------------------------------------------------------------------

class StreamDescriptorTests(unittest.TestCase):
    def _client(self, tmp):
        from fastapi.testclient import TestClient
        from cvti.api.app import create_app
        app = create_app(db_path=str(Path(tmp) / "events.db"),
                         site_path="configs/site_live.json")
        token, _ = app.state.tokens.mint("t", "owner")
        c = TestClient(app)
        c.headers["Authorization"] = f"Bearer {token}"
        return c

    def test_gateway_up_means_webrtc_with_the_mjpeg_fallback_riding_along(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "frames.json").write_text(
                json.dumps({"port": 7777, "token": "tk"}))
            (Path(tmp) / "stream_gateway.json").write_text(json.dumps(
                {"api_port": 1984, "rtsp_port": 8554, "webrtc_port": 8555,
                 "streams": {"Front Door": "front_door"},
                 "generated_at": time.time()}))
            got = self._client(tmp).get("/api/v1/cameras/Front Door/stream").json()
            self.assertEqual(got["kind"], "webrtc")
            self.assertEqual(got["url"],
                             "http://127.0.0.1:1984/api/webrtc?src=front_door")
            self.assertIn("ws://127.0.0.1:1984/api/ws?src=front_door", got["ws"])
            self.assertIn("token=tk", got["mjpeg_fallback"])

    def test_a_camera_the_gateway_does_not_carry_stays_mjpeg(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "frames.json").write_text(
                json.dumps({"port": 7777, "token": "tk"}))
            (Path(tmp) / "stream_gateway.json").write_text(json.dumps(
                {"api_port": 1984, "streams": {"Other": "other"},
                 "generated_at": time.time()}))
            got = self._client(tmp).get("/api/v1/cameras/Demo Clip/stream").json()
            self.assertEqual(got["kind"], "mjpeg")

    def test_no_descriptor_means_exactly_the_old_behaviour(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "frames.json").write_text(
                json.dumps({"port": 7777, "token": "tk"}))
            got = self._client(tmp).get("/api/v1/cameras/X/stream").json()
            self.assertEqual(got["kind"], "mjpeg")

    def test_nothing_at_all_is_still_a_503(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = self._client(tmp).get("/api/v1/cameras/X/stream")
            self.assertEqual(r.status_code, 503)

    def test_stopping_the_gateway_retracts_the_advertisement(self):
        with tempfile.TemporaryDirectory() as tmp:
            gw = _live(Go2rtcGateway(CAMS, tmp))
            self.assertIsNotNone(gw.write_descriptor())
            self.assertTrue((Path(tmp) / "stream_gateway.json").exists())
            gw._proc.poll.return_value = None
            gw.stop()
            self.assertFalse((Path(tmp) / "stream_gateway.json").exists())


# ---------------------------------------------------------------------------
# W1.5 — one upstream session per camera, pinned at the wiring
# ---------------------------------------------------------------------------

class OneSessionPerCameraPins(unittest.TestCase):
    """These became true on 4 Sep (the pilot's Tapo allows two sessions) and
    almost lost their reason to exist when W1's charter was written assuming
    they were still false. Pinned so they cannot quietly become false again."""

    def test_every_secondary_consumer_reads_the_engines_frames(self):
        src = (ROOT / "cvti" / "serving" / "pipeline.py").read_text()
        self.assertIn("mapping_service.frame_source = _mapper_frame", src)
        self.assertIn("frame_source=_scanner_frame", src)
        self.assertIn("frame_source=_latest_frame", src)

    def test_ingest_prefers_the_gateway_and_keeps_the_cameras_url_in_hand(self):
        src = (ROOT / "cvti" / "serving" / "pipeline.py").read_text()
        self.assertIn("gateway.restream_url(", src)
        self.assertIn("fallback_sources=_gw_fallbacks", src)
        self.assertIn("gateway.write_descriptor()", src)
        self.assertIn("gateway.stop()", src)

    def test_health_carries_the_gateway_row(self):
        src = (ROOT / "cvti" / "serving" / "pipeline.py").read_text()
        self.assertIn('"stream_gateway": gateway.status()', src)

    def test_the_console_wall_prefers_engine_frames(self):
        src = (ROOT / "cvti" / "app" / "console_backend.py").read_text()
        self.assertIn("Prefer the engine's already-decoded frames", src)


if __name__ == "__main__":
    unittest.main()
