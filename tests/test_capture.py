"""One capture path, every platform (27 Aug).

'It has to be good for Windows and all, not just Apple.' Three call sites each
opened captures their own way and inherited OpenCV's per-platform defaults:
deeper buffering on Windows (latency), the MSMF webcam backend (slow to open,
poor frame rates), and no low-latency flags for network streams.
"""
import sys
import unittest
from unittest import mock

from cvti.serving import capture


class SourceKindTest(unittest.TestCase):
    def test_it_tells_live_streams_from_files_and_webcams(self):
        self.assertTrue(capture.is_live_source("rtsp://cam/1"))
        self.assertTrue(capture.is_live_source("https://x/y.m3u8"))
        self.assertFalse(capture.is_live_source("0"))
        self.assertFalse(capture.is_live_source("/clips/a.mp4"))


class BackendSelectionTest(unittest.TestCase):
    """The platform differences that made the same code feel worse on Windows."""

    def _open(self, source, platform):
        cap = mock.MagicMock()
        cap.isOpened.return_value = True
        fake_cv2 = mock.MagicMock()
        fake_cv2.VideoCapture.return_value = cap
        fake_cv2.CAP_DSHOW = 700
        fake_cv2.CAP_FFMPEG = 1900
        fake_cv2.CAP_PROP_BUFFERSIZE = 38
        with mock.patch.dict(sys.modules, {"cv2": fake_cv2}), \
             mock.patch.object(capture.sys, "platform", platform):
            capture.open_capture(source)
        return fake_cv2, cap

    def test_windows_webcams_use_directshow_not_msmf(self):
        cv2, _ = self._open("0", "win32")
        cv2.VideoCapture.assert_called_with(0, cv2.CAP_DSHOW)

    def test_other_platforms_leave_webcam_backend_to_opencv(self):
        cv2, _ = self._open("0", "darwin")
        cv2.VideoCapture.assert_called_with(0)

    def test_network_streams_always_use_ffmpeg(self):
        # Since 3 Sep the open also requests GPU decode (ANY negotiates and
        # falls back to software by itself); since 7 Sep it also pins the
        # software decoder to ONE thread — see the single-thread tests below.
        for platform in ("win32", "darwin", "linux"):
            cv2, _ = self._open("rtsp://cam/1", platform)
            cv2.VideoCapture.assert_called_with(
                "rtsp://cam/1", cv2.CAP_FFMPEG,
                [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY,
                 cv2.CAP_PROP_N_THREADS, 1])

    def test_live_decode_is_single_threaded(self):
        """OpenCV defaults the decoder to thread_count = CPU count, and H.264
        frame-threading delays output ~one frame per thread: 810ms standing
        latency on an 11-core dev Mac at 15fps, ~250ms on the pilot's 4-core
        box — on every frame, alert evidence included. Found 7 Sep by the
        latency baseline's first run; threads=1 measured 810 -> 145ms p50."""
        cv2, _ = self._open("rtsp://cam/1", "win32")
        args = cv2.VideoCapture.call_args[0]
        params = args[2]
        idx = params.index(cv2.CAP_PROP_N_THREADS)
        self.assertEqual(params[idx + 1], 1)

    def test_files_keep_default_decode_threading(self):
        """Offline work (eval over clips) wants throughput, and nobody is
        waiting at the glass — the eval harness must not get slower."""
        cv2, _ = self._open("/clips/a.mp4", "darwin")
        for call in cv2.VideoCapture.call_args_list:
            for arg in call[0]:
                if isinstance(arg, list):
                    self.assertNotIn(cv2.CAP_PROP_N_THREADS, arg)

    def test_old_opencv_without_the_knob_still_opens(self):
        """The single-thread ask is an optimisation, never a requirement."""
        cap = mock.MagicMock()
        cap.isOpened.return_value = True
        fake_cv2 = mock.MagicMock()
        fake_cv2.VideoCapture.return_value = cap
        fake_cv2.CAP_FFMPEG = 1900
        fake_cv2.CAP_PROP_BUFFERSIZE = 38
        del fake_cv2.CAP_PROP_N_THREADS          # an OpenCV from before the knob
        with mock.patch.dict(sys.modules, {"cv2": fake_cv2}):
            capture.open_capture("rtsp://cam/1")
        fake_cv2.VideoCapture.assert_called_with(
            "rtsp://cam/1", fake_cv2.CAP_FFMPEG,
            [fake_cv2.CAP_PROP_HW_ACCELERATION, fake_cv2.VIDEO_ACCELERATION_ANY])

    def test_live_sources_request_a_one_frame_buffer(self):
        # The queue IS the latency: deeper on Windows, which is why the same
        # code felt laggier there.
        for source in ("0", "rtsp://cam/1"):
            cv2, cap = self._open(source, "win32")
            cap.set.assert_any_call(cv2.CAP_PROP_BUFFERSIZE, 1)

    def test_files_are_not_starved_of_buffer(self):
        cv2, cap = self._open("/clips/a.mp4", "darwin")
        self.assertFalse(cap.set.called, "a file should keep normal buffering")

    def test_low_latency_flags_are_set_for_live_streams(self):
        import os
        os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
        self._open("rtsp://cam/1", "linux")
        opts = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS", "")
        for flag in ("rtsp_transport;tcp", "nobuffer", "low_delay"):
            self.assertIn(flag, opts)

    def test_an_operator_env_var_always_wins(self):
        import os
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
        try:
            self._open("rtsp://cam/1", "linux")
            self.assertEqual(os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"],
                             "rtsp_transport;udp")
        finally:
            os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)


class EveryCallSiteUsesItTest(unittest.TestCase):
    def test_no_module_opens_its_own_capture_any_more(self):
        import inspect
        from cvti.app import live_wall
        from cvti.serving import custom_rules, streams
        for mod in (streams, custom_rules, live_wall):
            src = inspect.getsource(mod)
            body = src.split("def _open")[1].split("\n    def ")[0]
            self.assertIn("open_capture", body,
                          f"{mod.__name__} still opens captures its own way")


if __name__ == "__main__":
    unittest.main()
