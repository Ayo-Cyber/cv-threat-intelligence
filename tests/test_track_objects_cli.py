from __future__ import annotations

import json

import numpy as np
import pytest

from cvti.cli.track_objects import RuntimeDependencies, main


class FakeCapture:
    def __init__(self, frames, fps=10.0, opened=True, positions_ms=None):
        self.frames = list(frames)
        self.fps = fps
        self.opened = opened
        self.released = False
        self.read_count = 0
        self.positions_ms = positions_ms

    def isOpened(self):
        return self.opened

    def get(self, property_id):
        if property_id == FakeCV2.CAP_PROP_FPS:
            return self.fps
        if self.positions_ms is not None and self.read_count:
            return self.positions_ms[self.read_count - 1]
        return max(0, self.read_count - 1) * 1000.0 / self.fps

    def read(self):
        if not self.frames:
            return False, None
        self.read_count += 1
        return True, self.frames.pop(0)

    def release(self):
        self.released = True


class FakeWriter:
    def __init__(self, opened=True):
        self.opened = opened
        self.frames = []
        self.released = False
        self.args: tuple = ()

    def isOpened(self):
        return self.opened

    def write(self, frame):
        self.frames.append(frame)

    def release(self):
        self.released = True


class FakeCV2:
    CAP_PROP_FPS = 5
    CAP_PROP_POS_MSEC = 6
    FONT_HERSHEY_SIMPLEX = 0

    def __init__(self, capture, writer=None):
        self.capture = capture
        self.writer = writer or FakeWriter()
        self.opened_source = None
        self.destroyed = False

    def VideoCapture(self, source):
        self.opened_source = source
        return self.capture

    def VideoWriter(self, *_args):
        self.writer.args = _args
        return self.writer

    @staticmethod
    def VideoWriter_fourcc(*_args):
        return 1

    @staticmethod
    def rectangle(*_args):
        return None

    @staticmethod
    def putText(*_args):
        return None

    @staticmethod
    def imshow(*_args):
        return None

    @staticmethod
    def waitKey(_delay):
        return -1

    def destroyAllWindows(self):
        self.destroyed = True


class FakeModel:
    names = {0: "person", 39: "bottle"}

    def __init__(self, fail_at=None, failure=None):
        self.calls = []
        self.fail_at = fail_at
        self.failure = failure or RuntimeError("inference failed")

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_at == len(self.calls):
            raise self.failure
        return [object()]


class FakeTracker:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.timestamps = []
        self.__class__.instances.append(self)

    def update(self, _detections, timestamp):
        self.timestamps.append(timestamp)

    def snapshot(self, timestamp):
        return {
            "schema_version": 1,
            "camera_id": self.kwargs["camera_id"],
            "session_id": "fake",
            "timestamp": timestamp,
            "status": "ok",
            "tracks": [],
            "counters": {},
        }

    def overlays(self, _timestamp):
        return []


def _clock():
    values = iter(index / 1000 for index in range(1000))
    return lambda: next(values)


def _deps(capture, model=None, writer=None, tracker_factory=FakeTracker):
    cv2 = FakeCV2(capture, writer)
    model = model or FakeModel()
    deps = RuntimeDependencies(cv2, lambda _path: model, lambda _result: object(), tracker_factory, _clock())
    return deps, cv2, model


def _argv(tmp_path, *, source=None, output=None, extra=()):
    weights = tmp_path / "local.pt"
    weights.write_bytes(b"local weights")
    source = source or tmp_path / "clip.mp4"
    if not isinstance(source, str):
        source.write_bytes(b"video")
    output = output or tmp_path / "run"
    return [
        "--source", str(source), "--weights", str(weights), "--device", "cpu",
        "--output-dir", str(output), *extra,
    ]


def test_samples_once_per_selected_frame_and_writes_explicit_source_end(tmp_path):
    frames = [np.zeros((8, 12, 3), dtype=np.uint8) for _ in range(7)]
    deps, _cv2, model = _deps(FakeCapture(frames, fps=10.0))
    output = tmp_path / "run"

    assert main(_argv(tmp_path, output=output, extra=("--every-n-frames", "2", "--max-frames", "3")), dependencies=deps) == 0

    records = [json.loads(line) for line in (output / "snapshots.jsonl").read_text().splitlines()]
    assert [item["timestamp"] for item in records[:-1]] == [0.0, 0.2, 0.4]
    assert [item["source_frame_index"] for item in records[:-1]] == [0, 2, 4]
    assert records[-1]["event"] == "source_end"
    assert records[-1]["source_end"] == {"outcome": "complete", "reason": "max_frames"}
    assert len(model.calls) == 3
    assert all(call["device"] == "cpu" and call["verbose"] is False for call in model.calls)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["counts"] == {
        "processed_frames": 3, "skipped_frames": 2,
        "source_frames_read": 5, "tracks_created": 0,
    }
    assert summary["video"]["timestamp_basis"] == "media_pos_msec_with_frame_index_fallback"


def test_webcam_uses_monotonic_elapsed_without_touching_a_real_camera(tmp_path):
    deps, cv2, _model = _deps(FakeCapture([np.zeros((2, 3, 3), dtype=np.uint8)]))
    assert main(_argv(tmp_path, source="0", extra=("--max-frames", "1")), dependencies=deps) == 0
    assert cv2.opened_source == 0
    assert FakeTracker.instances[-1].kwargs["camera_id"] == "webcam"
    summary = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert summary["video"]["timestamp_basis"] == "monotonic_elapsed"


@pytest.mark.parametrize(
    "change, message",
    [
        ({"weights": "missing.pt"}, "automatic downloads are disabled"),
        ({"source": "missing.mp4"}, "existing video file"),
        ({"extra": ("--max-frames", "0")}, "positive integer"),
        ({"extra": ("--every-n-frames", "0")}, "positive integer"),
    ],
)
def test_invalid_files_and_bounds_fail_before_loading_runtime(tmp_path, capsys, change, message):
    argv = _argv(tmp_path, extra=change.get("extra", ()))
    if "weights" in change:
        argv[argv.index("--weights") + 1] = change["weights"]
    if "source" in change:
        argv[argv.index("--source") + 1] = change["source"]
    with pytest.raises(SystemExit) as caught:
        main(argv)
    assert caught.value.code == 2
    assert message in capsys.readouterr().err


def test_existing_output_is_never_overwritten(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep")
    with pytest.raises(SystemExit):
        main(_argv(tmp_path, output=output))
    assert marker.read_text() == "keep"


def test_failure_releases_resources_and_preserves_partial_evidence(tmp_path):
    capture = FakeCapture([np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(2)])
    writer = FakeWriter()
    deps, _cv2, _model = _deps(capture, model=FakeModel(fail_at=2), writer=writer)
    output = tmp_path / "run"
    assert main(_argv(tmp_path, output=output, extra=("--save-video",)), dependencies=deps) == 2
    assert capture.released and writer.released
    assert (output / "snapshots.jsonl").exists()
    summary = json.loads((output / "summary.json").read_text())
    assert summary["outcome"] == "error"
    assert summary["source_end"] == "error"
    assert summary["counts"]["processed_frames"] == 1
    assert summary["artifacts"]["annotated_video"] == "annotated.mp4"
    records = [json.loads(line) for line in (output / "snapshots.jsonl").read_text().splitlines()]
    assert records[-1]["event"] == "source_end"


def test_save_video_writes_sampled_annotated_frames_and_releases(tmp_path):
    capture = FakeCapture([np.zeros((4, 5, 3), dtype=np.uint8)])
    writer = FakeWriter()
    deps, _cv2, _model = _deps(capture, writer=writer)
    assert main(_argv(tmp_path, extra=("--save-video",)), dependencies=deps) == 0
    assert len(writer.frames) == 1
    assert writer.released and capture.released
    summary = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert summary["artifacts"]["annotated_video"] == "annotated.mp4"


def test_opened_file_returning_zero_frames_is_an_error_with_summary(tmp_path, capsys):
    capture = FakeCapture([])
    deps, _cv2, _model = _deps(capture)
    output = tmp_path / "run"
    assert main(_argv(tmp_path, output=output), dependencies=deps) == 2
    assert "returned zero frames" in capsys.readouterr().err
    summary = json.loads((output / "summary.json").read_text())
    assert summary["outcome"] == "error"
    assert summary["source_end"] == "capture_failure"
    assert summary["artifacts"]["annotated_video"] is None
    assert capture.released


def test_webcam_read_failure_is_not_reported_as_eof(tmp_path, capsys):
    capture = FakeCapture([])
    deps, _cv2, _model = _deps(capture)
    assert main(_argv(tmp_path, source="0"), dependencies=deps) == 2
    assert "webcam capture failed" in capsys.readouterr().err
    summary = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert summary["source_end"] == "capture_failure"


def test_interrupt_preserves_partial_video_jsonl_and_interrupted_summary(tmp_path, capsys):
    capture = FakeCapture([np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(2)])
    writer = FakeWriter()
    model = FakeModel(fail_at=2, failure=KeyboardInterrupt())
    deps, _cv2, _model = _deps(capture, model=model, writer=writer)
    assert main(_argv(tmp_path, extra=("--save-video",)), dependencies=deps) == 130
    assert "partial artifacts were preserved" in capsys.readouterr().err
    summary = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert summary["outcome"] == "interrupted"
    assert summary["source_end"] == "interrupted"
    assert summary["artifacts"]["annotated_video"] == "annotated.mp4"
    assert writer.released and capture.released


def test_nonmonotonic_media_timestamps_fall_back_to_frame_index(tmp_path):
    frames = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(3)]
    capture = FakeCapture(frames, fps=10.0, positions_ms=[50.0, 40.0, float("nan")])
    deps, _cv2, _model = _deps(capture)
    assert main(_argv(tmp_path), dependencies=deps) == 0
    records = [json.loads(line) for line in (tmp_path / "run" / "snapshots.jsonl").read_text().splitlines()]
    assert [record["timestamp"] for record in records[:-1]] == pytest.approx([0.05, 0.1, 0.2])
    summary = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert summary["video"]["media_timestamp_frames"] == 1
    assert summary["video"]["frame_index_fallback_frames"] == 2


def test_timing_and_track_bookkeeping_are_bounded(tmp_path, monkeypatch):
    import cvti.cli.track_objects as subject

    monkeypatch.setattr(subject, "TIMING_WINDOW_SIZE", 3)

    class SequentialTracker(FakeTracker):
        def snapshot(self, timestamp):
            result = super().snapshot(timestamp)
            counter = len(self.timestamps)
            result["tracks"] = [{
                "id": f"{self.kwargs['camera_id']}/{self.kwargs['session_id']}/{counter}"
            }] if counter else []
            return result

    frames = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(10)]
    deps, _cv2, _model = _deps(FakeCapture(frames), tracker_factory=SequentialTracker)
    assert main(_argv(tmp_path), dependencies=deps) == 0
    summary = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert summary["counts"]["tracks_created"] == 10
    assert summary["timing_ms"]["inference"]["sample_count"] == 3
    assert summary["timing_ms"]["inference"]["observations_total"] == 10
    assert summary["timing_ms"]["inference"]["window"] == "most_recent"


def test_writer_uses_sampled_constant_rate_and_duration(tmp_path):
    frames = [np.zeros((4, 5, 3), dtype=np.uint8) for _ in range(5)]
    writer = FakeWriter()
    deps, _cv2, _model = _deps(FakeCapture(frames, fps=10.0), writer=writer)
    assert main(_argv(tmp_path, extra=("--every-n-frames", "2", "--save-video")), dependencies=deps) == 0
    assert writer.args[2] == 5.0
    assert len(writer.frames) == 3
    assert len(writer.frames) / writer.args[2] == pytest.approx(0.6)
