from __future__ import annotations

import io
import json
import math
from types import SimpleNamespace

from PIL import Image

import cvti.cli.watch_objects as watch_cli
from cvti.cli.watch_objects import RunDependencies, main
from cvti.object_watch.matcher import ObjectCandidate
from cvti.object_watch.runtime_config import ObjectWatchConfig
from cvti.object_watch.store import load_targets


def _png(path, value=40):
    data = io.BytesIO()
    Image.new("RGB", (10, 8), (value, value, value)).save(data, "PNG")
    path.write_bytes(data.getvalue())
    return path


def test_help_lists_recognition_workflow(capsys):
    try:
        main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    text = capsys.readouterr().out
    assert "configure" in text and "eval-crops" in text
    assert "semantic object-recognition" in text


def test_doctor_missing_model_is_truthful_json(tmp_path, capsys):
    assert main(["doctor", "--site-dir", str(tmp_path)]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["ready"] is False
    assert result["status"] == "unavailable"
    assert result["model_fingerprint"] is None
    assert result["downloads"] is False


def test_enroll_requires_explicit_review_and_keeps_negative_separate(tmp_path, capsys):
    positive = _png(tmp_path / "positive.png")
    negative = _png(tmp_path / "lookalike.png", 80)
    base = ["enroll", "--site-dir", str(tmp_path), "--object-id", "carton-a",
            "--label", "Carton A", "--image", str(positive),
            "--negative-image", str(negative)]
    assert main(base) == 0
    target = load_targets(tmp_path)[0]
    assert not target.examples[0].reviewed
    assert not target.negative_examples[0].reviewed
    result = json.loads(capsys.readouterr().out)
    assert result["review_explicit"] is False


def test_enroll_review_and_explicit_bbox_use_canonical_crop(tmp_path):
    image = _png(tmp_path / "photo.png")
    assert main(["enroll", "--site-dir", str(tmp_path), "--object-id", "box",
                 "--label", "Box", "--image", str(image), "--bbox", "1", "2", "7", "6",
                 "--review"]) == 0
    example = load_targets(tmp_path)[0].examples[0]
    assert example.reviewed is True
    assert example.bbox == (1, 2, 7, 6)
    assert example.bbox_format == "pixel_xyxy"


def test_configure_rejects_remote_model_without_creating_library(tmp_path, capsys):
    assert main(["configure", "--site-dir", str(tmp_path),
                 "--model-path", "https://example.invalid/model", "--device", "cpu"]) == 2
    assert "local filesystem" in capsys.readouterr().err
    assert not (tmp_path / "object_library").exists()


def test_configure_persists_local_cpu_first_configuration(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    assert main(["configure", "--site-dir", str(tmp_path), "--model-path", str(model),
                 "--device", "cpu"]) == 0
    config = json.loads((tmp_path / "object_library" / "runtime.json").read_text(encoding="utf-8"))
    assert config["backend"] == "siglip"
    assert config["device"] == "cpu"
    assert config["model_path"] == str(model.resolve())


class _Frame:
    shape = (24, 32, 3)

    def copy(self):
        return self


class _Capture:
    def __init__(self, frames):
        self.frames = list(frames)

    def isOpened(self): return True
    def get(self, _key): return 10.0
    def read(self): return (True, self.frames.pop(0)) if self.frames else (False, None)
    def release(self): pass


class _Cv2:
    CAP_PROP_FPS = 5

    def __init__(self, frames):
        self.frames = frames

    def VideoCapture(self, _source):
        return _Capture(self.frames)


class _Matcher:
    seen = []

    def __init__(self, _site, _backend, **_kwargs):
        target = SimpleNamespace(grounding_description="red branded carton")
        self.index = SimpleNamespace(targets=(SimpleNamespace(target=target),))
        self.last_decisions = ()

    def match(self, _camera, _frame, candidates, _timestamp):
        self.seen.append(candidates)
        candidate = candidates[0]
        assert all(math.isfinite(value) for value in candidate.bbox)
        self.last_decisions = (SimpleNamespace(
            status="matched", reason="positive_margin", object_id="carton-a",
            best_similarity=.9, runner_up_similarity=None, candidate=candidate,
        ),)
        return []


def _run_dependencies(frames, factory):
    return RunDependencies(
        _Cv2(frames), backend_loader=lambda _config: SimpleNamespace(fingerprint="siglip-test"),
        monotonic=lambda: 1.0, proposal_provider_factory=factory,
    )


def test_yolo_world_proposals_only_uses_active_descriptions_and_loads_once(
    tmp_path, monkeypatch, capsys,
):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"local")
    world = tmp_path / "world.pt"; world.write_bytes(b"local")
    clip = tmp_path / "clip.pt"; clip.write_bytes(b"local")
    config = ObjectWatchConfig(model_path=tmp_path, world_weights=world,
                               clip_weights=clip, max_candidates=1)
    monkeypatch.setattr(watch_cli, "_config_and_backend",
                        lambda *_args: (config, SimpleNamespace(fingerprint="siglip-test")))
    monkeypatch.setattr(watch_cli, "ObjectMatcher", _Matcher)
    _Matcher.seen = []
    constructed = []
    calls = []

    class Provider:
        def propose(self, frame, phrases, *, limit_candidates):
            calls.append((frame, phrases, limit_candidates))
            return (ObjectCandidate((1, 2, 20, 22), "red branded carton", .8),)

    def factory(_config):
        constructed.append(True)
        return Provider()

    output = tmp_path / "run"
    assert main([
        "run", "--site-dir", str(tmp_path), "--source", str(source),
        "--output-dir", str(output), "--proposals", "yolo_world", "--max-frames", "2",
    ], dependencies=_run_dependencies([_Frame(), _Frame()], factory)) == 0

    result = json.loads(capsys.readouterr().out)
    decisions = [json.loads(line) for line in (output / "decisions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert result["proposal_mode"] == "yolo_world"
    assert [row["object_id"] for row in decisions] == ["carton-a", "carton-a"]
    assert constructed == [True]
    assert [item[1] for item in calls] == [("red branded carton",)] * 2
    assert all(item[2] == 1 for item in calls)
    assert all(len(rows) == 1 for rows in _Matcher.seen)


def test_yolo_world_missing_config_fails_before_output(tmp_path, monkeypatch, capsys):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"local")
    config = ObjectWatchConfig(model_path=tmp_path)
    monkeypatch.setattr(watch_cli, "_config_and_backend",
                        lambda *_args: (config, SimpleNamespace(fingerprint="siglip-test")))
    monkeypatch.setattr(watch_cli, "ObjectMatcher", _Matcher)
    built = []
    output = tmp_path / "run"

    assert main([
        "run", "--site-dir", str(tmp_path), "--source", str(source),
        "--output-dir", str(output), "--proposals", "yolo_world",
    ], dependencies=_run_dependencies([], lambda _config: built.append(True))) == 2

    assert "configured local YOLO-World and CLIP weights are required" in capsys.readouterr().err
    assert not output.exists()
    assert built == []


def test_untracked_matches_do_not_create_cli_presence(tmp_path, monkeypatch, capsys):
    source = tmp_path / "source.mp4"; source.write_bytes(b"local")
    world = tmp_path / "world.pt"; world.write_bytes(b"local")
    clip = tmp_path / "clip.pt"; clip.write_bytes(b"local")
    config = ObjectWatchConfig(model_path=tmp_path, world_weights=world,
                               clip_weights=clip, max_candidates=1)
    monkeypatch.setattr(watch_cli, "_config_and_backend",
                        lambda *_args: (config, SimpleNamespace(fingerprint="siglip-test")))
    monkeypatch.setattr(watch_cli, "load_targets", lambda *_args: [
        SimpleNamespace(id="carton-a", examples=[]),
    ])

    class Matcher(_Matcher):
        def match(self, _camera, _frame, candidates, timestamp):
            candidate = candidates[0]
            self.last_decisions = (SimpleNamespace(
                status="matched", reason="positive_margin", object_id="carton-a",
                best_similarity=.9, runner_up_similarity=None, candidate=candidate,
            ),)
            return [SimpleNamespace(
                object_id="carton-a", track_id=candidate.track_id,
                bbox=candidate.bbox, similarity=.9, reference_example_ids=(),
            )]

        @staticmethod
        def _crop_png(_frame, bbox):
            return None, bbox

    monkeypatch.setattr(watch_cli, "ObjectMatcher", Matcher)
    confidence = iter((.9, .9, .01, .01, .9, .9))

    class Provider:
        def propose(self, *_args, **_kwargs):
            return (ObjectCandidate((1, 2, 20, 22), "carton", next(confidence)),)

    output = tmp_path / "run"
    assert main([
        "run", "--site-dir", str(tmp_path), "--source", str(source),
        "--output-dir", str(output), "--proposals", "yolo_world", "--max-frames", "6",
    ], dependencies=_run_dependencies([_Frame()] * 6, lambda _config: Provider())) == 0

    summary = json.loads(capsys.readouterr().out)
    rows = [json.loads(line) for line in (output / "decisions.jsonl").read_text(encoding="utf-8").splitlines()]
    decisions = [row for row in rows if row["status"] == "matched"]
    presence = [row for row in rows if row["status"] == "presence_candidate"]
    assert [row["track_id"] is None for row in decisions] == [False, False, True, True, False, False]
    assert summary["counts"]["presence_candidates"] == 1
    assert len(presence) == 1 and presence[0]["track_id"] is not None
