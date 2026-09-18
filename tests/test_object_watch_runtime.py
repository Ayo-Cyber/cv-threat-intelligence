import time
import threading

import numpy as np
from PIL import Image
import io

from cvti.object_watch.runtime import ObjectWatchRuntime, WatchSample
from cvti.object_watch.runtime_config import ObjectWatchConfig


def sample(sequence: int, frame) -> WatchSample:
    return WatchSample("cam", 1, sequence, time.monotonic(), float(sequence), frame)


def test_submit_owns_frame_and_unavailable_worker_does_not_block(tmp_path) -> None:
    runtime = ObjectWatchRuntime(
        ObjectWatchConfig(library_path=tmp_path / "object_library"),
        backend_factory=lambda _config: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    runtime.start()
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    item = sample(1, frame)
    frame[:] = 255
    started = time.monotonic()
    assert runtime.submit(item)
    assert time.monotonic() - started < 0.1
    deadline = time.monotonic() + 1
    rows = []
    while not rows and time.monotonic() < deadline:
        rows = runtime.drain("cam", 1, 1.0)
        time.sleep(0.01)
    runtime.stop()
    assert not item.frame.any()
    assert rows[0].status == "unavailable"


def test_latest_pending_sample_replaces_older(tmp_path) -> None:
    runtime = ObjectWatchRuntime(
        ObjectWatchConfig(library_path=tmp_path / "object_library"),
        backend_factory=lambda _config: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    runtime.start()
    assert runtime.submit(sample(1, frame))
    assert runtime.submit(sample(2, frame))
    assert not runtime.submit(sample(2, frame))
    runtime.reset_camera("cam", 2)
    assert not runtime.submit(WatchSample("cam", 1, 3, 3.0, 3.0, frame))
    assert runtime.submit(WatchSample("cam", 2, 1, 3.0, 3.0, frame))
    runtime.stop()


def test_slow_backend_initialization_does_not_block_start_or_submit(tmp_path) -> None:
    release = threading.Event()

    def slow_backend(_config):
        release.wait(1.0)
        raise RuntimeError("offline")

    runtime = ObjectWatchRuntime(
        ObjectWatchConfig(library_path=tmp_path / "object_library"),
        backend_factory=slow_backend,
    )
    started = time.monotonic()
    runtime.start()
    assert time.monotonic() - started < 0.1
    assert runtime.submit(sample(1, np.zeros((4, 4, 3), dtype=np.uint8)))
    assert time.monotonic() - started < 0.1
    release.set()
    runtime.stop()


def test_drain_uses_worker_snapshot_without_filesystem_reads(tmp_path, monkeypatch) -> None:
    runtime = ObjectWatchRuntime(
        ObjectWatchConfig(library_path=tmp_path / "object_library"),
        backend_factory=lambda _config: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    runtime.start()
    assert runtime.submit(sample(1, np.zeros((4, 4, 3), dtype=np.uint8)))
    deadline = time.monotonic() + 1
    while runtime.status()["processed"] == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    monkeypatch.setattr("cvti.object_watch.runtime.library_revision",
                        lambda *_args: (_ for _ in ()).throw(AssertionError("filesystem read")))
    rows = runtime.drain("cam", 1, 1.0)
    runtime.stop()
    assert rows and rows[0].status == "unavailable"


def test_siglip_backend_owns_lock_without_runtime_nesting(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace
    from cvti.object_watch.embeddings import SiglipEmbeddingBackend
    from cvti.object_watch.matcher import ObjectCandidate

    class Value:
        def to(self, _device):
            return self

    active = 0
    peak = 0
    guard = threading.Lock()

    class Features:
        def __getitem__(self, _index):
            return self
        def float(self):
            return self
        def cpu(self):
            return self
        def tolist(self):
            return [1.0, 0.0]

    class Model:
        def get_image_features(self, **_inputs):
            nonlocal active, peak
            with guard:
                active += 1
                peak = max(peak, active)
            time.sleep(0.03)
            with guard:
                active -= 1
            return Features()

    class NoGrad:
        def __enter__(self): return self
        def __exit__(self, *_args): return None

    backend = SiglipEmbeddingBackend.__new__(SiglipEmbeddingBackend)
    backend._image_cls = Image
    backend._processor = lambda **_kwargs: {"pixels": Value()}
    backend._model = Model()
    backend._torch = SimpleNamespace(no_grad=lambda: NoGrad())
    backend._lock = threading.Lock()
    backend.device = "cpu"
    backend.model_name = str(tmp_path / "model")
    backend.fingerprint = "siglip-test"
    backend.dimensions = 2
    backend.inference_lock_root = tmp_path / "object_library"

    encoded = io.BytesIO()
    Image.new("RGB", (4, 4)).save(encoded, "PNG")
    threads = [threading.Thread(target=backend.embed_image, args=(encoded.getvalue(),))
               for _ in range(2)]
    for thread in threads: thread.start()
    for thread in threads: thread.join(1)
    assert all(not thread.is_alive() for thread in threads)
    assert peak == 1

    index = SimpleNamespace(library_revision=1, model_fingerprint=backend.fingerprint,
                            targets=())
    monkeypatch.setattr("cvti.object_watch.runtime.build_recognition_index",
                        lambda *_args: index)

    class Matcher:
        def match(self, *_args):
            backend.embed_image(encoded.getvalue())
            return []

    monkeypatch.setattr("cvti.object_watch.runtime.ObjectMatcher.from_index",
                        lambda *_args, **_kwargs: Matcher())
    runtime = ObjectWatchRuntime(
        ObjectWatchConfig(library_path=tmp_path / "object_library"),
        backend_factory=lambda _config: backend,
    )
    runtime.start()
    now = time.monotonic()
    assert runtime.submit(WatchSample("cam", 0, 1, now, now,
                                     np.zeros((4, 4, 3), np.uint8),
                                     (ObjectCandidate((0, 0, 4, 4), track_id=1),)))
    deadline = time.monotonic() + 1
    while runtime.status()["processed"] == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    runtime.stop()
    assert runtime.status()["processed"] == 1


def test_result_queue_keeps_complete_latest_sample_group(tmp_path) -> None:
    from cvti.object_watch.matcher import ObjectMatch
    from cvti.object_watch.runtime import WatchResult

    runtime = ObjectWatchRuntime(ObjectWatchConfig(library_path=tmp_path / "object_library"),
                                 backend_factory=lambda _config: object())
    runtime.reset_camera("cam", 0)
    runtime._snapshot_signature = ()
    runtime._snapshot_revision = 1
    runtime._snapshot_active = {str(i): 1 for i in range(6)}
    now = time.monotonic()
    rows = []
    for i in range(6):
        match = ObjectMatch("cam", str(i), str(i), "item", (0, 0, 2, 2), .9,
                            now, track_id=i, target_revision=1, library_revision=1)
        rows.append(WatchResult("cam", 0, 1, now, now, 1, 1, "", (match,),
                                (np.zeros((2, 2, 3)), np.zeros((2, 4, 3))), "matched"))
    runtime._admit_results(rows)
    assert [row.matches[0].object_id for row in runtime.drain("cam", 0)] == [
        str(i) for i in range(6)
    ]


def test_yolo_world_provider_cached_until_local_assets_change(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace

    from cvti.object_watch.runtime import WatchSample

    library = tmp_path / "object_library"
    world = tmp_path / "world.pt"
    clip = tmp_path / "clip.pt"
    world.write_bytes(b"world-1")
    clip.write_bytes(b"clip-1")
    config = ObjectWatchConfig(
        library_path=library, proposal_provider="yolo_world",
        world_weights=world, clip_weights=clip,
    )
    backend = SimpleNamespace(fingerprint="model", preprocessing_version=1, dimensions=2)
    index = SimpleNamespace(library_revision=1, model_fingerprint="model", targets=())
    monkeypatch.setattr("cvti.object_watch.runtime.build_recognition_index", lambda *_: index)
    monkeypatch.setattr("cvti.object_watch.runtime.ObjectMatcher.from_index",
                        lambda *_a, **_k: SimpleNamespace(match=lambda *_: []))
    constructions = []

    class Provider:
        def __init__(self, *_args, **_kwargs):
            constructions.append(self)
        def propose(self, *_args):
            return ()

    monkeypatch.setattr("cvti.object_watch.runtime.YoloWorldProposalProvider", Provider)
    runtime = ObjectWatchRuntime(config, backend_factory=lambda _config: backend)
    runtime.reset_camera("cam", 0)
    signature = runtime._worker_config_signature()
    loaded_backend, proposal, signature = runtime._reload_runtime(signature)
    frame = np.zeros((4, 4, 3), np.uint8)
    for sequence in (1, 2):
        now = time.monotonic()
        runtime._process(WatchSample("cam", 0, sequence, now, now, frame,
                                     proposal_provider="yolo_world"),
                         loaded_backend, proposal)
    assert len(constructions) == 1

    world.write_bytes(b"world-asset-changed")
    changed = runtime._worker_config_signature()
    loaded_backend, proposal, changed = runtime._reload_runtime(changed)
    now = time.monotonic()
    runtime._process(WatchSample("cam", 0, 3, now, now, frame,
                                  proposal_provider="yolo_world"),
                      loaded_backend, proposal)
    assert len(constructions) == 2

    now = time.monotonic()
    runtime._process(WatchSample("cam", 0, 4, now, now, frame,
                                 proposal_provider="none"),
                     loaded_backend, proposal)
    assert len(constructions) == 2


def test_candidate_tracks_reset_with_source_generation(tmp_path) -> None:
    from cvti.object_watch.matcher import ObjectCandidate

    runtime = ObjectWatchRuntime(ObjectWatchConfig(
        library_path=tmp_path / "object_library", sample_fps=5,
    ))
    frame = np.zeros((24, 24, 3), np.uint8)
    first_sample = WatchSample("cam", 1, 1, 0.0, 0.0, frame)
    first = runtime._track(first_sample, (
        ObjectCandidate((0, 0, 20, 20), confidence=.9),
    ))[0]

    runtime.reset_camera("cam", 2)
    second_sample = WatchSample("cam", 2, 1, 0.0, 0.0, frame)
    second = runtime._track(second_sample, (
        ObjectCandidate((0, 0, 20, 20), confidence=.9),
    ))[0]

    assert first.track_id is not None
    assert second.track_id is not None
    assert first.track_id != second.track_id


def test_untracked_runtime_matches_are_diagnostic_but_do_not_prime_stability(
    tmp_path, monkeypatch,
) -> None:
    from types import SimpleNamespace

    from cvti.object_watch.matcher import ObjectCandidate, ObjectMatch
    from cvti.object_watch.presence import PresenceToken

    config = ObjectWatchConfig(
        library_path=tmp_path / "object_library", sample_fps=5,
    )
    runtime = ObjectWatchRuntime(config)
    runtime.reset_camera("cam", 1)
    runtime._snapshot_revision = 1
    runtime._snapshot_active = {"target": 1}
    runtime._snapshot_model_fingerprint = "model"
    index = SimpleNamespace(
        library_revision=1, model_fingerprint="model",
        targets=(SimpleNamespace(target=SimpleNamespace(
            id="target", revision=1, grounding_description="",
        )),),
    )
    monkeypatch.setattr("cvti.object_watch.runtime.build_recognition_index", lambda *_: index)

    class Matcher:
        def match(self, camera_id, _frame, candidates, timestamp):
            candidate = candidates[0]
            return [ObjectMatch(
                camera_id, "target", "Target", "custom", candidate.bbox, .9,
                timestamp, track_id=candidate.track_id, target_revision=1,
                library_revision=1, model_fingerprint="model",
            )]

    monkeypatch.setattr("cvti.object_watch.runtime.ObjectMatcher.from_index",
                        lambda *_args, **_kwargs: Matcher())
    monkeypatch.setattr(runtime, "_evidence", lambda *_args: [
        np.zeros((2, 2, 3), np.uint8), np.zeros((2, 4, 3), np.uint8),
    ])
    backend = SimpleNamespace(fingerprint="model")
    frame = np.zeros((140, 140, 3), np.uint8)

    untracked = []
    for sequence, bbox in ((1, (0, 0, 20, 20)), (2, (100, 0, 120, 20))):
        item = WatchSample("cam", 1, sequence, sequence / 5, sequence / 5, frame,
                           (ObjectCandidate(bbox, confidence=.30),))
        result = runtime._process(item, backend, None)[0]
        assert result.status == "matched"
        assert result.matches[0].track_id is None
        assert runtime.reserve(result, "rule") is None
        untracked.append(result)

    assert runtime._stability._states == {}
    assert runtime._presence._states == {}
    assert runtime.commit(PresenceToken(("cam", 1, "rule", "target", None), 1)) is False

    first_tracked = WatchSample(
        "cam", 1, 3, .6, .6, frame,
        (ObjectCandidate((40, 40, 60, 60), confidence=.9, track_id=99),),
    )
    second_tracked = WatchSample(
        "cam", 1, 4, .8, .8, frame,
        (ObjectCandidate((40, 40, 60, 60), confidence=.9, track_id=99),),
    )
    assert runtime._process(first_tracked, backend, None)[0].status == "no_match"
    stable = runtime._process(second_tracked, backend, None)[0]
    assert stable.status == "matched"
    token = runtime.reserve(stable, "rule")
    assert token is not None and runtime.commit(token)
    assert runtime.reserve(stable, "rule") is None


def test_generated_proposals_are_zoned_after_overlap_nms_and_admitted(
    tmp_path, monkeypatch,
) -> None:
    from types import SimpleNamespace

    from cvti.object_watch.matcher import ObjectCandidate, ObjectMatch
    from cvti.object_watch.runtime import WatchZone

    runtime = ObjectWatchRuntime(ObjectWatchConfig(
        library_path=tmp_path / "object_library", sample_fps=5,
    ))
    runtime.reset_camera("cam", 1)
    target = SimpleNamespace(id="target", revision=1, grounding_description="blue box")
    index = SimpleNamespace(
        library_revision=1, model_fingerprint="model",
        targets=(SimpleNamespace(target=target),),
    )
    monkeypatch.setattr("cvti.object_watch.runtime.build_recognition_index", lambda *_: index)
    seen = []

    class Matcher:
        def match(self, camera_id, _frame, candidates, timestamp):
            seen.append(tuple(candidates))
            return [ObjectMatch(
                camera_id, "target", "Target", "custom", candidate.bbox, .9,
                timestamp, track_id=candidate.track_id, zone_id=candidate.zone_id,
                target_revision=1, library_revision=1, model_fingerprint="model",
            ) for candidate in candidates]

    class Provider:
        def propose(self, *_args):
            return (
                ObjectCandidate((10, 10, 30, 30), "generated", .95),
                ObjectCandidate((70, 70, 90, 90), "outside", .90),
            )

    monkeypatch.setattr("cvti.object_watch.runtime.ObjectMatcher.from_index",
                        lambda *_args, **_kwargs: Matcher())
    monkeypatch.setattr(runtime, "_evidence", lambda *_args: [
        np.zeros((2, 2, 3), np.uint8), np.zeros((2, 4, 3), np.uint8),
    ])
    backend = SimpleNamespace(fingerprint="model")
    frame = np.zeros((100, 100, 3), np.uint8)
    zone = WatchZone("door", ((0, 0), (50, 0), (50, 50), (0, 50)))

    results = []
    for sequence in (1, 2):
        item = WatchSample(
            "cam", 1, sequence, sequence / 5, sequence / 5, frame,
            # This lower-confidence generic box must lose NMS without losing
            # the generated box's eventual zone membership.
            (ObjectCandidate((10, 10, 30, 30), "generic", .40, 9, "door"),),
            (zone,), "yolo_world",
        )
        results = runtime._process(item, backend, Provider())

    assert [(candidate.label_hint, candidate.zone_id) for candidate in seen[-1]] == [
        ("generated", "door"), ("outside", None),
    ]
    matched = next(result for result in results
                   if result.status == "matched" and result.matches[0].zone_id == "door")
    assert matched.matches[0].zone_id == "door"
    assert runtime.reserve(matched, "door-rule") is not None

    from cvti.rules.customization import CustomizationEngine
    from cvti.serving.event_adapters import object_watch_result_events

    rules = tmp_path / "rules.json"
    rules.write_text('{"rules":[{"name":"door-rule","trigger":'
                     '{"detector":"object_watch","state":"object_seen",'
                     '"object_id":"target","zone":"door"}}]}')
    alerts = CustomizationEngine(rules).evaluate([
        event
        for result in results if result.status == "matched"
        for event in object_watch_result_events(result)
    ])
    assert len(alerts) == 1
    assert alerts[0].metadata["zone"] == "door"


def test_generated_proposal_without_configured_zones_uses_whole_camera(
    tmp_path, monkeypatch,
) -> None:
    from types import SimpleNamespace

    from cvti.object_watch.matcher import ObjectCandidate

    runtime = ObjectWatchRuntime(ObjectWatchConfig(
        library_path=tmp_path / "object_library",
    ))
    runtime.reset_camera("cam", 0)
    index = SimpleNamespace(library_revision=1, model_fingerprint="model", targets=())
    monkeypatch.setattr("cvti.object_watch.runtime.build_recognition_index", lambda *_: index)
    seen = []
    monkeypatch.setattr(
        "cvti.object_watch.runtime.ObjectMatcher.from_index",
        lambda *_args, **_kwargs: SimpleNamespace(
            match=lambda _camera, _frame, candidates, _timestamp: seen.extend(candidates) or []
        ),
    )
    provider = SimpleNamespace(propose=lambda *_: (ObjectCandidate((2, 2, 8, 8), "box", .9),))
    frame = np.zeros((10, 10, 3), np.uint8)
    result = runtime._process(
        WatchSample("cam", 0, 1, 0.0, 0.0, frame, proposal_provider="yolo_world"),
        SimpleNamespace(fingerprint="model"), provider,
    )

    assert result[0].status == "no_match"
    assert len(seen) == 1 and seen[0].zone_id is None
