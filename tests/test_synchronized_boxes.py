from types import SimpleNamespace

import numpy as np
import pytest

from cvti.serving.pipeline import MultiStreamPipeline


class Publisher:
    def __init__(self):
        self.tracking = True
        self.calls = []

    def has_viewers(self, camera_id):
        return True

    def has_tracking_viewers(self, camera_id):
        return self.tracking

    def publish(self, camera_id, image, overlays):
        self.calls.append((camera_id, image, overlays))


def make_pipe():
    publisher = Publisher()
    state = SimpleNamespace(process=lambda *a, **k: [], _motion_overlays=[], _box_by_track={})
    pipe = MultiStreamPipeline({"crowd": 0}, publisher=publisher,
                               camera_states={"crowd": state},
                               synchronized_box_cameras={"crowd"})
    pipe._names = {0: "person"}
    pipe._threat_classes = set()
    return pipe, publisher


def test_only_requested_camera_and_annotated_view_is_synchronized():
    pipe, publisher = make_pipe()
    assert pipe._synchronized_boxes("crowd")
    assert not pipe._synchronized_boxes("night")
    publisher.tracking = False
    assert not pipe._synchronized_boxes("crowd")
    publisher.tracking = True
    pipe.view_only.add("crowd")
    assert not pipe._synchronized_boxes("crowd")


def test_inference_publishes_exact_frame_not_newer_decoder_frame():
    import torch
    from ultralytics.engine.results import Results
    pipe, publisher = make_pipe()
    original = np.zeros((100, 100, 3), dtype=np.uint8)
    frame = SimpleNamespace(camera_id="crowd", image=original, timestamp=1.0)
    result = Results(original, path="", names={0: "person"}, boxes=torch.empty((0, 6)))
    pipe._route_to_queue(frame, result)
    assert len(publisher.calls) == 1
    assert publisher.calls[0][1] is original
    publisher.tracking = False
    pipe._route_to_queue(frame, result)
    assert len(publisher.calls) == 1  # raw smooth path owns publishing again


def test_smooth_path_cannot_replace_synchronized_frame(monkeypatch):
    from cvti.serving import pipeline
    pipe, publisher = make_pipe()
    class EndLoop(BaseException):
        pass
    def fail_peek():
        pytest.fail("synchronized display read a newer frame")
    pipe._decoders = {"crowd": SimpleNamespace(display_fps=0, peek_latest=fail_peek)}
    def stop(_):
        raise EndLoop
    monkeypatch.setattr(pipeline.time, "sleep", stop)
    with pytest.raises(EndLoop):
        pipe._smooth_publish_loop()
    assert publisher.calls == []
