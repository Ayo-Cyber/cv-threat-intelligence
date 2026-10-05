import pytest

from cvti.app.console_backend import _desktop_inference_args


def test_defaults_are_unchanged():
    assert _desktop_inference_args({}) == [
        "--target-fps", "4", "--imgsz", "512", "--conf", "0.4"]


def test_high_detail_site():
    assert _desktop_inference_args({"inference": {
        "imgsz": 960, "target_fps": 2, "confidence": 0.25}}) == [
        "--target-fps", "2", "--imgsz", "960", "--conf", "0.25"]


@pytest.mark.parametrize("settings", [None, [], {"imgsz": 513}, {"imgsz": True},
    {"imgsz": 4096}, {"target_fps": 0}, {"confidence": float("nan")},
    {"confidence": 0.01}, {"confidence": "0.25"}])
def test_invalid_settings_rejected(settings):
    with pytest.raises(ValueError):
        _desktop_inference_args({"inference": settings})
