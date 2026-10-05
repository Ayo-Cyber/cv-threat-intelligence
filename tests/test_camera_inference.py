from types import SimpleNamespace

import numpy as np
import pytest

from cvti.serving.camera_inference import CameraInference, restore_boxes, rotate_for_detection


def test_default_does_not_transform_images():
    image = np.zeros((30, 50, 3), dtype=np.uint8)
    result = object()
    seen = []
    model = SimpleNamespace(predict=lambda images, **kwargs: seen.append(images) or [result])
    predictor = CameraInference({})
    assert predictor.predict(model, [SimpleNamespace(camera_id="a", image=image)], conf=0.25) == [result]
    assert seen[0][0] is image


def test_right_angle_restores_source_coordinates():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    rotated, inverse = rotate_for_detection(image, -90)
    assert rotated.shape[:2] == (200, 100)
    np.testing.assert_allclose(restore_boxes([[60., 10., 80., 30.]], inverse, image.shape),
                               [[10, 20, 30, 40]], atol=1e-5)
    assert restore_boxes(np.empty((0, 4)), inverse, image.shape).shape == (0, 4)


def test_boxes_clipped_to_original_frame():
    inverse = np.array([[1., 0., 0.], [0., 1., 0.]])
    np.testing.assert_array_equal(restore_boxes([[-5., -4., 999., 888.]], inverse, (100, 200)),
                                  [[0, 0, 200, 100]])


@pytest.mark.parametrize("numpy_boxes", [False, True])
def test_profile_isolated_and_result_mapped_back(numpy_boxes):
    import torch
    from ultralytics.engine.results import Results

    original = np.zeros((100, 200, 3), dtype=np.uint8)
    options = []
    normal = object()
    base = SimpleNamespace(predict=lambda images, **kwargs: [normal])
    def predict(images, **kwargs):
        options.append(kwargs)
        boxes = [[60., 10., 80., 30., 0.8, 0.]]
        return [Results(images[0], path="", names={0: "person"},
                        boxes=np.array(boxes) if numpy_boxes else torch.tensor(boxes))]
    predictor = CameraInference({"angled": {"rotation_degrees": -90, "confidence": 0.5}})
    predictor.models[None] = SimpleNamespace(predict=predict)
    results = predictor.predict(base, [SimpleNamespace(camera_id="angled", image=original),
                                      SimpleNamespace(camera_id="night", image=original)], conf=0.25)
    assert results[1] is normal
    assert options[0]["conf"] == 0.5
    assert results[0].orig_img is original
    assert results[0].orig_shape == (100, 200)
    coordinates = results[0].boxes.xyxy
    np.testing.assert_allclose(coordinates if numpy_boxes else coordinates.numpy(),
                               [[10, 20, 30, 40]], atol=1e-5)


@pytest.mark.parametrize("profile", [[], {"rotation_degrees": float("nan")},
    {"rotation_degrees": True}, {"confidence": 0}, {"weights": "missing.pt"}, {"bogus": 1}])
def test_bad_profiles_rejected(profile):
    with pytest.raises(ValueError):
        CameraInference({"camera": profile})
