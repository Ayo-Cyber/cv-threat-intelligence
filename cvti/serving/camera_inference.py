"""Opt-in camera orientation/model calibration, preserving source coordinates."""
from pathlib import Path
import math

import cv2
import numpy as np


def rotate_for_detection(image, angle):
    if angle == 0:
        return image, None
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    c, s = abs(matrix[0, 0]), abs(matrix[0, 1])
    nw, nh = round(h * s + w * c), round(h * c + w * s)
    matrix[0, 2] += (nw - w) / 2
    matrix[1, 2] += (nh - h) / 2
    return cv2.warpAffine(image, matrix, (nw, nh)), cv2.invertAffineTransform(matrix)


def restore_boxes(boxes, inverse, shape):
    """Inverse-map all four corners, not just the box diagonal."""
    out = np.asarray(boxes).copy().reshape(-1, 4)
    if not len(out):
        return out
    corners = out[:, [0, 1, 2, 1, 2, 3, 0, 3]].reshape(-1, 4, 2)
    points = corners @ inverse[:, :2].T + inverse[:, 2]
    out[:, :2], out[:, 2:] = points.min(axis=1), points.max(axis=1)
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, shape[1])
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, shape[0])
    return out


class CameraInference:
    def __init__(self, profiles):
        self.profiles = profiles or {}
        self.models = {}
        for camera, profile in self.profiles.items():
            if not isinstance(profile, dict) or set(profile) - {"weights", "rotation_degrees", "confidence"}:
                raise ValueError(f"camera {camera}: invalid detection_inference settings")
            angle = profile.get("rotation_degrees", 0)
            confidence = profile.get("confidence", 0.4)
            for value, low, high in ((angle, -180, 180), (confidence, 0.1, 0.9)):
                if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
                    raise ValueError(f"camera {camera}: invalid detection rotation/confidence")
            weights = profile.get("weights")
            if weights is not None and (not isinstance(weights, str) or
                    Path(weights).suffix != ".pt" or not Path(weights).is_file()):
                raise ValueError(f"camera {camera}: detection weights must be an existing local .pt file")

    def load(self, base):
        from ultralytics import YOLO
        for profile in self.profiles.values():
            weights = profile.get("weights")
            if weights and weights not in self.models:
                model = YOLO(weights)
                if model.task != "detect" or model.names != base.names:
                    raise ValueError("camera detector must use the shared detector's detection classes")
                self.models[weights] = model

    def predict(self, base, frames, **kwargs):
        results = [None] * len(frames)
        ordinary = [i for i, frame in enumerate(frames) if frame.camera_id not in self.profiles]
        if ordinary:
            predictions = base.predict([frames[i].image for i in ordinary], **kwargs)
            for i, result in zip(ordinary, predictions):
                results[i] = result
        for i, frame in enumerate(frames):
            profile = self.profiles.get(frame.camera_id)
            if profile is None:
                continue
            model = self.models.get(profile.get("weights"), base)
            image, inverse = rotate_for_detection(frame.image, profile.get("rotation_degrees", 0))
            options = dict(kwargs, conf=profile.get("confidence", kwargs["conf"]))
            result = model.predict([image], **options)[0]
            if inverse is not None:
                data = result.boxes.data
                boxes = data.clone() if hasattr(data, "clone") else data.copy()
                coordinates = boxes[:, :4]
                if hasattr(coordinates, "cpu"):
                    coordinates = coordinates.cpu().numpy()
                mapped = restore_boxes(coordinates, inverse, frame.image.shape)
                boxes[:, :4] = boxes.new_tensor(mapped) if hasattr(boxes, "new_tensor") else mapped
                # Everything downstream (tracking, zones, evidence, UI) uses
                # the original unrotated camera image and its pixel coordinates.
                result.orig_img = frame.image
                result.orig_shape = frame.image.shape[:2]
                result.update(boxes=boxes)
            results[i] = result
        return results
