"""Video action model wrapper for temporal threat candidates.

This module is intentionally standalone. It lets us test pretrained video
classifiers on clips before wiring them into the live detector loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
from cvti.console import emit

from cvti.logging_setup import get_logger

log = get_logger(__name__)


DEFAULT_VIDEOMAE_MODEL = "MCG-NJU/videomae-base-finetuned-kinetics"
DEFAULT_X3D_MODEL = "x3d_s"
DEFAULT_FRAME_COUNT = 16


@dataclass(frozen=True)
class VideoActionPrediction:
    label: str
    confidence: float
    rank: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 6),
            "rank": self.rank,
        }


@dataclass(frozen=True)
class SampledFrame:
    index: int
    frame: np.ndarray


@dataclass(frozen=True)
class FrameWindow:
    name: str
    start_index: int
    end_index: int
    sampled: list[SampledFrame]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "sampled_frame_indices": [item.index for item in self.sampled],
        }


class MissingVideoActionDependency(RuntimeError):
    """Raised when optional VideoMAE dependencies have not been installed."""


def read_video_frames(
    video_path: str | Path,
    *,
    max_frames: int | None = None,
    stride: int = 1,
) -> list[np.ndarray]:
    """Read RGB frames from a video file.

    OpenCV decodes frames as BGR. VideoMAE processors expect RGB-like image
    arrays, so this function converts frames before returning them.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(path)
    if stride < 1:
        raise ValueError("stride must be >= 1")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    frames: list[np.ndarray] = []
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % stride == 0:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if max_frames is not None and len(frames) >= max_frames:
                    break
            frame_idx += 1
    finally:
        cap.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from video: {path}")
    return frames


def sample_evenly(frames: list[np.ndarray], *, count: int = DEFAULT_FRAME_COUNT) -> list[np.ndarray]:
    """Return exactly `count` frames, spread across the input sequence."""
    return [sample.frame for sample in sample_evenly_with_indices(frames, count=count)]


def sample_evenly_with_indices(
    frames: list[np.ndarray],
    *,
    count: int = DEFAULT_FRAME_COUNT,
) -> list[SampledFrame]:
    """Return exactly `count` frames plus their original source indices."""
    if count < 1:
        raise ValueError("count must be >= 1")
    if not frames:
        raise ValueError("sample_evenly needs at least one frame")

    if len(frames) == 1:
        return [SampledFrame(index=0, frame=frames[0]) for _ in range(count)]

    indices = np.linspace(0, len(frames) - 1, num=count)
    int_indices = [int(i) for i in indices]
    return [SampledFrame(index=index, frame=frames[index]) for index in int_indices]


def build_segment_windows(
    frames: list[np.ndarray],
    *,
    count: int = DEFAULT_FRAME_COUNT,
) -> list[FrameWindow]:
    """Build beginning/middle/ending contiguous windows from a decoded video."""
    if not frames:
        raise ValueError("build_segment_windows needs at least one frame")
    total = len(frames)
    bounds = [
        ("beginning", 0, max(1, int(np.ceil(total / 3)))),
        ("middle", int(total // 3), max(int(total // 3) + 1, int(np.ceil(2 * total / 3)))),
        ("ending", int(2 * total // 3), total),
    ]

    windows: list[FrameWindow] = []
    for name, start, end in bounds:
        start = max(0, min(start, total - 1))
        end = max(start + 1, min(end, total))
        segment = frames[start:end]
        sampled = [
            SampledFrame(index=start + item.index, frame=item.frame)
            for item in sample_evenly_with_indices(segment, count=count)
        ]
        windows.append(FrameWindow(name=name, start_index=start, end_index=end - 1, sampled=sampled))
    return windows


def build_centered_window(
    frames: list[np.ndarray],
    *,
    center_index: int,
    radius_frames: int,
    count: int = DEFAULT_FRAME_COUNT,
    name: str = "event",
) -> FrameWindow:
    """Build one window around a detector-selected event frame."""
    if not frames:
        raise ValueError("build_centered_window needs at least one frame")
    if radius_frames < 0:
        raise ValueError("radius_frames must be >= 0")

    total = len(frames)
    center = max(0, min(center_index, total - 1))
    start = max(0, center - radius_frames)
    end = min(total, center + radius_frames + 1)
    segment = frames[start:end]
    sampled = [
        SampledFrame(index=start + item.index, frame=item.frame)
        for item in sample_evenly_with_indices(segment, count=count)
    ]
    return FrameWindow(name=name, start_index=start, end_index=end - 1, sampled=sampled)


class VideoMAEActionModel:
    """Thin lazy wrapper around Hugging Face VideoMAE video classification."""

    def __init__(
        self,
        model_name: str = DEFAULT_VIDEOMAE_MODEL,
        *,
        device: str | None = None,
        frame_count: int = DEFAULT_FRAME_COUNT,
        verbose: bool = False,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.frame_count = frame_count
        self.verbose = verbose
        self._processor: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None

    def predict_frames(
        self,
        frames: list[np.ndarray],
        *,
        top_k: int = 5,
    ) -> list[VideoActionPrediction]:
        """Classify a sequence of frames and return top-k labels."""
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        self._load()
        assert self._processor is not None
        assert self._model is not None
        assert self._torch is not None

        sampled = sample_evenly(frames, count=self.frame_count)
        inputs = self._processor(sampled, return_tensors="pt")
        inputs = {key: value.to(self._model.device) for key, value in inputs.items()}

        with self._torch.no_grad():
            outputs = self._model(**inputs)
            probabilities = outputs.logits.softmax(dim=-1)[0]

        top = self._torch.topk(probabilities, k=min(top_k, probabilities.shape[-1]))
        predictions: list[VideoActionPrediction] = []
        for rank, (idx, score) in enumerate(zip(top.indices.tolist(), top.values.tolist()), start=1):
            predictions.append(
                VideoActionPrediction(
                    label=self._model.config.id2label.get(idx, str(idx)),
                    confidence=float(score),
                    rank=rank,
                )
            )
        return predictions

    def predict_video(
        self,
        video_path: str | Path,
        *,
        top_k: int = 5,
        max_decode_frames: int | None = 240,
        stride: int = 1,
    ) -> list[VideoActionPrediction]:
        frames = read_video_frames(video_path, max_frames=max_decode_frames, stride=stride)
        return self.predict_frames(frames, top_k=top_k)

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            self._log("Loading torch...")
            import torch
            self._log("Loading Hugging Face VideoMAE classes...")
            from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
        except ImportError as exc:
            raise MissingVideoActionDependency(
                "VideoMAE needs optional dependencies. Install them with: "
                "`./.venv/bin/python -m pip install transformers accelerate safetensors`"
            ) from exc

        self._log(f"Loading cached processor: {self.model_name}")
        processor = VideoMAEImageProcessor.from_pretrained(self.model_name, local_files_only=True)
        self._log(f"Loading cached model weights: {self.model_name}")
        model = VideoMAEForVideoClassification.from_pretrained(self.model_name, local_files_only=True)

        device = self.device or _default_device(torch)
        self._log(f"Moving model to device: {device}")
        model.to(device)
        model.eval()

        self._processor = processor
        self._model = model
        self._torch = torch

    def _log(self, message: str) -> None:
        if self.verbose:
            emit(f"[VideoMAE] {message}", file=sys.stderr, flush=True)


class X3DActionModel:
    """Lazy wrapper around PyTorchVideo X3D Kinetics-400 models."""

    def __init__(
        self,
        model_name: str = DEFAULT_X3D_MODEL,
        *,
        device: str | None = None,
        frame_count: int = DEFAULT_FRAME_COUNT,
        verbose: bool = False,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.frame_count = frame_count
        self.verbose = verbose
        self._model: Any | None = None
        self._torch: Any | None = None
        self._labels: list[str] = []

    def predict_frames(
        self,
        frames: list[np.ndarray],
        *,
        top_k: int = 5,
    ) -> list[VideoActionPrediction]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        self._load()
        assert self._model is not None
        assert self._torch is not None

        sampled = sample_evenly(frames, count=self.frame_count)
        tensor = _frames_to_x3d_tensor(sampled, self._torch).to(next(self._model.parameters()).device)
        with self._torch.no_grad():
            probabilities = self._model(tensor).softmax(dim=-1)[0]

        top = self._torch.topk(probabilities, k=min(top_k, probabilities.shape[-1]))
        predictions: list[VideoActionPrediction] = []
        for rank, (idx, score) in enumerate(zip(top.indices.tolist(), top.values.tolist()), start=1):
            label = self._labels[idx] if idx < len(self._labels) else str(idx)
            predictions.append(VideoActionPrediction(label=label, confidence=float(score), rank=rank))
        return predictions

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            self._log("Loading torch...")
            import torch
            self._log("Loading PyTorchVideo X3D...")
            from pytorchvideo.models.hub import x3d_m, x3d_s, x3d_xs
        except ImportError as exc:
            raise MissingVideoActionDependency(
                "X3D needs optional PyTorchVideo dependencies. Install them with: "
                "`./.venv/bin/python -m pip install -r requirements-video.txt`"
            ) from exc

        loaders = {
            "x3d_xs": x3d_xs,
            "x3d_s": x3d_s,
            "x3d_m": x3d_m,
        }
        if self.model_name not in loaders:
            raise ValueError(f"Unsupported X3D model '{self.model_name}'. Use one of: {', '.join(loaders)}")

        self._log(f"Loading cached/pretrained weights: {self.model_name}")
        model = loaders[self.model_name](pretrained=True)
        # PyTorchVideo X3D currently hits unsupported avg_pool3d ops on Apple MPS.
        # Default to CPU unless the caller explicitly opts into another device.
        device = self.device or "cpu"
        self._log(f"Moving model to device: {device}")
        model.to(device)
        model.eval()

        self._model = model
        self._torch = torch
        self._labels = _kinetics400_labels()

    def _log(self, message: str) -> None:
        if self.verbose:
            emit(f"[X3D] {message}", file=sys.stderr, flush=True)


def _frames_to_x3d_tensor(frames: list[np.ndarray], torch: Any) -> Any:
    processed = []
    for frame in frames:
        resized = _resize_short_side(frame, short_side=256)
        cropped = _center_crop(resized, size=256)
        arr = cropped.astype(np.float32) / 255.0
        arr = (arr - np.array([0.45, 0.45, 0.45], dtype=np.float32)) / np.array([0.225, 0.225, 0.225], dtype=np.float32)
        processed.append(arr)
    video = np.stack(processed, axis=0)  # T,H,W,C
    video = np.transpose(video, (3, 0, 1, 2))  # C,T,H,W
    return torch.from_numpy(video).unsqueeze(0).float()


def _resize_short_side(frame: np.ndarray, *, short_side: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if height <= width:
        new_height = short_side
        new_width = int(round(width * short_side / height))
    else:
        new_width = short_side
        new_height = int(round(height * short_side / width))
    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)


def _center_crop(frame: np.ndarray, *, size: int) -> np.ndarray:
    height, width = frame.shape[:2]
    y0 = max(0, (height - size) // 2)
    x0 = max(0, (width - size) // 2)
    return frame[y0:y0 + size, x0:x0 + size]


def _kinetics400_labels() -> list[str]:
    try:
        from torchvision.models.video import R3D_18_Weights

        return list(R3D_18_Weights.KINETICS400_V1.meta["categories"])
    except Exception as exc:
        log.warning("video-action inference failed; no predictions", exc_info=True)
        return []


def _default_device(torch: Any) -> str:
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
