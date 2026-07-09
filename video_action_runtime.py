"""Runtime bridge for hybrid video-action evidence in the live detector."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import cv2

from customization import RawEvent
from video_action_hybrid import predictions_to_events
from video_action_model import (
    DEFAULT_FRAME_COUNT,
    DEFAULT_VIDEOMAE_MODEL,
    DEFAULT_X3D_MODEL,
    FrameWindow,
    SampledFrame,
    VideoMAEActionModel,
    X3DActionModel,
    sample_evenly_with_indices,
)


@dataclass(frozen=True)
class BufferedFrame:
    index: int
    frame: np.ndarray


class VideoActionRuntime:
    """Keeps recent frames and runs a weak video classifier around trigger frames."""

    def __init__(
        self,
        *,
        model: Any,
        backend: str,
        model_name: str,
        fps: float,
        window_seconds: float = 4.0,
        frame_count: int = DEFAULT_FRAME_COUNT,
        top_k: int = 5,
        cooldown_seconds: float = 2.0,
        verbose: bool = False,
    ) -> None:
        self.model = model
        self.backend = backend
        self.model_name = model_name
        self.fps = fps if fps > 0 else 30.0
        self.window_seconds = window_seconds
        self.frame_count = frame_count
        self.top_k = top_k
        self.cooldown_seconds = cooldown_seconds
        self.verbose = verbose
        self._last_analysis_timestamp = -1_000_000.0
        buffer_size = max(frame_count, int(round(self.fps * window_seconds * 2)) + frame_count)
        self._frames: deque[BufferedFrame] = deque(maxlen=buffer_size)

    def add_frame(self, frame: np.ndarray, *, frame_index: int) -> None:
        # Store RGB frames because VideoMAE/X3D wrappers expect RGB arrays.
        self._frames.append(BufferedFrame(index=frame_index, frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

    def analyze_event(self, *, center_frame_index: int, timestamp: float) -> list[RawEvent]:
        if timestamp - self._last_analysis_timestamp < self.cooldown_seconds:
            return []
        if not self._frames:
            return []

        window = self._build_buffered_window(center_frame_index)
        predictions = self.model.predict_frames([item.frame for item in window.sampled], top_k=self.top_k)
        self._last_analysis_timestamp = timestamp
        if self.verbose:
            top = predictions[0] if predictions else None
            if top is not None:
                print(
                    f"[VideoAction] {self.backend}:{self.model_name} {window.start_index}..{window.end_index} "
                    f"top={top.label} ({top.confidence:.3f})"
                )

        return predictions_to_events(
            predictions,
            backend=self.backend,
            model_name=self.model_name,
            window_name=window.name,
            sampled_frame_indices=[item.index for item in window.sampled],
            timestamp=timestamp,
        )

    def _build_buffered_window(self, center_frame_index: int) -> FrameWindow:
        radius = max(0, int(round((self.window_seconds * self.fps) / 2)))
        start = center_frame_index - radius
        end = center_frame_index + radius

        candidates = [item for item in self._frames if start <= item.index <= end]
        if not candidates:
            candidates = list(self._frames)

        sampled_local = sample_evenly_with_indices([item.frame for item in candidates], count=self.frame_count)
        sampled = [
            SampledFrame(index=candidates[item.index].index, frame=item.frame)
            for item in sampled_local
        ]
        return FrameWindow(
            name="event",
            start_index=candidates[0].index,
            end_index=candidates[-1].index,
            sampled=sampled,
        )


def build_video_action_runtime(
    *,
    backend: str,
    model_name: str,
    fps: float,
    window_seconds: float,
    frame_count: int,
    top_k: int,
    cooldown_seconds: float,
    device: str | None,
    verbose: bool,
) -> VideoActionRuntime:
    if backend == "videomae":
        resolved_model = model_name or DEFAULT_VIDEOMAE_MODEL
        model = VideoMAEActionModel(resolved_model, device=device, frame_count=frame_count, verbose=verbose)
    elif backend == "x3d":
        resolved_model = model_name or DEFAULT_X3D_MODEL
        model = X3DActionModel(resolved_model, device=device, frame_count=frame_count, verbose=verbose)
    else:
        raise ValueError(f"Unsupported video action backend: {backend}")

    return VideoActionRuntime(
        model=model,
        backend=backend,
        model_name=resolved_model,
        fps=fps,
        window_seconds=window_seconds,
        frame_count=frame_count,
        top_k=top_k,
        cooldown_seconds=cooldown_seconds,
        verbose=verbose,
    )
