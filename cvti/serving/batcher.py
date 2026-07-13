"""Cross-camera batcher.

Each pipeline tick, collect the newest unread frame from every camera and pack
them into one batch. Only genuinely-new frames are included, so idle cameras
don't waste a batch slot. The detector then runs ONE forward pass over the
whole batch, and results are scattered back per camera.
"""
from __future__ import annotations

from typing import Any, Protocol

from cvti.serving.streams import Frame


class _Readable(Protocol):
    def read_latest(self) -> Frame | None: ...


def collect_batch(decoders: dict[str, _Readable], *, max_batch: int = 32) -> list[Frame]:
    """Gather up to `max_batch` fresh frames across all cameras (one per camera)."""
    batch: list[Frame] = []
    for decoder in decoders.values():
        frame = decoder.read_latest()
        if frame is not None:
            batch.append(frame)
            if len(batch) >= max_batch:
                break
    return batch
