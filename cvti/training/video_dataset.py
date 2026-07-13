"""Shared CamNuvem clip dataset for temporal-model fine-tuning (plan.md Phase 6).

Backend-agnostic on purpose: it yields (sampled_frames, label) where frames are
`count` RGB frames evenly spread across each clip. Both the VideoMAE and X3D
trainers consume the SAME clips + labels, so their results stay comparable.

CamNuvem layout (already on disk):
    CamNuvem Robbery Dataset/videos/samples/<split>/<class>/*.mp4
    split ∈ {training, test}     class ∈ {normal, theft}
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

DEFAULT_DATA_ROOT = "CamNuvem Robbery Dataset/videos/samples"
# CamNuvem is robbery-vs-normal; "theft" is the positive (robbery) class.
DEFAULT_CLASS_MAP = {"normal": 0, "theft": 1}
_VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def scan_clips(root: str | Path, split: str, *, class_map: dict[str, int] | None = None,
               per_class_limit: int | None = None) -> list[tuple[Path, int]]:
    """List (video_path, label) for a split. `per_class_limit` caps each class
    (keeps the smoke set balanced)."""
    class_map = class_map or DEFAULT_CLASS_MAP
    base = Path(root) / split
    items: list[tuple[Path, int]] = []
    for cls, label in class_map.items():
        cls_dir = base / cls
        if not cls_dir.exists():
            continue
        clips = sorted(p for p in cls_dir.iterdir() if p.suffix.lower() in _VIDEO_EXTS)
        if per_class_limit:
            clips = clips[:per_class_limit]
        items.extend((p, label) for p in clips)
    return items


def read_clip_evenly(path: str | Path, count: int) -> list[np.ndarray]:
    """Return `count` RGB frames evenly spaced across the clip.

    Seeks to `count` positions instead of decoding the whole (possibly minutes-
    long) robbery video — keeps memory + time bounded per clip.
    """
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames: list[np.ndarray] = []
    try:
        if total >= 1:
            idxs = [round(k * (total - 1) / max(1, count - 1)) for k in range(count)]
            for idx in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, f = cap.read()
                if ok:
                    frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        else:  # unknown length — read sequentially then sample
            seq = []
            while len(seq) < count * 20:
                ok, f = cap.read()
                if not ok:
                    break
                seq.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            if seq:
                idxs = [round(k * (len(seq) - 1) / max(1, count - 1)) for k in range(count)]
                frames = [seq[i] for i in idxs]
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded: {path}")
    while len(frames) < count:          # pad short clips by repeating the last frame
        frames.append(frames[-1])
    return frames[:count]


class RobberyClipDataset:
    """Lazily decodes clips; index i -> (frames [T,H,W,3] uint8 RGB, label int)."""

    def __init__(self, root: str | Path, split: str, *, frames: int = 16,
                 class_map: dict[str, int] | None = None, per_class_limit: int | None = None) -> None:
        self.class_map = class_map or DEFAULT_CLASS_MAP
        self.items = scan_clips(root, split, class_map=self.class_map,
                                per_class_limit=per_class_limit)
        self.frames = frames

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> tuple[np.ndarray, int]:
        path, label = self.items[i]
        clip = read_clip_evenly(path, self.frames)     # list of [H,W,3] RGB
        return np.stack(clip), label                   # [T,H,W,3], int

    def labels(self) -> list[int]:
        return [label for _, label in self.items]
