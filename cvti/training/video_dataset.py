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


def pool_clips(root: str | Path, *, class_map: dict[str, int] | None = None,
               splits: tuple[str, ...] = ("training", "test")) -> list[tuple[Path, int]]:
    """All (path, label) across every split — used to build a fresh stratified
    split. CamNuvem's own training/test split is class-inverted (train 69% theft,
    test 75% normal), which makes metrics untrustworthy; re-splitting fixes that."""
    items: list[tuple[Path, int]] = []
    for split in splits:
        items.extend(scan_clips(root, split, class_map=class_map))
    return items


def stratified_split(items: list[tuple[Path, int]], *, val_fraction: float = 0.25,
                     seed: int = 1234) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    """Split into (train, test) keeping each class's ratio the same in both —
    so the test set isn't class-skewed relative to train. Deterministic (seed)."""
    import random

    by_class: dict[int, list[tuple[Path, int]]] = {}
    for path, label in items:
        by_class.setdefault(label, []).append((path, label))
    rng = random.Random(seed)
    train: list[tuple[Path, int]] = []
    test: list[tuple[Path, int]] = []
    for label, group in by_class.items():
        group = sorted(group, key=lambda x: str(x[0]))
        rng.shuffle(group)
        n_test = max(1, round(len(group) * val_fraction))
        test.extend(group[:n_test])
        train.extend(group[n_test:])
    return train, test


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

    def __init__(self, root: str | Path | None = None, split: str = "training", *,
                 frames: int = 16, class_map: dict[str, int] | None = None,
                 per_class_limit: int | None = None,
                 items: list[tuple[Path, int]] | None = None) -> None:
        self.class_map = class_map or DEFAULT_CLASS_MAP
        # Explicit `items` (e.g. a stratified split) take precedence over scanning
        # the dataset's own split directories.
        if items is not None:
            self.items = list(items)
        else:
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
