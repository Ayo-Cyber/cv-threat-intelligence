"""Dataset export — turn labeled events into a fine-tuning dataset (offline path).

Each decisive label + its saved evidence becomes a training clip:
  review 'true'  -> positive class (a real threat)
  review 'false' -> negative class (a normal / false-alarm scene)

Output layout mirrors what cvti.training.video_finetune expects — one mp4 per
clip under a per-class folder — plus a manifest.json:

    <out>/threat/<event_id>.mp4
    <out>/normal/<event_id>.mp4
    <out>/manifest.json

Clips with no saved mp4 fall back to encoding their frames. Point the trainer at
`<out>` with `--data-root`.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from cvti.feedback.store import FeedbackStore

from cvti.logging_setup import get_logger

log = get_logger(__name__)

DEFAULT_CLASS_MAP = {"true": "threat", "false": "normal"}


@dataclass
class ExportResult:
    out_dir: str
    total: int
    per_class: dict
    skipped: int

    def to_dict(self) -> dict:
        return {"out_dir": self.out_dir, "total": self.total,
                "per_class": self.per_class, "skipped": self.skipped}


def export_dataset(store: FeedbackStore, out_dir: str | Path,
                   class_map: dict | None = None) -> ExportResult:
    class_map = class_map or DEFAULT_CLASS_MAP
    out = Path(out_dir)
    per_class: dict = {}
    manifest = []
    skipped = 0
    for e in store.labeled_events():
        cls = class_map.get(e.review)
        if not cls or not e.evidence_dir:
            skipped += 1
            continue
        ev = Path(e.evidence_dir)
        clip = ev / "clip.mp4"
        cls_dir = out / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        dst = cls_dir / f"{e.id}.mp4"
        wrote = False
        if clip.exists():
            shutil.copyfile(clip, dst)
            wrote = True
        else:
            wrote = _frames_to_mp4(sorted(ev.glob("frame_*.jpg")), dst)
        if not wrote:
            skipped += 1
            continue
        per_class[cls] = per_class.get(cls, 0) + 1
        manifest.append({"event_id": e.id, "class": cls, "camera_id": e.camera_id,
                         "rule": e.rule, "review": e.review, "file": str(dst.relative_to(out))})
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(
        {"class_map": class_map, "count": len(manifest), "clips": manifest}, indent=2))
    return ExportResult(str(out), len(manifest), per_class, skipped)


def _frames_to_mp4(frames: list[Path], dst: Path, fps: int = 8) -> bool:
    if not frames:
        return False
    try:
        import cv2
        import numpy as np
        first = cv2.imread(str(frames[0]))
        if first is None:
            return False
        h, w = first.shape[:2]
        for fourcc in ("avc1", "mp4v"):
            vw = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
            if vw.isOpened():
                for fp in frames:
                    im = cv2.imread(str(fp))
                    if im is not None:
                        vw.write(im if im.shape[:2] == (h, w) else cv2.resize(im, (w, h)))
                vw.release()
                return True
    except Exception as exc:  # noqa: BLE001
        log.debug("dataset export step failed", exc_info=True)
        return False
    return False
