"""Labeled clip sets for evaluation.

Ground truth is per-clip and binary: does this clip contain the threat, or not.
Two sources, both usable without any manual labelling:

  camnuvem  — the dataset's own *test* split (theft/ vs normal/). VideoMAE was
              fine-tuned on the *training* split only, so this is genuinely
              held out and is what runs/video_finetune/videomae/eval.json used.
  local     — data/test_clips, where the filename carries the label
              (theft_*, violence_*, weapons_* = threat; normal_*, empty_* = normal).

A clip's `expects` names which threat it should raise, so a fire clip isn't
counted as a miss just because the theft detector stayed quiet.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CAMNUVEM_TEST = ROOT / "CamNuvem Robbery Dataset" / "videos" / "samples" / "test"
LOCAL_CLIPS = ROOT / "data" / "test_clips"

# filename prefix -> (is_threat, threat kind)
_LOCAL_LABELS = [
    ("theft_", True, "theft"),
    ("violence_", True, "violence"),
    ("stabbing", True, "violence"),
    ("weapons_", True, "weapons"),
    ("synthetic_fire", True, "fire"),
    ("fire_", True, "fire"),
    ("crowd_", True, "crowd"),
    ("fall_", True, "fall"),
    ("normal_", False, ""),
    ("empty_", False, ""),
]


@dataclass
class EvalClip:
    path: str
    is_threat: bool          # ground truth
    kind: str = ""           # 'theft' | 'violence' | 'weapons' | 'fire' | ''
    source: str = ""         # which set it came from
    expects: tuple = field(default_factory=tuple)   # detectors that may legitimately fire

    @property
    def name(self) -> str:
        return Path(self.path).name


def _camnuvem_clips(limit_per_class: int = 0) -> list[EvalClip]:
    out: list[EvalClip] = []
    for cls, is_threat in (("theft", True), ("normal", False)):
        d = CAMNUVEM_TEST / cls
        if not d.is_dir():
            continue
        clips = sorted(p for p in d.glob("*.mp4"))
        if limit_per_class:
            clips = clips[:limit_per_class]
        for p in clips:
            out.append(EvalClip(str(p), is_threat, "theft" if is_threat else "",
                                "camnuvem-test",
                                ("concealment", "video_action", "theft") if is_threat else ()))
    return out


def _local_clips() -> list[EvalClip]:
    out: list[EvalClip] = []
    if not LOCAL_CLIPS.is_dir():
        return out
    for p in sorted(LOCAL_CLIPS.glob("*.mp4")):
        for prefix, is_threat, kind in _LOCAL_LABELS:
            if p.name.startswith(prefix):
                out.append(EvalClip(str(p), is_threat, kind, "local-clips"))
                break
        # unlabelled filenames are skipped rather than guessed
    return out


def load_dataset(which: str = "camnuvem", limit: int = 0,
                 limit_per_class: int = 0, kind: str = "") -> list[EvalClip]:
    """Build the labeled clip list.

    which: 'camnuvem' (held-out, the honest one) | 'local' | 'all'
    limit: cap the TOTAL clips (interleaved so both classes stay represented).
    kind:  measure one threat only ('fire', 'crowd', 'theft'...) — keeps that
           kind's positives plus all negatives.
    """
    if which == "camnuvem":
        clips = _camnuvem_clips(limit_per_class)
    elif which == "local":
        clips = _local_clips()
    else:
        clips = _camnuvem_clips(limit_per_class) + _local_clips()

    if kind:
        # Measuring ONE threat: keep that kind's positives, and every negative.
        # Without this a theft clip (is_threat=True) counts as a missed fire.
        clips = [c for c in clips if (c.kind == kind) or not c.is_threat]

    if limit and len(clips) > limit:
        # interleave threat/normal so a small --limit still measures both
        threat = [c for c in clips if c.is_threat]
        normal = [c for c in clips if not c.is_threat]
        mixed: list[EvalClip] = []
        while (threat or normal) and len(mixed) < limit:
            if threat:
                mixed.append(threat.pop(0))
            if normal and len(mixed) < limit:
                mixed.append(normal.pop(0))
        clips = mixed
    return clips


def describe(clips: list[EvalClip]) -> dict:
    return {"total": len(clips),
            "threat": sum(1 for c in clips if c.is_threat),
            "normal": sum(1 for c in clips if not c.is_threat),
            "sources": sorted({c.source for c in clips})}
