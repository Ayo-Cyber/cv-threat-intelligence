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

# EP-07-T3: the two critical always-on detectors, measured at last.
# data/critical/<kind>/{threat,normal}/*.mp4 is the curated home for clips
# whatever their origin — self-captured footage, the Seville mock-attack set,
# hand-picked UCF-Crime clips. Curation is deliberate: a dataset's own idea of
# "Robbery" is not the same claim as "a weapon is visible in frame".
CRITICAL_CLIPS = ROOT / "data" / "critical"

# data/ucf_crime/: the official UCF-Crimes layout (category folders). Only the
# categories whose LABEL matches what our detector actually claims are mapped —
# Fighting/Assault are visible violence; Shooting is a visible firearm. Robbery
# is deliberately NOT mapped to weapons: many robbery clips show no weapon at
# all, and counting them as misses would punish the detector for the dataset's
# labelling, not its own performance.
UCF_CRIME = ROOT / "data" / "ucf_crime"
_UCF_THREATS = {"Fighting": "violence", "Assault": "violence", "Shooting": "weapons"}

_KIND_EXPECTS = {"violence": ("violence",), "weapons": ("weapons",),
                 "fire": ("fire_smoke",), "crowd": ("crowd_formation",),
                 "fall": ("fall",), "theft": ("concealment", "video_action", "theft")}

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


def _critical_clips(base: Path | None = None) -> list[EvalClip]:
    """data/critical/<kind>/{threat,normal}/*.mp4 — the curated measurement set."""
    base = Path(base) if base else CRITICAL_CLIPS
    out: list[EvalClip] = []
    if not base.is_dir():
        return out
    for kind_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        kind = kind_dir.name
        for label, is_threat in (("threat", True), ("normal", False)):
            for clip in sorted((kind_dir / label).glob("*.mp4")):
                out.append(EvalClip(str(clip), is_threat, kind if is_threat else "",
                                    f"critical/{kind}",
                                    _KIND_EXPECTS.get(kind, ()) if is_threat else ()))
    return out


def _ucf_crime_clips(base: Path | None = None) -> list[EvalClip]:
    """The official UCF-Crimes folder layout, mapped conservatively."""
    base = Path(base) if base else UCF_CRIME
    out: list[EvalClip] = []
    if not base.is_dir():
        return out
    for cat_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        name = cat_dir.name
        if name in _UCF_THREATS:
            kind = _UCF_THREATS[name]
            for clip in sorted(cat_dir.rglob("*.mp4")):
                out.append(EvalClip(str(clip), True, kind, f"ucf-crime/{name}",
                                    _KIND_EXPECTS.get(kind, ())))
        elif name.lower().startswith(("normal", "testing_normal")):
            for clip in sorted(cat_dir.rglob("*.mp4")):
                out.append(EvalClip(str(clip), False, "", "ucf-crime/normal"))
        # every other category (Arson, Road Accident…) is out of scope, skipped
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
    elif which == "critical":
        clips = _critical_clips()
    elif which == "ucf_crime":
        # UCF clips ride together with the curated set: curation can override
        # or supplement, and the normals pool is shared.
        clips = _ucf_crime_clips() + _critical_clips()
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
