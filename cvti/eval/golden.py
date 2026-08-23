"""A frozen, labelled corpus of real gate inputs — the prompt regression set.

The headline precision figure is a function of a string literal in `gate.py`.
Three prompt revisions moved theft precision 37.5% -> 53.3% -> 63.6%: a 26-point
swing on wording alone, with nothing guarding it. Anyone can edit `_QUESTIONS`,
run the app, watch it work, and ship a regression invisibly.

The fix is to be able to re-answer "what did that wording cost us?" in minutes
rather than hours. A full evaluation re-runs YOLO and VideoMAE over every clip
to rediscover the same candidate moments; only the last step — the VLM reading
some frames — depends on the prompt at all.

So the detector stage is frozen once: every candidate the detectors produced,
with its evidence frames and the label of the clip it came from. Replaying that
through a changed prompt measures the prompt, and nothing else.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cvti.logging_setup import get_logger

log = get_logger(__name__)

DEFAULT_GOLDEN_DIR = Path("runs/eval/golden")
MANIFEST = "manifest.json"


@dataclass
class GoldenCase:
    """One candidate the detectors raised, and whether it was really a threat."""
    case_id: str
    clip: str
    is_threat: bool           # ground truth, from the clip's label
    rule_name: str
    detector: str
    priority: str
    title: str
    frames: list = field(default_factory=list)      # paths, relative to the set root
    scene: dict = field(default_factory=dict)
    object_label: str = ""
    person_id: Any = None

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["frames"] = [str(f) for f in self.frames]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "GoldenCase":
        return cls(**d)


class GoldenSetWriter:
    """Captures candidates as an evaluation runs. Idempotent per case id."""

    def __init__(self, root: str | Path = DEFAULT_GOLDEN_DIR) -> None:
        self.root = Path(root)
        self.cases_dir = self.root / "cases"
        self.cases: list[GoldenCase] = []
        self._seen: set[str] = set()
        # Checkpoint (capture is an hour of GPU time — a crash at clip 90 must
        # not cost 90 clips): every case is appended to cases.partial.jsonl the
        # moment it lands, and a new writer resumes from it. write() finalises
        # into the manifest and removes the partial.
        self._partial = self.root / "cases.partial.jsonl"
        if self._partial.exists():
            import json as _json
            for line in self._partial.read_text().splitlines():
                if not line.strip():
                    continue
                case = GoldenCase.from_dict(_json.loads(line))
                self.cases.append(case)
                self._seen.add(case.case_id)

    @property
    def captured_clips(self) -> set:
        """Clips already fully captured (for skip-on-resume)."""
        return {c.clip for c in self.cases}

    def add(self, *, clip_name: str, is_threat: bool, candidate: Any,
            frames: list, scene: dict | None = None) -> GoldenCase | None:
        import cv2

        index = len(self._seen)
        case_id = f"{Path(clip_name).stem}__{getattr(candidate, 'rule_name', 'rule')}__{index:03d}"
        if case_id in self._seen:
            return None
        self._seen.add(case_id)

        case_dir = self.cases_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, frame in enumerate(frames or []):
            rel = f"cases/{case_id}/frame_{i}.jpg"
            cv2.imwrite(str(self.root / rel), frame)
            paths.append(rel)

        case = GoldenCase(
            case_id=case_id, clip=clip_name, is_threat=bool(is_threat),
            rule_name=getattr(candidate, "rule_name", ""),
            detector=getattr(candidate, "detector", ""),
            priority=getattr(candidate, "priority", "high"),
            title=getattr(candidate, "title", ""),
            frames=paths, scene=scene or {},
            object_label=getattr(candidate, "object_label", "") or "",
            person_id=getattr(candidate, "person_id", None))
        self.cases.append(case)
        import json as _json
        self.root.mkdir(parents=True, exist_ok=True)
        with self._partial.open("a") as fh:
            fh.write(_json.dumps(case.to_dict()) + "\n")
        return case

    def write(self, meta: dict | None = None) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / MANIFEST
        path.write_text(json.dumps({
            "meta": meta or {},
            "cases": [c.to_dict() for c in self.cases],
        }, indent=2))
        log.info("golden set: %d case(s) written to %s", len(self.cases), self.root)
        self._partial.unlink(missing_ok=True)   # banked in the manifest now
        return path


class GoldenSet:
    """Reads a frozen set and replays it through a gate."""

    def __init__(self, root: str | Path = DEFAULT_GOLDEN_DIR) -> None:
        self.root = Path(root)
        data = json.loads((self.root / MANIFEST).read_text())
        self.meta: dict = data.get("meta", {})
        self.cases: list[GoldenCase] = [GoldenCase.from_dict(c) for c in data["cases"]]

    def __len__(self) -> int:
        return len(self.cases)

    def load_frames(self, case: GoldenCase) -> list:
        import cv2
        out = []
        for rel in case.frames:
            img = cv2.imread(str(self.root / rel))
            if img is not None:
                out.append(img)
        return out

    def candidate(self, case: GoldenCase):
        from cvti.contracts import CandidateAlert
        return CandidateAlert(
            rule_name=case.rule_name, priority=case.priority, detector=case.detector,
            title=case.title, person_id=case.person_id,
            object_label=case.object_label or None, timestamp=0.0)

    def replay(self, gate, progress=None, *, resume_path: str | Path | None = None,
               limit: int = 0) -> list[dict]:
        """Verify every case with `gate`. Returns per-case verdicts.

        A gate error is recorded as an error, never as a rejection — scoring a
        transport failure as "the model said no" is how a broken gate reports
        excellent precision.

        **Resumable.** With `resume_path`,every verdict is appended to that .jsonl
        the moment it lands, and a later call skips cases already answered — a
        twenty-minute run that dies at case 90 costs six cases, not ninety.
        Error rows are NOT treated as done (a transport failure is not a
        measurement; it is retried), except "no frames on disk", which retrying
        cannot fix. `limit` caps how many NEW verifications this call performs,
        so the full set can be measured in short chunks.
        """
        done: dict = {}
        resume_path = Path(resume_path) if resume_path else None
        if resume_path and resume_path.exists():
            for line in resume_path.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if not row.get("error") or row.get("error") == "no frames on disk":
                    done[row["case_id"]] = row
        out = []
        fresh = 0
        for i, case in enumerate(self.cases, 1):
            if case.case_id in done:
                out.append(done[case.case_id])
                continue
            if limit and fresh >= limit:
                continue
            frames = self.load_frames(case)
            row = {"case_id": case.case_id, "is_threat": case.is_threat,
                   "rule_name": case.rule_name, "confirmed": False, "error": ""}
            if not frames:
                row["error"] = "no frames on disk"
            else:
                try:
                    verdict = gate.verify(frames, self.candidate(case), case.scene)
                    if getattr(verdict, "errored", False):
                        row["error"] = verdict.error
                    else:
                        row["confirmed"] = bool(verdict.confirmed)
                        row["confidence"] = round(float(verdict.confidence), 3)
                        row["reason"] = (verdict.reason or "")[:200]
                except Exception as exc:  # noqa: BLE001 - an error is not a rejection
                    log.debug("replay case errored: %s", case.case_id, exc_info=True)
                    row["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
            fresh += 1
            out.append(row)
            if resume_path:
                with resume_path.open("a") as fh:
                    fh.write(json.dumps(row) + "\n")
            if progress:
                progress(i, len(self.cases), row)
        return out


def score(verdicts: list[dict]) -> dict:
    """Candidate-level precision/recall for the gate stage, with intervals.

    Candidate-level, not clip-level: this measures the prompt's judgement on the
    inputs it was actually given. Clip-level numbers are what NUMBERS.md
    publishes, and mixing the two would invite comparing figures that are not
    comparable.
    """
    from cvti.eval.metrics import wilson_interval

    usable = [v for v in verdicts if not v["error"]]
    tp = sum(1 for v in usable if v["is_threat"] and v["confirmed"])
    fn = sum(1 for v in usable if v["is_threat"] and not v["confirmed"])
    fp = sum(1 for v in usable if not v["is_threat"] and v["confirmed"])
    tn = sum(1 for v in usable if not v["is_threat"] and not v["confirmed"])

    def rate(k, n):
        if not n:
            return None, None, 0
        iv = wilson_interval(k, n)
        return round(k / n, 4), [round(iv[0], 4), round(iv[1], 4)], n

    precision, p_ci, p_n = rate(tp, tp + fp)
    recall, r_ci, r_n = rate(tp, tp + fn)
    return {
        "cases": len(verdicts), "scored": len(usable),
        "errors": len(verdicts) - len(usable),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "precision": precision, "precision_ci": p_ci, "precision_n": p_n,
        "recall": recall, "recall_ci": r_ci, "recall_n": r_n,
    }


def clear(root: str | Path = DEFAULT_GOLDEN_DIR) -> None:
    shutil.rmtree(Path(root), ignore_errors=True)
