"""Open-vocabulary detection for English rules (W3) — YOLO-World.

The custom-rule scanner's VLM answers every English sentence today, which is
why object/attribute rules inherit the VLM's failure modes: cap false
positives, glasses misses, a "bus" box drawn around the whole frame. An
object/attribute phrase ("person wearing a cap", "white bus") is a DETECTION
question — a grounded open-vocab detector answers it with a real box, a real
score, in milliseconds, every frame if asked. Scene/behaviour sentences
("someone climbing over the counter") stay with the VLM: they need reasoning
about motion and intent, which no single-frame detector has.

Routing is decided per rule by `route_rule` — deliberately conservative: only
a sentence that is CLEARLY about visible appearance routes to the detector;
anything with behaviour, interaction, or time in it keeps the VLM path, so
existing scene rules cannot change behaviour by accident.

Runtime note: YOLO-World needs the CLIP text encoder ONCE per phrase set (at
rule creation/change), then detection is pure YOLO. `set_classes` downloads
CLIP weights on first use — an offline pilot box needs them vendored like the
YOLO weights themselves (W8 packaging concern, flagged in the PR).
"""
from __future__ import annotations

import re
import threading
import time
from pathlib import Path
from typing import Any

from cvti.logging_setup import get_logger

log = get_logger(__name__)

WEIGHTS_DEFAULT = "models/yolov8s-worldv2.pt"

# The confidence floor for a grounded claim. YOLO-World phrase scores are not
# calibrated like the VLM's self-reported certainty; 0.30 kills the sub-0.2
# noise floor open-vocab models emit on busy scenes while keeping real
# attribute hits (typically 0.35-0.8). The attribute manifest (W6) is what
# tunes this number with measurements instead of vibes.
MIN_SCORE = 0.30

# --- routing ---------------------------------------------------------------
# Behaviour/interaction/time words: ANY of these keeps the rule on the VLM
# path. Gerunds are matched as words so "building" (noun) does not trip
# "build". This list errs long: a wrongly-VLM'd object rule merely stays as
# accurate as today, while a wrongly-detector'd behaviour rule would go blind.
_BEHAVIOUR = re.compile(
    r"\b("
    r"climb\w*|fight\w*|punch\w*|kick\w*|chas\w*|run\w*|sneak\w*|hid\w*|"
    r"crawl\w*|jump\w*|fall\w*|throw\w*|break\w*|smash\w*|enter\w*|exit\w*|"
    r"leav\w*|steal\w*|theft|rob\w*|shoplift\w*|conceal\w*|tamper\w*|"
    r"vandal\w*|trespass\w*|loiter\w*|linger\w*|wait\w*|dwell\w*|"
    r"fleeing|flee\w*|attack\w*|assault\w*|struggl\w*|wrestl\w*|push\w*|"
    r"shov\w*|grab\w*|snatch\w*|point\w*|aim\w*|threat\w*|argu\w*|shout\w*|"
    r"panic\w*|collaps\w*|faint\w*|smok\w*|fire|flame\w*|"
    r"crowd\w*|gather\w*|group of|more than|fewer than|count\w*|"
    r"after hours|at night|during|unattended|abandoned|left behind|"
    r"follow\w*|approach\w*|open\w*|clos\w*|touch\w*|reach\w*"
    r")\b", re.IGNORECASE)

# Appearance/object markers: at least one of these must be present for the
# detector route — attribute verbs (wearing/carrying/holding/with), colours,
# clothing and carried-object nouns, vehicles, animals.
_ATTRIBUTE = re.compile(
    r"\b("
    r"wear\w*|carry\w*|hold\w*|with a|with an|in a|dressed|"
    r"red|blue|green|white|black|yellow|orange|grey|gray|brown|pink|purple|"
    r"cap|hat|hood\w*|helmet|mask|glasses|sunglasses|scarf|glove\w*|"
    r"backpack|bag|handbag|suitcase|luggage|box|package|parcel|umbrella|"
    r"ladder|tool\w*|crowbar|"
    r"jacket|coat|vest|uniform|shirt|trousers|shorts|skirt|dress|"
    r"bus|car|truck|lorry|van|vehicle|bicycle|bike|motorcycle|scooter|"
    r"aeroplane|airplane|plane|drone|boat|"
    r"dog|cat|animal"
    r")\b", re.IGNORECASE)


def route_rule(description: str) -> str:
    """'openvocab' when the sentence is purely about visible appearance,
    'vlm' for everything else (behaviour, interaction, time, or unclear)."""
    text = (description or "").strip()
    if not text:
        return "vlm"
    if _BEHAVIOUR.search(text):
        return "vlm"
    if _ATTRIBUTE.search(text):
        return "openvocab"
    return "vlm"


# --- the detector ----------------------------------------------------------

def _phrase_for(description: str) -> str:
    """The open-vocab class string for a rule sentence.

    YOLO-World wants a noun phrase, not an instruction: strip the imperative
    scaffolding customers type ("Detect the...", "Alert me if you see...")."""
    text = (description or "").strip().strip("?!.")
    text = re.sub(r"^(detect|find|spot|watch for|look for|alert( me)?( if| when)?"
                  r"( you see| there is| there's)?|is there|see if)\s+",
                  "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(a|an|the)\s+", "", text, flags=re.IGNORECASE)
    return text or description


class OpenVocabDetector:
    """YOLO-World behind a phrase cache: set_classes only when phrases change.

    `model_factory` is the test seam — production leaves it None and gets
    ultralytics' YOLOWorld, lazily, so importing this module costs nothing.
    """

    def __init__(self, weights: str = WEIGHTS_DEFAULT, *,
                 min_score: float = MIN_SCORE, imgsz: int = 640,
                 device: str = "cpu", model_factory: Any = None) -> None:
        self.weights = weights
        self.min_score = min_score
        self.imgsz = imgsz
        # CPU by default: the scanner cadence is ~12s, a few hundred ms is
        # nothing there, and the pilot box has no CUDA. mps/cuda callers opt in.
        self.device = device
        self._factory = model_factory
        self._model = None
        self._classes: tuple = ()
        self._lock = threading.Lock()
        self.last_ms: float = 0.0
        self.load_error: str = ""

    def _ensure(self, phrases: tuple) -> bool:
        with self._lock:
            if self._model is None:
                try:
                    if self._factory is not None:
                        self._model = self._factory(self.weights)
                    else:
                        from ultralytics import YOLOWorld
                        self._model = YOLOWorld(self.weights)
                except Exception as exc:  # noqa: BLE001 - a missing model must not kill the scanner
                    self.load_error = str(exc)[:200]
                    log.warning(f"openvocab unavailable ({self.load_error}); "
                                "object rules fall back to the VLM", exc_info=True)
                    return False
            if phrases != self._classes:
                try:
                    self._model.set_classes(list(phrases))
                    self._classes = phrases
                except Exception as exc:  # noqa: BLE001 - CLIP missing/offline
                    self.load_error = str(exc)[:200]
                    log.warning("openvocab set_classes failed; object rules fall "
                                "back to the VLM", exc_info=True)
                    return False
        return True

    def detect(self, frame: Any, phrases: list[str]) -> list[dict] | None:
        """Grounded detections for `phrases` on one frame.

        Returns None when the detector cannot answer (model/CLIP missing) —
        the caller's signal to fall back to the VLM path. An answered frame
        with nothing in it returns [] — a real, grounded 'no'."""
        wanted = tuple(sorted({p for p in phrases if p}))
        if not wanted or not self._ensure(wanted):
            return None
        t0 = time.monotonic()
        try:
            res = self._model.predict(frame, device=self.device, conf=self.min_score,
                                      imgsz=self.imgsz, verbose=False)[0]
        except Exception as exc:  # noqa: BLE001 - inference failure is an unanswered cycle
            self.load_error = str(exc)[:200]
            log.warning("openvocab inference failed", exc_info=True)
            return None
        self.last_ms = (time.monotonic() - t0) * 1000.0
        out = []
        boxes = getattr(res, "boxes", None)
        names = getattr(res, "names", {}) or {}
        if boxes is None:
            return out
        for i in range(len(boxes)):
            cls_i = int(boxes.cls[i])
            name = names[cls_i] if isinstance(names, dict) else names[cls_i]
            score = float(boxes.conf[i])
            x1, y1, x2, y2 = (float(v) for v in boxes.xyxy[i][:4])
            out.append({"phrase": str(name), "score": score,
                        "box": (x1, y1, x2, y2)})
        return out

    def status(self) -> dict:
        return {"loaded": self._model is not None and not self.load_error,
                "classes": list(self._classes), "last_ms": round(self.last_ms, 1),
                "error": self.load_error, "weights": Path(self.weights).name}
