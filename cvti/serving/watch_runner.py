"""Runs the watches: bind plain-English descriptions to tracked people, keep cases.

Sits beside the detection pipeline rather than inside it — a VLM call takes
seconds, so this samples on its own cadence and never holds up detection.

Per cycle, per camera with watches:
    latest frame + the tracker's current person boxes
      -> draw numbered boxes
      -> ask the model which number matches each watch
      -> bind the answer to a track id
      -> open or update that subject's case
      -> alert only when a case OPENS (not on every sighting)
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any

from cvti.serving.watches import CaseBook, Watch, annotate_candidates, build_prompt, parse_matches
from cvti.logging_setup import get_logger

log = get_logger(__name__)


class WatchRunner:
    def __init__(self, cameras: list[dict], states: dict, sink: Any, *, model: str,
                 base_url: str = "http://localhost:11434/v1",
                 interval: float = 10.0, stale_after: float = 60.0,
                 frame_source: Any = None) -> None:
        # only cameras that actually define watches
        self.cameras = [c for c in cameras if c.get("watches")]
        self.states = states or {}
        self.sink = sink
        self.model = model
        self.base_url = base_url
        self.interval = interval
        self.book = CaseBook(stale_after=stale_after)
        # how to get a frame for a camera; injectable so tests need no video
        self.frame_source = frame_source
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.cycles = 0
        self.opened = 0

    # --- one camera, one cycle -------------------------------------------
    def _watches_for(self, cam: dict) -> list[Watch]:
        return [Watch.from_config(w) for w in (cam.get("watches") or [])]

    def _boxes_for(self, cam_id: str) -> list:
        """Current tracked person boxes, from the live per-camera state."""
        state = self.states.get(cam_id)
        by_track = getattr(state, "_box_by_track", None) or {}
        return [(tid, *box) for tid, box in by_track.items()]

    def _ask(self, prompt: str, frame_bytes: bytes) -> str:
        os.environ.setdefault("OLLAMA_API_KEY", "ollama")
        from cvti.scene.agent_mapper import call_openai_compatible
        # 256 output tokens: the answer is a short JSON list of watch matches.
        return call_openai_compatible(prompt=prompt, frame_bytes=frame_bytes,
                                      model=self.model, api_key_env="OLLAMA_API_KEY",
                                      api_base_url=self.base_url, require_key=False,
                                      max_tokens=256)

    def scan_camera(self, cam: dict, frame: Any, now: float | None = None) -> list[dict]:
        """Bind watches to tracked people for one camera. Returns newly opened cases."""
        import cv2
        watches = self._watches_for(cam)
        boxes = self._boxes_for(cam["id"])
        if not watches or not boxes:
            return []                       # nothing to describe, or nobody to bind to
        annotated, mapping = annotate_candidates(frame, boxes)
        # Downscale AFTER annotating so the numbered boxes travel with the
        # frame; 896 long-edge is the model's native tile size, the numbers
        # stay legible.
        from cvti.scene.agent_mapper import downscale_for_vlm
        ok, buf = cv2.imencode(".jpg", downscale_for_vlm(annotated),
                               [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return []
        scene = ""
        state = self.states.get(cam["id"])
        ctx = getattr(state, "scene_context", None) or {}
        if isinstance(ctx, dict):
            from cvti.scene.context_store import render_scene_context
            scene = render_scene_context(ctx)[:400]
        raw = self._ask(build_prompt(watches, len(mapping), scene), buf.tobytes())
        opened = []
        box_by_track = {b[0]: tuple(b[1:]) for b in boxes}      # tid -> (x1,y1,x2,y2)
        for hit in parse_matches(raw, watches, mapping):
            case, is_new = self.book.observe(
                cam["id"], hit["watch"], hit["track_id"],
                bbox=box_by_track.get(hit["track_id"]), reason=hit["reason"], now=now)
            if is_new:
                self.opened += 1
                opened.append(case.to_dict())
                self._alert(cam, frame, case)
        return opened

    def _alert(self, cam: dict, frame: Any, case: Any) -> None:
        """Raise one alert when a case opens — later sightings just update it."""
        if self.sink is None:
            return
        from cvti.contracts import VerificationResult
        from cvti.serving.alert_queue import QueuedAlert
        alert = QueuedAlert(
            camera_id=cam["id"], rule_name=f"watch:{case.watch}", priority="high",
            title=f"WATCH: {case.watch}", timestamp=time.time(), track_id=case.track_id,
            payload={"frames": [frame], "bbox": case.bbox})
        result = VerificationResult(
            confirmed=True, confidence=0.9,
            reason=f"Matched watch '{case.watch}': {case.reason}",
            alert_priority="high", timestamp=time.time(), raw_response="watch-vlm")
        try:
            self.sink.handle(alert, result)
        except Exception as exc:  # noqa: BLE001 - a sink error must not stop watching
            log.error(f"[watch] sink failed: {str(exc)[:110]}", exc_info=True)

    # --- background loop --------------------------------------------------
    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self.cycles += 1
            for cam in self.cameras:
                try:
                    frame = self.frame_source(cam["id"]) if self.frame_source else None
                    if frame is None:
                        continue
                    self.scan_camera(cam, frame)
                except Exception as exc:  # noqa: BLE001
                    log.error(f"[watch] {cam.get('id')} failed: {str(exc)[:110]}", exc_info=True)
            for case in self.book.expire():
                log.info(f"[watch] case closed: {case.watch} #{case.track_id} on "
                      f"{case.camera_id} after {case.duration:.0f}s / {case.sightings} sightings")

    def start(self) -> "WatchRunner":
        if self.cameras:
            self._thread = threading.Thread(target=self._loop, name="watches", daemon=True)
            self._thread.start()
            names = ", ".join(w.get("name", "?") for c in self.cameras for w in c["watches"])
            log.info(f"[watch] following {len(self.cameras)} camera(s): {names}")
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)

    def active_cases(self, now: float | None = None) -> list[dict]:
        """Currently-open cases (subjects seen recently), newest sighting first."""
        cases = sorted(self.book.active(now), key=lambda c: c.last_seen, reverse=True)
        return [c.to_dict() for c in cases]
