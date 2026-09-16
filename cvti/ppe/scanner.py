"""The PPE worker: bounded, off the camera path, honest about coverage.

One thread for the whole site. Each camera with a policy is assessed at its
own cadence (default 1 Hz) on the frame the engine ALREADY decoded and the
person boxes + zone membership its tracker ALREADY computed — the camera loop
never waits for this. One inference in flight at a time; if a cycle overruns
its cadence the worker skips ahead to the newest frame and counts the drop,
instead of queueing stale frames that would be judged late.

What it emits: ONE alert per person per violation (a change in what is
missing re-alerts; the same missing set does not). Compliant and unable
people are never alerted — they are counted, and the counts (assessed /
unable / unknown reasons / dropped cycles / detector health) are written to
`ppe_status.json` every cycle so an operator can see when coverage is
degraded rather than trusting a quiet board.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Callable

from cvti.logging_setup import get_logger
from cvti.ppe.assess import (COMPLIANT, UNABLE, VIOLATION, Compliance, TrackEvidence,
                             assess_compliance, observe_people, violation_confidence)
from cvti.ppe.policy import PPEPolicy, load_ppe_policy

log = get_logger(__name__)

SITE_REFRESH_SECONDS = 10.0


class PPEScanner:
    def __init__(self, cameras: list[dict], sink: Any, *,
                 frame_source: Callable[[str], Any],
                 boxes_source: Callable[[str], Any],
                 zones_source: Callable[[str], Any] | None = None,
                 detector: Any = None,
                 site_config_path: str | None = None,
                 status_path: str | Path | None = None,
                 clock: Callable[[], float] = time.time) -> None:
        self.sink = sink
        self.frame_source = frame_source
        self.boxes_source = boxes_source
        self.zones_source = zones_source
        self.site_config_path = site_config_path
        self.status_path = Path(status_path) if status_path else None
        self.clock = clock
        # detector.detect(frame, phrases, floor=...) -> [{phrase, score, box}] | None.
        # None here = build the default open-vocab detector lazily on first
        # use; a trained PPE model wraps into the same shape later.
        self._detector = detector
        self._detector_broken = ""
        self.policies: dict[str, PPEPolicy] = {}
        self._cameras: list[dict] = []
        self.refresh_cameras(cameras)
        self._evidence: dict[tuple[str, int], TrackEvidence] = {}
        # (camera, track) -> {opened_at, last_seen, missing, zone}
        self.open_violations: dict[tuple[str, int], dict] = {}
        self.metrics: dict[str, dict] = {}
        self._last_run: dict[str, float] = {}
        self._last_site_refresh = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.emitted = 0

    # --- configuration ------------------------------------------------------

    def refresh_cameras(self, cameras: list[dict]) -> None:
        policies: dict[str, PPEPolicy] = {}
        for cam in cameras or []:
            try:
                pol = load_ppe_policy(cam)
            except ValueError as exc:
                log.warning(f"[ppe {cam.get('id')}] policy rejected: {exc}")
                continue
            if pol is not None:
                policies[str(cam["id"])] = pol
        gone = set(self.policies) - set(policies)
        for cid in gone:
            for key in [k for k in self._evidence if k[0] == cid]:
                self._evidence.pop(key, None)
            for key in [k for k in self.open_violations if k[0] == cid]:
                self.open_violations.pop(key, None)
        self.policies = policies
        self._cameras = list(cameras or [])

    def _refresh_from_site(self, now: float) -> None:
        if not self.site_config_path or now - self._last_site_refresh < SITE_REFRESH_SECONDS:
            return
        self._last_site_refresh = now
        try:
            site = json.loads(Path(self.site_config_path).read_text())
        except (OSError, ValueError):
            log.debug("[ppe] site re-read failed; keeping current policies", exc_info=True)
            return
        self.refresh_cameras(site.get("cameras", []))

    # --- lifecycle ----------------------------------------------------------

    def start(self) -> "PPEScanner":
        if not self.policies and not self.site_config_path:
            return self
        self._thread = threading.Thread(target=self._loop, name="ppe-compliance", daemon=True)
        self._thread.start()
        names = ", ".join(self.policies) or "none yet — watching the site file"
        log.info(f"[ppe] assessing [{names}]")
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        self._stop.wait(2.0)                    # let the tracker produce boxes
        while not self._stop.is_set():
            now = self.clock()
            self._refresh_from_site(now)
            next_due = now + 1.0
            for cam_id, pol in list(self.policies.items()):
                due = self._last_run.get(cam_id, 0.0) + pol.cadence_seconds
                if now >= due:
                    t0 = time.monotonic()
                    try:
                        self.step(cam_id, now)
                    except Exception:  # noqa: BLE001 - one camera must not stop the worker
                        log.error(f"[ppe {cam_id}] cycle failed", exc_info=True)
                    took = time.monotonic() - t0
                    m = self._metrics_for(cam_id)
                    m["cycle_ms"] = round(took * 1000.0, 1)
                    if took > pol.cadence_seconds:
                        # Overran: the frames we would have judged are stale.
                        # Skip them (the next cycle sees the newest) and say so.
                        m["dropped_cycles"] += int(took // pol.cadence_seconds)
                    self._last_run[cam_id] = self.clock()
                    due = self._last_run[cam_id] + pol.cadence_seconds
                next_due = min(next_due, due)
            self._stop.wait(max(0.05, next_due - self.clock()))

    # --- one cycle ----------------------------------------------------------

    def _metrics_for(self, cam_id: str) -> dict:
        return self.metrics.setdefault(cam_id, {
            "people_seen": 0, "assessed": 0, "compliant": 0, "violations": 0,
            "unable": 0, "unknown_reasons": {}, "not_required": 0,
            "alerts": 0, "dropped_cycles": 0, "cycle_ms": 0.0, "last_frame_at": None,
            "skipped": ""})

    def _ensure_detector(self) -> Any:
        if self._detector is None and not self._detector_broken:
            try:
                from cvti.detector.openvocab import OpenVocabDetector
                self._detector = OpenVocabDetector()
            except Exception as exc:  # noqa: BLE001 - keep counting, visibly degraded
                self._detector_broken = str(exc)[:160]
                log.warning(f"[ppe] detector unavailable ({self._detector_broken})")
        return self._detector

    def step(self, cam_id: str, now: float | None = None) -> list[dict]:
        """Assess one camera once. Returns the violation alerts it emitted."""
        now = self.clock() if now is None else now
        pol = self.policies[cam_id]
        m = self._metrics_for(cam_id)
        frame = self.frame_source(cam_id)
        if frame is None:
            m["skipped"] = "camera gave no frames"
            self._expire(cam_id, pol, now)
            self._write_status()
            return []
        m["skipped"] = ""
        m["last_frame_at"] = now
        people = list(self.boxes_source(cam_id) or [])
        zones = dict((self.zones_source(cam_id) if self.zones_source else None) or {})
        if not people:
            self._expire(cam_id, pol, now)
            self._write_status()
            return []
        detector = self._ensure_detector()
        dets = None
        if detector is not None:
            floor = min(pol.items[k].min_score for k in pol.all_items())
            dets = detector.detect(frame, pol.phrases(), floor=floor)
        frame_hw = frame.shape[:2]
        # A detector that does not say otherwise is treated as zero-shot: its
        # silence on low-recall items is shadowed, never alerted.
        zero_shot = bool(getattr(detector, "zero_shot", True))
        observations = observe_people(people, dets, pol, frame_hw, zones, zero_shot=zero_shot)
        emitted: list[dict] = []
        for obs in observations:
            m["people_seen"] += 1
            key = (cam_id, obs.track_id)
            if not obs.required:
                m["not_required"] += 1
                continue
            ev = self._evidence.get(key)
            if ev is None:
                ev = self._evidence[key] = TrackEvidence(pol.window_seconds, pol.confirm_observations)
            ev.add(obs, now)
            verdicts = {k: ev.verdict(k, now) for k in obs.required}
            comp = assess_compliance(obs.required, verdicts)
            zone = next((z for z in obs.zones if z in pol.zone_required), None)
            if comp.status == VIOLATION:
                m["assessed"] += 1
                m["violations"] += 1
                if self._open_or_update(key, comp, zone, now):
                    record = self._emit(cam_id, pol, frame, obs.track_id, obs.box, zone, comp,
                                        violation_confidence(ev, comp.missing), now, dets)
                    emitted.append(record)
                    m["alerts"] += 1
            elif comp.status == COMPLIANT:
                m["assessed"] += 1
                m["compliant"] += 1
                self._close(key, "compliant", now)
            elif comp.status == UNABLE:
                m["unable"] += 1
                for why in comp.unknown.values():
                    m["unknown_reasons"][why] = m["unknown_reasons"].get(why, 0) + 1
                # An open violation stays open through 'unable' frames — we did
                # not see them put the helmet on; we stopped being able to see.
        self._expire(cam_id, pol, now)
        self._write_status()
        return emitted

    def _open_or_update(self, key: tuple, comp: Compliance, zone: str | None, now: float) -> bool:
        """True when this violation should ALERT (new, or the missing set changed)."""
        cur = self.open_violations.get(key)
        if cur is not None and tuple(cur["missing"]) == tuple(comp.missing):
            cur["last_seen"] = now
            return False
        self.open_violations[key] = {"opened_at": now, "last_seen": now,
                                     "missing": list(comp.missing), "zone": zone}
        return True

    def _close(self, key: tuple, why: str, now: float) -> None:
        if self.open_violations.pop(key, None) is not None:
            log.info(f"[ppe {key[0]}] track {key[1]} violation closed ({why})")

    def _expire(self, cam_id: str, pol: PPEPolicy, now: float) -> None:
        for key in [k for k, ev in self._evidence.items()
                    if k[0] == cam_id and now - ev.last_seen > pol.clear_after_seconds]:
            self._evidence.pop(key, None)
            self._close(key, "track gone", now)

    # --- output -------------------------------------------------------------

    def _emit(self, cam_id: str, pol: PPEPolicy, frame: Any, tid: int, box: tuple,
              zone: str | None, comp: Compliance, confidence: float, now: float,
              dets: list[dict] | None) -> dict:
        from cvti.contracts import VerificationResult
        from cvti.serving.alert_queue import QueuedAlert
        labels = [pol.items[k].label for k in comp.missing]
        where = f" — {zone.replace('_', ' ')}" if zone else ""
        title = f"PPE: MISSING {', '.join(labels).upper()}{where}"
        reason = comp.summary(pol.items)
        notes = [pol.items[k].visual_only_note for k in comp.required if pol.items[k].visual_only_note]
        evidence = _annotate(frame, box, f"#{tid} missing {', '.join(labels)}")
        pixel_box = tuple(int(v) for v in box)
        payload: dict[str, Any] = {"frames": [evidence], "bbox": pixel_box,
                                   "ppe": comp.to_dict(), "zone": zone,
                                   "visual_only_notes": notes}
        engine = "yolo-world"
        det_status = getattr(self._detector, "status", None)
        if callable(det_status):
            try:
                engine = str(det_status().get("weights") or engine)
            except Exception:  # noqa: BLE001 - status is decoration
                pass
        alert = QueuedAlert(camera_id=cam_id, rule_name=f"ppe:{zone or 'camera'}",
                            priority=pol.priority, title=title, timestamp=now,
                            track_id=tid, zone=zone, object_label="ppe", payload=payload)
        result = VerificationResult(confirmed=True, confidence=float(confidence), reason=reason,
                                    alert_priority=pol.priority, timestamp=now,
                                    raw_response=f"ppe-{engine}")
        self.sink.handle(alert, result)
        self.emitted += 1
        log.info(f"[ppe {cam_id}] ALERT track {tid}: {title} ({reason}; conf {confidence:.2f})")
        return {"camera_id": cam_id, "track_id": tid, "zone": zone, "title": title,
                "missing": list(comp.missing), "confidence": confidence, "reason": reason}

    def status(self) -> dict:
        det: dict = {"loaded": False, "error": self._detector_broken}
        st = getattr(self._detector, "status", None)
        if callable(st):
            try:
                det = st()
            except Exception:  # noqa: BLE001
                pass
        cams = {}
        for cid, pol in self.policies.items():
            m = dict(self._metrics_for(cid))
            seen = max(1, m["people_seen"] - m["not_required"])
            m["unable_rate"] = round(m["unable"] / seen, 3)
            m["open_violations"] = sum(1 for k in self.open_violations if k[0] == cid)
            m["degraded"] = bool(m["dropped_cycles"] or m["skipped"] or not det.get("loaded", False))
            m["policy"] = pol.to_dict()
            m["shadow_items"] = list(pol.shadow_items(bool(getattr(self._detector, "zero_shot", True))))
            cams[cid] = m
        return {"updated_at": self.clock(), "detector": det, "cameras": cams}

    def _write_status(self) -> None:
        if self.status_path is None:
            return
        try:
            tmp = self.status_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.status(), indent=1, default=str))
            tmp.replace(self.status_path)
        except OSError:
            log.debug("[ppe] status write failed", exc_info=True)


def _annotate(frame: Any, box: tuple, label: str) -> Any:
    """Evidence copy: the person outlined, what is missing written above them."""
    try:
        import cv2
        out = frame.copy()
        x1, y1, x2, y2 = (int(v) for v in box)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        ty = max(th + 6, y1 - 6)
        cv2.rectangle(out, (x1, ty - th - 6), (x1 + tw + 8, ty + 4), (0, 0, 255), -1)
        cv2.putText(out, label, (x1 + 4, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        return out
    except Exception:  # noqa: BLE001 - an unannotated frame is still evidence
        return frame
