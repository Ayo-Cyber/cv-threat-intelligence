"""Custom (customer-written) rule execution — VLM-as-detector.

Built-in detectors are fast CV/pose/video models. Customer rules are written in
plain English ("someone climbing over the counter"), so there's no trained
detector for them — the local VLM IS the detector. This scans each camera that
has custom rules on a slow cadence (default every ~12s, one VLM call), asks the
model whether any of that camera's rules is happening right now IN ITS SCENE
CONTEXT, and fires a confirmed alert if so. Because the VLM already judged it,
the alert goes straight to the sink (no second gate pass). A per-(camera,rule)
cooldown stops it re-firing the same thing every cycle.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from cvti.logging_setup import get_logger

log = get_logger(__name__)


# A hesitant yes is a no (3 Sep, 'not flagging things that aren't it'): the
# scanner asks the model for certainty in its OWN answer and drops claims
# below this. Chosen next to the gate's min_confidence philosophy — the floor
# kills the borderline "reported at 0.5" noise, not real sightings.
MIN_RULE_CONFIDENCE = 0.7

# Incident lifecycle (3 Sep). Reminders for an UNACKNOWLEDGED ongoing
# situation widen geometrically from the base cooldown (90s -> 4.5min ->
# 13.5min, capped at 15) — insistent enough to get a human, quiet enough not
# to bury them. Two consecutive clear scans close the incident; the next
# sighting is genuinely new information and alerts fresh.
REMINDER_WIDENING = 3.0
REMINDER_CAP_SECONDS = 900.0
CLEAR_AFTER_MISSES = 2

# Evidence marks, colour-coded by WHAT was flagged (BGR): the person in blue,
# a loose object in amber, a held instrument (weapon-shaped things) in red —
# so the operator's eye lands on the right thing before reading a word.
TARGET_COLOURS = {"person": (255, 140, 40),
                  "object": (0, 165, 255),
                  "instrument": (0, 0, 255)}

# Only claims the model can actually PLACE earn a located box (4 Sep): it
# anchors on people and on what a person is holding, but its box for a
# free-standing scene object — the operator's bus rule, live — landed nowhere
# near the subject. A wrong box drawn over evidence is worse than none, so an
# object claim is tagged by name and colour in the frame corner instead of
# pretending we know where it is.
LOCATED_TARGETS = ("person", "instrument")

# The VLM says WHAT, the detector says WHERE (4 Sep evening, after a day of
# model boxes landing off-target): when the engine's tracked person boxes are
# available, a person claim snaps to one of THOSE — the model's own box is
# used only to pick WHICH person, never drawn. A model box covering nearly
# the whole frame located nothing (the cap event drew a border around the
# entire image) and grounds nothing.
UNGROUNDED_AREA_FRAC = 0.9
# On a crowded street a garbage model box grazes SOME person box; a graze is
# not a pick. Raised from 0.05 after the first grounded evidence still boxed
# the wrong spot (Dublin, 20:04) — in a crowd, tag-only beats a coin flip.
MIN_SNAP_IOU = 0.15


def _iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def ground_person_box(vlm_pixel_box: tuple, person_boxes: list) -> tuple | None:
    """The detector box a person claim may draw, or None (tag only).

    Exactly one tracked person: unambiguous — theirs. Several: the model's
    box picks which, and only when it genuinely overlaps one (a pick below
    MIN_SNAP_IOU is a guess, and a wrong person boxed as the subject is the
    exact lie this exists to end). Nobody tracked: the claim may still be
    real (detection sampled a different instant) but there is nothing
    trustworthy to point at — tag, don't point.
    """
    boxes = [tuple(b[-4:]) for b in (person_boxes or [])]
    if not boxes:
        return None
    if len(boxes) == 1:
        return boxes[0]
    best = max(boxes, key=lambda b: _iou(vlm_pixel_box, b))
    return best if _iou(vlm_pixel_box, best) >= MIN_SNAP_IOU else None


def _claim_confidence(claim: dict) -> float:
    """The model's stated certainty, 0..1. Absent or malformed = 1.0 — the
    floor tightens compliant answers, it doesn't strand non-compliant ones."""
    raw = claim.get("confidence")
    if raw is None:
        return 1.0
    try:
        return max(0.0, min(1.0, float(raw)))
    except (TypeError, ValueError):
        return 1.0


def _claim_target(claim: dict) -> str:
    target = str(claim.get("target") or "").strip().lower()
    return target if target in TARGET_COLOURS else "object"


def _normalized_box(raw) -> tuple | None:
    """A model-supplied [x1,y1,x2,y2] on the 0-1000 grid, or None.

    Strict on purpose: a fabricated or degenerate box drawn over evidence is
    worse than no box, so anything malformed is dropped silently and the
    frame ships unboxed — exactly the pre-3-Sep behaviour."""
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in raw)
    except (TypeError, ValueError):
        return None
    if not all(0.0 <= v <= 1000.0 for v in (x1, y1, x2, y2)):
        return None
    if x2 - x1 < 5 or y2 - y1 < 5:      # degenerate sliver, not a subject
        return None
    return (x1, y1, x2, y2)


def annotate_hit(frame, hit: dict, person_boxes: list | None = None):
    """(evidence_frame, pixel_box|None): the hit drawn colour-coded on a COPY
    of the frame.

    Where a box may come from, in trust order (4 Sep, twice in one day):
    - a PERSON claim with the engine's tracked person boxes available snaps
      to a DETECTOR box (ground_person_box) — the model's coordinates only
      pick which person, they are never drawn;
    - an INSTRUMENT claim keeps the model's sanity-checked box (until
      grounded open-vocabulary detection lands);
    - an OBJECT claim, an ungrounded near-whole-frame box, or a person claim
      the detector can't corroborate: corner tag naming what was seen, no
      located box. A wrong box over evidence is worse than none.
    No box from the model = the original frame untouched and no pixel box."""
    import cv2
    box = hit.get("box")
    if box is None:
        return frame, None
    target = hit.get("target", "object")
    colour = TARGET_COLOURS.get(target, TARGET_COLOURS["object"])
    label = f"{target}: {hit.get('name', '')}"[:48]
    height, width = frame.shape[:2]

    def _tag_only():
        evidence = frame.copy()
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                             0.5, 1)
        pad = 5
        cv2.rectangle(evidence, (0, 0),
                      (min(width - 1, tw + 2 * pad), th + baseline + 2 * pad),
                      colour, -1)
        cv2.putText(evidence, label, (pad, pad + th),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        return evidence, None

    if target not in LOCATED_TARGETS:
        # Tag, don't point: the sighting is real, its coordinates are not.
        return _tag_only()
    x1 = int(round(box[0] / 1000.0 * width))
    y1 = int(round(box[1] / 1000.0 * height))
    x2 = int(round(box[2] / 1000.0 * width))
    y2 = int(round(box[3] / 1000.0 * height))
    x1, x2 = max(0, min(x1, width - 1)), max(0, min(x2, width - 1))
    y1, y2 = max(0, min(y1, height - 1)), max(0, min(y2, height - 1))
    if x2 <= x1 or y2 <= y1:
        return frame, None
    area_frac = ((x2 - x1) * (y2 - y1)) / float(width * height)
    if target == "person" and person_boxes is not None:
        grounded = ground_person_box((x1, y1, x2, y2), person_boxes)
        if grounded is None:
            return _tag_only()
        x1, y1, x2, y2 = (int(v) for v in grounded)
        x1, x2 = max(0, min(x1, width - 1)), max(0, min(x2, width - 1))
        y1, y2 = max(0, min(y1, height - 1)), max(0, min(y2, height - 1))
        if x2 <= x1 or y2 <= y1:
            return _tag_only()
    elif area_frac >= UNGROUNDED_AREA_FRAC:
        # A border around the whole image is not a location (the cap event).
        return _tag_only()
    evidence = frame.copy()
    cv2.rectangle(evidence, (x1, y1), (x2, y2), colour, 2)
    cv2.putText(evidence, label, (x1 + 3, max(14, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    return evidence, (x1, y1, x2, y2)


def _rules_for(cam: dict) -> list[dict]:
    """Every English rule on a camera, whatever field it arrived in.

    `custom_threats` is the original hand-written config shape
    ({name, description}); `custom_rules` is what the app's DESCRIBE IT IN
    ENGLISH box writes ({question, dwell}). The two never met — the box wrote
    one field, the scanner read the other, and a sentence like "Detect the
    white aeroplane" sat configured while nothing scanned for it (user report,
    23 Aug). Questions become threats named by their own words.
    """
    out = list(cam.get("custom_threats") or [])
    for r in cam.get("custom_rules") or []:
        q = (r.get("question") or "").strip()
        if not q:
            continue
        name = " ".join(q.strip("?!. ").split()[:6]).lower()
        out.append({"name": name, "description": q})
    legacy = cam.get("custom_rule")
    if legacy and (legacy.get("question") or "").strip():
        q = legacy["question"].strip()
        if q not in [t["description"] for t in out]:
            out.append({"name": " ".join(q.strip("?!. ").split()[:6]).lower(),
                        "description": q})
    return out


class CustomRuleScanner:
    def __init__(self, cameras: list[dict], sink, *, model: str,
                 base_url: str = "http://localhost:11434/v1",
                 interval: float = 12.0, cooldown: float = 90.0,
                 site_config_path: str | None = None,
                 frame_source=None, context_provider=None,
                 boxes_source=None) -> None:
        # boxes_source(camera_id) -> [(track_id, x1, y1, x2, y2), ...] pixel
        # person boxes from the engine's tracker: the grounded WHERE for a
        # person claim's evidence box (the VLM's own coordinates only pick
        # which person). None = standalone scanner, model boxes sanity-checked.
        self.boxes_source = boxes_source
        # Person boxes captured at the same instant as the scanned frame —
        # the VLM call between capture and emit takes ~10s, and boxes fetched
        # at emit time describe a DIFFERENT street (Dublin, 20:04: the box
        # landed on the cobbles everyone had walked away from).
        self._scan_boxes: dict = {}
        # frame_source(camera_id) -> frame|None: when the engine provides it,
        # the scanner PEEKS the frames the engine already decoded instead of
        # opening its own VideoCapture per camera — which doubled network
        # bandwidth and decode CPU for every RTSP camera with English rules.
        self.frame_source = frame_source
        self.context_provider = context_provider
        self.site_config_path = site_config_path
        self.cameras = [c for c in cameras if _rules_for(c)]
        self._all_cameras = list(cameras)
        self.sink = sink
        self.model = model
        self.base_url = base_url
        self.interval = interval
        self.cooldown = cooldown
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # One ongoing situation = ONE incident, not an alert every cooldown
        # (3 Sep, 'why do we constantly keep sending that alert?'). A rule
        # that keeps answering yes updates its open incident; it re-alerts
        # only as a REMINDER, at widening intervals, and only while nobody
        # has acknowledged the alert — repetition's real job is making sure
        # a human sees it, and an acknowledged incident has one. Cleared
        # for CLEAR_AFTER_MISSES scans = closed; the next sighting is new
        # information and opens a new incident.
        # (cam, rule) -> {opened_at, last_seen, misses, reminders, next_reminder_at}
        self._incidents: dict[tuple, dict] = {}
        # The heartbeat file. 'My English rule hasn't fired' arrived three
        # times in two days (28-30 Aug) and the product offered no way to tell
        # 'the model answers none every cycle' from 'every call fails' from
        # 'nothing is scanning'. Every cycle now writes what actually happened
        # per camera; the Rules panel shows it live.
        self.status_path: Path | None = None
        self._status: dict = {}
        # Adaptive backoff. During the 29 Aug demo every scanner call timed
        # out for minutes on end: four cameras' gate verifications and the
        # scanner all contend for OLLAMA_NUM_PARALLEL=2 slots, so under load
        # the scanner queues behind long verifies, times out, and immediately
        # queues again — adding pressure to the exact resource it is starving
        # on. On failure the effective interval doubles (cap 10x); one success
        # resets it. Alert verification keeps priority; sentences catch up
        # when the model has headroom.
        self._backoff = 1.0
        # Site-file mtime as of the last look — the fast-path's change signal.
        self._site_mtime: float = 0.0

    def _refresh_cameras(self) -> None:
        """Re-read the site file so a sentence typed in the app starts scanning
        within one cycle — no restart. Cheap: one JSON read per interval."""
        if not self.site_config_path:
            return
        try:
            site = json.loads(Path(self.site_config_path).read_text())
            self._all_cameras = site.get("cameras", [])
        except (OSError, ValueError):
            log.debug("[custom-rules] site re-read failed; keeping current", exc_info=True)
            return
        self.cameras = [c for c in self._all_cameras if _rules_for(c)]
        # Incidents for renamed/deleted rules are dead — prune to the live set.
        live = {(c["id"], t["name"]) for c in self.cameras for t in _rules_for(c)}
        for k in [k for k in self._incidents if k not in live]:
            self._incidents.pop(k, None)

    def start(self) -> "CustomRuleScanner":
        if not self.cameras and not self.site_config_path:
            return self
        self._thread = threading.Thread(target=self._loop, name="custom-rules", daemon=True)
        self._thread.start()
        names = ", ".join(c["id"] for c in self.cameras) or "none yet — watching the site file"
        log.info(f"[custom-rules] scanning [{names}] every {self.interval:.0f}s")
        return self

    def _open(self, source):
        from cvti.serving.capture import open_capture
        return open_capture(source)

    def _grab(self, cap):
        import cv2
        ok, fr = cap.read()
        if not ok:                                   # loop files at EOF
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, fr = cap.read()
        return fr if ok else None

    def _timed_scan(self, c: dict, caps: dict, dead_since: dict) -> None:
        """_scan_camera plus a perf observation — the scanner's entry in
        perf_report.json (4 Sep instrumentation build)."""
        from cvti.serving.perf import BOARD
        _t0 = time.monotonic()
        try:
            self._scan_camera(c, caps, dead_since)
        finally:
            BOARD.observe("english_scan", c.get("id", "?"),
                          (time.monotonic() - _t0) * 1000.0)

    def _scan_camera(self, c: dict, caps: dict, dead_since: dict) -> None:
        if self.frame_source is not None:
            # the engine already decoded this stream — just look at it
            frame = self.frame_source(c["id"])
            if frame is None:
                return
            self._capture_boxes(c["id"])
        else:
            if c["id"] not in caps:              # standalone fallback: own decode
                caps[c["id"]] = self._open(c["source"])
                log.info(f"[custom-rules] now scanning {c['id']} "
                         f"({len(_rules_for(c))} rule(s))")
            cap = caps.get(c["id"])
            if cap is None:
                return
            frame = self._grab(cap)
            if frame is None:
                # The main pipeline's decoder reconnects; this scanner used
                # to hold one dead VideoCapture forever, so the customer's
                # English rules silently stopped scanning that camera after
                # any stream drop. Reopen with a 30s backoff.
                # (Audit 23 Aug, #8.)
                import time as _t
                first = dead_since.setdefault(c["id"], _t.time())
                if _t.time() - first >= 30:
                    log.info(f"[custom-rules {c['id']}] no frames for 30s — reopening stream")
                    try:
                        cap.release()
                    except Exception:  # noqa: BLE001
                        log.debug("release failed", exc_info=True)
                    caps[c["id"]] = self._open(c["source"])
                    dead_since[c["id"]] = _t.time()
                return
            dead_since.pop(c["id"], None)
        try:
            hits = self._check(c, frame)
            self._record(c, hits)
        except Exception as exc:  # noqa: BLE001 - a scan error must not kill the loop
            log.info(f"[custom-rules {c['id']}] {str(exc)[:120]}")
            self._record(c, None, error=str(exc)[:200])
            return
        self._route_hits(c, frame, hits)

    def _capture_boxes(self, cam_id: str) -> None:
        """Snapshot the tracker's person boxes for the frame just captured."""
        if self.boxes_source is None:
            return
        try:
            self._scan_boxes[cam_id] = self.boxes_source(cam_id)
        except Exception:  # noqa: BLE001 - grounding is best-effort
            log.debug("boxes_source failed", exc_info=True)
            self._scan_boxes[cam_id] = None

    def _rule_keys(self) -> set:
        return {(c["id"], t["name"]) for c in self.cameras for t in _rules_for(c)}

    def _site_changed(self) -> bool:
        if not self.site_config_path:
            return False
        try:
            mtime = Path(self.site_config_path).stat().st_mtime
        except OSError:
            return False
        changed = mtime != self._site_mtime
        self._site_mtime = mtime
        return changed

    def _wait_for_next_pass(self, seconds: float, caps: dict, dead_since: dict) -> None:
        """The inter-pass sleep — with a fast path for a freshly typed rule.

        'Typed it, saw it work' (audit 1 Sep): a rule typed in the app used to
        wait out the remainder of this sleep — up to interval x backoff, two
        minutes under load — before its first scan. Now the sleep polls the
        site file's mtime once a second (one stat call); when a change ADDS a
        rule, the camera that gained it is scanned right here, and the sleep
        then resumes to its original deadline, so one typed sentence never
        doubles the whole site's scan rate. Edits and deletions don't cut the
        sleep — only a new sentence has a person watching for its first
        answer.
        """
        deadline = time.monotonic() + seconds
        while not self._stop.is_set() and time.monotonic() < deadline:
            self._stop.wait(min(1.0, max(0.05, deadline - time.monotonic())))
            if not self._site_changed():
                continue
            before = self._rule_keys()
            self._refresh_cameras()
            fresh = self._rule_keys() - before
            if not fresh:
                continue
            fresh_cams = {cam_id for cam_id, _ in fresh}
            log.info(f"[custom-rules] new sentence on {', '.join(sorted(fresh_cams))} "
                     "— scanning now instead of next cycle")
            for c in self.cameras:
                if c["id"] in fresh_cams and not self._stop.is_set():
                    self._timed_scan(c, caps, dead_since)

    def _loop(self) -> None:
        caps: dict = {}
        dead_since: dict = {}     # camera_id -> when frames stopped
        self._stop.wait(self.interval)               # let models load / scene settle
        while not self._stop.is_set():
            self._refresh_cameras()
            wanted = {c["id"] for c in self.cameras}
            for cid in list(caps):                   # camera lost its rules: stop decoding it
                if cid not in wanted:
                    try:
                        caps.pop(cid).release()
                    except Exception:  # noqa: BLE001
                        log.debug("release failed", exc_info=True)
            for c in self.cameras:
                self._timed_scan(c, caps, dead_since)
            self._wait_for_next_pass(self.interval * self._backoff, caps, dead_since)
        for cap in caps.values():
            try:
                cap.release()
            except Exception as exc:  # noqa: BLE001
                log.debug("releasing a capture failed during teardown", exc_info=True)
                pass

    def _record(self, cam: dict, hits, error: str | None = None) -> None:
        """One line of truth per camera per cycle, flushed to status_path."""
        from cvti.health import component
        comp = component(f"english_rules.{cam['id']}")
        entry = self._status.setdefault(cam["id"], {"scans": 0, "hits": 0, "errors": 0})
        entry["scans"] += 1
        entry["rules"] = len(_rules_for(cam))
        # Open incidents, so the Rules panel can say "ongoing: hoodie, 12 min"
        # instead of the operator wondering why the alerts went quiet.
        entry["ongoing"] = [
            {"rule": key[1],
             "for_s": round(time.time() - incident["opened_at"]),
             "reminders": incident["reminders"]}
            for key, incident in self._incidents.items() if key[0] == cam["id"]]
        entry["last_scan_at"] = time.time()
        if error is not None:
            entry["errors"] += 1
            entry["last_error"] = error
            entry["last_outcome"] = "call failed"
            comp.failed(RuntimeError(error))
            self._backoff = min(self._backoff * 2.0, 10.0)
            entry["backoff_s"] = round(self.interval * self._backoff)
        else:
            comp.ok()
            self._backoff = 1.0
            entry.pop("backoff_s", None)
            entry.pop("last_error", None)
            if hits:
                entry["hits"] += len(hits)
                entry["last_hit_at"] = time.time()
                entry["last_outcome"] = "matched: " + ", ".join(h["name"] for h in hits)[:120]
            else:
                entry["last_outcome"] = "model answered none"
        if self.status_path is not None:
            try:
                tmp = self.status_path.with_suffix(".tmp")
                tmp.write_text(json.dumps({"generated_at": time.time(),
                                           "interval_s": self.interval,
                                           "cameras": self._status}))
                tmp.replace(self.status_path)
            except OSError:
                log.debug("english-rules status write failed", exc_info=True)

    def _triage_state(self, cam_id: str, rule_name: str) -> str | None:
        """The newest matching alert's triage state, best-effort."""
        lookup = getattr(self.sink, "triage_state", None)
        if lookup is None:
            return None
        try:
            return lookup(cam_id, f"custom:{rule_name}")
        except Exception:  # noqa: BLE001 - a state lookup must never stop scanning
            log.debug("[custom-rules] triage-state lookup failed", exc_info=True)
            return None

    def _route_hits(self, c: dict, frame, hits: list[dict],
                    now: float | None = None) -> None:
        """Incident lifecycle: first sighting alerts, persistence updates,
        reminders escalate only while unacknowledged, clearance closes."""
        now = time.time() if now is None else now
        cam_id = c["id"]
        hit_names = set()
        for hit in hits:
            hit_names.add(hit["name"])
            key = (cam_id, hit["name"])
            incident = self._incidents.get(key)
            if incident is None:
                # New information: a situation that wasn't there last scan.
                self._incidents[key] = {"opened_at": now, "last_seen": now,
                                        "misses": 0, "reminders": 0,
                                        "next_reminder_at": now + self.cooldown}
                self._emit(c, frame, hit)
                continue
            incident["last_seen"] = now
            incident["misses"] = 0
            state = self._triage_state(cam_id, hit["name"])
            if state in ("acknowledged", "resolved"):
                continue         # a human owns it — re-confirming is noise
            if now >= incident["next_reminder_at"]:
                incident["reminders"] += 1
                interval = min(self.cooldown * (REMINDER_WIDENING
                                                ** incident["reminders"]),
                               REMINDER_CAP_SECONDS)
                incident["next_reminder_at"] = now + interval
                self._emit(c, frame, hit, ongoing_since=incident["opened_at"])
        # Rules scanned this pass but NOT seen: count toward clearance. A rule
        # that was never checked (deleted mid-flight) is pruned by refresh.
        scanned = {t["name"] for t in _rules_for(c)}
        for key in [k for k in self._incidents if k[0] == cam_id]:
            rule = key[1]
            if rule in hit_names or rule not in scanned:
                continue
            incident = self._incidents[key]
            incident["misses"] += 1
            if incident["misses"] >= CLEAR_AFTER_MISSES:
                held = now - incident["opened_at"]
                log.info(f"[custom-rules {cam_id}] incident closed: '{rule}' "
                         f"clear after {held / 60:.1f} min")
                self._incidents.pop(key, None)

    def _scene(self, cam_id: str) -> str:
        from cvti.scene.context_store import render_scene_context
        context = self.context_provider(cam_id) if self.context_provider else None
        return render_scene_context(context)

    def _check(self, cam: dict, frame) -> list[dict]:
        import cv2
        threats = _rules_for(cam)
        if not threats:
            return []
        lines = "\n".join(f'- {t["name"]}: {t["description"]}' for t in threats)
        # Plural on purpose. This used to ask for THE threat (singular), so a
        # camera with several true rules got exactly one answer per cycle —
        # whichever the model found most salient — and the rest were shadowed
        # every scan. Caught live on 27 Aug: an operator wearing glasses AND a
        # hoodie had written a glasses rule; the hoodie rule answered every
        # cycle and the glasses rule never fired once. The evidence frame shows
        # both, plainly.
        # Tightened 3 Sep ('not flagging things that aren't it'): the model
        # must point at concrete visible evidence, say how sure it is, and
        # SHOW us where — near-matches are named as the failure mode they are,
        # and 'none' is framed as the normal answer, not a disappointment.
        prompt = (
            "You are a security camera analyst. Verify each watch item against ONLY "
            f"what is clearly visible in this image.\nScene: {self._scene(cam['id'])}.\n"
            f"Watch specifically for:\n{lines}\n"
            "Check EVERY listed item independently — more than one can be true at once.\n"
            "Rules:\n"
            "- Report an item ONLY when you can point at concrete visible evidence in "
            "THIS image. Near-matches do not count: a dark jacket is not a hoodie, a "
            "phone is not a weapon, a person merely being present does not match an "
            "action.\n"
            "- If you are not sure, do not report it. An empty list is the normal "
            "answer for a normal scene.\n"
            'Reply ONLY compact JSON: {"threats": [{"name": "<exact name from the list>", '
            '"reason": "<one short sentence naming the visible evidence>", '
            '"confidence": <0.0-1.0, certainty in your own answer>, '
            '"target": "person" | "object" | "instrument", '
            '"box": [x1, y1, x2, y2] <the matching person or object, coordinates '
            "normalized to 0-1000 over the image>}]} — an empty list if none."
        )
        from cvti.scene.agent_mapper import call_openai_compatible, downscale_for_vlm
        ok, buf = cv2.imencode(".jpg", downscale_for_vlm(frame),
                               [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return []
        os.environ.setdefault("OLLAMA_API_KEY", "ollama")
        # 256 output tokens: a compact JSON list of a handful of threats with
        # one-sentence reasons — never a CoT ramble queueing behind verifies.
        # No retries and a 120s budget (audit 1 Sep, V2): the scanner reruns
        # every ~12s anyway, and alert verifies must never find every Ollama
        # slot held by a scan that is being patient. A failed cycle is what
        # the adaptive backoff and the heartbeat file are for.
        raw = call_openai_compatible(prompt=prompt, frame_bytes=buf.tobytes(), model=self.model,
                                     api_key_env="OLLAMA_API_KEY", api_base_url=self.base_url,
                                     require_key=False, max_tokens=320,
                                     max_retries=0, timeout=120.0)
        m = re.search(r"\{.*\}", raw or "", re.S)
        if not m:
            return []
        try:
            d = json.loads(m.group(0))
        except (ValueError, TypeError):
            return []
        claims = d.get("threats")
        if claims is None:
            # A model that saw the old prompt in its context, or free-styles the
            # shape, still gets its single answer honoured.
            claims = [{"name": d.get("threat", "none"), "reason": d.get("reason", "")}]
        if not isinstance(claims, list):
            return []
        hits, seen = [], set()
        for c in claims:
            if not isinstance(c, dict):
                continue
            threat = str(c.get("name", "none")).strip().lower()
            if not threat or threat in ("none", "no", "null", "n/a", "nothing"):
                continue
            # Only fire a threat the customer actually defined — never one the
            # model invents. Match by name overlap or description word overlap.
            match = next((t for t in threats
                          if t["name"].lower() in threat or threat in t["name"].lower()), None)
            if match is None:
                # Paraphrase fallback ("black hoodie" for the hoodie rule) —
                # but demand REAL overlap. A single generic word used to be
                # enough, so an invented "person holding a rifle" matched any
                # rule containing "person" and fired as the customer's rule.
                words = [w for w in threat.split() if len(w) > 3]
                def _overlap(t):
                    return sum(w in t["description"].lower() for w in words)
                match = next((t for t in threats
                              if _overlap(t) >= (2 if len(words) > 1 else 1)), None)
            if match is None or match["name"] in seen:
                continue
            # Confidence floor (3 Sep): the model now states certainty in its
            # OWN answer, and a hesitant yes is a no. Absent confidence (an
            # older model free-styling the shape) passes — the floor tightens
            # compliant answers, it doesn't strand non-compliant ones.
            conf = _claim_confidence(c)
            if conf < MIN_RULE_CONFIDENCE:
                log.info(f"[custom-rules {cam['id']}] '{match['name']}' below the "
                         f"confidence floor ({conf:.2f} < {MIN_RULE_CONFIDENCE}) — dropped")
                continue
            seen.add(match["name"])
            hit = {"name": match["name"], "reason": str(c.get("reason", ""))[:240],
                   "confidence": conf}
            box = _normalized_box(c.get("box"))
            if box is not None:
                hit["box"] = box
                hit["target"] = _claim_target(c)
            hits.append(hit)
        return hits

    def _emit(self, cam: dict, frame, hit: dict,
              ongoing_since: float | None = None) -> None:
        from cvti.contracts import VerificationResult
        from cvti.serving.alert_queue import QueuedAlert
        # The evidence points at WHAT the model saw (3 Sep): a person or
        # instrument claim gets its colour-coded box drawn on the evidence
        # copy, and the pixel bbox rides in the payload so the sink's subject
        # shot points at it too. An object claim is corner-tagged, never
        # located (4 Sep — see LOCATED_TARGETS); no box from the model = an
        # unboxed frame, honestly.
        # Grounding uses the boxes captured WITH the frame (see _capture_boxes);
        # a live fetch here would describe the street ten VLM-seconds later.
        if cam["id"] in self._scan_boxes:
            person_boxes = self._scan_boxes[cam["id"]]
        else:
            person_boxes = None
            if self.boxes_source is not None:      # direct-emit path (tests, tools)
                try:
                    person_boxes = self.boxes_source(cam["id"])
                except Exception:  # noqa: BLE001 - grounding is best-effort
                    log.debug("boxes_source failed", exc_info=True)
        evidence, pixel_box = annotate_hit(frame, hit, person_boxes=person_boxes)
        payload = {"frames": [evidence]}
        if pixel_box is not None:
            payload["bbox"] = pixel_box
        title = f"CUSTOM: {hit['name']}"
        reason = hit["reason"]
        if ongoing_since is not None:
            # A reminder, not a discovery: nobody has acknowledged the open
            # incident, so it asks again — and says how long it has waited.
            minutes = max(1, round((time.time() - ongoing_since) / 60))
            title = f"STILL: {hit['name']}"
            reason = f"ongoing {minutes} min, unacknowledged — {hit['reason']}"
        alert = QueuedAlert(
            camera_id=cam["id"], rule_name=f"custom:{hit['name']}",
            priority="high", title=title, timestamp=time.time(),
            payload=payload)
        result = VerificationResult(
            confirmed=True, confidence=float(hit.get("confidence", 0.9)),
            reason=reason,
            alert_priority="high", timestamp=time.time(), raw_response="custom-vlm")
        self.sink.handle(alert, result)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
