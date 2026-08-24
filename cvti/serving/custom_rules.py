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
                 site_config_path: str | None = None) -> None:
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
        self._last_fire: dict[tuple, float] = {}   # (cam, rule) -> ts

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
        # Cooldown keys for renamed/deleted rules are dead — prune to the live set.
        live = {(c["id"], t["name"]) for c in self.cameras for t in _rules_for(c)}
        for k in [k for k in self._last_fire if k not in live]:
            self._last_fire.pop(k, None)

    def start(self) -> "CustomRuleScanner":
        if not self.cameras and not self.site_config_path:
            return self
        self._thread = threading.Thread(target=self._loop, name="custom-rules", daemon=True)
        self._thread.start()
        names = ", ".join(c["id"] for c in self.cameras) or "none yet — watching the site file"
        log.info(f"[custom-rules] scanning [{names}] every {self.interval:.0f}s")
        return self

    def _open(self, source):
        import cv2
        if str(source).isdigit():
            return cv2.VideoCapture(int(source))
        if str(source).lower().startswith("rtsp"):
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
            return cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        return cv2.VideoCapture(source)

    def _grab(self, cap):
        import cv2
        ok, fr = cap.read()
        if not ok:                                   # loop files at EOF
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, fr = cap.read()
        return fr if ok else None

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
                if c["id"] not in caps:              # camera gained rules: start decoding
                    caps[c["id"]] = self._open(c["source"])
                    log.info(f"[custom-rules] now scanning {c['id']} "
                             f"({len(_rules_for(c))} rule(s))")
                cap = caps.get(c["id"])
                if cap is None:
                    continue
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
                    continue
                dead_since.pop(c["id"], None)
                try:
                    hit = self._check(c, frame)
                except Exception as exc:  # noqa: BLE001 - a scan error must not kill the loop
                    log.info(f"[custom-rules {c['id']}] {str(exc)[:120]}")
                    continue
                if hit and not self._cooling(c["id"], hit["name"]):
                    self._emit(c, frame, hit)
            self._stop.wait(self.interval)
        for cap in caps.values():
            try:
                cap.release()
            except Exception as exc:  # noqa: BLE001
                log.debug("releasing a capture failed during teardown", exc_info=True)
                pass

    def _cooling(self, cam_id: str, name: str) -> bool:
        key = (cam_id, name)
        now = time.time()
        if now - self._last_fire.get(key, 0.0) < self.cooldown:
            return True
        self._last_fire[key] = now
        return False

    def _scene(self, cam_id: str) -> str:
        p = Path("runs/context") / cam_id / "scene_context.json"
        if p.exists():
            try:
                return json.loads(p.read_text()).get("scene_description", "a monitored area")
            except (ValueError, OSError):
                pass
        return "a monitored area"

    def _check(self, cam: dict, frame) -> dict | None:
        import cv2
        threats = _rules_for(cam)
        if not threats:
            return None
        lines = "\n".join(f'- {t["name"]}: {t["description"]}' for t in threats)
        prompt = (
            "You are a security camera analyst. Only report a threat if it is clearly "
            f"happening in the image; otherwise say none.\nScene: {self._scene(cam['id'])}.\n"
            f"Watch specifically for:\n{lines}\n"
            'Reply ONLY compact JSON: {"threat": "<exact threat name that is happening, or none>", '
            '"reason": "<one short sentence of what you see>"}'
        )
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return None
        os.environ.setdefault("OLLAMA_API_KEY", "ollama")
        from cvti.scene.agent_mapper import call_openai_compatible
        raw = call_openai_compatible(prompt=prompt, frame_bytes=buf.tobytes(), model=self.model,
                                     api_key_env="OLLAMA_API_KEY", api_base_url=self.base_url,
                                     require_key=False)
        m = re.search(r"\{.*\}", raw or "", re.S)
        if not m:
            return None
        try:
            d = json.loads(m.group(0))
        except (ValueError, TypeError):
            return None
        threat = str(d.get("threat", "none")).strip().lower()
        if not threat or threat in ("none", "no", "null", "n/a", "nothing"):
            return None
        # Only fire a threat the customer actually defined — never one the model
        # invents. Match by name overlap or description word overlap.
        match = next((t for t in threats
                      if t["name"].lower() in threat or threat in t["name"].lower()), None)
        if match is None:
            desc_words = threat.split()
            match = next((t for t in threats
                          if any(w in t["description"].lower() for w in desc_words if len(w) > 3)), None)
        if match is None:
            return None
        return {"name": match["name"], "reason": str(d.get("reason", ""))[:240]}

    def _emit(self, cam: dict, frame, hit: dict) -> None:
        from cvti.contracts import VerificationResult
        from cvti.serving.alert_queue import QueuedAlert
        alert = QueuedAlert(
            camera_id=cam["id"], rule_name=f"custom:{hit['name']}",
            priority="high", title=f"CUSTOM: {hit['name']}", timestamp=time.time(),
            payload={"frames": [frame]})
        result = VerificationResult(
            confirmed=True, confidence=0.9, reason=hit["reason"],
            alert_priority="high", timestamp=time.time(), raw_response="custom-vlm")
        self.sink.handle(alert, result)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
