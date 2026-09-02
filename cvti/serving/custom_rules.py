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
                 site_config_path: str | None = None,
                 frame_source=None, context_provider=None) -> None:
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
        self._last_fire: dict[tuple, float] = {}   # (cam, rule) -> ts
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
        from cvti.serving.capture import open_capture
        return open_capture(source)

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
                if self.frame_source is not None:
                    # the engine already decoded this stream — just look at it
                    frame = self.frame_source(c["id"])
                    if frame is None:
                        continue
                    try:
                        hits = self._check(c, frame)
                        self._record(c, hits)
                    except Exception as exc:  # noqa: BLE001 - a scan error must not kill the loop
                        log.info(f"[custom-rules {c['id']}] {str(exc)[:120]}")
                        self._record(c, None, error=str(exc)[:200])
                        continue
                    for hit in hits:
                        if not self._cooling(c["id"], hit["name"]):
                            self._emit(c, frame, hit)
                    continue
                if c["id"] not in caps:              # standalone fallback: own decode
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
                    hits = self._check(c, frame)
                    self._record(c, hits)
                except Exception as exc:  # noqa: BLE001 - a scan error must not kill the loop
                    log.info(f"[custom-rules {c['id']}] {str(exc)[:120]}")
                    self._record(c, None, error=str(exc)[:200])
                    continue
                for hit in hits:
                    if not self._cooling(c["id"], hit["name"]):
                        self._emit(c, frame, hit)
            self._stop.wait(self.interval * self._backoff)
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

    def _cooling(self, cam_id: str, name: str) -> bool:
        key = (cam_id, name)
        now = time.time()
        if now - self._last_fire.get(key, 0.0) < self.cooldown:
            return True
        self._last_fire[key] = now
        return False

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
        prompt = (
            "You are a security camera analyst. Only report a threat if it is clearly "
            f"happening in the image; otherwise report none.\nScene: {self._scene(cam['id'])}.\n"
            f"Watch specifically for:\n{lines}\n"
            "Check EVERY listed item independently — more than one can be true at once.\n"
            'Reply ONLY compact JSON: {"threats": [{"name": "<exact name from the list>", '
            '"reason": "<one short sentence of what you see>"}]} — an empty list if none.'
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
                                     require_key=False, max_tokens=256,
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
            seen.add(match["name"])
            hits.append({"name": match["name"], "reason": str(c.get("reason", ""))[:240]})
        return hits

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
