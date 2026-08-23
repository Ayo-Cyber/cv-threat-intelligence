"""Live agent mapping — infer each camera's scene once, so the VLM gate reasons
with real context ("retail counter, overhead CCTV") instead of "unknown".

Runs in the background at monitoring startup and reuses the SAME local VLM as the
gate (Ollama Gemma), so there's no extra model / RAM — just one call per camera.
Cameras start with whatever static scene_context they have (usually none) and get
upgraded the moment their inference returns; it never blocks detection.
"""
from __future__ import annotations

import json
import os
import re
import threading

import cv2

from cvti.logging_setup import get_logger

log = get_logger(__name__)

SCENE_PROMPT = (
    "You are configuring a security camera. Look at this frame and reply with ONLY "
    "compact JSON, no prose:\n"
    '{"environment_type": "<one word: retail, warehouse, street, office, parking, '
    'entrance, home, or other>", "scene_description": "<one sentence: what this '
    'camera watches and what normal activity looks like>"}'
)


def _sample_frame(source, at: float = 0.15, width: int = 512) -> bytes | None:
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    try:
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        if n > 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * at))
        ok, fr = cap.read()
    finally:
        cap.release()
    if not ok:
        return None
    h, w = fr.shape[:2]
    if w > width:
        fr = cv2.resize(fr, (width, int(h * width / w)))
    ok2, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes() if ok2 else None


def _parse(raw: str | None) -> dict:
    m = re.search(r"\{.*\}", raw or "", re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            return {"environment_type": str(d.get("environment_type", "unknown"))[:40],
                    "scene_description": str(d.get("scene_description", ""))[:240]}
        except (ValueError, TypeError):
            pass
    return {"environment_type": "unknown", "scene_description": (raw or "")[:160]}


def infer_scene(source, *, model: str,
                base_url: str = "http://localhost:11434/v1",
                api_key_env: str = "OLLAMA_API_KEY") -> dict | None:
    jb = _sample_frame(source)
    if jb is None:
        return None
    os.environ.setdefault(api_key_env, "ollama")
    from cvti.scene.agent_mapper import call_openai_compatible
    raw = call_openai_compatible(prompt=SCENE_PROMPT, frame_bytes=jb, model=model,
                                 api_key_env=api_key_env, api_base_url=base_url,
                                 require_key=False)
    return _parse(raw)


def map_cameras_async(cams_cfg: list[dict], states: dict, *, model: str,
                      base_url: str = "http://localhost:11434/v1") -> threading.Thread:
    """Background: infer + attach scene_context for each camera. Non-blocking."""
    def worker():
        for c in cams_cfg:
            cid = c.get("id")
            try:
                scene = infer_scene(c["source"], model=model, base_url=base_url)
                if scene and cid in states:
                    states[cid].scene_context = scene
                    log.info("[agent-map] %s: %s — %s", cid, scene["environment_type"],
                             scene["scene_description"][:90])
                    # Persist what the mapper learned: the UI's scene panel and
                    # the custom-rule scanner both read this file, and it was
                    # never written by the live path — they showed "a monitored
                    # area" forever after a successful mapping.
                    # (Audit 23 Aug, #7.)
                    try:
                        import json as _json
                        from pathlib import Path as _P
                        d = _P("runs/context") / str(cid)
                        d.mkdir(parents=True, exist_ok=True)
                        (d / "scene_context.json").write_text(_json.dumps(scene, indent=2))
                    except OSError:
                        log.warning("[agent-map] %s: scene file write failed", cid, exc_info=True)
            except Exception as exc:  # noqa: BLE001 - mapping must never break monitoring
                log.warning("[agent-map] %s failed: %s", cid, str(exc)[:100], exc_info=True)
    t = threading.Thread(target=worker, name="agent-map", daemon=True)
    t.start()
    return t
