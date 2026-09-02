"""Local VLM (Ollama) status + model pull — for the app's first-run VLM step.

Qt-free and dependency-free (stdlib urllib), so it's testable headless and safe
to call from a bundled app. Talks to the Ollama HTTP API at localhost:11434,
so it works whether Ollama was installed via brew, the app, or a service — no
dependence on the `ollama` CLI being on PATH.
"""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

from cvti.contracts import LOCAL_VLM_MODEL

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = LOCAL_VLM_MODEL

# model -> {"state": "pulling"|"done"|"error", "percent": int, "detail": str}
_pulls: dict[str, dict] = {}
_lock = threading.Lock()


def _http_json(url: str, timeout: float = 2.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def gate_status(model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST) -> dict:
    """Is the local VLM gate usable right now?

    Returns {ollama, model_present, model, models, mode}. mode is:
      'live'    — server up AND model present (real verification available)
      'no-model'— server up but the model isn't pulled yet
      'offline' — Ollama server not reachable
    """
    try:
        data = _http_json(f"{host}/api/tags")
    except (urllib.error.URLError, OSError, ValueError):
        return {"ollama": False, "model_present": False, "model": model,
                "models": [], "mode": "offline"}
    names = [m.get("name", "") for m in data.get("models", [])]
    present = _model_matches(model, names)
    return {"ollama": True, "model_present": present, "model": model,
            "models": names, "mode": "live" if present else "no-model"}


def _model_matches(model: str, names: list[str]) -> bool:
    # Ollama tags carry ":latest"; treat "gemma3:4b" and "gemma3:4b" (or a
    # bare "gemma3") as matches so status is forgiving of tag suffixes.
    base = model.split(":")[0]
    for n in names:
        if n == model or n.split(":")[0] == base:
            return True
    return False


def pull_progress(model: str = DEFAULT_MODEL) -> dict:
    with _lock:
        return dict(_pulls.get(model, {"state": "idle", "percent": 0, "detail": ""}))


def start_pull(model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST) -> dict:
    """Kick off a background model download via Ollama's streaming pull API.

    Returns immediately with the current progress record; poll pull_progress().
    A pull already in flight is not restarted.
    """
    with _lock:
        cur = _pulls.get(model)
        if cur and cur.get("state") == "pulling":
            return dict(cur)
        _pulls[model] = {"state": "pulling", "percent": 0, "detail": "starting"}
    t = threading.Thread(target=_pull_worker, args=(model, host), daemon=True)
    t.start()
    return pull_progress(model)


def _set(model: str, **kw) -> None:
    with _lock:
        _pulls.setdefault(model, {"state": "pulling", "percent": 0, "detail": ""}).update(kw)


def _pull_worker(model: str, host: str) -> None:
    body = json.dumps({"name": model, "stream": True}).encode()
    req = urllib.request.Request(f"{host}/api/pull", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=None) as r:
            for raw in r:  # newline-delimited JSON progress events
                line = raw.decode(errors="replace").strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                if ev.get("error"):
                    _set(model, state="error", detail=str(ev["error"])[:160])
                    return
                total, done = ev.get("total"), ev.get("completed")
                pct = int(done * 100 / total) if total and done else None
                _set(model, detail=str(ev.get("status", ""))[:80],
                     **({"percent": pct} if pct is not None else {}))
        _set(model, state="done", percent=100, detail="ready")
    except (urllib.error.URLError, OSError) as exc:
        _set(model, state="error", detail=f"{exc}"[:160])
