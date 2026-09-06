"""A mock Argus Engine API — canned data, no engine, no cameras.

`python -m cvti.api --mock` serves realistic responses in the exact shapes of
contract v0.2, and the WebSocket emits a fresh fake alert every few seconds.
Demi's Electron app builds and demos the whole UI against this today, while
the real endpoints grow behind the same contract.

Any username/password is accepted (role owner) so the login screen is testable.
"""

from __future__ import annotations

from typing import Optional

import asyncio
import json
import time

from fastapi import Depends, FastAPI, Header, HTTPException, Query, WebSocket, WebSocketDisconnect

from cvti.api.app import API_PREFIX, _iso, _send, register_index
from cvti.api.tokens import TokenStore

_CAMERAS = [
    {"id": "Dublin Street", "source": "rtsp://***@10.0.0.11/stream1", "view_only": False,
     "state": "connected", "last_frame_age_s": 0.2, "reconnects": 0,
     "ingest": {"width": 1280, "height": 720, "source_fps": 30, "sampling_fps": 12, "limited": False}},
    {"id": "Loading Bay 3", "source": "rtsp://***@10.0.0.12/stream2", "view_only": False,
     "state": "connected", "last_frame_age_s": 0.3, "reconnects": 1,
     "ingest": {"width": 1920, "height": 1080, "source_fps": 25, "sampling_fps": 6, "limited": True}},
    {"id": "Warehouse A", "source": "rtsp://***@10.0.0.13/stream1", "view_only": True,
     "state": "connected", "last_frame_age_s": 0.1, "reconnects": 0,
     "ingest": {"width": 1280, "height": 720, "source_fps": 15, "sampling_fps": 1, "view_only": True, "limited": False}},
]

_RULES = ["custom:black hoodie", "loitering_watch", "video_theft_candidate", "baseline_fire_smoke"]
_REASONS = ["a person in a black hoodie, right side", "person dwelling in the loading bay",
            "a parked vehicle, not theft", "multiple people moving quickly"]


def _health() -> dict:
    return {
        "status": "degraded", "reasons": ["disk 78.0% used"], "version": "mock-0.2",
        "uptime_s": 4212, "generated_at": time.time(),
        "engine": {"phase": "monitoring", "cameras": 3, "frames_processed": 91234, "alerts_queued": 0},
        "gate": {"provider": "ollama", "model": "gemma3:4b", "reachable": True,
                 "verified": 58, "confirmed": 44, "rejected": 14, "unverified": 0},
        "disk": {"used_pct": 78.0, "free_gb": 120.4},
        "memory": {"available_gb": 9.1, "level": "ok"},
        "cameras": _CAMERAS,
    }


def _mock_event(i: int) -> dict:
    eid = f"evt_{9000 + i}"
    return {
        "id": eid, "ts": time.time(), "iso": _iso(time.time()),
        "camera_id": _CAMERAS[i % len(_CAMERAS)]["id"],
        "rule": _RULES[i % len(_RULES)], "priority": ("critical" if i % 5 == 0 else "high"),
        "confidence": 0.8 + (i % 3) * 0.05, "reason": _REASONS[i % len(_REASONS)],
        "zone": None, "verdict": ("rejected" if i % 4 == 0 else "confirmed"),
        "evidence": {"dir": f"/mock/{eid}", "thumb": f"{API_PREFIX}/events/{eid}/evidence/thumb", "clip": True},
        "triage": {"state": "new"},
    }


def create_mock_app() -> FastAPI:
    app = FastAPI(title="Argus Engine API (mock)", version="0.2.0-mock")
    app.state.tokens = TokenStore()
    app.state.counter = {"n": 24}

    def require(authorization: Optional[str] = Header(default=None)):
        token = authorization[7:].strip() if authorization and authorization.lower().startswith("bearer ") else None
        if app.state.tokens.resolve(token) is None:
            raise HTTPException(status_code=401, detail="unauthorized")
        return True

    @app.post(f"{API_PREFIX}/auth/session")
    async def sign_in(body: dict):
        token, p = app.state.tokens.mint(body.get("username", "demo"), "owner")
        return {"token": token, "expires_at": _iso(p.expires_at),
                "user": {"username": p.username, "role": "owner",
                         "permissions": ["control_engine", "configure_site", "manage_users", "view_alerts"]}}

    @app.get(f"{API_PREFIX}/auth/me")
    async def me(_=Depends(require)):
        return {"username": "demo", "role": "owner", "permissions": ["view_alerts"]}

    @app.get(f"{API_PREFIX}/system/health")
    async def health(_=Depends(require)):
        return _health()

    @app.get(f"{API_PREFIX}/system/info")
    async def info(_=Depends(require)):
        return {"version": "mock-0.2", "node_id": "mock", "platform": "mock",
                "capabilities": ["events", "health", "live_mjpeg", "websocket"]}

    @app.get(f"{API_PREFIX}/monitor")
    async def monitor(_=Depends(require)):
        return {"running": True, "starting": False, "phase": "monitoring", "health_age_s": 1.0}

    @app.get(f"{API_PREFIX}/cameras")
    async def cameras(_=Depends(require)):
        return _CAMERAS

    @app.get(f"{API_PREFIX}/cameras/{{cid}}")
    async def camera(cid: str, _=Depends(require)):
        return next((c for c in _CAMERAS if c["id"] == cid),
                    {"error": {"code": "not_found", "message": cid}})

    @app.get(f"{API_PREFIX}/events")
    async def events(_=Depends(require), limit: int = Query(50, ge=1, le=200)):
        n = app.state.counter["n"]
        return {"events": [_mock_event(i) for i in range(n, max(0, n - limit), -1)],
                "next_cursor": None}

    @app.get(f"{API_PREFIX}/events/{{eid}}")
    async def event(eid: str, _=Depends(require)):
        return _mock_event(int(eid.removeprefix("evt_")) - 9000 if eid.removeprefix("evt_").isdigit() else 1)

    @app.get(f"{API_PREFIX}/triage")
    async def triage(_=Depends(require)):
        return {"to_review": 7, "total": app.state.counter["n"], "by_priority": {"critical": 3, "high": 21}}

    @app.get(f"{API_PREFIX}/cameras/{{cid}}/stream")
    async def stream(cid: str, _=Depends(require)):
        return {"kind": "mjpeg", "url": f"http://127.0.0.1:5599/stream/{cid}?token=mock"}

    @app.websocket(f"{API_PREFIX}/stream")
    async def ws(ws: WebSocket, token: Optional[str] = Query(None)):
        if app.state.tokens.resolve(token) is None:
            await ws.close(code=4401); return
        await ws.accept()
        await _send(ws, "health", _health())
        await _send(ws, "triage", {"to_review": 7, "total": app.state.counter["n"], "by_priority": {}})
        try:
            while True:
                await asyncio.sleep(4.0)
                app.state.counter["n"] += 1
                await _send(ws, "alert.new", _mock_event(app.state.counter["n"]))
        except WebSocketDisconnect:
            return

    register_index(app, mock=True)
    return app
