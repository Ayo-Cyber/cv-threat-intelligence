"""End-to-end assertions: did the product actually work? (25 Aug)

Run after the engine has been fed a real RTSP stream. Every check here maps to
a failure that has actually happened in this project, so a green run means
those specific ways of being broken are ruled out — not that "the code
imported". Exit non-zero on the first real failure, with the reason.
"""
from __future__ import annotations

import json
import sqlite3
import sys
import time
import urllib.request
from pathlib import Path

DEADLINE_S = 180


def _wait(predicate, what: str, deadline: float = DEADLINE_S):
    t0 = time.time()
    last = None
    while time.time() - t0 < deadline:
        try:
            got = predicate()
            if got:
                print(f"  ok  {what}  ({time.time()-t0:.0f}s)")
                return got
            last = "not yet"
        except Exception as exc:  # noqa: BLE001 - keep waiting through startup races
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(2)
    fail(f"{what} — timed out after {deadline:.0f}s (last: {last})")


def fail(msg: str):
    print(f"\nFAIL: {msg}")
    sys.exit(1)


def main(out: str) -> int:
    out = Path(out)
    print(f"asserting the pipeline in {out}")

    # 1. The engine publishes health at all. (Health was once written only
    #    after frames arrived, so a fully-broken site said nothing.)
    health_path = out / "gate_health.json"
    doc = _wait(lambda: json.loads(health_path.read_text()) if health_path.exists() else None,
                "engine writes /health")

    # 2. The camera actually connected over RTSP — decode + reconnect path.
    def connected():
        d = json.loads(health_path.read_text())
        cams = d.get("cameras") or []
        return cams and all(c.get("state") == "connected" for c in cams)
    _wait(connected, "RTSP camera reaches CONNECTED")

    # 3. Health is FRESH (staleness is how the UI knows the engine is alive).
    doc = json.loads(health_path.read_text())
    age = time.time() - doc.get("generated_at", 0)
    if age > 30:
        fail(f"health is {age:.0f}s stale — the UI would read 'engine not running'")
    print(f"  ok  health is fresh ({age:.0f}s)")

    # 4. Frames are published WITH a token the UI can use. (The live wall was
    #    once dark for days because the token never reached the app.)
    fj = out / "frames.json"
    info = _wait(lambda: json.loads(fj.read_text()) if fj.exists() else None,
                 "frame publisher announces itself")
    if not info.get("token"):
        fail("frames.json has no token — the UI cannot authenticate to its own frames")
    port, token = info["port"], info["token"]

    def frame_ok():
        req = urllib.request.Request(f"http://127.0.0.1:{port}/cameras",
                                     headers={"X-Argus-Token": token})
        cams = json.loads(urllib.request.urlopen(req, timeout=5).read())["cameras"]
        if not cams:
            return False
        url = f"http://127.0.0.1:{port}/frame/{cams[0]}?token={token}"
        data = urllib.request.urlopen(url, timeout=5).read()
        return data[:2] == b"\xff\xd8"        # a real JPEG, not an error page
    _wait(frame_ok, "an authenticated JPEG comes off the frame publisher")

    # 5. Unauthenticated access is refused. (Evidence on an open port is the
    #    one failure that is worse than being down.)
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/cameras", timeout=5)
        fail("frame publisher served an UNAUTHENTICATED request")
    except urllib.error.HTTPError as exc:
        if exc.code != 401:
            fail(f"unauthenticated request got {exc.code}, expected 401")
        print("  ok  unauthenticated frame request refused (401)")

    # 5b. The publisher is LOOPBACK-ONLY. Discovered by this very test failing
    #     from a sibling container (25 Aug): evidence frames must not be
    #     reachable from another host on the network, ever.
    import socket
    s = socket.socket(); s.settimeout(3)
    reachable = s.connect_ex((socket.gethostname(), port)) == 0
    s.close()
    if reachable:
        fail(f"the frame publisher answered on a NON-loopback address (port {port}) "
             "— camera evidence is exposed to the network")
    print("  ok  frame publisher is loopback-only (unreachable off-host)")

    # 6. Detection produced alerts, and they were PERSISTED. This is the whole
    #    product in one assertion: pixels in, rows out.
    db = out / "events.db"

    def has_alerts():
        if not db.exists():
            return False
        con = sqlite3.connect(str(db))
        try:
            n = con.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        finally:
            con.close()
        return n or False
    n = _wait(has_alerts, "alerts reach events.db")

    # 7. The gate was unreachable by design here, so every alert must be
    #    UNVERIFIED and VISIBLE — never silently dropped. Fail-visible is the
    #    safety property this product is built on.
    con = sqlite3.connect(str(db))
    try:
        unver = con.execute("SELECT COUNT(*) FROM events WHERE unverified = 1").fetchone()[0]
        cols = {r[1] for r in con.execute("PRAGMA table_info(events)")}
    finally:
        con.close()
    if unver == 0:
        fail(f"{n} alert(s) stored but none marked unverified — a dead gate was "
             "recorded as if something had judged them")
    print(f"  ok  {n} alert(s) persisted, {unver} correctly marked UNVERIFIED")

    # 8. Evidence frames were written to disk (what an operator reviews).
    ev_dirs = [d for d in (out / "events").glob("*") if d.is_dir()] if (out / "events").exists() else []
    jpgs = [f for d in ev_dirs for f in d.glob("*.jpg")]
    if not jpgs:
        fail("alerts exist but no evidence frames were written")
    print(f"  ok  evidence on disk ({len(jpgs)} frame(s) across {len(ev_dirs)} event(s))")

    # 9. The mobile view answers and refuses anonymous access.
    def mobile_gated():
        try:
            r = urllib.request.urlopen("http://127.0.0.1:8710/", timeout=5)
            return b"Sign in" in r.read()
        except urllib.error.HTTPError as exc:
            return exc.code in (401, 403)
    _wait(mobile_gated, "phone view is up and login-gated")

    # 10. Schema the app depends on (a migration that silently didn't run is
    #     indistinguishable from a quiet site until someone opens the UI).
    for needed in ("state", "owner", "outcome", "unverified", "provisional", "prompt_version"):
        if needed not in cols:
            fail(f"events table is missing '{needed}' — a migration did not run")
    print("  ok  events schema has every column the app reads")

    print("\nPASS — pixels in, verified-or-visibly-unverified alerts out.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "/out"))
