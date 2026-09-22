"""Boot the PACKAGED desktop app cold and make it do a customer's first hour.

Until 22 Sep the only thing CI launched on Windows was the retired PyQt
`Argus.exe --smoke`; the Electron shell a customer double-clicks was checked
for FILES (packaged_app_check.py). So "it might not start on Windows" could
not be answered by anyone. This runs the shell itself, from the packaged
tree, with a clean data directory, and asserts the chain a first user walks:

  shell starts -> it spawns argus-api -> the API answers -> first owner is
  created -> sign in -> a camera is added -> the engine is started -> the
  engine reports that camera CONNECTED.

Every step has a deadline and a plain-English failure. On failure the app's
own output, the API support log and the engine's monitor.log are printed, so
the build log is the diagnosis. Usage: packaged_app_boot.py <release dir>.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from packaged_app_check import _find_app

ROOT = Path(__file__).resolve().parents[2]
API_UP_S = 180.0          # main.ts gives a packaged API 180 s to come up
ENGINE_CONNECT_S = 300.0  # cold engine: model load on a CPU runner, then decode


def app_binary(app_root: Path, platform: str = sys.platform) -> Path:
    """The executable electron-builder produced for this OS."""
    if platform == "darwin":
        return app_root / "Contents" / "MacOS" / "Argus"
    if platform == "win32":
        return app_root / "Argus.exe"
    for name in ("argus", "Argus"):
        if (app_root / name).exists():
            return app_root / name
    return app_root / "argus"


def camera_connected(health: dict | None) -> bool:
    """The engine's own health doc says at least one camera is streaming."""
    return any(c.get("state") == "connected" for c in ((health or {}).get("cameras") or []))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _call(method: str, url: str, body: dict | None = None, token: str | None = None,
          timeout: float = 8.0) -> tuple[int, dict | list | None]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"content-type": "application/json",
                                          **({"authorization": f"Bearer {token}"} if token else {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            return exc.code, json.loads(raw)
        except ValueError:
            return exc.code, {"raw": raw[:300].decode(errors="replace")}
    except (urllib.error.URLError, OSError, TimeoutError):
        return 0, None


def _kill_tree(pid: int) -> None:
    try:
        import psutil
    except ImportError:
        return
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    procs = parent.children(recursive=True) + [parent]
    for p in procs:
        try:
            p.terminate()
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(procs, timeout=20)
    for p in alive:
        try:
            p.kill()
        except psutil.Error:
            pass


def _tail(path: Path, n: int = 40) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return f"    ({path.name}: not written)"
    return "\n".join(f"    {ln}" for ln in lines[-n:]) or f"    ({path.name}: empty)"


def main(release_dir: str) -> int:
    app_root, _ = _find_app(Path(release_dir).resolve())
    binary = app_binary(app_root)
    if not binary.exists():
        print(f"FAIL: packaged app binary missing: {binary}")
        return 1

    cmd = [str(binary)]
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        if shutil.which("xvfb-run"):
            cmd = ["xvfb-run", "-a", *cmd]
        else:
            print("SKIP: no display and no xvfb-run on this Linux runner — the shell cannot open")
            return 0

    clips = sorted((ROOT / "data" / "test_clips").glob("*.mp4"))
    tmp = Path(tempfile.mkdtemp(prefix="argus-boot-"))
    site_dir = tmp / "site"
    site_dir.mkdir()
    port = free_port()
    env = dict(os.environ,
               ARGUS_API_PORT=str(port),
               ARGUS_SITE_CONFIG=str(site_dir / "site.json"),
               ARGUS_DB=str(site_dir / "events.db"))
    if sys.platform == "win32":
        # Both the shell (install-layout) and the Python side (user_data_dir)
        # derive their data root from APPDATA: point them at the sandbox too.
        (tmp / "appdata").mkdir()
        env["APPDATA"] = str(tmp / "appdata")

    base = f"http://127.0.0.1:{port}/api/v1"
    problems: list[str] = []
    timings: dict[str, float] = {}
    out_path, err_path = tmp / "app.stdout", tmp / "app.stderr"
    print(f"launching: {binary} (API on :{port}, data in {tmp})")
    t0 = time.monotonic()
    with out_path.open("w", encoding="utf-8", errors="replace") as out_f, \
            err_path.open("w", encoding="utf-8", errors="replace") as err_f:
        proc = subprocess.Popen(cmd, env=env, cwd=str(tmp), stdout=out_f, stderr=err_f)
    token = None
    try:
        # 1. the shell brings up its API
        deadline = time.monotonic() + API_UP_S
        state = None
        while time.monotonic() < deadline and proc.poll() is None:
            code, state = _call("GET", f"{base}/auth/state", timeout=3.0)
            if code == 200:
                break
            time.sleep(1.0)
        if proc.poll() is not None:
            problems.append(f"the app exited by itself with code {proc.returncode} before its API answered")
        elif not isinstance(state, dict):
            problems.append(f"the app never brought its API up on :{port} within {API_UP_S:.0f} s")
        else:
            timings["api_up_s"] = round(time.monotonic() - t0, 1)
            print(f"  API answered after {timings['api_up_s']} s: configured={state.get('configured')}")
            if state.get("configured"):
                problems.append("a fresh data directory already reports configured=true — first-run state leaked")

        # 2. first owner + sign in
        if not problems:
            code, made = _call("POST", f"{base}/auth/first-owner",
                               {"username": "smoke", "password": "Smoke-Pass-2026!"})
            if code != 200 or not (isinstance(made, dict) and made.get("ok")):
                problems.append(f"first owner could not be created: {code} {made}")
        if not problems:
            code, sess = _call("POST", f"{base}/auth/session",
                               {"username": "smoke", "password": "Smoke-Pass-2026!"})
            token = (sess or {}).get("token") if isinstance(sess, dict) else None
            if code != 200 or not token:
                problems.append(f"sign-in failed for the owner just created: {code} {sess}")
            else:
                print("  first owner created and signed in")

        # 3. a camera, then the engine
        if not problems and not clips:
            print("  (no data/test_clips in the checkout — boot and sign-in proven, engine not exercised)")
        elif not problems:
            code, cams = _call("GET", f"{base}/cameras", token=token)
            if code != 200 or cams:
                problems.append(f"camera list on a fresh site was not empty: {code} {cams}")
        if not problems and clips:
            cam = {"id": "boot_cam", "source": str(clips[0]),
                   "config": "configs/all_threats_v1.json"}
            code, added = _call("POST", f"{base}/cameras", {"camera": cam}, token=token)
            if code not in (200, 201):
                problems.append(f"adding a camera failed: {code} {added}")
            else:
                print(f"  camera added on {clips[0].name}")
        if not problems and clips:
            t1 = time.monotonic()
            code, started = _call("POST", f"{base}/engine/start", {}, token=token, timeout=60.0)
            if code not in (200, 201):
                problems.append(f"engine start refused: {code} {started}")
            else:
                print(f"  engine start accepted: {json.dumps(started)[:160]}")
                deadline = time.monotonic() + ENGINE_CONNECT_S
                health = monitor = None
                while time.monotonic() < deadline and proc.poll() is None:
                    _, monitor = _call("GET", f"{base}/monitor", token=token)
                    _, health = _call("GET", f"{base}/system/health", token=token)
                    if camera_connected(health if isinstance(health, dict) else None):
                        break
                    time.sleep(3.0)
                if camera_connected(health if isinstance(health, dict) else None):
                    timings["camera_connected_s"] = round(time.monotonic() - t1, 1)
                    print(f"  engine reported the camera CONNECTED after {timings['camera_connected_s']} s")
                else:
                    states = [c.get("state") for c in ((health or {}).get("cameras") or [])] \
                        if isinstance(health, dict) else None
                    problems.append(
                        f"the engine never reported the camera connected within {ENGINE_CONNECT_S:.0f} s "
                        f"(camera states: {states}; monitor: {json.dumps(monitor)[:300] if monitor else None})")
            _call("POST", f"{base}/engine/stop", {}, token=token, timeout=30.0)
    finally:
        _kill_tree(proc.pid)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()

    if problems:
        print("\nFAIL — the packaged app did not complete a first run:")
        for p in problems:
            print("  -", p)
        print("\n  app stdout:");   print(_tail(out_path))
        print("  app stderr:");     print(_tail(err_path))
        for log in sorted(site_dir.glob("*.log")):
            print(f"  {log.name}:");   print(_tail(log, 60))
        for log in sorted(tmp.rglob("support*.log")):
            print(f"  {log.relative_to(tmp)}:"); print(_tail(log, 60))
        return 1
    print(f"PASS — the packaged app started cold, its API answered, an owner signed in, "
          f"a camera was added and the engine connected to it. {json.dumps(timings)}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: packaged_app_boot.py <electron-builder release dir>")
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
