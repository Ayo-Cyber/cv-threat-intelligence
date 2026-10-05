"""Install the object-recognition model (SigLIP) as part of Download AI.

Object watch loads SigLIP from a local directory with downloads disabled, so a
fresh customer install had the recognition CODE and none of its FILES: the
feature was simply "unavailable" with nothing offering to fix it (Demi's
handoff, 5 Oct 2026). This module is the missing piece: a pinned checkpoint,
fetched into the per-user application-data directory, resumed if interrupted,
checksum-verified, loaded once and asked for one embedding before it is ever
called ready. The verification model (Gemma, via Ollama) keeps its own path in
cvti/serving/vlm.py; the two share the Download AI experience, not a runtime.

State is process-local like the VLM pull: poll `status()`.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib import error as urlerror
from urllib import request as urlrequest

from cvti.logging_setup import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class ModelFile:
    name: str
    size: int
    sha256: str


@dataclass(frozen=True)
class ModelSpec:
    repo: str
    revision: str
    files: tuple[ModelFile, ...]
    license: str
    display_size: str

    @property
    def total_bytes(self) -> int:
        return sum(f.size for f in self.files)


# google/siglip-base-patch16-224, Apache-2.0. The image tower is all object
# watch uses, so the tokenizer files are not fetched. Sizes and digests were
# read from the Hub's tree listing for this exact revision (safetensors is an
# LFS object; the two JSON files were fetched and hashed) on 5 Oct 2026.
SIGLIP = ModelSpec(
    repo="google/siglip-base-patch16-224",
    revision="7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed",
    files=(
        ModelFile("config.json", 432,
                  "cd85b3d28829722820bcb89a2cfbb4160e55fd359249a3044da724166a8d9688"),
        ModelFile("preprocessor_config.json", 368,
                  "d11ccb80f15d358a11bdb070e92e2d889005874b7db15823d5f10d9b2533b14a"),
        ModelFile("model.safetensors", 812_672_320,
                  "2c63cb7d1f2e95ba501893cbb8faeb4ea9a3af295498d35097126228659c2af8"),
    ),
    license="apache-2.0",
    display_size="0.8 GB",
)

MARKER = "installed.json"
CHUNK = 1024 * 1024
DISK_HEADROOM = 1.15          # need total * 1.15 free before starting


def default_models_dir() -> Path:
    """Per-user, writable, survives reinstalls: <app data>/models. ARGUS_MODELS_DIR
    overrides it (tests, and sites that keep models on another volume)."""
    override = os.environ.get("ARGUS_MODELS_DIR")
    if override:
        return Path(override).expanduser()
    from cvti.utils import user_data_dir
    return user_data_dir() / "models"


def default_install_dir() -> Path:
    return default_models_dir() / "siglip"


def file_url(spec: ModelSpec, name: str) -> str:
    base = os.environ.get("ARGUS_SIGLIP_BASE_URL", "").rstrip("/")
    if base:
        return f"{base}/{name}"
    return f"https://huggingface.co/{spec.repo}/resolve/{spec.revision}/{name}"


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_state: dict = {"state": "idle", "percent": 0, "detail": "", "bytes_done": 0, "bytes_total": 0}
_worker: threading.Thread | None = None


def _set(**kw) -> None:
    with _lock:
        _state.update(kw)


def _installed_marker(install_dir: Path) -> dict | None:
    try:
        doc = json.loads((install_dir / MARKER).read_text())
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def is_installed(install_dir: Path | None = None, spec: ModelSpec = SIGLIP) -> bool:
    """Ready = the marker says this revision was verified AND every file is still
    there at its recorded size. Cheap (no hashing) because it runs on every poll."""
    install_dir = install_dir or default_install_dir()
    marker = _installed_marker(install_dir)
    if not marker or marker.get("revision") != spec.revision or not marker.get("verified"):
        return False
    for f in spec.files:
        p = install_dir / f.name
        try:
            if p.stat().st_size != f.size:
                return False
        except OSError:
            return False
    return True


def status(install_dir: Path | None = None, spec: ModelSpec = SIGLIP) -> dict:
    install_dir = install_dir or default_install_dir()
    with _lock:
        cur = dict(_state)
    installed = is_installed(install_dir, spec)
    if installed and cur["state"] not in ("downloading", "verifying"):
        marker = _installed_marker(install_dir) or {}
        cur.update(state="ready", percent=100, detail="ready",
                   fingerprint=marker.get("fingerprint"), dimensions=marker.get("dimensions"))
    cur.update(path=str(install_dir), model=spec.repo, revision=spec.revision,
               license=spec.license, display_size=spec.display_size,
               bytes_total=cur.get("bytes_total") or spec.total_bytes,
               installed=installed)
    return cur


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
def start_install(install_dir: Path | None = None, *, spec: ModelSpec = SIGLIP,
                  smoke: Callable[[Path], dict] | None = None) -> dict:
    """Begin (or resume) the install on a background thread; poll status().
    An install already running is not restarted. An installed model returns
    ready at once."""
    global _worker
    install_dir = install_dir or default_install_dir()
    if is_installed(install_dir, spec):
        return status(install_dir, spec)
    with _lock:
        if _worker is not None and _worker.is_alive():
            return dict(_state)
        _state.update(state="downloading", percent=0, detail="starting",
                      bytes_done=0, bytes_total=spec.total_bytes, error="")
        _worker = threading.Thread(target=_install_worker, name="siglip-install",
                                   args=(install_dir, spec, smoke or run_smoke_test), daemon=True)
        _worker.start()
    return status(install_dir, spec)


def _install_worker(install_dir: Path, spec: ModelSpec, smoke: Callable[[Path], dict]) -> None:
    try:
        install(install_dir, spec=spec, smoke=smoke, progress=_set)
    except Exception as exc:  # noqa: BLE001 - the state record is the report
        detail = f"{type(exc).__name__}: {str(exc)[:160]}"
        log.error("[recognition model] install failed: %s", detail, exc_info=True)
        _set(state="error", detail=detail, error=detail)


def install(install_dir: Path, *, spec: ModelSpec = SIGLIP,
            smoke: Callable[[Path], dict] | None = None,
            progress: Callable[..., None] | None = None) -> dict:
    """Synchronous install: download+verify every file, smoke-test, write the
    marker. Raises on failure; nothing is marked ready until the smoke test passes."""
    progress = progress or (lambda **kw: None)
    install_dir.mkdir(parents=True, exist_ok=True)
    _check_disk(install_dir, spec)
    done_before = 0
    for f in spec.files:
        target = install_dir / f.name
        if target.is_file() and target.stat().st_size == f.size and _sha256(target) == f.sha256:
            done_before += f.size
            progress(bytes_done=done_before, percent=int(done_before * 100 / spec.total_bytes))
            continue
        _download(spec, f, target, base_done=done_before, total=spec.total_bytes, progress=progress)
        done_before += f.size
        progress(bytes_done=done_before, percent=int(done_before * 100 / spec.total_bytes))
    progress(state="verifying", percent=100, detail="loading the model once to check it works")
    result = (smoke or run_smoke_test)(install_dir)
    if not result.get("ok"):
        raise RuntimeError(f"model loaded but failed its check: {result.get('error') or 'unknown'}")
    marker = {"repo": spec.repo, "revision": spec.revision, "license": spec.license,
              "verified": True, "verified_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "fingerprint": result.get("fingerprint"), "dimensions": result.get("dimensions"),
              "files": {f.name: f.sha256 for f in spec.files}}
    tmp = install_dir / (MARKER + ".tmp")
    tmp.write_text(json.dumps(marker, indent=2) + "\n")
    tmp.replace(install_dir / MARKER)
    progress(state="ready", percent=100, detail="ready", fingerprint=marker["fingerprint"],
             dimensions=marker["dimensions"])
    log.info("[recognition model] %s@%s installed at %s", spec.repo, spec.revision[:8], install_dir)
    return marker


def _check_disk(install_dir: Path, spec: ModelSpec) -> None:
    have = sum((install_dir / f.name).stat().st_size for f in spec.files if (install_dir / f.name).is_file())
    need = int((spec.total_bytes - have) * DISK_HEADROOM)
    free = shutil.disk_usage(install_dir).free
    if free < need:
        raise RuntimeError(f"not enough disk space: {need // (1024 * 1024)} MB needed, "
                           f"{free // (1024 * 1024)} MB free at {install_dir}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(spec: ModelSpec, f: ModelFile, target: Path, *, base_done: int, total: int,
              progress: Callable[..., None], timeout: float = 60.0) -> None:
    """Fetch one file to <name>.part with HTTP Range resume, then verify its
    digest and move it into place. A wrong digest deletes the part file so the
    retry starts clean instead of resuming a corrupt download forever."""
    part = target.with_name(target.name + ".part")
    have = part.stat().st_size if part.is_file() else 0
    if have > f.size:
        part.unlink(); have = 0
    url = file_url(spec, f.name)
    if have < f.size:
        headers = {"User-Agent": "argus-model-install/1"}
        if have:
            headers["Range"] = f"bytes={have}-"
        req = urlrequest.Request(url, headers=headers)
        try:
            resp = urlrequest.urlopen(req, timeout=timeout)
        except urlerror.HTTPError as exc:
            if exc.code == 416 and have:        # server says we already have it all
                resp = None
            else:
                raise RuntimeError(f"download of {f.name} failed: HTTP {exc.code}") from exc
        if resp is not None:
            with resp:
                resumed = resp.status == 206
                mode = "ab" if (have and resumed) else "wb"
                if not resumed:
                    have = 0
                with part.open(mode) as out:
                    while True:
                        chunk = resp.read(CHUNK)
                        if not chunk:
                            break
                        out.write(chunk)
                        have += len(chunk)
                        progress(state="downloading", detail=f.name, bytes_done=base_done + have,
                                 percent=int((base_done + have) * 100 / max(1, total)))
    if part.stat().st_size != f.size:
        size = part.stat().st_size
        part.unlink()
        raise RuntimeError(f"{f.name}: got {size} bytes, expected {f.size}")
    progress(detail=f"checking {f.name}")
    digest = _sha256(part)
    if digest != f.sha256:
        part.unlink()
        raise RuntimeError(f"{f.name}: checksum mismatch — the download was corrupt or not the pinned file")
    part.replace(target)


# ---------------------------------------------------------------------------
# Smoke test — runs in the ENGINE interpreter, which is the one that ships torch
# ---------------------------------------------------------------------------
def smoke_command(install_dir: Path) -> list[str]:
    """The packaged app excludes torch/transformers (they ride in the engine
    bundle), so the load test runs as an engine subprocess. In development
    both are the same interpreter."""
    if getattr(sys, "frozen", False):
        from cvti.app.console_backend import ConsoleBackend
        engine = ConsoleBackend._bundled_engine()
        if engine is not None:
            return [str(engine), "object-watch-smoke", str(install_dir)]
    return [sys.executable, "-m", "cvti.object_watch.smoke", str(install_dir)]


def run_smoke_test(install_dir: Path, timeout: float = 600.0) -> dict:
    try:
        proc = subprocess.run(smoke_command(install_dir), capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "the model did not load within 10 minutes"}
    except OSError as exc:
        return {"ok": False, "error": f"could not start the engine for the check: {exc}"}
    line = (proc.stdout or "").strip().splitlines()
    try:
        out = json.loads(line[-1]) if line else {}
    except ValueError:
        out = {}
    if proc.returncode != 0 or not out.get("ok"):
        err = out.get("error") or (proc.stderr or "").strip().splitlines()[-1:] or ["unknown"]
        return {"ok": False, "error": str(err if isinstance(err, str) else err[0])[:200]}
    return out
