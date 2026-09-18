"""Exercise object-watch enrollment through a relocated frozen ``argus-api``.

This controller deliberately uses only the standard library after seeding the
temporary account database. It is a release affordance, not a substitute for a
real frozen-bundle run.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

APPROVED_TEMP = Path("/var/folders/1m/nxpw6v2x4zd55cd3479np3j00000gn/T/opencode")
TERMINAL_JOB_STATES = {"completed", "failed"}
TARGET_CATEGORY = "product"
TORCH_VERSION = "2.8.0"
TORCH_GIT_VERSION = "a1cb3cc05d46d198467bebbb6e8fba50a325d4e7"
TORCH_GENERATED_MODULE = "_remote_module_non_scriptable"
DEFAULT_STARTUP_TIMEOUT = 180.0
PYINSTALLER_VIRTUAL_BOOTSTRAP = {
    "pyimod01_archive", "pyimod02_importers", "pyimod03_ctypes", "struct",
}
TORCH_VIRTUAL_ALIASES = {
    "torch.classes": ("_classes.py", "torch._classes"),
    "torch.ops": ("_ops.py", "torch._ops"),
}
SOURCE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE_ROOT))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def validate_fixture_contract() -> None:
    """Fail before copying/running a bundle if this controller fixture drifted."""
    from cvti.object_watch.store import VALID_CATEGORIES
    require(TARGET_CATEGORY in VALID_CATEGORIES,
            f"harness target category is not supported by the backend: {TARGET_CATEGORY}")


def request(base: str, method: str, path: str, body=None, token: str | None = None,
            expected: int | tuple[int, ...] = 200) -> tuple[int, dict]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(base + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            status, raw = int(response.status), response.read()
    except urllib.error.HTTPError as exc:
        status, raw = exc.code, exc.read()
    wanted = (expected,) if isinstance(expected, int) else expected
    require(status in wanted, f"{method} {path}: expected {wanted}, got {status}: {raw[:300]!r}")
    decoded = json.loads(raw) if raw else {}
    require(isinstance(decoded, dict), f"{method} {path}: response was not a JSON object")
    return status, decoded


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def find_api(bundle: Path) -> Path:
    names = {"argus-api", "argus-api.exe"}
    direct = [bundle / name for name in names]
    hits = [path for path in direct if path.is_file()]
    if not hits:
        hits = [path for path in bundle.rglob("*") if path.is_file() and path.name in names]
    require(len(hits) == 1, f"expected exactly one argus-api executable, found {len(hits)}")
    return hits[0]


def clean_environment(home: Path, report: Path, temp_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    loader_names = {"DYLD_LIBRARY_PATH", "DYLD_INSERT_LIBRARIES", "LD_LIBRARY_PATH", "LD_PRELOAD"}
    for key in list(env):
        if (key in {"PYTHONPATH", "PYTHONHOME"} or key.startswith("ARGUS_")
                or key.startswith("DYLD_") or key.startswith("LD_") or key in loader_names):
            env.pop(key, None)
    cache = home / "cache"
    env.update({
        "HOME": str(home), "USERPROFILE": str(home), "APPDATA": str(home / "AppData"),
        "LOCALAPPDATA": str(home / "LocalAppData"), "XDG_CACHE_HOME": str(cache),
        "HF_HOME": str(cache / "huggingface"), "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1", "YOLO_AUTOINSTALL": "False",
        "ARGUS_FROZEN_IMPORT_REPORT": str(report),
        "TMPDIR": str(temp_root), "TEMP": str(temp_root), "TMP": str(temp_root),
    })
    return env


def _cleanup_startup_process(process: subprocess.Popen) -> int | None:
    """Best-effort cleanup which never replaces the primary startup failure."""
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
    return process.returncode


def start_api(executable: Path, run_dir: Path, site: Path, db: Path, report: Path,
              startup_timeout: float = DEFAULT_STARTUP_TIMEOUT, *,
              monotonic=time.monotonic, sleeper=time.sleep):
    require(startup_timeout > 0, "--startup-timeout must be positive")
    port = free_port()
    log = (run_dir / f"api-{report.stem}.log").open("wb")
    temp_root = run_dir / "tmp"
    temp_root.mkdir(exist_ok=True)
    process = subprocess.Popen(
        [str(executable), "--host", "127.0.0.1", "--port", str(port),
         "--db", str(db), "--site", str(site)],
        cwd=run_dir, env=clean_environment(run_dir / "home", report, temp_root),
        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
    )
    base = f"http://127.0.0.1:{port}"
    started = monotonic()
    deadline = started + startup_timeout
    while monotonic() < deadline:
        if process.poll() is not None:
            log.close()
            elapsed = monotonic() - started
            raise RuntimeError(
                f"argus-api exited during startup after {elapsed:.3f}s "
                f"(returncode={process.returncode}, executable={executable}, log={log.name})")
        try:
            request(base, "GET", "/api/v1/object-targets", expected=401)
            return process, log, base, monotonic() - started
        except (OSError, RuntimeError):
            sleeper(0.2)
    cleanup_returncode = _cleanup_startup_process(process)
    elapsed = monotonic() - started
    log.close()
    raise RuntimeError(
        f"timed out waiting for frozen argus-api after {elapsed:.3f}s "
        f"(limit={startup_timeout:.3f}s, executable={executable}, log={log.name}, "
        f"cleanup_returncode={cleanup_returncode})")


def stop_api(process: subprocess.Popen, log) -> None:
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=10)
    log.close()
    require(process.returncode == 0, f"argus-api shutdown returned {process.returncode}")


def _inside(path: Path, root: Path) -> bool:
    """Use resolved ancestry, never string prefixes (``Argus-evil`` is outside)."""
    resolved, resolved_root = path.resolve(), root.resolve()
    return resolved == resolved_root or resolved_root in resolved.parents


def _bundled_path(value, relocated: Path, label: str) -> Path:
    require(isinstance(value, str) and bool(value), f"{label} is not a filesystem path")
    path = Path(value)
    require(path.is_absolute(), f"{label} is not absolute: {value}")
    # resolve(strict=False) intentionally supports PyInstaller archive/virtual
    # descendants which need not exist as individual files, while still
    # resolving an existing symlinked ancestor so escapes cannot pass.
    require(_inside(path, relocated), f"{label} is outside relocated bundle: {value}")
    return path.resolve()


def _validate_torch_generated_source(report: dict, relocated: Path,
                                     controlled_temp_root: Path | None):
    evidence = report.get("torch_generated_remote_module")
    if evidence is None:
        return None
    require(isinstance(evidence, dict), "PyTorch generated-source evidence is not an object")
    if controlled_temp_root is None:
        raise RuntimeError("PyTorch generated-source evidence has no controlled temp root")
    require(evidence.get("torch_version") == TORCH_VERSION,
            "PyTorch generated-source exception has the wrong version")
    require(evidence.get("torch_git_version") == TORCH_GIT_VERSION,
            "PyTorch generated-source exception has the wrong git revision")
    require(evidence.get("module_name") == TORCH_GENERATED_MODULE,
            "PyTorch generated-source exception has the wrong module name")
    filename = f"{TORCH_GENERATED_MODULE}.py"
    require(evidence.get("module_filename") == filename,
            "PyTorch generated-source exception has the wrong filename")
    _bundled_path(evidence.get("generator_origin"), relocated, "PyTorch generator origin")
    _bundled_path(evidence.get("template_origin"), relocated, "PyTorch template origin")
    temp_dir = Path(str(evidence.get("temp_dir") or ""))
    resolved_temp_dir = Path(str(evidence.get("resolved_temp_dir") or ""))
    require(temp_dir.is_absolute() and resolved_temp_dir.is_absolute()
            and _inside(resolved_temp_dir, controlled_temp_root),
            "PyTorch generated-source temp directory is outside controlled TMPROOT")
    origin = Path(str(evidence.get("module_origin") or ""))
    resolved_origin = Path(str(evidence.get("resolved_module_origin") or ""))
    require(origin.is_absolute() and origin.parent == temp_dir and origin.name == filename
            and resolved_origin.is_absolute()
            and resolved_origin.parent == resolved_temp_dir
            and resolved_origin.name == filename,
            "PyTorch generated module origin is not the exact attested temp file")
    actual_hash = evidence.get("actual_content_sha256")
    expected_hash = evidence.get("expected_template_sha256")
    require(isinstance(actual_hash, str) and len(actual_hash) == 64
            and actual_hash == expected_hash,
            "PyTorch generated module content does not match its bundled template")
    entries = evidence.get("directory_entries")
    require(isinstance(entries, list) and set(entries) in ({filename}, {filename, "__pycache__"}),
            "PyTorch generated-source directory contains unexpected entries")
    pycache_entries = evidence.get("pycache_entries")
    require(isinstance(pycache_entries, list), "PyTorch generated-source pycache listing is invalid")
    require(all(isinstance(name, str)
                and name.startswith(TORCH_GENERATED_MODULE + ".") and name.endswith(".pyc")
                for name in pycache_entries),
            "PyTorch generated-source directory contains foreign bytecode")
    if "__pycache__" not in entries:
        require(not pycache_entries, "PyTorch generated-source evidence lists a missing pycache")
    return {"temp_dir": resolved_temp_dir, "origin": resolved_origin}


def _is_pyinstaller_virtual_bootstrap(name: str, origin, runtime_root: Path | None) -> bool:
    """Recognize PyInstaller's embedded bootstrap modules with virtual __file__ paths."""
    if name not in PYINSTALLER_VIRTUAL_BOOTSTRAP or runtime_root is None:
        return False
    path = Path(str(origin))
    expected = runtime_root / f"{name}.py"
    return path.is_absolute() and path.resolve() == expected.resolve() and not path.exists()


def _is_bundled_torch_virtual_alias(name: str, origin, modules: dict,
                                    relocated: Path) -> bool:
    expected = TORCH_VIRTUAL_ALIASES.get(name)
    if expected is None or origin != expected[0]:
        return False
    canonical_origin = modules.get(expected[1])
    try:
        _bundled_path(canonical_origin, relocated, f"canonical module {expected[1]!r} origin")
    except RuntimeError:
        return False
    return True


def check_import_report(
    report_path: Path,
    relocated: Path,
    expected_executable: Path,
    *,
    require_frozen: bool,
    validate_provenance: bool,
    require_runtime_modules: bool,
    controlled_temp_root: Path | None = None,
) -> dict:
    require(report_path.is_file(), "frozen API did not emit its import report")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if require_frozen:
        require(report.get("sys.frozen") is True, "argus-api did not report sys.frozen=true")
    if not validate_provenance:
        return {"checked": False, "sys_frozen": report.get("sys.frozen") is True}

    executable = _bundled_path(report.get("executable"), relocated, "executable")
    require(executable == expected_executable.resolve(),
            f"report executable differs from launched artifact: {executable}")
    _bundled_path(report.get("_MEIPASS"), relocated, "_MEIPASS")
    generated = _validate_torch_generated_source(report, relocated, controlled_temp_root)
    sys_path = report.get("sys_path")
    require(isinstance(sys_path, list), "report sys_path is not a list")
    for index, value in enumerate(sys_path):
        if generated is not None and Path(str(value)).resolve() == generated["temp_dir"]:
            continue
        _bundled_path(value, relocated, f"sys.path[{index}]")

    modules = report.get("modules") or {}
    require(isinstance(modules, dict), "report modules is not an object")
    for name, origin in modules.items():
        require(isinstance(name, str), "report contains a non-string module name")
        if origin in {"built-in", "frozen"}:
            continue
        if (generated is not None and name == TORCH_GENERATED_MODULE
                and Path(str(origin)).resolve() == generated["origin"]):
            continue
        runtime_root = controlled_temp_root.parent if controlled_temp_root is not None else None
        if _is_pyinstaller_virtual_bootstrap(name, origin, runtime_root):
            continue
        if _is_bundled_torch_virtual_alias(name, origin, modules, relocated):
            continue
        _bundled_path(origin, relocated, f"module {name!r} origin")

    relevant = {}
    if require_runtime_modules:
        required = {
            "cvti.api.app",
            "cvti.object_watch.embeddings",
            "torch",
            "transformers",
            "transformers.models.siglip.modeling_siglip",
        }
        missing = sorted(required - modules.keys())
        require(not missing, "required frozen runtime modules were not loaded: " + ", ".join(missing))
        image_processors = {
            "transformers.models.siglip.image_processing_siglip",
            "transformers.models.siglip.image_processing_siglip_fast",
        }
        loaded_processors = sorted(image_processors & modules.keys())
        require(bool(loaded_processors), "no supported SigLIP image-processing module was loaded")
        selected = required | set(loaded_processors)
        relevant = {name: modules[name] for name in sorted(selected)}
    return {"checked": True, "module_count": len(modules), "runtime_origins": relevant}


def validate_import_reports(
    initial: Path,
    restart: Path,
    relocated: Path,
    expected_executable: Path,
    *,
    require_frozen: bool,
    validate_provenance: bool,
    controlled_temp_root: Path | None = None,
) -> dict:
    """Validate each process independently; restart cannot inherit first-run proof."""
    return {
        "initial": check_import_report(
            initial, relocated, expected_executable,
            require_frozen=require_frozen,
            validate_provenance=validate_provenance,
            require_runtime_modules=validate_provenance,
            controlled_temp_root=controlled_temp_root,
        ),
        "restart": check_import_report(
            restart, relocated, expected_executable,
            require_frozen=require_frozen,
            validate_provenance=validate_provenance,
            require_runtime_modules=False,
            controlled_temp_root=controlled_temp_root,
        ),
    }


def run(args) -> dict:
    require(Path(args.model).expanduser().is_absolute(), "--model must be an absolute path")
    require(Path(args.image).expanduser().is_absolute(), "--image must be an absolute path")
    bundle, model, image = (Path(value).expanduser().resolve()
                            for value in (args.bundle, args.model, args.image))
    require(bundle.is_dir(), f"bundle is not a directory: {bundle}")
    require(model.is_absolute() and model.is_dir(), f"model must be an existing absolute directory: {model}")
    require(image.is_absolute() and image.is_file(), f"image must be an existing absolute file: {image}")
    validate_fixture_contract()
    APPROVED_TEMP.mkdir(parents=True, exist_ok=True)
    if args.output_dir:
        out = Path(args.output_dir).expanduser().resolve()
        require(APPROVED_TEMP.resolve() in out.parents, "--output-dir must be under the approved opencode temp root")
        require(not out.exists(), "refusing to overwrite an existing --output-dir")
        out.mkdir(parents=True)
    else:
        import tempfile
        out = Path(tempfile.mkdtemp(prefix="argus-frozen-object-watch-", dir=APPROVED_TEMP))
    relocated = out / "relocated" / bundle.name
    relocated.parent.mkdir()
    shutil.copytree(bundle, relocated, symlinks=True)
    executable = find_api(relocated)
    run_dir = out / "runtime"
    run_dir.mkdir()
    for directory in (run_dir / "home", run_dir / "home" / "AppData", run_dir / "home" / "LocalAppData"):
        directory.mkdir(parents=True, exist_ok=True)
    site = run_dir / "site.json"
    site.write_text(json.dumps({"name": "frozen-object-watch", "notify": "console", "cameras": []}), encoding="utf-8")
    db = run_dir / "events.db"

    # Seed only this disposable auth.db. Credentials and bearer tokens are never printed.
    from cvti.security.accounts import AccountStore
    username, password = "frozen-owner", secrets.token_urlsafe(24)
    accounts = AccountStore(run_dir / "auth.db")
    accounts.create_user(username, password, role="owner")
    accounts.close()

    report1 = out / "imports-first.json"
    process, log, base, first_startup_seconds = start_api(
        executable, run_dir, site, db, report1, args.startup_timeout)
    fingerprint = example_id = None
    try:
        request(base, "GET", "/api/v1/object-targets", expected=401)
        _, login = request(base, "POST", "/api/v1/auth/session",
                           {"username": username, "password": password})
        token = login["token"]
        _, initial = request(base, "GET", "/api/v1/object-targets", token=token)
        runtime = initial["runtime"]
        require(runtime["status"] == "unavailable", "missing default model was not explicitly unavailable")
        require(runtime["backend"] == "siglip" and not runtime.get("fingerprint"),
                "missing model unexpectedly fell back or acquired weights")
        _, configured = request(base, "PUT", "/api/v1/object-targets/runtime",
                                {"config": {"backend": "siglip", "device": "cpu", "model_path": str(model)}}, token)
        runtime = configured["runtime"]
        require(runtime["status"] == "structurally_available", f"unexpected structural readiness: {runtime}")
        fingerprint = runtime["fingerprint"]
        require(bool(fingerprint), "structurally available model has no fingerprint")
        request(base, "POST", "/api/v1/object-targets",
                {"object_id": "frozen-photo", "label": "Frozen photo",
                 "category": TARGET_CATEGORY}, token, 201)
        request(base, "POST", "/api/v1/object-targets/frozen-photo/activate", token=token, expected=400)
        payload = base64.b64encode(image.read_bytes()).decode("ascii")
        _, added = request(base, "POST", "/api/v1/object-targets/frozen-photo/examples",
                           {"image_b64": payload, "bbox": [0, 0, 1, 1],
                            "bbox_format": "normalized_xyxy", "source": "upload"}, token, 201)
        example_id = added["example"]["id"]
        request(base, "POST", "/api/v1/object-targets/frozen-photo/activate", token=token, expected=400)
        request(base, "PUT", f"/api/v1/object-targets/frozen-photo/examples/{example_id}/review",
                {"reviewed": True}, token)
        request(base, "POST", "/api/v1/object-targets/frozen-photo/activate", token=token, expected=400)
        _, queued = request(base, "POST", "/api/v1/object-targets/reembed", {}, token)
        deadline = time.time() + args.job_timeout
        while time.time() < deadline:
            _, job = request(base, "GET", f"/api/v1/object-targets/jobs/{queued['job_id']}", token=token)
            if job["status"] in TERMINAL_JOB_STATES:
                break
            time.sleep(0.5)
        else:
            raise RuntimeError("timed out waiting for SigLIP enrollment")
        require(job["status"] == "completed", f"enrollment failed: {job.get('error', job['status'])}")
        require(job.get("written") == 1 and job.get("model") == "siglip",
                f"empty or wrong enrollment result: {job}")
        require(job.get("model_fingerprint") == fingerprint, "job fingerprint differs from readiness fingerprint")
        _, activated = request(base, "POST", "/api/v1/object-targets/frozen-photo/activate", token=token)
        require(activated["target"]["review_state"] == "active", "target did not activate")
    finally:
        stop_api(process, log)

    embedding_path = run_dir / "object_library" / "embeddings" / fingerprint / "frozen-photo.json"
    stored = json.loads(embedding_path.read_text(encoding="utf-8"))["embeddings"][example_id]
    vector = stored["vector"]
    require(len(vector) == 768 and all(math.isfinite(float(v)) for v in vector),
            "stored SigLIP embedding is not 768 finite components")
    require(abs(math.sqrt(sum(float(v) ** 2 for v in vector)) - 1.0) < 1e-5,
            "stored SigLIP embedding is not unit normalized")
    crop = run_dir / "object_library" / "examples" / "frozen-photo" / f"{example_id}.png"
    require(crop.is_file(), "canonical enrollment crop is missing")
    require(stored["crop_sha256"] == hashlib.sha256(crop.read_bytes()).hexdigest(),
            "stored embedding crop digest does not match canonical crop")
    require(stored["model_name"] == "siglip" and stored["model_fingerprint"] == fingerprint,
            "stored embedding model metadata does not match runtime")

    report2 = out / "imports-restart.json"
    process, log, base, restart_startup_seconds = start_api(
        executable, run_dir, site, db, report2, args.startup_timeout)
    try:
        _, login = request(base, "POST", "/api/v1/auth/session",
                           {"username": username, "password": password})
        _, listing = request(base, "GET", "/api/v1/object-targets", token=login["token"])
        target = next(row for row in listing["targets"] if row["id"] == "frozen-photo")
        require(target["review_state"] == "active", "active target did not persist across restart")
        require(listing["runtime"]["fingerprint"] == fingerprint, "runtime fingerprint did not persist")
    finally:
        stop_api(process, log)

    diagnostics = validate_import_reports(
        report1, report2, relocated, executable,
        require_frozen=args.require_frozen,
        validate_provenance=args.assert_no_source_imports,
        controlled_temp_root=run_dir / "tmp",
    )
    summary = {"ok": True, "output_dir": str(out), "runtime_status": "structurally_available",
               "embedding_dimensions": len(vector), "written": 1, "persistence": "active",
               "startup_seconds": {
                   "initial": round(first_startup_seconds, 3),
                   "restart": round(restart_startup_seconds, 3),
               },
               "import_diagnostics": diagnostics}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--require-frozen", action="store_true")
    parser.add_argument("--assert-no-source-imports", action="store_true")
    parser.add_argument("--output-dir")
    parser.add_argument("--job-timeout", type=float, default=600)
    parser.add_argument("--startup-timeout", type=float, default=DEFAULT_STARTUP_TIMEOUT)
    args = parser.parse_args()
    try:
        summary = run(args)
    except Exception as exc:  # noqa: BLE001 - command-line checker reports one safe failure
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
