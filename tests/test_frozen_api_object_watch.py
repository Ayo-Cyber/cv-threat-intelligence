from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import pytest

_HARNESS_PATH = Path(__file__).parent / "e2e" / "frozen_api_object_watch.py"
_SPEC = importlib.util.spec_from_file_location("frozen_api_object_watch", _HARNESS_PATH)
assert _SPEC and _SPEC.loader
_HARNESS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_HARNESS)
check_import_report = _HARNESS.check_import_report
validate_import_reports = _HARNESS.validate_import_reports


RUNTIME_MODULES = {
    "cvti.api.app",
    "cvti.object_watch.embeddings",
    "torch",
    "transformers",
    "transformers.models.siglip.modeling_siglip",
    "transformers.models.siglip.image_processing_siglip",
}


class _StartupProcess:
    def __init__(self, cleanup_returncode=1):
        self.returncode = None
        self.cleanup_returncode = cleanup_returncode
        self.signals = []

    def poll(self):
        return self.returncode

    def send_signal(self, value):
        self.signals.append(value)

    def wait(self, timeout):
        self.returncode = self.cleanup_returncode
        return self.returncode

    def terminate(self):
        raise AssertionError("graceful cleanup unexpectedly timed out")

    def kill(self):
        raise AssertionError("graceful cleanup unexpectedly timed out")


def test_startup_timeout_uses_monotonic_config_and_preserves_primary_failure(tmp_path, monkeypatch):
    process = _StartupProcess(cleanup_returncode=1)
    ticks = iter((10.0, 10.0, 10.6, 10.7))
    monkeypatch.setattr(_HARNESS.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(_HARNESS, "free_port", lambda: 12345)
    monkeypatch.setattr(_HARNESS, "request", lambda *args, **kwargs: (_ for _ in ()).throw(OSError()))
    executable = tmp_path / "Argus" / "argus-api"
    run_dir = tmp_path / "runtime"
    run_dir.mkdir()
    (run_dir / "home").mkdir()
    with pytest.raises(RuntimeError) as caught:
        _HARNESS.start_api(
            executable, run_dir, tmp_path / "site.json", tmp_path / "events.db",
            tmp_path / "report.json", startup_timeout=0.5,
            monotonic=lambda: next(ticks), sleeper=lambda _seconds: None,
        )
    message = str(caught.value)
    assert "timed out waiting for frozen argus-api" in message
    assert "after 0.700s" in message
    assert "limit=0.500s" in message
    assert f"executable={executable}" in message
    assert "api-report.log" in message
    assert "cleanup_returncode=1" in message
    assert process.signals == [_HARNESS.signal.SIGINT]


def test_default_startup_timeout_matches_packaged_caller_bound():
    assert _HARNESS.DEFAULT_STARTUP_TIMEOUT == 180.0


def test_accepts_only_nonexistent_pyinstaller_virtual_bootstrap_at_runtime_root(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    origin = runtime / "pyimod01_archive.py"
    assert _HARNESS._is_pyinstaller_virtual_bootstrap(
        "pyimod01_archive", str(origin), runtime)
    assert not _HARNESS._is_pyinstaller_virtual_bootstrap(
        "unrelated", str(origin), runtime)
    origin.write_text("# external source", encoding="utf-8")
    assert not _HARNESS._is_pyinstaller_virtual_bootstrap(
        "pyimod01_archive", str(origin), runtime)


def test_accepts_torch_virtual_alias_only_with_bundled_canonical_module(tmp_path):
    root, _ = _artifact(tmp_path)
    modules = {"torch._ops": str(root / "_internal/torch/_ops.py")}
    assert _HARNESS._is_bundled_torch_virtual_alias(
        "torch.ops", "_ops.py", modules, root)
    assert not _HARNESS._is_bundled_torch_virtual_alias(
        "torch.ops", "wrong.py", modules, root)
    modules["torch._ops"] = "/external/torch/_ops.py"
    assert not _HARNESS._is_bundled_torch_virtual_alias(
        "torch.ops", "_ops.py", modules, root)


def test_harness_category_is_valid_backend_contract_fixture():
    from cvti.object_watch.store import VALID_CATEGORIES

    _HARNESS.validate_fixture_contract()
    assert _HARNESS.TARGET_CATEGORY == "product"
    assert _HARNESS.TARGET_CATEGORY in VALID_CATEGORIES


def test_harness_rejects_invalid_category_before_bundle_run(monkeypatch):
    monkeypatch.setattr(_HARNESS, "TARGET_CATEGORY", "test")
    with pytest.raises(RuntimeError, match="not supported by the backend"):
        _HARNESS.validate_fixture_contract()


def _artifact(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "Argus"
    root.mkdir()
    executable = root / "argus-api"
    executable.write_bytes(b"binary")
    return root, executable


def _report(root: Path, executable: Path, *, frozen=True, modules=None) -> dict:
    origins = {
        name: str(root / "_internal" / (name.replace(".", "/") + ".pyc"))
        for name in (RUNTIME_MODULES if modules is None else modules)
    }
    origins["sys"] = "built-in"
    origins["importlib._bootstrap"] = "frozen"
    return {
        "sys.frozen": frozen,
        "executable": str(executable),
        "_MEIPASS": str(root / "_internal"),
        "sys_path": [str(root / "_internal"), str(root / "_internal" / "base_library.zip")],
        "modules": origins,
    }


def _write(path: Path, report: dict) -> Path:
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def _check(path: Path, root: Path, executable: Path, *, runtime=True):
    return check_import_report(
        path, root, executable, require_frozen=True,
        validate_provenance=True, require_runtime_modules=runtime,
    )


def _with_torch_generated(report: dict, root: Path, temp_root: Path) -> dict:
    temp_dir = temp_root / "tmp-torch-known"
    origin = temp_dir / "_remote_module_non_scriptable.py"
    report["sys_path"].append(str(temp_dir))
    report["modules"]["_remote_module_non_scriptable"] = str(origin)
    report["torch_generated_remote_module"] = {
        "torch_version": _HARNESS.TORCH_VERSION,
        "torch_git_version": _HARNESS.TORCH_GIT_VERSION,
        "generator_origin": str(root / "_internal/torch/distributed/nn/jit/instantiator.py"),
        "template_origin": str(root / "_internal/torch/distributed/nn/jit/templates/remote_module_template.py"),
        "temp_dir": str(temp_dir),
        "resolved_temp_dir": str(temp_dir),
        "module_name": "_remote_module_non_scriptable",
        "module_filename": "_remote_module_non_scriptable.py",
        "module_origin": str(origin),
        "resolved_module_origin": str(origin),
        "actual_content_sha256": "a" * 64,
        "expected_template_sha256": "a" * 64,
        "directory_entries": ["__pycache__", "_remote_module_non_scriptable.py"],
        "pycache_entries": ["_remote_module_non_scriptable.cpython-312.pyc"],
    }
    return report


def _check_generated(path: Path, root: Path, executable: Path, temp_root: Path):
    return check_import_report(
        path, root, executable, require_frozen=True, validate_provenance=True,
        require_runtime_modules=True, controlled_temp_root=temp_root,
    )


def test_valid_bundle_report_accepts_virtual_zip_paths_and_builtin_origins(tmp_path):
    root, executable = _artifact(tmp_path)
    result = _check(_write(tmp_path / "report.json", _report(root, executable)), root, executable)
    assert result["checked"] is True
    assert result["module_count"] == len(RUNTIME_MODULES) + 2


def test_accepts_exact_attested_pytorch_generated_source(tmp_path):
    root, executable = _artifact(tmp_path)
    temp_root = tmp_path / "controlled-tmp"
    report = _with_torch_generated(_report(root, executable), root, temp_root)
    result = _check_generated(_write(tmp_path / "report.json", report), root, executable, temp_root)
    assert result["checked"] is True


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("module_name", "_remote_module_other", "wrong module name"),
        ("torch_version", "2.8.1", "wrong version"),
        ("expected_template_sha256", "b" * 64, "does not match"),
        ("directory_entries", ["_remote_module_non_scriptable.py", "foreign.py"], "unexpected entries"),
        ("directory_entries", ["_remote_module_non_scriptable.py", "inject.pth"], "unexpected entries"),
        ("directory_entries", ["_remote_module_non_scriptable.py", "other_module.py"], "unexpected entries"),
        ("pycache_entries", ["foreign.cpython-312.pyc"], "foreign bytecode"),
    ],
)
def test_rejects_malformed_pytorch_generated_source_evidence(tmp_path, field, value, message):
    root, executable = _artifact(tmp_path)
    temp_root = tmp_path / "controlled-tmp"
    report = _with_torch_generated(_report(root, executable), root, temp_root)
    report["torch_generated_remote_module"][field] = value
    with pytest.raises(RuntimeError, match=message):
        _check_generated(_write(tmp_path / "report.json", report), root, executable, temp_root)


def test_rejects_external_pytorch_generator(tmp_path):
    root, executable = _artifact(tmp_path)
    temp_root = tmp_path / "controlled-tmp"
    report = _with_torch_generated(_report(root, executable), root, temp_root)
    report["torch_generated_remote_module"]["generator_origin"] = "/external/torch/instantiator.py"
    with pytest.raises(RuntimeError, match="outside relocated bundle"):
        _check_generated(_write(tmp_path / "report.json", report), root, executable, temp_root)


def test_rejects_sibling_pytorch_temp_directory(tmp_path):
    root, executable = _artifact(tmp_path)
    temp_root = tmp_path / "controlled-tmp"
    report = _with_torch_generated(_report(root, executable), root, temp_root)
    sibling = tmp_path / "controlled-tmp-sibling" / "tmp-torch-known"
    evidence = report["torch_generated_remote_module"]
    evidence["temp_dir"] = str(sibling)
    evidence["resolved_temp_dir"] = str(sibling)
    evidence["module_origin"] = str(sibling / evidence["module_filename"])
    evidence["resolved_module_origin"] = evidence["module_origin"]
    report["sys_path"][-1] = str(sibling)
    report["modules"]["_remote_module_non_scriptable"] = evidence["module_origin"]
    with pytest.raises(RuntimeError, match="outside controlled TMPROOT"):
        _check_generated(_write(tmp_path / "report.json", report), root, executable, temp_root)


def test_rejects_escaping_pytorch_temp_symlink(tmp_path):
    root, executable = _artifact(tmp_path)
    temp_root = tmp_path / "controlled-tmp"
    temp_root.mkdir()
    outside = tmp_path / "outside-tmp"
    outside.mkdir()
    escape = temp_root / "escape"
    try:
        escape.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    report = _with_torch_generated(_report(root, executable), root, temp_root)
    evidence = report["torch_generated_remote_module"]
    evidence["temp_dir"] = str(escape)
    evidence["resolved_temp_dir"] = str(outside)
    evidence["module_origin"] = str(escape / evidence["module_filename"])
    evidence["resolved_module_origin"] = str(outside / evidence["module_filename"])
    with pytest.raises(RuntimeError, match="outside controlled TMPROOT"):
        _check_generated(_write(tmp_path / "report.json", report), root, executable, temp_root)


def test_attestation_does_not_allow_broad_temp_sys_path(tmp_path):
    root, executable = _artifact(tmp_path)
    temp_root = tmp_path / "controlled-tmp"
    report = _with_torch_generated(_report(root, executable), root, temp_root)
    report["sys_path"].append(str(temp_root))
    with pytest.raises(RuntimeError, match=r"sys.path\[.*outside relocated bundle"):
        _check_generated(_write(tmp_path / "report.json", report), root, executable, temp_root)


@pytest.mark.parametrize(
    ("module", "bad_origin"),
    [
        ("cvti.api.app", "{source}/cvti/api/app.py"),
        ("torch", "/external/site-packages/torch/__init__.py"),
        ("cvti.api.app", "{sibling}/cvti/api/app.py"),
        ("torch", "namespace"),
    ],
)
def test_rejects_nonbundle_module_origins(tmp_path, module, bad_origin):
    root, executable = _artifact(tmp_path)
    report = _report(root, executable)
    if "{source}" in bad_origin:
        bad_origin = bad_origin.format(source=str(_HARNESS.SOURCE_ROOT))
    if "{sibling}" in bad_origin:
        bad_origin = bad_origin.format(sibling=str(root) + "-sibling")
    report["modules"][module] = bad_origin
    path = _write(tmp_path / "report.json", report)
    with pytest.raises(RuntimeError, match="outside relocated bundle|not absolute"):
        _check(path, root, executable)


def test_rejects_module_origin_through_escaping_symlink(tmp_path):
    root, executable = _artifact(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    escape = root / "escape"
    try:
        escape.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    report = _report(root, executable)
    report["modules"]["torch"] = str(escape / "torch.py")
    with pytest.raises(RuntimeError, match="outside relocated bundle"):
        _check(_write(tmp_path / "report.json", report), root, executable)


def test_suffix_impostors_do_not_satisfy_exact_siglip_modules(tmp_path):
    root, executable = _artifact(tmp_path)
    modules = (RUNTIME_MODULES - {
        "transformers.models.siglip.modeling_siglip",
        "transformers.models.siglip.image_processing_siglip",
    }) | {"third_party.modeling_siglip", "third_party.image_processing_siglip"}
    report = _report(root, executable, modules=modules)
    with pytest.raises(RuntimeError, match="required frozen runtime modules"):
        _check(_write(tmp_path / "report.json", report), root, executable)


def test_accepts_fast_siglip_processor_without_requiring_unused_base(tmp_path):
    root, executable = _artifact(tmp_path)
    modules = (RUNTIME_MODULES - {"transformers.models.siglip.image_processing_siglip"}) | {
        "transformers.models.siglip.image_processing_siglip_fast"
    }
    result = _check(_write(tmp_path / "report.json", _report(root, executable, modules=modules)),
                    root, executable)
    assert "transformers.models.siglip.image_processing_siglip_fast" in result["runtime_origins"]


def test_require_frozen_alone_checks_restart_independently(tmp_path):
    root, executable = _artifact(tmp_path)
    initial = _write(tmp_path / "initial.json", _report(root, executable))
    restart = _write(tmp_path / "restart.json", _report(root, executable, frozen=False, modules={"cvti.api.app"}))
    with pytest.raises(RuntimeError, match="sys.frozen=true"):
        validate_import_reports(
            initial, restart, root, executable,
            require_frozen=True, validate_provenance=False,
        )


def test_provenance_checks_restart_for_source_contamination(tmp_path):
    root, executable = _artifact(tmp_path)
    initial = _write(tmp_path / "initial.json", _report(root, executable))
    restart_report = _report(root, executable, modules={"cvti.api.app"})
    restart_report["modules"]["cvti.api.app"] = "/checkout/cvti/api/app.py"
    restart = _write(tmp_path / "restart.json", restart_report)
    with pytest.raises(RuntimeError, match="outside relocated bundle"):
        validate_import_reports(
            initial, restart, root, executable,
            require_frozen=True, validate_provenance=True,
        )


def test_rejects_sibling_prefix_executable(tmp_path):
    root, executable = _artifact(tmp_path)
    sibling = tmp_path / "Argus-sibling"
    sibling.mkdir()
    wrong = sibling / "argus-api"
    wrong.write_bytes(b"binary")
    report = _report(root, executable)
    report["executable"] = str(wrong)
    with pytest.raises(RuntimeError, match="outside relocated bundle"):
        _check(_write(tmp_path / "report.json", report), root, executable)
