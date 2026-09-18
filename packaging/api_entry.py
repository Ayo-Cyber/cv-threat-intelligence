"""Frozen entry point for the Argus Engine API — the desktop UI's backend.

PyInstaller needs a script, not a module path, so this is the executable
surface of `python -m cvti.api` inside the installed bundle. The Electron
shell spawns THIS binary: an installed customer has no repo and no venv, so
the module invocation the dev path uses cannot exist on their machine.

Two bundle-only concerns, the same ones engine_entry.py carries and for the
same reasons:

- `freeze_support()` first, before anything can fork.
- chdir to the bundle's resource root, so the repo-relative defaults baked
  into the backend (configs/, prompts/, models/) resolve. Everything the API
  WRITES arrives as an absolute path from the shell (--db, --site), so
  nothing is ever written back into the bundle.

Starting the engine needs no special handling here: the API runs frozen, and
console_backend._bundled_engine() already resolves `argus-engine` beside its
own executable — which is exactly where COLLECT puts it.
"""
import multiprocessing
import json
import os
import sys


def _module_origin(module):
    origin = getattr(getattr(module, "__spec__", None), "origin", None)
    return origin if origin is not None else getattr(module, "__file__", None)


def _torch_generated_source_report():
    """Attest PyTorch 2.8's known non-scriptable RemoteModule source file.

    Importantly, this only inspects modules already loaded by inference. It does
    not import torch or invoke the instantiator's code-generation method.
    """
    torch = sys.modules.get("torch")
    instantiator = sys.modules.get("torch.distributed.nn.jit.instantiator")
    templates = sys.modules.get(
        "torch.distributed.nn.jit.templates.remote_module_template")
    generated = sys.modules.get("_remote_module_non_scriptable")
    if torch is None or instantiator is None or templates is None or generated is None:
        return None
    version = getattr(torch, "__version__", None)
    git_version = getattr(getattr(torch, "version", None), "git_version", None)
    if (version != "2.8.0"
            or git_version != "a1cb3cc05d46d198467bebbb6e8fba50a325d4e7"):
        return None
    generated_origin = _module_origin(generated)
    temp_dir = getattr(instantiator, "INSTANTIATED_TEMPLATE_DIR_PATH", None)
    if not isinstance(generated_origin, str) or not isinstance(temp_dir, str):
        return None
    try:
        expected = templates.get_remote_module_template(True).format(
            assign_module_interface_cls="module_interface_cls = None",
            args="*args", kwargs="**kwargs", arg_types="*args, **kwargs",
            arrow_and_return_type="", arrow_and_future_return_type="",
            jit_script_decorator="",
        )
        import hashlib
        with open(generated_origin, "rb") as generated_file:
            actual = generated_file.read()
        entries = sorted(os.listdir(temp_dir))
        pycache = os.path.join(temp_dir, "__pycache__")
        pycache_entries = sorted(os.listdir(pycache)) if os.path.isdir(pycache) else []
    except (AttributeError, KeyError, OSError, TypeError):
        return None
    return {
        "torch_version": version,
        "torch_git_version": git_version,
        "generator_origin": _module_origin(instantiator),
        "template_origin": _module_origin(templates),
        "temp_dir": temp_dir,
        "resolved_temp_dir": os.path.realpath(temp_dir),
        "module_name": "_remote_module_non_scriptable",
        "module_filename": "_remote_module_non_scriptable.py",
        "module_origin": generated_origin,
        "resolved_module_origin": os.path.realpath(generated_origin),
        "actual_content_sha256": hashlib.sha256(actual).hexdigest(),
        "expected_template_sha256": hashlib.sha256(expected.encode()).hexdigest(),
        "directory_entries": entries,
        "pycache_entries": pycache_entries,
    }


def _write_import_report(path: str) -> None:
    """Write an opt-in, local-only account of where frozen imports came from."""
    if not path or not os.path.isabs(path):
        return
    modules = {}
    for name, module in sorted(sys.modules.items()):
        origin = _module_origin(module)
        if origin is not None:
            modules[name] = os.fspath(origin)
    report = {
        "sys.frozen": bool(getattr(sys, "frozen", False)),
        "executable": sys.executable,
        "_MEIPASS": getattr(sys, "_MEIPASS", None),
        "sys_path": list(sys.path),
        "modules": modules,
    }
    torch_generated = _torch_generated_source_report()
    if torch_generated is not None:
        report["torch_generated_remote_module"] = torch_generated
    # The caller creates the private destination directory. Do not create or
    # inspect any other path and do not include environment or request data.
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    # A frozen app must never pip-install at runtime (10 Sep field diagnostics).
    os.environ.setdefault("YOLO_AUTOINSTALL", "False")
    if getattr(sys, "frozen", False):
        os.chdir(getattr(sys, "_MEIPASS", os.path.dirname(sys.executable)))
    # Account recovery for an installed machine. The dev path is a loose
    # script under Frontend/scripts, which a customer with no repo and no
    # Python cannot run — before this flag a locked-out installation had no
    # way back in. Same flow, same guards (cvti/security/recovery.py):
    #   argus-api --recover-account --db "<events.db>"
    if "--recover-account" in sys.argv[1:]:
        import argparse
        p = argparse.ArgumentParser(prog="argus-api --recover-account")
        p.add_argument("--recover-account", action="store_true")
        p.add_argument("--db", required=True,
                       help="the events.db of the site to recover; auth.db sits beside it")
        args, _ = p.parse_known_args()
        from cvti.security.recovery import run_interactive
        raise SystemExit(run_interactive(args.db))
    try:
        from cvti.api.__main__ import main
        main()
    finally:
        _report = os.environ.get("ARGUS_FROZEN_IMPORT_REPORT", "")
        if _report:
            try:
                _write_import_report(_report)
            except OSError:
                # Diagnostics must never change normal API shutdown behavior.
                pass
