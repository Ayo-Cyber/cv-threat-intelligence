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
import os
import sys

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
    from cvti.api.__main__ import main
    main()
