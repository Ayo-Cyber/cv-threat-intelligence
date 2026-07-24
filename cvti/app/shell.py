"""CVTI Console — native desktop app shell (Phase 9).

A native window hosting the operator UI (web frontend) via Qt WebEngine, wired
to the Python backend through QWebChannel. Runs offline on the edge box; it's a
real installable desktop app (web tech inside a native shell, like VS Code).

    cvti-console --site-config configs/site_live.json --db runs/site/events.db
    # or: python -m cvti.app.shell

Robustness notes (macOS / Qt 6.10):
  * AA_ShareOpenGLContexts must be set BEFORE the QApplication or WebEngine
    renders a blank window.
  * qtwebchannel.js is a qrc-only resource; a file:// page loading a qrc://
    script is blocked cross-origin, so we inject it via QWebEngineScript at
    document creation instead of referencing it from the HTML.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _web_index() -> Path:
    """Locate web/index.html in dev AND in a PyInstaller bundle.

    Frozen builds unpack data under sys._MEIPASS (the assets ship at
    cvti/app/web via packaging/cvti-console.spec); in dev it sits next to this
    file.
    """
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", ".")) / "cvti" / "app" / "web" / "index.html"
    return Path(__file__).resolve().parent / "web" / "index.html"


def _qwebchannel_js() -> str:
    """Read the qtwebchannel.js runtime out of the Qt resource system."""
    from PyQt6.QtCore import QFile, QIODevice
    f = QFile(":/qtwebchannel/qwebchannel.js")
    if f.open(QIODevice.OpenModeFlag.ReadOnly):
        try:
            return bytes(f.readAll()).decode("utf-8")
        finally:
            f.close()
    return ""


def main() -> None:
    p = argparse.ArgumentParser(description="CVTI operator console (desktop app).")
    p.add_argument("--site-config", default="configs/site_live.json")
    p.add_argument("--db", default="runs/site/events.db")
    args = p.parse_args()

    try:
        from PyQt6.QtCore import Qt, QUrl
        from PyQt6.QtWebChannel import QWebChannel
        from PyQt6.QtWebEngineCore import QWebEngineScript
        from PyQt6.QtWebEngineWidgets import QWebEngineView
        from PyQt6.QtWidgets import QApplication, QMainWindow
    except ImportError as exc:
        print(f"CVTI Console needs PyQt6 + WebEngine ({exc}):\n"
              "  pip install PyQt6 PyQt6-WebEngine")
        sys.exit(1)

    from cvti.app.bridge import Backend
    from cvti.app.console_backend import ConsoleBackend

    # Must be set before the QApplication is constructed (else blank render).
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    app = QApplication(sys.argv)
    app.setApplicationName("CVTI Console")

    win = QMainWindow()
    win.setWindowTitle("CVTI Console")
    win.resize(1320, 860)

    view = QWebEngineView()
    page = view.page()

    backend = Backend(ConsoleBackend(site_path=args.site_config, db_path=args.db))
    channel = QWebChannel(page)
    channel.registerObject("backend", backend)
    page.setWebChannel(channel)

    # Inject the web-channel runtime at document creation so index.html doesn't
    # have to load it via qrc:// (blocked cross-origin from a file:// page).
    js = _qwebchannel_js()
    if js:
        script = QWebEngineScript()
        script.setName("qwebchannel")
        script.setSourceCode(js)
        script.setInjectionPoint(QWebEngineScript.InjectionPoint.DocumentCreation)
        script.setWorldId(QWebEngineScript.ScriptWorldId.MainWorld)
        script.setRunsOnSubFrames(False)
        page.scripts().insert(script)
    else:
        print("[warn] could not read qtwebchannel.js from Qt resources; UI bridge may not connect")

    def _loaded(ok: bool) -> None:
        print(f"[cvti-console] page loaded ok={ok}")
        if not ok:
            print("[cvti-console] load FAILED — check the index.html path above")
    page.loadFinished.connect(_loaded)

    index = _web_index()
    print(f"[cvti-console] loading {index}")
    if not index.exists():
        print(f"[cvti-console] ERROR: {index} not found")
        sys.exit(1)
    view.load(QUrl.fromLocalFile(str(index)))

    win.setCentralWidget(view)
    win.show()
    win.raise_()
    win.activateWindow()
    print("[cvti-console] window shown — if you don't see it, check other desktops/Spaces")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
