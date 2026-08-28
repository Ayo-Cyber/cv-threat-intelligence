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
from cvti.logging_setup import get_logger

log = get_logger(__name__)


def _web_index() -> Path:
    """Locate web/index.html in dev AND in a PyInstaller bundle.

    Frozen builds unpack data under sys._MEIPASS (the assets ship at
    cvti/app/web via packaging/argus.spec); in dev it sits next to this
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
    p = argparse.ArgumentParser(description="Argus operator console (desktop app).")
    p.add_argument("--site-config", default=None)
    p.add_argument("--db", default=None)
    p.add_argument("--smoke", action="store_true",
                   help="boot, load the UI, exit 0 if the page rendered — CI's "
                        "proof that the SHIPPED APP starts, not just the engine")
    args = p.parse_args()

    # Dev keeps the repo-relative defaults. A frozen app is launched from
    # Finder/Explorer with cwd '/', so relative paths would try to write into
    # the filesystem root (and fail) — its site lives in the per-user data dir.
    if getattr(sys, "frozen", False):
        from cvti.utils import user_data_dir
        site_dir = user_data_dir() / "site"
        site_dir.mkdir(parents=True, exist_ok=True)
        args.site_config = args.site_config or str(site_dir / "site.json")
        args.db = args.db or str(site_dir / "events.db")
    else:
        args.site_config = args.site_config or "configs/site_live.json"
        args.db = args.db or "runs/site/events.db"

    from cvti.logging_setup import setup_logging
    setup_logging(Path(args.db).parent, component="argus-app")

    try:
        from PyQt6.QtCore import Qt, QUrl
        from PyQt6.QtWebChannel import QWebChannel
        from PyQt6.QtWebEngineCore import QWebEngineScript
        from PyQt6.QtWebEngineWidgets import QWebEngineView
        from PyQt6.QtWidgets import QApplication, QMainWindow
    except ImportError as exc:
        log.info(f"Argus Console needs PyQt6 + WebEngine ({exc}):\n"
              "  pip install PyQt6 PyQt6-WebEngine")
        sys.exit(1)

    from cvti.app.bridge import Backend
    from cvti.app.console_backend import ConsoleBackend

    # Must be set before the QApplication is constructed (else blank render).
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    app = QApplication(sys.argv)
    app.setApplicationName("Argus Console")

    win = QMainWindow()
    win.setWindowTitle("Argus Console")
    win.resize(1320, 860)

    view = QWebEngineView()
    page = view.page()

    # The live wall fetches JPEG frames from a localhost HTTP server via <img>.
    # A file:// page can't load remote (http) URLs unless we allow it.
    from PyQt6.QtWebEngineCore import QWebEngineSettings
    st = view.settings()
    st.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
    st.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
    # Let event-clip <video> autoplay without a click (else playback looks broken).
    st.setAttribute(QWebEngineSettings.WebAttribute.PlaybackRequiresUserGesture, False)

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
        log.error("[warn] could not read qtwebchannel.js from Qt resources; UI bridge may not connect")

    def _loaded(ok: bool) -> None:
        log.info(f"[cvti-console] page loaded ok={ok}")
        if not ok:
            log.error("[cvti-console] load FAILED — check the index.html path above")
        if args.smoke:
            # Until 28 Aug nothing anywhere launched the app itself: CI proved
            # the ENGINE ran on a clean machine while Argus.exe had only ever
            # been started on developer laptops. This flag makes "the app
            # boots and renders its UI" a release gate on all three OSes.
            log.info(f"[cvti-console] smoke: exiting {0 if ok else 1}")
            QApplication.instance().exit(0 if ok else 1)
    page.loadFinished.connect(_loaded)

    index = _web_index()
    log.info(f"[cvti-console] loading {index}")
    if not index.exists():
        log.error(f"[cvti-console] ERROR: {index} not found")
        sys.exit(1)
    view.load(QUrl.fromLocalFile(str(index)))

    win.setCentralWidget(view)
    win.show()
    win.raise_()
    win.activateWindow()
    log.info("[cvti-console] window shown — if you don't see it, check other desktops/Spaces")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
