# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the CVTI Console desktop app (web-in-native-shell).

Build from the repo ROOT:
    pyinstaller packaging/cvti-console.spec --noconfirm

Produces:
    dist/CVTI Console/            (onedir, all platforms)
    dist/CVTI Console.app         (macOS bundle, additionally)

PyInstaller does NOT cross-compile — run this on each target OS. The GitHub
Actions workflow (.github/workflows/build-app.yml) builds all three from one push.

The console app is deliberately lean: it views events.db and onboards cameras
(sqlite + OpenCV). It does NOT contain the detection engine (torch / ultralytics
/ transformers) — those run in the separate `cvti.serving.pipeline` service — so
they're excluded to keep the bundle small.
"""
import os
import sys

# Paths in a spec resolve relative to the spec file, not the CWD. Anchor
# everything to the repo root (the parent of packaging/) so the build works
# regardless of where PyInstaller is invoked from.
ROOT = os.path.abspath(os.path.join(SPECPATH, os.pardir))

# Qt WebEngine is the fragile part of the bundle: it needs its widgets/core/
# channel modules plus the Quick/QML runtime it uses internally. PyInstaller's
# PyQt6 hooks collect the WebEngine helper process + resources; we just make sure
# these submodules are seen.
hiddenimports = [
    "PyQt6.QtWebEngineWidgets",
    "PyQt6.QtWebEngineCore",
    "PyQt6.QtWebChannel",
    "PyQt6.QtNetwork",
    "PyQt6.QtPrintSupport",
    "PyQt6.QtQml",
    "PyQt6.QtQuick",
    "PyQt6.QtGui",
]

# The web frontend assets ship inside the bundle at the same relative path
# cvti/app/shell.py loads them from (cvti/app/web/index.html).
datas = [(os.path.join(ROOT, "cvti", "app", "web"), "cvti/app/web")]

# Self-contained playback demo (clips + recorded alerts), if it's been built
# (packaging/build_demo_data.py). Lets the app demo itself on any machine.
_demo = os.path.join(ROOT, "packaging", "demo_data")
if os.path.isdir(_demo):
    datas.append((_demo, "demo_data"))

# The detection stack never runs inside the desktop app — keep it out.
excludes = [
    "torch", "torchvision", "ultralytics", "transformers", "pytorchvideo",
    "matplotlib", "scipy", "pandas", "tkinter",
    "PyQt5", "PySide6", "PySide2", "IPython", "pytest", "notebook",
]

a = Analysis(
    [os.path.join(ROOT, "cvti", "app", "shell.py")],
    pathex=[ROOT],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="CVTI Console",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,            # windowed GUI app (no terminal window)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="CVTI Console",
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="CVTI Console.app",
        icon=None,
        bundle_identifier="com.cvti.console",
        info_plist={
            "NSHighResolutionCapable": True,
            "CFBundleShortVersionString": "0.9.0",
            "CFBundleVersion": "0.9.0",
        },
    )
