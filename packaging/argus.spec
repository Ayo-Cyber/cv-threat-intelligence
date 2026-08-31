# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Argus — ONE bundle containing the whole product (EP-05-T1).

    python packaging/build.py            # builds with this spec
    pyinstaller packaging/argus.spec     # same thing, by hand

Produces (onedir):
    dist/Argus/            Windows / Linux
    dist/Argus.app         macOS

Two executables ship side by side in the same bundle:

    Argus          the operator console (PyQt6/QtWebEngine GUI shell)
    argus-engine   the detection pipeline (YOLO + VideoMAE + the TrueSight gate)

The console finds and launches argus-engine next to its own executable
(console_backend._bundled_engine), so "Start monitoring" works on a machine
with no Python installed. This replaces the earlier lean viewer-only spec —
the audit's words for that artifact were "a complete product failure at first
contact": an installer whose intelligence did not come with it.

What is bundled vs. pulled on first run:
    bundled     detector weights (models/*.pt, the VideoMAE fine-tune if built),
                configs/prompts/schemas, the web UI, the Ollama runtime if
                scripts/fetch_ollama.* has been run
    first run   the TrueSight verifier model (~3.3 GB) — downloaded in-app with
                progress, and Ollama resumes partial pulls natively

PyInstaller does not cross-compile: each OS builds its own artifact
(.github/workflows/build-app.yml runs the three-OS matrix on tags).
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(SPECPATH, os.pardir))

APP_VERSION = "1.0.0"


def _tree(src_rel, dest_rel):
    """Recursively collect a directory into datas, if it exists."""
    out = []
    src = os.path.join(ROOT, src_rel)
    for base, _dirs, files in os.walk(src):
        for f in files:
            full = os.path.join(base, f)
            rel_dir = os.path.relpath(base, src)
            dest = dest_rel if rel_dir == "." else os.path.join(dest_rel, rel_dir)
            out.append((full, dest))
    return out


# ---------------------------------------------------------------------------
# Shared data files — bundled once, visible to both executables.
# ---------------------------------------------------------------------------
datas = [(os.path.join(ROOT, "cvti", "app", "web"), "cvti/app/web")]

for pattern, dest in (("models/*.pt", "models"), ("configs/*.json", "configs"),
                      ("configs/*.yaml", "configs"), ("prompts/*.txt", "prompts"),
                      ("schemas/*.json", "schemas")):
    import glob as _glob
    for f in _glob.glob(os.path.join(ROOT, pattern)):
        datas.append((f, dest))

# VideoMAE fine-tune (fights/falls) — large, and only present on a machine
# that has trained or copied it. Bundle when there; the engine degrades
# loudly (logs, /health) when absent.
if os.path.isdir(os.path.join(ROOT, "runs", "video_finetune", "videomae")):
    datas += _tree("runs/video_finetune/videomae", "runs/video_finetune/videomae")

# The Ollama runtime (~binary + GPU runners), if a build script fetched it.
# Model weights are NOT here — they pull on first run into the user data dir.
_plat = {"win32": "windows", "darwin": "darwin"}.get(sys.platform, "linux")
if os.path.isdir(os.path.join(ROOT, "vendor", "ollama", _plat)):
    datas += _tree(f"vendor/ollama/{_plat}", f"vendor/ollama/{_plat}")

# Self-contained playback demo (clips + recorded alerts), if built
# (packaging/build_demo_data.py) — lets the app demo itself anywhere.
if os.path.isdir(os.path.join(ROOT, "packaging", "demo_data")):
    datas += _tree("packaging/demo_data", "demo_data")
# The weapon detector loads through torch.hub from this vendored repo; without
# it every install failed with "No module named 'hubconf'" (pilot, 29 Aug) —
# tracked in git all along, never bundled.
if os.path.isdir(os.path.join(ROOT, "external", "yolov5")):
    datas += _tree("external/yolov5", "external/yolov5")


# ---------------------------------------------------------------------------
# The console (GUI shell). Keeps the detection stack out of ITS import graph;
# the shared COLLECT below still carries the engine's libraries.
# ---------------------------------------------------------------------------
app_a = Analysis(
    [os.path.join(ROOT, "cvti", "app", "shell.py")],
    pathex=[ROOT],
    datas=datas,
    hiddenimports=[
        "PyQt6.QtWebEngineWidgets", "PyQt6.QtWebEngineCore", "PyQt6.QtWebChannel",
        "PyQt6.QtNetwork", "PyQt6.QtPrintSupport", "PyQt6.QtQml", "PyQt6.QtQuick",
        "PyQt6.QtGui",
        # Imported inside the live-feed resolver, so PyInstaller cannot see it
        # statically. Without it, Live EarthCams exists as a button and fails
        # as a feature on every install.
        "yt_dlp",
        # Stdlib, but only imported dynamically (the yolov5 weapon-model
        # loader) — the frozen engine failed with "No module named
        # 'logging.config'" and the weapons detector was dead on every
        # install, cascading per-frame errors on any camera with weapons
        # enabled (pilot health panel, 30 Aug: detector.cam1 480 errors).
        "logging.config", "logging.handlers",
    ],
    excludes=["torch", "torchvision", "ultralytics", "transformers", "pytorchvideo",
              "matplotlib", "scipy", "pandas", "tkinter", "polars",
              "PyQt5", "PySide6", "PySide2", "IPython", "pytest", "notebook"],
    noarchive=False,
)

# ---------------------------------------------------------------------------
# The engine. Everything the pipeline imports lazily or via strings.
# ---------------------------------------------------------------------------
engine_a = Analysis(
    [os.path.join(ROOT, "packaging", "engine_entry.py")],
    pathex=[ROOT],
    datas=[],                      # shared datas ride with the app Analysis
    hiddenimports=[
        "cvti.serving.pipeline",
        # ultralytics internals reached by name
        "ultralytics", "ultralytics.models.yolo", "ultralytics.models.yolo.detect",
        "ultralytics.models.yolo.classify", "ultralytics.models.yolo.pose",
        "ultralytics.nn.tasks", "ultralytics.data.loaders",
        "ultralytics.utils", "ultralytics.utils.plotting",
        "ultralytics.trackers.byte_tracker", "ultralytics.trackers.utils",
        # VideoMAE
        "transformers", "torch", "torchvision",
        # ByteTrack assignment + image plumbing
        "scipy.optimize", "scipy.spatial", "scipy.linalg",
        "PIL.Image", "PIL.ImageDraw", "PIL.ImageFont", "PIL.ImageOps",
        "cv2",
    ],
    # polars rides in via an optional pandas/arrow path nothing here uses:
    # 156 MB of a customer's download for a dependency the product never calls.
    excludes=["PyQt6", "PyQt5", "PySide6", "PySide2", "tkinter", "polars",
              "IPython", "pytest", "notebook", "matplotlib.backends.backend_qtagg"],
    noarchive=False,
)

app_pyz = PYZ(app_a.pure)
engine_pyz = PYZ(engine_a.pure)

app_exe = EXE(
    app_pyz,
    app_a.scripts,
    [],
    exclude_binaries=True,
    name="Argus",
    debug=False,
    strip=False,
    upx=False,
    console=False,               # windowed GUI app (no terminal window)
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

engine_exe = EXE(
    engine_pyz,
    engine_a.scripts,
    [],
    exclude_binaries=True,
    name="argus-engine",
    debug=False,
    strip=False,
    upx=False,
    console=True,                # headless subprocess; stdout goes to monitor.log
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# One COLLECT: both executables share one set of libraries and data files.
def _strip_dead_weight(datas):
    """Drop what a customer downloads but never uses (bundle audit, 25 Aug):
    QtWebEngine's devtools DEBUG resources are 76 MB of symbols for a devtools
    panel the app never opens."""
    drop = ("qtwebengine_devtools_resources.debug.pak",)
    return [d for d in datas if not str(d[0]).endswith(drop)]


app_a.datas = _strip_dead_weight(app_a.datas)
engine_a.datas = _strip_dead_weight(engine_a.datas)

coll = COLLECT(
    app_exe,
    engine_exe,
    app_a.binaries,
    app_a.datas,
    engine_a.binaries,
    engine_a.datas,
    strip=False,
    upx=False,
    name="Argus",
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="Argus.app",
        icon=None,
        bundle_identifier="com.argus.desktop",
        info_plist={
            # PyInstaller stamps LSBackgroundOnly=true because the bundled
            # engine EXE is console=True — and a background-only app can NEVER
            # take keyboard focus, so every keystroke fell through to whatever
            # window was behind Argus (field report, 31 Aug: "I am typing into
            # the app but it's typing here"). Force it off.
            "LSBackgroundOnly": False,
            "NSHighResolutionCapable": True,
            "CFBundleShortVersionString": APP_VERSION,
            "CFBundleVersion": APP_VERSION,
            "NSLocalNetworkUsageDescription":
                "Argus connects to the security cameras on your local network.",
        },
    )
