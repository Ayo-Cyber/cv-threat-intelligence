# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for CV Threat Intelligence desktop app.

Build:
  Mac/Linux:  pyinstaller cvti.spec
  Windows:    pyinstaller cvti.spec
"""

import sys
from pathlib import Path

block_cipher = None

# ---------------------------------------------------------------------------
# Data files bundled inside the app
# ---------------------------------------------------------------------------
datas = [
    ("models/*.pt",        "models"),
    ("configs/*.json",     "configs"),
    ("configs/*.yaml",     "configs"),
    ("prompts/*.txt",      "prompts"),
    ("schemas/*.json",     "schemas"),
]

# ---------------------------------------------------------------------------
# Hidden imports — packages that PyInstaller misses through static analysis
# ---------------------------------------------------------------------------
hidden = [
    # ultralytics internals
    "ultralytics",
    "ultralytics.models",
    "ultralytics.models.yolo",
    "ultralytics.models.yolo.detect",
    "ultralytics.models.yolo.classify",
    "ultralytics.models.yolo.pose",
    "ultralytics.nn",
    "ultralytics.nn.tasks",
    "ultralytics.data",
    "ultralytics.data.loaders",
    "ultralytics.utils",
    "ultralytics.utils.plotting",
    "ultralytics.trackers",
    "ultralytics.trackers.byte_tracker",
    "ultralytics.trackers.utils",
    # scipy (used by ByteTrack for Hungarian assignment)
    "scipy",
    "scipy.optimize",
    "scipy.spatial",
    "scipy.linalg",
    # PIL / Pillow (used by ultralytics internally)
    "PIL",
    "PIL.Image",
    "PIL.ImageDraw",
    "PIL.ImageFont",
    "PIL.ImageOps",
    # matplotlib (used by ultralytics for plotting)
    "matplotlib",
    "matplotlib.pyplot",
    "matplotlib.backends.backend_agg",
    # pandas (used by ultralytics for results)
    "pandas",
    # seaborn (used by ultralytics for confusion matrix)
    "seaborn",
    # supervision
    "supervision",
    "supervision.detection",
    "supervision.annotators",
    # cv2
    "cv2",
    # torch
    "torch",
    "torch.nn",
    "torchvision",
    # PyQt6
    "PyQt6",
    "PyQt6.QtCore",
    "PyQt6.QtGui",
    "PyQt6.QtWidgets",
    "PyQt6.sip",
    # cvti package
    "cvti",
    "cvti.utils",
    "cvti.app",
    "cvti.app.main",
    "cvti.app.window",
    "cvti.app.worker",
    "cvti.app.widgets",
    "cvti.app.widgets.feed",
    "cvti.app.widgets.alerts",
    "cvti.app.widgets.mapper",
    "cvti.app.widgets.config",
    "cvti.scene",
    "cvti.scene.agent_mapper",
    "cvti.retail",
    "cvti.retail.zones",
    "cvti.retail.concealment",
    "cvti.rules",
    "cvti.rules.customization",
    "cvti.verification",
    "cvti.verification.gate",
    "cvti.event_adapters",
    "cvti.contracts",
]

a = Analysis(
    ["cvti/app/main.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "jupyter", "IPython", "tkinter", "wx",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# ---------------------------------------------------------------------------
# Single EXE (Windows) / binary (Linux)
# ---------------------------------------------------------------------------
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="CVTI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="cvti/app/assets/icon.ico" if sys.platform == "win32" else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="CVTI",
)

# ---------------------------------------------------------------------------
# Mac .app bundle
# ---------------------------------------------------------------------------
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="CVTI.app",
        icon="cvti/app/assets/icon.icns",
        bundle_identifier="com.cvti.threat-intelligence",
        info_plist={
            "CFBundleName": "CV Threat Intelligence",
            "CFBundleDisplayName": "CV Threat Intelligence",
            "CFBundleVersion": "0.1.0",
            "CFBundleShortVersionString": "0.1.0",
            "CFBundleExecutable": "CVTI",
            "NSHighResolutionCapable": True,
            "NSCameraUsageDescription": "CVTI needs camera access for live threat detection.",
            "LSMinimumSystemVersion": "10.15.0",
            "LSApplicationCategoryType": "public.app-category.utilities",
        },
    )
