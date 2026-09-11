"""Frozen entry point for the Argus detection engine (EP-05-T1).

PyInstaller needs a script, not a module path, so this is the executable
surface of `python -m cvti.serving.pipeline` inside the installed bundle.

Two bundle-only concerns live here and nowhere else:

- `freeze_support()` first: torch/ultralytics fork worker processes, and on
  Windows a frozen child re-executes this entry — without this line, starting
  the engine would recursively launch engines until the machine fell over.

- chdir to the bundle's resource root: the pipeline's default paths
  (models/yolov8n.pt, runs/video_finetune/videomae, prompts/…) are
  repo-relative by design. Inside the bundle those same relative paths exist
  under sys._MEIPASS, so making that the working directory lets every default
  resolve without teaching the whole pipeline about frozen mode. Everything
  the engine WRITES arrives as an absolute path (--output-dir, --site-config)
  from the app, so nothing is ever written into the bundle.
"""
import multiprocessing
import os
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    # A frozen app must NEVER pip-install at runtime. Without this, a missing
    # optional dep (onnxruntime absent from the 1.8.8 Windows bundle) sent
    # ultralytics' AutoUpdate off downloading wheels INSIDE the installed app
    # at engine start — minutes of 100% CPU that read as "the engine is not
    # starting" (Windows diagnostics, 10 Sep). Missing dep = clean fallback,
    # visibly, never a runtime pip.
    os.environ.setdefault("YOLO_AUTOINSTALL", "False")
    if getattr(sys, "frozen", False):
        os.chdir(getattr(sys, "_MEIPASS", os.path.dirname(sys.executable)))
        # Offline object rules (W3/W8): ultralytics loads CLIP with a bare
        # clip.load("ViT-B/32") — no path parameter — so the checkpoint must
        # sit in ~/.cache/clip. The bundle ships it under vendor/clip; copy
        # it across once so first use needs no network. Copy, not symlink:
        # the bundle dir may be replaced by an upgrade while the cache lives on.
        try:
            import shutil
            _src = os.path.join(os.getcwd(), "vendor", "clip", "ViT-B-32.pt")
            _dst_dir = os.path.join(os.path.expanduser("~"), ".cache", "clip")
            _dst = os.path.join(_dst_dir, "ViT-B-32.pt")
            if os.path.exists(_src) and not os.path.exists(_dst):
                os.makedirs(_dst_dir, exist_ok=True)
                shutil.copyfile(_src, _dst)
        except Exception:  # noqa: BLE001 - seeding is best-effort; the rule
            pass           # falls back to the VLM path, visibly, if it fails
    from cvti.serving.pipeline import main
    main()
