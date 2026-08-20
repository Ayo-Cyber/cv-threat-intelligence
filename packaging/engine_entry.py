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
    if getattr(sys, "frozen", False):
        os.chdir(getattr(sys, "_MEIPASS", os.path.dirname(sys.executable)))
    from cvti.serving.pipeline import main
    main()
