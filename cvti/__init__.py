"""CV Threat Intelligence package."""

import os


# Production runtimes must never let Ultralytics install packages implicitly.
# Assignment (rather than setdefault) deliberately enforces that policy even
# when a parent process supplied a conflicting value.
os.environ["YOLO_AUTOINSTALL"] = "false"
