"""App-level exceptions the API bridge maps to HTTP statuses.

Kept free of cv2/Qt imports so the API process can name them at startup.
"""
from __future__ import annotations


class PreviewBusy(RuntimeError):
    """A camera preview is still releasing its capture.

    Raised when the engine is about to start and the preview's decode thread
    has not let go of the camera within the grace period (a blocked RTSP read
    takes seconds to return). Starting anyway would open a second handle on a
    camera that may only allow one (the pilot's Tapo caps RTSP sessions), so
    the start is refused — but it is a retry-in-a-moment refusal, not a fault.
    """
