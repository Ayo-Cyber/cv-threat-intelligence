"""Phase 8 multi-stream serving pipeline.

Runs many camera streams on one edge box in near real-time:

    decode threads (latest-frame-drop)  -> cross-camera batcher
      -> ONE batched detector forward pass
      -> per-camera tracker + rules
      -> alert queue (dedup + throttle)  -> async VLM gate pool

The single-stream `cvti-detect` path is untouched; this is an additive layer.
"""

from cvti.serving.alert_queue import AlertQueue, QueuedAlert
from cvti.serving.batcher import collect_batch
from cvti.serving.camera import PerCameraState, build_camera_states, load_site_config
from cvti.serving.gate_pool import GatePool
from cvti.serving.streams import StreamDecoder

__all__ = [
    "AlertQueue", "QueuedAlert", "collect_batch", "StreamDecoder",
    "PerCameraState", "build_camera_states", "load_site_config", "GatePool",
]
