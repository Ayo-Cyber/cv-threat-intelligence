"""Feedback / reinforcement-training subsystem.

Closes the loop between operator labels (True threat / False alarm) and the
detection stack:

  store       — read labeled events + their evidence from the events DB
  calibration — per-(camera, rule) precision -> demote chronically-wrong rules (online)
  dataset     — turn labeled clips into a fine-tuning dataset (offline)
  registry    — track model checkpoints + rollback
  manager     — orchestrate all of the above (FeedbackManager + CLI)
"""
from cvti.feedback.calibration import Calibration
from cvti.feedback.store import FeedbackStore

__all__ = ["FeedbackStore", "Calibration"]
