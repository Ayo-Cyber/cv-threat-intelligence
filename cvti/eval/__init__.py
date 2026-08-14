"""Evaluation harness — measure what Argus actually catches, and what TrueSight suppresses.

The product claim is "context-aware verification kills false alarms". This package
measures that on HELD-OUT clips, at two stages:

    clips -> detectors -> Stage 1 (raw candidates)   what a plain CV system would alert on
                       -> TrueSight -> Stage 2       what actually reaches the operator

Reporting both is the point: the delta between them IS the product.
"""
from cvti.eval.dataset import EvalClip, load_dataset
from cvti.eval.metrics import StageMetrics, compare_stages

__all__ = ["EvalClip", "load_dataset", "StageMetrics", "compare_stages"]
