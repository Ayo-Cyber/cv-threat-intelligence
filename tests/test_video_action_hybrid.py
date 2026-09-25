from __future__ import annotations

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvti.rules.customization import CustomizationEngine
from cvti.video_action_hybrid import predictions_to_events
from cvti.video_action_model import VideoActionPrediction


class VideoActionHybridTests(unittest.TestCase):
    def test_predictions_to_events_maps_violence_label_as_weak_video_action_event(self) -> None:
        events = predictions_to_events(
            [VideoActionPrediction(label="punching person (boxing)", confidence=0.16, rank=1)],
            backend="videomae",
            model_name="test-model",
            window_name="event",
            sampled_frame_indices=[100, 110, 120],
            timestamp=4.2,
        )

        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event.detector, "video_action")
        self.assertEqual(event.level, "low")
        self.assertEqual(event.extra["signal_type"], "violence_candidate")
        self.assertEqual(event.extra["raw_confidence"], 0.16)
        self.assertAlmostEqual(event.extra["adjusted_confidence"], 0.056)
        self.assertEqual(event.extra["sampled_frame_indices"], [100, 110, 120])

    def test_predictions_to_events_ignores_unmapped_low_value_label(self) -> None:
        events = predictions_to_events(
            [VideoActionPrediction(label="folding paper", confidence=0.24, rank=1)],
            backend="videomae",
            model_name="test-model",
            window_name="event",
            sampled_frame_indices=[1, 2, 3],
        )

        self.assertEqual(events, [])

    def test_video_action_events_can_flow_through_customization_engine(self) -> None:
        engine = CustomizationEngine()
        engine.rules = [
            {
                "name": "weak_video_violence",
                "trigger": {"detector": "video_action"},
                "context_filter": "signal_type == 'violence_candidate' and adjusted_confidence >= 0.05",
                "priority": "medium",
            }
        ]
        events = predictions_to_events(
            [VideoActionPrediction(label="punching person (boxing)", confidence=0.16, rank=1)],
            backend="videomae",
            model_name="test-model",
            window_name="event",
            sampled_frame_indices=[100, 110, 120],
        )

        alerts = engine.evaluate(events)

        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].rule_name, "weak_video_violence")
        self.assertEqual(alerts[0].detector, "video_action")


if __name__ == "__main__":
    unittest.main()


# --- per-signal thresholds (25 Sep) -----------------------------------------
# Every false positive in the 24 Sep KPI scorecard came from the theft signal,
# and each one costs a ~12s VLM verification on the pilot's 4-core box. A sweep
# over 170 normals + 162 theft clips put its AUC at 0.659 — barely above a coin
# flip — so theft now carries its own, much higher bar while the signals that
# measure well keep the permissive one.

def _prediction(label, confidence):
    from cvti.video_action_model import VideoActionPrediction
    return VideoActionPrediction(label=label, confidence=confidence, rank=1)


def _events(label, confidence, **kw):
    from cvti.video_action_hybrid import predictions_to_events
    return predictions_to_events(
        [_prediction(label, confidence)], backend="videomae", model_name="m",
        window_name="w", sampled_frame_indices=[0], **kw)


def test_a_weak_theft_score_no_longer_raises_a_candidate():
    # 0.20 cleared the old global 0.05 bar and was 12.9% false positives.
    assert _events("theft", 0.20) == []


def test_a_confident_theft_score_still_raises_one():
    events = _events("theft", 0.97)
    assert len(events) == 1
    assert events[0].extra["signal_type"] == "theft_candidate"


def test_violence_keeps_the_permissive_bar():
    # Row 9 measures the violence path at 97.2%; the theft bar must not touch it.
    assert len(_events("punching", 0.20)) == 1


def test_a_site_can_override_the_bar_in_either_direction():
    assert _events("theft", 0.20, signal_thresholds={"theft_candidate": 0.10})
    assert _events("theft", 0.99, signal_thresholds={"theft_candidate": 1.01}) == []
