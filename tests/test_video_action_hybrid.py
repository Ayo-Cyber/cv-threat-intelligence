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
