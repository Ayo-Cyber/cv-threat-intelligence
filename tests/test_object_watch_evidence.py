from __future__ import annotations

from unittest import mock

import cv2
import numpy as np

from cvti.contracts import CandidateAlert
from cvti.object_watch.evidence import build_object_watch_evidence
from cvti.object_watch.images import encode_rgb_array
from cvti.verification.gate import VerificationGate


def _alert(detector="object_watch"):
    return CandidateAlert(
        rule_name="chi_seen",
        priority="high",
        detector=detector,
        title="CHI CARTON OBJECT SEEN",
        person_id=None,
        object_label="Chi carton",
        timestamp=1.0,
    )


def test_comparison_panel_survives_one_frame_gate_cap_with_both_images():
    # Candidate is red in BGR; enrolled reference is green.
    observation = np.full((80, 100, 3), (0, 0, 255), dtype=np.uint8)
    reference = np.full((50, 70, 3), (0, 255, 0), dtype=np.uint8)
    evidence = build_object_watch_evidence(
        observation,
        (10, 10, 90, 70),
        encode_rgb_array(reference),
        "Chi carton",
        include_context=True,
    )
    assert len(evidence) == 2

    seen = {}
    gate = VerificationGate(provider="ollama", cot=False, max_frames=1)

    def provider(prompt, frames_bytes, alert):
        seen["prompt"] = prompt
        seen["frames"] = frames_bytes
        return ('{"confirmed": false, "confidence": 0.9, '
                '"reason": "unknown", "alert_priority": "high"}')

    with mock.patch.object(gate, "_call_provider", side_effect=provider):
        gate.verify(evidence, _alert())

    assert len(seen["frames"]) == 1
    panel = cv2.imdecode(np.frombuffer(seen["frames"][0], dtype=np.uint8), cv2.IMREAD_COLOR)
    assert np.count_nonzero((panel[:, :, 2] > 200) & (panel[:, :, 1] < 40)) > 100
    assert np.count_nonzero((panel[:, :, 1] > 200) & (panel[:, :, 2] < 40)) > 100
    prompt = seen["prompt"].lower()
    assert "labelled comparison panel" in prompt
    assert "not a chronological pair" in prompt
    assert "resemblance is not identity proof" in prompt
    assert "unknown" in prompt


def test_object_seen_uses_object_prompt_with_reference_panel_for_both_modes():
    observation = np.full((80, 100, 3), (0, 0, 255), dtype=np.uint8)
    reference = np.full((50, 70, 3), (0, 255, 0), dtype=np.uint8)
    evidence = build_object_watch_evidence(
        observation,
        (10, 10, 90, 70),
        encode_rgb_array(reference),
        "Chi carton",
        include_context=True,
    )

    for cot in (True, False):
        seen = {}
        gate = (VerificationGate(provider="ollama", max_frames=1)
                if cot else VerificationGate(provider="ollama", cot=False, max_frames=1))

        def provider(prompt, frames_bytes, alert):
            seen["prompt"] = prompt
            seen["frames"] = frames_bytes
            return ('{"confirmed": false, "confidence": 0.9, '
                    '"reason": "unknown", "alert_priority": "high"}')

        with mock.patch.object(gate, "_call_provider", side_effect=provider):
            gate.verify(evidence, _alert(), {"scene_description": "No people visible."})

        prompt = " ".join(seen["prompt"].lower().split())
        assert "candidate and enrolled reference" in prompt
        assert "stationary matching object" in prompt
        assert "a person" in prompt and "is not required" in prompt
        assert "visual resemblance" in prompt
        assert "not proof of identity" in prompt
        assert "theft" in prompt
        assert '"confirmed" to false' in prompt
        assert "never confirm merely because the comparison is ambiguous" in prompt
        assert "when in doubt, confirm" not in prompt
        assert "are there any people" not in prompt
        assert "use the motion across" not in prompt
        assert "ordinary activity" not in prompt
        assert "specific threat" not in prompt

        assert len(seen["frames"]) == 1
        panel = cv2.imdecode(
            np.frombuffer(seen["frames"][0], dtype=np.uint8), cv2.IMREAD_COLOR
        )
        assert np.count_nonzero((panel[:, :, 2] > 200) & (panel[:, :, 1] < 40)) > 100
        assert np.count_nonzero((panel[:, :, 1] > 200) & (panel[:, :, 2] < 40)) > 100


def test_context_is_optional_and_inputs_are_not_mutated():
    observation = np.zeros((40, 40, 3), dtype=np.uint8)
    before = observation.copy()
    reference = encode_rgb_array(np.zeros((20, 20, 3), dtype=np.uint8))
    evidence = build_object_watch_evidence(
        observation, (5, 5, 30, 30), reference, "box", include_context=False
    )
    assert len(evidence) == 1
    assert np.array_equal(observation, before)


def test_unrelated_gate_prompt_does_not_describe_a_reference_panel():
    seen = {}
    gate = VerificationGate(provider="ollama", cot=False)

    def provider(prompt, frames_bytes, alert):
        seen["prompt"] = prompt
        return ('{"confirmed": false, "confidence": 0.9, '
                '"reason": "ordinary", "alert_priority": "high"}')

    with mock.patch.object(gate, "_call_provider", side_effect=provider):
        gate.verify(np.zeros((20, 20, 3), dtype=np.uint8), _alert("presence"))

    assert "Object-watch evidence instructions" not in seen["prompt"]
    assert "catch its mistakes" in seen["prompt"]
