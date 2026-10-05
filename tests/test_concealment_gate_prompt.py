from unittest.mock import patch
import json
import numpy as np
import pytest
from cvti.contracts import CandidateAlert
from cvti.verification.gate import VerificationGate


@pytest.mark.parametrize("cot", [False, True])
def test_concealment_does_not_use_generic_lean_confirm_prompt(cot):
    gate = VerificationGate(provider="mock", cot=cot)
    alert = CandidateAlert(rule_name="shoplifting", priority="high",
                           detector="concealment", title="Possible concealment",
                           person_id=1, object_label=None, timestamp=0)
    with patch.object(gate, "_call_provider", return_value='{"confirmed":false,"confidence":0.9,"reason":"No insertion","alert_priority":"high"}') as call:
        result = gate.verify([np.zeros((100, 100, 3), dtype=np.uint8)], alert)
    prompt = call.call_args.args[0]
    assert "lean CONFIRM" not in prompt
    assert "Ignore captions" in prompt
    assert "occluded or falls outside the frames" in prompt
    assert not result.confirmed


@pytest.mark.parametrize("fields,expected", [
    ({}, False),
    ({"item_visible": True, "action": "touching", "destination": "pocket"}, False),
    ({"item_visible": True, "action": "removal", "destination": "personal_bag"}, False),
    ({"item_visible": "true", "action": "insertion", "destination": "pocket"}, False),
    ({"item_visible": True, "action": "insertion", "destination": "personal_bag"}, False),
])
def test_confirmation_requires_structured_insertion(fields, expected):
    gate = VerificationGate(provider="ollama")
    alert = CandidateAlert(rule_name="shoplifting", priority="high",
                           detector="concealment", title="Possible concealment",
                           person_id=1, object_label=None, timestamp=0)
    raw = json.dumps({"confirmed": True, "confidence": .9, "reason": "test",
                      "alert_priority": "high", **fields})
    with patch.object(gate, "_call_provider", return_value=raw):
        result = gate.verify(np.zeros((100,100,3),dtype=np.uint8), alert)
    assert result.confirmed is expected


@pytest.mark.parametrize("destination", ["pocket", "waistband", "clothing", "personal_bag"])
def test_supported_concealment_uses_fields_not_free_text(destination):
    gate = VerificationGate(provider="ollama")
    alert = CandidateAlert(rule_name="product_concealment", priority="high",
                           detector="concealment", title="Possible concealment",
                           person_id=1, object_label=None, timestamp=0)
    raw = json.dumps({"confirmed": True, "confidence": .9,
                      "reason": "She puts a wig in her pocket", "alert_priority": "high",
                      "item_visible": True, "action": "insertion", "destination": destination,
                      "same_subject": True, "opening_visible": True,
                      "start_frame": 1, "end_frame": 2, "limitation": "none"})
    with patch.object(gate, "_call_provider", return_value=raw):
        result = gate.verify([np.zeros((100, 100, 3), dtype=np.uint8)] * 3, alert)
    assert result.confirmed
    assert "appears to move an item" in result.reason
    assert "wig" not in result.reason
    if destination != "pocket":
        assert "pocket" not in result.reason
    assert result.raw_response == raw


def test_structured_verification_keeps_one_call_and_existing_image_token_budgets():
    gate = VerificationGate(provider="ollama")
    alert = CandidateAlert(rule_name="product_concealment", priority="high",
                           detector="concealment", title="Possible concealment",
                           person_id=1, object_label=None, timestamp=0)
    raw = json.dumps({"confirmed": False, "confidence": .9, "reason": "No item"})
    with patch("cvti.verification.gate._call_ollama", return_value=raw) as provider:
        result = gate.verify([np.zeros((100, 100, 3), dtype=np.uint8)] * 4, alert)
    assert not result.confirmed
    assert provider.call_count == 1
    assert len(provider.call_args.args[1]) == gate.LOCAL_CONCEALMENT_MAX_FRAMES
    assert provider.call_args.kwargs["max_tokens"] == max(
        192, gate.MAX_TOKENS_COT if gate.cot else gate.MAX_TOKENS_JSON)


@pytest.mark.parametrize("confirmed", [False, "false", "true"])
def test_structured_observations_never_upgrade_rejection_or_string_booleans(confirmed):
    gate = VerificationGate(provider="ollama")
    alert = CandidateAlert(rule_name="product_concealment", priority="high",
                           detector="concealment", title="Possible concealment",
                           person_id=1, object_label=None, timestamp=0)
    raw = json.dumps(dict(confirmed=confirmed, confidence=.9, item_visible=True,
                          action="insertion", destination="clothing", same_subject=True,
                          start_frame=1, end_frame=2, limitation="none"))
    with patch.object(gate, "_call_provider", return_value=raw):
        result = gate.verify([np.zeros((20, 20, 3), dtype=np.uint8)] * 3, alert)
    assert not result.confirmed
    assert "Conflicting observations" in result.reason
