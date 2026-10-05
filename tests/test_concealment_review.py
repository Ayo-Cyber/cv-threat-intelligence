import json
import sqlite3
from unittest.mock import patch

import numpy as np
import pytest

from cvti.contracts import CandidateAlert, VerificationResult
from cvti.verification.gate import VerificationGate
from cvti.serving.alert_queue import QueuedAlert
from cvti.serving.alert_sink import AlertSink
from cvti.api.sources import _verdict_from_row


@pytest.mark.parametrize("changes,review", [
    ({}, True), ({"confirmed": False}, False), ({"item_visible": False}, False),
    ({"action": "holding"}, False), ({"confidence": .01}, False),
    ({"start_frame": 8, "end_frame": 8}, False), ({"same_subject": False}, False),
])
def test_only_disputed_positive_becomes_review(changes, review):
    data = dict(confirmed=True, confidence=.85, item_visible=True, action="insertion",
                destination="clothing", same_subject=True, start_frame=3, end_frame=3,
                limitation="none")
    data.update(changes)
    candidate = CandidateAlert("product_concealment", "high", "concealment", "Possible", 1, None, 0)
    gate = VerificationGate(provider="ollama")
    with patch.object(gate, "_call_provider", return_value=json.dumps(data)):
        result = gate.verify([np.zeros((20, 20, 3), dtype=np.uint8)] * 4, candidate)
    assert result.review_required is review
    assert not result.confirmed
    assert not result.errored


def test_review_persisted_without_paging_or_confirmed_verdict(tmp_path):
    sink = AlertSink(str(tmp_path), save_evidence=False, routing_path=None)
    alert = QueuedAlert(camera_id="one", rule_name="product_concealment", priority="high",
                        title="Possible", timestamp=0, payload={})
    result = VerificationResult(False, .85, "NEEDS REVIEW: Conflicting frame references.", "high", "now",
                                review_required=True)
    try:
        with patch.object(sink, "_dispatch") as dispatch:
            event_id = sink.handle(alert, result)
        assert event_id
        dispatch.assert_not_called()
        with sqlite3.connect(tmp_path / "events.db") as db:
            reason, unverified = db.execute("SELECT reason,unverified FROM events WHERE id=?", (event_id,)).fetchone()
        assert unverified == 1
        assert _verdict_from_row({"reason": reason}) == "review_required"
    finally:
        sink.close()
