import json
import time
import zipfile

from cvti.app.live_wall import LiveWall
from cvti.diagnostics import build_bundle


def test_preview_history_is_bounded_and_contains_no_source():
    wall = LiveWall([{"id": "camera", "source": "rtsp://user:private@host/live"}])
    for i in range(50):
        wall._record("camera", str(i))
    report = wall.diagnostics()
    assert len(report["camera"]["events"]) == 20
    assert report["camera"]["last_decoded_age_s"] is None
    assert "private" not in json.dumps(report)
    report["camera"]["events"].clear()
    assert len(wall.diagnostics()["camera"]["events"]) == 20


def test_export_marks_old_metrics_and_redacts_preview(tmp_path):
    (tmp_path / "perf_report.json").write_text(json.dumps({"generated_at": time.time() - 7200}))
    archive = build_bundle(tmp_path, preview={"error": "rtsp://user:private@host/live"})
    with zipfile.ZipFile(archive) as bundle:
        assert "private" not in bundle.read("preview_diagnostics.json").decode()
        health = json.loads(bundle.read("health.json"))
        assert health["saved_report_freshness"]["perf_report.json"]["status"] == "stale"
