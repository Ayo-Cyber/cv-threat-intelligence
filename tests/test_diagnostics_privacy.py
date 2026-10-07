import json
import zipfile
from unittest.mock import patch

from cvti.diagnostics import build_bundle, redact_text


def test_export_redacts_credentials_and_does_not_copy_arbitrary_files(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "app.log").write_text('rtsp://admin:camera-secret@host/stream?token=query-secret\nAuthorization: Bearer bearer-secret\npassword="quoted secret"\n')
    (logs / "accounts.json").write_text('{"password":"never-export"}')
    (logs / "frame.png").write_bytes(b"image")
    (tmp_path / "monitor.log").write_text("Stream failed; retrying\n")
    (tmp_path / "gate_health.json").write_text(json.dumps({"token": "private-token", "cameras": [{"camera_id": "bay", "state": "offline"}]}))
    site = tmp_path / "site.json"
    site.write_text(json.dumps({"cameras": [{"id": "bay", "source": "rtsp://admin:never-export@host/stream", "normal_movement": True}]}))
    with patch("cvti.diagnostics.resolve_log_dir", return_value=logs):
        bundle = build_bundle(tmp_path, site_path=site)
    with zipfile.ZipFile(bundle) as archive:
        contents = "\n".join(archive.read(name).decode() for name in archive.namelist())
        for secret in ("camera-secret", "query-secret", "bearer-secret", "quoted secret", "never-export", "private-token"):
            assert secret not in contents
        assert "accounts.json" not in archive.namelist()
        cameras = json.loads(archive.read("camera_configuration.json"))
        assert cameras[0]["detectors"]["normal_movement"] is True
        assert "source" not in cameras[0]


def test_export_is_bounded_and_unique(tmp_path):
    (tmp_path / "monitor.log").write_text("old\n" * 500 + "recent failure\n")
    with patch("cvti.diagnostics.MAX_FILE_BYTES", 100):
        first = build_bundle(tmp_path)
        second = build_bundle(tmp_path)
    assert first != second
    with zipfile.ZipFile(first) as archive:
        tail = archive.read("logs/monitor.log").decode()
        assert "Older log lines omitted" in tail
        assert "recent failure" in tail
        assert len(tail) < 200


def test_inline_media_is_omitted():
    assert "image-secret" not in redact_text("data:image/jpeg;base64,image-secret")
