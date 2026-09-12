from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from cvti.api.app import create_app


class CameraProbeBodyCompatibilityTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        site = self.tmp / "site.json"
        site.write_text('{"name":"probe-test","cameras":[]}')
        self.app = create_app(
            db_path=str(self.tmp / "events.db"), site_path=str(site)
        )
        self.client = TestClient(self.app)
        made = self.client.post(
            "/api/v1/auth/first-owner",
            json={"username": "ayo", "password": "Argus-Fresh-2026"},
        )
        self.assertEqual(made.status_code, 200, made.text)
        signed_in = self.client.post(
            "/api/v1/auth/session",
            json={"username": "ayo", "password": "Argus-Fresh-2026"},
        )
        self.headers = {
            "Authorization": f"Bearer {signed_in.json()['token']}"
        }

    def test_source_is_canonical_and_url_remains_an_alias(self):
        captured = []

        def call(_principal, method, **kwargs):
            captured.append((method, kwargs))
            return {"ok": True}

        with patch.object(self.app.state.backend_host, "call", side_effect=call):
            canonical = self.client.post(
                "/api/v1/cameras/probe",
                headers=self.headers,
                json={"source": "rtsp://canonical"},
            )
            alias = self.client.post(
                "/api/v1/cameras/probe",
                headers=self.headers,
                json={"url": "rtsp://legacy"},
            )
            precedence = self.client.post(
                "/api/v1/cameras/probe",
                headers=self.headers,
                json={"source": "rtsp://canonical", "url": "rtsp://legacy"},
            )

        self.assertEqual(
            [canonical.status_code, alias.status_code, precedence.status_code],
            [200, 200, 200],
        )
        self.assertEqual(
            captured,
            [
                ("test", {"url": "rtsp://canonical"}),
                ("test", {"url": "rtsp://legacy"}),
                ("test", {"url": "rtsp://canonical"}),
            ],
        )


if __name__ == "__main__":
    unittest.main()
