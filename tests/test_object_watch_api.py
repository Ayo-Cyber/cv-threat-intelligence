from __future__ import annotations

import base64
import io
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from cvti.api.app import create_app
from cvti.object_watch.embeddings import HashEmbeddingBackend


_image = io.BytesIO()
Image.new("RGB", (2, 2), (20, 40, 60)).save(_image, format="PNG")
JPEG_1X1 = base64.b64encode(_image.getvalue()).decode()


def _mint_app(tmp: Path):
    site = tmp / "site.json"
    site.write_text('{"name": "object-api-test", "notify": "console", "cameras": []}')
    app = create_app(db_path=str(tmp / "events.db"), site_path=str(site))
    return app, TestClient(app)


def _token(client: TestClient, username: str, password: str) -> dict:
    r = client.post(
        "/api/v1/auth/session",
        json={"username": username, "password": password},
    )
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}


class ObjectWatchApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.app, self.client = _mint_app(self.tmp)
        self.client.post(
            "/api/v1/auth/first-owner",
            json={"username": "ayo", "password": "pw-123456"},
        )
        self.owner = _token(self.client, "ayo", "pw-123456")
        self.client.post(
            "/api/v1/users",
            headers=self.owner,
            json={"username": "op1", "password": "pw-123456", "role": "operator"},
        )
        self.operator = _token(self.client, "op1", "pw-123456")

    def test_owner_can_create_object_target_and_operator_cannot(self):
        owner = self.client.post(
            "/api/v1/object-targets",
            headers=self.owner,
            json={
                "target": {
                    "id": "chi-carton",
                    "label": "Chi carton",
                    "category": "product",
                    "aliases": [],
                    "allowed_zone_ids": ["storage"],
                    "min_similarity": 0.72,
                }
            },
        )
        self.assertEqual(owner.status_code, 201, owner.text)
        self.assertEqual(owner.json()["target"]["id"], "chi-carton")

        operator = self.client.post(
            "/api/v1/object-targets",
            headers=self.operator,
            json={"target": {"id": "x", "label": "X", "category": "product"}},
        )
        self.assertEqual(operator.status_code, 403)
        self.assertEqual(
            operator.json()["error"]["detail"]["permission"],
            "configure_cameras",
        )

    def test_object_target_reads_are_allowed_with_view_live_and_redacted(self):
        self.client.post(
            "/api/v1/object-targets",
            headers=self.owner,
            json={"target": {"id": "chi-carton", "label": "Chi carton", "category": "product"}},
        )
        self.client.post(
            "/api/v1/object-targets/chi-carton/examples",
            headers=self.owner,
            json={"image_b64": JPEG_1X1, "bbox": [0, 0, 1, 1], "source": "upload",
                  "bbox_format": "legacy"},
        )

        read = self.client.get("/api/v1/object-targets", headers=self.operator)

        self.assertEqual(read.status_code, 200, read.text)
        targets = read.json()["targets"]
        self.assertEqual(targets[0]["id"], "chi-carton")
        self.assertIn("sha256", targets[0]["examples"][0])
        self.assertNotIn("path", targets[0]["examples"][0])

    def test_add_example_rejects_non_image_base64(self):
        self.client.post(
            "/api/v1/object-targets",
            headers=self.owner,
            json={"target": {"id": "chi-carton", "label": "Chi carton", "category": "product"}},
        )

        bad = self.client.post(
            "/api/v1/object-targets/chi-carton/examples",
            headers=self.owner,
            json={
                "image_b64": base64.b64encode(b"not an image").decode(),
                "bbox": [0, 0, 1, 1],
                "source": "upload",
            },
        )

        self.assertEqual(bad.status_code, 400)
        self.assertIn("image", bad.json()["error"]["message"].lower())

    def test_activation_requires_reviewed_example_then_reembed_runs_locally(self):
        self.client.post(
            "/api/v1/object-targets",
            headers=self.owner,
            json={"target": {"id": "chi-carton", "label": "Chi carton", "category": "product"}},
        )

        early = self.client.post(
            "/api/v1/object-targets/chi-carton/activate",
            headers=self.owner,
        )
        self.assertEqual(early.status_code, 400)
        self.assertIn("active target requires", early.json()["error"]["message"])

        example = self.client.post(
            "/api/v1/object-targets/chi-carton/examples",
            headers=self.owner,
            json={"image_b64": JPEG_1X1, "bbox": [0, 0, 1, 1], "source": "upload"},
        )
        self.assertEqual(example.status_code, 201, example.text)
        example_id = example.json()["example"]["id"]
        reviewed = self.client.put(
            f"/api/v1/object-targets/chi-carton/examples/{example_id}/review",
            headers=self.owner, json={"reviewed": True},
        )
        self.assertEqual(reviewed.status_code, 200, reviewed.text)

        with patch("cvti.object_watch.runtime_config.load_configured_backend",
                   return_value=HashEmbeddingBackend()):
            reembed = self.client.post(
                "/api/v1/object-targets/reembed", headers=self.owner, json={},
            )
            self.assertEqual(reembed.status_code, 200, reembed.text)
            self.assertEqual(set(reembed.json()), {"job_id", "status"})
            deadline = time.time() + 3
            while time.time() < deadline:
                status = self.client.get(
                    f"/api/v1/object-targets/jobs/{reembed.json()['job_id']}",
                    headers=self.owner,
                )
                if status.json()["status"] not in {"queued", "running"}:
                    break
                time.sleep(0.01)
        self.assertEqual(status.json()["status"], "completed", status.text)
        self.assertEqual(status.json()["written"], 1)

        backend = self.app.state.backend_host._backend
        backend._object_watch_metadata = lambda: HashEmbeddingBackend()
        activated = self.client.post(
            "/api/v1/object-targets/chi-carton/activate", headers=self.owner,
        )
        self.assertEqual(activated.status_code, 200, activated.text)
        self.assertEqual(activated.json()["target"]["review_state"], "active")


if __name__ == "__main__":
    unittest.main()
