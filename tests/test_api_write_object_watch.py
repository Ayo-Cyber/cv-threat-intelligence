from __future__ import annotations

import base64
import io
import json

from fastapi.testclient import TestClient
from PIL import Image

from cvti.api.app import create_app


def _png() -> str:
    out = io.BytesIO()
    Image.new("RGB", (3, 3), (1, 2, 3)).save(out, format="PNG")
    return base64.b64encode(out.getvalue()).decode()


def _token(client, username, password):
    response = client.post("/api/v1/auth/session", json={"username": username, "password": password})
    return {"Authorization": f"Bearer {response.json()['token']}"}


def test_explicit_object_watch_api_and_operator_guards(tmp_path):
    site = tmp_path / "site.json"
    site.write_text(json.dumps({"name": "test", "notify": "console", "cameras": []}))
    app = create_app(db_path=str(tmp_path / "events.db"), site_path=str(site))
    client = TestClient(app)
    client.post("/api/v1/auth/first-owner", json={"username": "owner", "password": "pw-123456"})
    owner = _token(client, "owner", "pw-123456")
    client.post("/api/v1/users", headers=owner,
                json={"username": "op", "password": "pw-123456", "role": "operator"})
    operator = _token(client, "op", "pw-123456")

    created = client.post("/api/v1/object-targets", headers=owner, json={
        "object_id": "red-box", "label": "Red box", "category": "product",
        "grounding_description": "red cardboard carton",
    })
    assert created.status_code == 201, created.text
    added = client.post("/api/v1/object-targets/red-box/examples", headers=owner, json={
        "image_b64": _png(), "bbox": [0, 0, 2, 2], "bbox_format": "pixel_xyxy",
        "source": "upload", "negative": False,
    })
    assert added.status_code == 201, added.text
    example_id = added.json()["example"]["id"]
    assert client.put(
        f"/api/v1/object-targets/red-box/examples/{example_id}/review",
        headers=owner, json={"reviewed": True},
    ).status_code == 200
    preview = client.get(
        f"/api/v1/object-targets/red-box/examples/{example_id}/preview", headers=operator,
    )
    assert preview.status_code == 200
    assert base64.b64decode(preview.json()["image_b64"]).startswith(b"\x89PNG")

    listing = client.get("/api/v1/object-targets", headers=operator)
    assert listing.status_code == 200
    assert set(listing.json()["runtime"]) == {"status", "backend", "fingerprint", "reason_codes"}
    refused = client.post("/api/v1/object-targets/red-box/deactivate", headers=operator)
    assert refused.status_code == 403
    assert refused.json()["error"]["detail"]["permission"] == "configure_cameras"
    refused_job = client.post("/api/v1/object-targets/reembed", headers=operator, json={})
    assert refused_job.status_code == 403
    assert refused_job.json()["error"]["detail"]["permission"] == "configure_cameras"


def test_runtime_config_rejects_hash_and_remote_paths(tmp_path):
    site = tmp_path / "site.json"
    site.write_text('{"name":"test","notify":"console","cameras":[]}')
    app = create_app(db_path=str(tmp_path / "events.db"), site_path=str(site))
    client = TestClient(app)
    client.post("/api/v1/auth/first-owner", json={"username": "owner", "password": "pw-123456"})
    owner = _token(client, "owner", "pw-123456")

    rejected = client.put("/api/v1/object-targets/runtime", headers=owner,
                          json={"config": {"backend": "hash"}})
    assert rejected.status_code == 400
    remote = client.put("/api/v1/object-targets/runtime", headers=owner,
                        json={"config": {"model_path": "https://example/model"}})
    assert remote.status_code == 400


def test_http_normalized_bbox_preserves_fractional_coordinates(tmp_path):
    site = tmp_path / "site.json"
    site.write_text('{"name":"test","notify":"console","cameras":[]}')
    app = create_app(db_path=str(tmp_path / "events.db"), site_path=str(site))
    client = TestClient(app)
    client.post("/api/v1/auth/first-owner",
                json={"username": "owner", "password": "pw-123456"})
    owner = _token(client, "owner", "pw-123456")
    client.post("/api/v1/object-targets", headers=owner, json={
        "object_id": "red-box", "label": "Red box", "category": "product",
    })

    added = client.post("/api/v1/object-targets/red-box/examples", headers=owner, json={
        "image_b64": _png(), "bbox": [0.34, 0.34, 0.67, 0.67],
        "bbox_format": "normalized_xyxy", "source": "upload",
    })

    assert added.status_code == 201, added.text
    assert added.json()["example"]["bbox"] == [1, 1, 3, 3]
