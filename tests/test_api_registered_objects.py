import json
from fastapi.testclient import TestClient
from cvti.api.app import create_app


def test_registration_permissions_validation_and_remove(tmp_path):
    site = tmp_path / "site.json"
    site.write_text(json.dumps({"cameras": [{"id": "cam", "source": "video.mp4"}]}))
    client = TestClient(create_app(site_path=str(site), db_path=str(tmp_path / "events.db")))
    client.post("/api/v1/auth/first-owner", json={"username":"owner", "password":"pw-123456"})
    token = client.post("/api/v1/auth/session", json={"username":"owner", "password":"pw-123456"}).json()["token"]
    headers = {"Authorization": f"Bearer {token}"}
    url = "/api/v1/cameras/cam/registered-objects"
    body = {"name":"Laptop", "region":[30,30,70,70], "frame_hw":[100,100], "confirm_seconds":8}
    assert client.post(url, json=body).status_code == 401
    client.post("/api/v1/users", headers=headers, json={"username":"operator", "password":"pw-123456", "role":"operator"})
    operator = client.post("/api/v1/auth/session", json={"username":"operator", "password":"pw-123456"}).json()["token"]
    denied = client.post(url, json=body, headers={"Authorization":f"Bearer {operator}"})
    assert denied.status_code == 403
    response = client.post(url, headers=headers, json=body)
    assert response.status_code == 200, response.text
    key = response.json()["id"]
    entries = client.get(url, headers=headers).json()
    assert entries[0]["state"] == "pending_monitoring"
    assert "reference_path" not in entries[0]
    assert client.post(url, headers=headers, json={**body, "region":[-1,0,2,3]}).status_code == 400
    assert client.post(f"{url}/{key}/recapture", headers=headers).status_code == 200
    assert client.delete(f"{url}/{key}", headers=headers).status_code == 200
    assert client.get(url, headers=headers).json() == []
