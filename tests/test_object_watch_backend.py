from __future__ import annotations

import base64
import io
import json
import time

import pytest
from PIL import Image

from cvti.app.console_backend import ConsoleBackend
from cvti.object_watch.embeddings import HashEmbeddingBackend
from cvti.security.permissions import PermissionDenied


def _png(size=(4, 4)) -> str:
    out = io.BytesIO()
    Image.new("RGB", size, (210, 30, 20)).save(out, format="PNG")
    return base64.b64encode(out.getvalue()).decode()


def _backend(tmp_path, role="owner") -> ConsoleBackend:
    site = tmp_path / "site.json"
    site.write_text(json.dumps({"name": "test", "notify": "console", "cameras": []}))
    backend = ConsoleBackend(str(site), str(tmp_path / "events.db"), enable_demo=False)
    backend.accounts.create_user("user", "pw-123456", role=role)
    assert backend.sign_in("user", "pw-123456")["ok"]
    return backend


def _target(backend: ConsoleBackend):
    return backend.create_object_target(
        "red-box", "Red box", "product", grounding_description="red carton",
    )


def test_crop_preview_review_reembed_and_activation_lifecycle(tmp_path, monkeypatch):
    backend = _backend(tmp_path)
    _target(backend)
    added = backend.add_object_example(
        "red-box", _png(), [1, 1, 3, 3], "upload", bbox_format="pixel_xyxy",
    )
    example_id = added["example"]["id"]
    assert added["example"]["reviewed"] is False

    preview = backend.object_example_preview("red-box", example_id)
    crop = Image.open(io.BytesIO(base64.b64decode(preview["image_b64"])))
    assert preview["mime_type"] == "image/png"
    assert crop.size == (2, 2)

    backend.review_object_example("red-box", example_id)
    monkeypatch.setattr(
        "cvti.object_watch.runtime_config.load_configured_backend",
        lambda _config: HashEmbeddingBackend(),
    )
    queued = backend.reembed_object_targets()
    deadline = time.time() + 3
    while time.time() < deadline:
        job = backend.object_watch_job_status(queued["job_id"])
        if job["status"] not in {"queued", "running"}:
            break
        time.sleep(0.01)
    assert job["status"] == "completed"
    assert job["written"] == 1
    backend._object_watch_metadata = lambda: HashEmbeddingBackend()  # type: ignore[method-assign]
    assert backend.activate_object_target("red-box")["target"]["review_state"] == "active"
    assert backend.deactivate_object_target("red-box")["target"]["review_state"] == "disabled"


def test_default_runtime_is_unavailable_and_hash_selector_is_rejected(tmp_path):
    backend = _backend(tmp_path)
    _target(backend)
    status = backend.object_targets()
    assert status["runtime"]["backend"] == "siglip"
    assert status["runtime"]["status"] == "unavailable"
    assert status["runtime"]["fingerprint"] is None
    assert status["targets"][0]["can_activate"] is False
    with pytest.raises(ValueError, match="configured production backend"):
        backend.reembed_object_targets("hash")
    added = backend.add_object_example(
        "red-box", _png(), [0, 0, 4, 4], "upload", bbox_format="pixel_xyxy",
    )
    backend.review_object_example("red-box", added["example"]["id"])
    with pytest.raises(ValueError, match="unavailable"):
        backend.activate_object_target("red-box")


def test_operator_reads_but_cannot_mutate_and_ids_do_not_traverse(tmp_path):
    owner = _backend(tmp_path)
    _target(owner)
    owner.sign_out()
    owner.accounts.create_user("op", "pw-123456", role="operator")
    owner.sign_in("op", "pw-123456")
    assert owner.object_targets()["targets"][0]["id"] == "red-box"
    with pytest.raises(PermissionDenied):
        owner.add_object_example("red-box", _png(), [0, 0, 4, 4], "upload")
    owner.sign_out()
    owner.sign_in("user", "pw-123456")
    with pytest.raises(ValueError, match="unsafe object id"):
        owner.create_object_target("../escape", "Escape", "custom")
    with pytest.raises(ValueError):
        owner.object_example_preview("../escape", "../../secret")


def test_scoped_rule_survives_regeneration_and_disables_only_it(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    backend = _backend(tmp_path)
    _target(backend)
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"rules": [{"name": "baseline", "trigger": {"detector": "fire"}}]}))
    backend.add_camera({"id": "cam-1", "name": "Cam", "source": "0", "config": str(base)})
    backend.add_zone("cam-1", "door", [[0, 0], [10, 0], [10, 10]])

    backend.set_object_watch_rule("cam-1", "red-box", True, "door")
    camera = backend.list_cameras()[0]
    rules = json.loads((tmp_path / camera["config"]).read_text())["rules"]
    scoped = next(rule for rule in rules if rule["name"].startswith("object_watch_"))
    assert scoped["trigger"] == {"detector": "object_watch", "state": "object_seen",
                                  "object_id": "red-box", "zone": "door"}
    assert camera["object_watch"] is True
    assert camera["object_watch_enabled"] is True
    assert camera["object_watch_library"] == str((tmp_path / "object_library").resolve())

    backend.set_object_watch_rule("cam-1", "red-box", False, "door")
    camera = backend.list_cameras()[0]
    rules = json.loads((tmp_path / camera["config"]).read_text())["rules"]
    assert any(rule["name"] == "baseline" for rule in rules)
    assert not any(rule["name"].startswith("object_watch_") for rule in rules)


def test_legacy_detector_toggle_disables_persisted_object_watch_alias(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    backend = _backend(tmp_path)
    _target(backend)
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"rules": []}))
    backend.add_camera({"id": "cam-1", "name": "Cam", "source": "0", "config": str(base)})
    backend.add_zone("cam-1", "door", [[0, 0], [10, 0], [10, 10]])

    backend.set_object_watch_rule("cam-1", "red-box", True, "door")
    backend.set_camera_rules("cam-1", {"object_watch": False})

    persisted = json.loads((tmp_path / "site.json").read_text())["cameras"][0]
    assert persisted["object_watch"] is False
    assert persisted["object_watch_enabled"] is False
    assert bool(persisted.get("object_watch_enabled", persisted.get("object_watch", False))) is False
    assert backend.list_cameras()[0]["object_watch_enabled"] is False


def test_preset_change_regenerates_camera_with_watchlist_rules_only(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    backend = _backend(tmp_path)
    _target(backend)
    old_base = tmp_path / "old-base.json"
    old_base.write_text(json.dumps({"rules": [
        {"name": "oldbaseline", "trigger": {"detector": "fire"}},
    ]}))
    new_base = tmp_path / "new-base.json"
    new_base.write_text(json.dumps({"rules": [
        {"name": "newbaseline", "trigger": {"detector": "tamper"}},
    ]}))
    backend.add_camera(
        {"id": "cam-1", "name": "Cam", "source": "0", "config": str(old_base)},
    )
    backend.set_object_watch_rule("cam-1", "red-box", True)

    backend.set_camera_rules("cam-1", {"config": str(new_base)})

    camera = backend.list_cameras()[0]
    generated = json.loads((tmp_path / camera["config"]).read_text())
    names = [rule["name"] for rule in generated["rules"]]
    assert camera["_base_config"] == str(new_base)
    assert camera["config"] != str(new_base)
    assert "newbaseline" in names
    assert "oldbaseline" not in names
    watch_rule = next(rule for rule in generated["rules"] if rule["name"].startswith("object_watch_"))
    assert watch_rule["trigger"] == {
        "detector": "object_watch", "state": "object_seen", "object_id": "red-box",
    }
