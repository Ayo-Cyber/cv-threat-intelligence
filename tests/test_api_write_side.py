"""W4 write-side: the API adds transport, never a second permission model.

The contract's promises, held as tests: every write goes through the REAL
ConsoleBackend as the bearer principal (same authz, same audit trail as the
desktop console); 403 carries the missing permission's NAME in
`detail.permission`; 401 for no token; the first-run endpoints are public
exactly until an owner exists; domain refusals map to 400/404, never 500.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from cvti.api.app import create_app


def _mint_app(tmp: Path):
    site = tmp / "site.json"
    site.write_text('{"name": "w4-test", "notify": "console", "cameras": []}')
    app = create_app(db_path=str(tmp / "events.db"), site_path=str(site))
    return app, TestClient(app)


def _token(client, username, password):
    r = client.post("/api/v1/auth/session",
                    json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}


class FirstRunIsPublicUntilAnOwnerExists(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.app, self.client = _mint_app(self.tmp)

    def test_the_first_owner_flow_needs_no_token(self):
        state = self.client.get("/api/v1/auth/state")
        self.assertEqual(state.status_code, 200)
        self.assertFalse(state.json()["configured"])
        made = self.client.post("/api/v1/auth/first-owner",
                                json={"username": "ayo", "password": "pw-123456"})
        self.assertEqual(made.status_code, 200, made.text)
        self.assertTrue(made.json().get("ok"))
        # and exactly once: the door closes behind the first owner
        again = self.client.post("/api/v1/auth/first-owner",
                                 json={"username": "eve", "password": "pw-123456"})
        self.assertEqual(again.status_code, 400)
        self.assertIn("already has accounts", again.json()["error"]["message"])


class TheBackendPermissionModelIsTheOnlyOne(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.app, self.client = _mint_app(self.tmp)
        self.client.post("/api/v1/auth/first-owner",
                         json={"username": "ayo", "password": "pw-123456"})
        self.owner = _token(self.client, "ayo", "pw-123456")
        # the real role vocabulary: operator (review/view, no config) and
        # installer (config, no alert access) split the 403 cases between them
        self.client.post("/api/v1/users", headers=self.owner,
                         json={"username": "op1", "password": "pw-123456",
                               "role": "operator"})
        self.client.post("/api/v1/users", headers=self.owner,
                         json={"username": "inst1", "password": "pw-123456",
                               "role": "installer"})
        self.operator = _token(self.client, "op1", "pw-123456")
        self.installer = _token(self.client, "inst1", "pw-123456")

    def test_no_token_is_401(self):
        r = self.client.post("/api/v1/cameras", json={"camera": {}})
        self.assertEqual(r.status_code, 401)

    def test_403_names_the_missing_permission(self):
        r = self.client.post("/api/v1/cameras", headers=self.operator,
                             json={"camera": {"id": "x", "source": "demo"}})
        self.assertEqual(r.status_code, 403, r.text)
        self.assertEqual(r.json()["error"]["detail"]["permission"],
                         "configure_cameras")

    def test_an_operator_cannot_manage_users_and_is_told_why(self):
        r = self.client.post("/api/v1/users", headers=self.operator,
                             json={"username": "x", "password": "pw-123456"})
        self.assertEqual(r.status_code, 403)
        self.assertEqual(r.json()["error"]["detail"]["permission"], "manage_users")

    def test_owner_camera_crud_round_trip(self):
        made = self.client.post("/api/v1/cameras", headers=self.owner,
                                json={"camera": {"id": "cam9", "source": "demo",
                                                 "name": "Test cam"}})
        self.assertEqual(made.status_code, 201, made.text)
        self.assertTrue(any(c["id"] == "cam9" for c in made.json()))
        gone = self.client.delete("/api/v1/cameras/cam9", headers=self.owner)
        self.assertEqual(gone.status_code, 200, gone.text)
        self.assertFalse(any(c["id"] == "cam9" for c in gone.json()))

    def test_zone_write_flows_through_the_backend(self):
        self.client.post("/api/v1/cameras", headers=self.owner,
                         json={"camera": {"id": "cam9", "source": "demo"}})
        z = self.client.post("/api/v1/cameras/cam9/zones", headers=self.owner,
                             json={"name": "till", "points": [[0, 0], [10, 0],
                                                              [10, 10], [0, 10]],
                                   "dwell_seconds": 7})
        self.assertEqual(z.status_code, 201, z.text)
        listed = self.client.get("/api/v1/cameras/cam9/zones", headers=self.owner)
        self.assertTrue(any(zz.get("name") == "till" for zz in listed.json()),
                        listed.text)

    def test_users_crud_and_audit_visibility(self):
        listed = self.client.get("/api/v1/users", headers=self.owner)
        self.assertEqual(listed.status_code, 200)
        self.assertTrue(any(u.get("username") == "op1" for u in listed.json()))
        removed = self.client.delete("/api/v1/users/op1", headers=self.owner)
        self.assertEqual(removed.status_code, 200, removed.text)
        audit = self.client.get("/api/v1/audit", headers=self.owner)
        self.assertEqual(audit.status_code, 200)
        # the API principal is the audit ACTOR — same paper trail as the console
        self.assertTrue(any(e.get("actor") == "ayo" for e in audit.json()),
                        "API-driven writes must be attributed to the bearer user")

    def test_domain_refusals_map_to_400_not_500(self):
        # the backend refuses weak passwords with a reasoned error — the API
        # must surface that as a 400, never a stack trace
        r = self.client.post("/api/v1/users", headers=self.owner,
                             json={"username": "weak", "password": "x"})
        self.assertEqual(r.status_code, 400, r.text)
        self.assertTrue(r.json()["error"]["message"])

    def test_search_requires_view_alerts(self):
        r = self.client.get("/api/v1/events?q=door", headers=self.installer)
        self.assertEqual(r.status_code, 403)
        self.assertEqual(r.json()["error"]["detail"]["permission"], "view_alerts")
        ok = self.client.get("/api/v1/events?q=door", headers=self.owner)
        self.assertEqual(ok.status_code, 200, ok.text)

    def test_installer_cannot_read_any_alert_or_triage_rest_route(self):
        for path in (
            "/api/v1/events",
            "/api/v1/events/evt_1",
            "/api/v1/events/evt_1/clip",
            "/api/v1/triage",
        ):
            with self.subTest(path=path):
                refused = self.client.get(path, headers=self.installer)
                self.assertEqual(refused.status_code, 403, refused.text)
                self.assertEqual(
                    refused.json()["error"]["detail"]["permission"],
                    "view_alerts",
                )

    def test_installer_cannot_open_the_alert_websocket(self):
        token = self.installer["Authorization"].removeprefix("Bearer ")
        with self.assertRaises(WebSocketDisconnect) as caught:
            with self.client.websocket_connect(
                "/api/v1/stream",
                subprotocols=["argus.v1", f"argus.token.{token}"],
            ) as websocket:
                websocket.receive_json()
        self.assertEqual(caught.exception.code, 4403)

    def test_site_write_needs_configure_site(self):
        refused = self.client.put("/api/v1/site", headers=self.installer,
                                  json={"name": "nope"})
        self.assertEqual(refused.status_code, 403)
        self.assertEqual(refused.json()["error"]["detail"]["permission"],
                         "configure_site")
        put = self.client.put("/api/v1/site", headers=self.owner,
                              json={"name": "Renamed Site"})
        self.assertEqual(put.status_code, 200, put.text)
        self.assertEqual(self.client.get("/api/v1/site",
                                         headers=self.owner).json()["name"],
                         "Renamed Site")

    def test_owner_can_update_organization_and_audit_uses_bearer_actor(self):
        updated = self.client.put(
            "/api/v1/organization", headers=self.owner,
            json={"organization": {"id": "customer", "name": "Customer Ltd"}},
        )
        self.assertEqual(updated.status_code, 200, updated.text)
        self.assertEqual(updated.json(), {"id": "customer", "name": "Customer Ltd"})
        read = self.client.get("/api/v1/organization", headers=self.owner)
        self.assertEqual(read.status_code, 200, read.text)
        self.assertEqual(read.json(), updated.json())
        audit = self.client.get("/api/v1/audit", headers=self.owner).json()
        entry = next(e for e in audit
                     if e.get("target") == "organization:customer")
        self.assertEqual(entry["actor"], "ayo")

    def test_installer_cannot_update_organization(self):
        refused = self.client.put(
            "/api/v1/organization", headers=self.installer,
            json={"organization": {"id": "customer", "name": "Customer Ltd"}},
        )
        self.assertEqual(refused.status_code, 403, refused.text)
        self.assertEqual(refused.json()["error"]["detail"]["permission"],
                         "configure_site")

    def test_owner_can_create_update_delete_branch(self):
        made = self.client.post(
            "/api/v1/branches", headers=self.owner,
            json={"branch": {"id": "ikeja", "name": "Ikeja"}},
        )
        self.assertEqual(made.status_code, 201, made.text)
        self.assertIn({"id": "ikeja", "name": "Ikeja"}, made.json())
        updated = self.client.put(
            "/api/v1/branches/ikeja", headers=self.owner,
            json={"branch": {"name": "Ikeja Mall"}},
        )
        self.assertEqual(updated.status_code, 200, updated.text)
        self.assertIn({"id": "ikeja", "name": "Ikeja Mall"}, updated.json())
        listed = self.client.get("/api/v1/branches", headers=self.owner)
        self.assertEqual(listed.json(), updated.json())
        removed = self.client.delete("/api/v1/branches/ikeja", headers=self.owner)
        self.assertEqual(removed.status_code, 200, removed.text)
        self.assertNotIn("ikeja", {branch["id"] for branch in removed.json()})
        audit = self.client.get("/api/v1/audit", headers=self.owner).json()
        branch_mutations = {
            entry["detail"]["branch"]: entry["actor"]
            for entry in audit
            if entry.get("target") == "branch:ikeja"
        }
        self.assertEqual(branch_mutations, {
            "created": "ayo", "updated": "ayo", "removed": "ayo",
        })

    def test_duplicate_branch_post_conflicts_but_put_still_updates(self):
        first = self.client.post(
            "/api/v1/branches", headers=self.owner,
            json={"branch": {"id": "ikeja", "name": "Ikeja"}},
        )
        self.assertEqual(first.status_code, 201, first.text)

        duplicate = self.client.post(
            "/api/v1/branches", headers=self.owner,
            json={"branch": {"id": "ikeja", "name": "Overwrite"}},
        )
        self.assertEqual(duplicate.status_code, 409, duplicate.text)
        self.assertEqual(duplicate.json()["error"]["code"], "conflict")

        updated = self.client.put(
            "/api/v1/branches/ikeja", headers=self.owner,
            json={"branch": {"name": "Ikeja Mall"}},
        )
        self.assertEqual(updated.status_code, 200, updated.text)
        self.assertIn({"id": "ikeja", "name": "Ikeja Mall"}, updated.json())

    def test_owner_can_create_branch_area_and_read_hierarchy(self):
        made = self.client.post(
            "/api/v1/branches", headers=self.owner,
            json={"branch": {"id": "ikeja", "name": "Ikeja"}},
        )
        self.assertEqual(made.status_code, 201, made.text)
        area = self.client.post(
            "/api/v1/areas", headers=self.owner,
            json={"area": {"id": "paint", "name": "Paint floor",
                           "branch_id": "ikeja"}},
        )
        self.assertEqual(area.status_code, 201, area.text)
        tree = self.client.get("/api/v1/hierarchy", headers=self.owner)
        self.assertEqual(tree.status_code, 200, tree.text)
        branch = next(branch for branch in tree.json()["branches"]
                      if branch["id"] == "ikeja")
        self.assertEqual(branch["areas"][0]["id"], "paint")

    def test_operator_cannot_write_branch(self):
        refused = self.client.post(
            "/api/v1/branches", headers=self.operator,
            json={"branch": {"id": "ikeja", "name": "Ikeja"}},
        )
        self.assertEqual(refused.status_code, 403, refused.text)
        self.assertEqual(refused.json()["error"]["detail"]["permission"],
                         "configure_cameras")

    def test_operator_cannot_update_or_delete_branch(self):
        made = self.client.post(
            "/api/v1/branches", headers=self.owner,
            json={"branch": {"id": "ikeja", "name": "Ikeja"}},
        )
        self.assertEqual(made.status_code, 201, made.text)
        updated = self.client.put(
            "/api/v1/branches/ikeja", headers=self.operator,
            json={"branch": {"name": "Ikeja Mall"}},
        )
        removed = self.client.delete(
            "/api/v1/branches/ikeja", headers=self.operator,
        )
        for response in (updated, removed):
            self.assertEqual(response.status_code, 403, response.text)
            self.assertEqual(
                response.json()["error"]["detail"]["permission"],
                "configure_cameras",
            )
        branches = self.client.get("/api/v1/branches", headers=self.owner).json()
        self.assertIn({"id": "ikeja", "name": "Ikeja"}, branches)

    def test_nonempty_branch_delete_returns_409(self):
        self.client.post(
            "/api/v1/branches", headers=self.owner,
            json={"branch": {"id": "ikeja", "name": "Ikeja"}},
        )
        self.client.post(
            "/api/v1/areas", headers=self.owner,
            json={"area": {"id": "paint", "name": "Paint floor",
                           "branch_id": "ikeja"}},
        )
        refused = self.client.delete("/api/v1/branches/ikeja",
                                     headers=self.owner)
        self.assertEqual(refused.status_code, 409, refused.text)
        self.assertEqual(refused.json(), {
            "error": {"code": "conflict", "message": "branch contains areas",
                      "detail": {}},
        })

    def test_unknown_branch_update_and_delete_return_404(self):
        updated = self.client.put(
            "/api/v1/branches/missing", headers=self.owner,
            json={"branch": {"name": "Missing"}},
        )
        removed = self.client.delete("/api/v1/branches/missing",
                                     headers=self.owner)
        self.assertEqual(updated.status_code, 404, updated.text)
        self.assertEqual(removed.status_code, 404, removed.text)

    def test_legacy_cameras_endpoint_includes_resolved_location_ids(self):
        site = Path(self.app.state.site_path)
        site.write_text(
            '{"name":"Legacy","cameras":['
            '{"id":"cam1","source":"rtsp://user:secret@example.test/live",'
            '"area":"Paint floor"}]}'
        )
        cameras = self.client.get("/api/v1/cameras", headers=self.owner)
        self.assertEqual(cameras.status_code, 200, cameras.text)
        camera = cameras.json()[0]
        self.assertEqual(camera["area_id"], "camera--cam1")
        self.assertEqual(camera["branch_id"], "branch--default")
        self.assertNotIn("secret", camera["source"])

    def test_legacy_areas_endpoint_preserves_its_frozen_flat_shape(self):
        site = Path(self.app.state.site_path)
        site.write_text(
            '{"name":"Legacy","branches":['
            '{"id":"same","name":"One"},{"id":"same","name":"Two"}],'
            '"cameras":[{"id":"cam1","source":"demo"}]}'
        )
        areas = self.client.get("/api/v1/areas", headers=self.owner)
        self.assertEqual(areas.status_code, 200, areas.text)
        self.assertEqual(areas.json(), [{
            "id": "camera--cam1",
            "name": "cam1",
            "implicit": True,
            "camera_ids": ["cam1"],
        }])

    def test_malformed_hierarchy_does_not_hide_flat_camera_reads(self):
        site = Path(self.app.state.site_path)
        site.write_text(
            '{"branches":[{"id":"same","name":"One"},'
            '{"id":"same","name":"Two"}],"cameras":['
            '{"id":"cam1","source":"demo"}]}'
        )
        cameras = self.client.get("/api/v1/cameras", headers=self.owner)
        self.assertEqual(cameras.status_code, 200, cameras.text)
        self.assertEqual([camera["id"] for camera in cameras.json()], ["cam1"])

    def test_hierarchy_redacts_camera_credentials(self):
        site = Path(self.app.state.site_path)
        site.write_text(
            '{"name":"Legacy","cameras":['
            '{"id":"cam1","source":"rtsp://user:secret@example.test/live",'
            '"detect_source":"rtsp://detector:othersecret@example.test/sub",'
            '"area":"Paint floor"}]}'
        )
        tree = self.client.get("/api/v1/hierarchy", headers=self.operator)
        self.assertEqual(tree.status_code, 200, tree.text)
        camera = tree.json()["branches"][0]["areas"][0]["cameras"][0]
        self.assertEqual(camera["source"], "rtsp://***@example.test/live")
        self.assertEqual(camera["detect_source"],
                         "rtsp://***@example.test/sub")


class AlertUpdateDiffTest(unittest.TestCase):
    def test_review_states_reads_the_review_column(self):
        import sqlite3
        tmp = Path(tempfile.mkdtemp())
        db = tmp / "events.db"
        con = sqlite3.connect(db)
        con.execute("CREATE TABLE events (id INTEGER PRIMARY KEY, review TEXT, "
                    "reason TEXT)")
        con.execute("INSERT INTO events (id, review, reason) VALUES "
                    "(1, NULL, NULL), (2, 'ack', 'seen')")
        con.commit(); con.close()
        from cvti.api.sources import review_states
        states = review_states(str(db))
        # review AND reason length: an enrichment that only extends the reason
        # must still change the state string (that IS alert.update's trigger)
        self.assertEqual(states, {1: "|0", 2: "ack|4"})

    def test_no_db_is_an_empty_map_not_a_crash(self):
        from cvti.api.sources import review_states
        self.assertEqual(review_states("/nowhere/events.db"), {})


if __name__ == "__main__":
    unittest.main()
