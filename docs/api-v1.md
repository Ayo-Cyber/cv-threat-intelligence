# Argus Engine API — Contract v1.0 (FROZEN 9 Sep 2026)

*The written agreement between the engine and every client of it — Demi's
Electron console first. Frozen means: changes to this file are deliberate,
announced in the PR that makes them, and versioned — never a casual rename
mid-build. Both sides code against this file; when the halves meet, they fit.*

**Provenance.** The read-side below shipped in #102/#103 as "contract v0.2",
which until this file existed only in code comments. The write-side is derived
from the 67 backend operations Demi's shipped UI actually invokes
(`Frontend/bridge.py` METHODS — his bridge's own docstring says to replace it
with this API). A consistency test holds this document equal to both sources:
every implemented route must appear here, and every bridge method must have a
mapped endpoint here.

## Conventions

- Base path `/api/v1`. JSON bodies both ways. UTF-8.
- **Auth**: `Authorization: Bearer <token>` from `POST /auth/session`.
  The WebSocket offers `argus.v1` and `argus.token.<token>` in
  `Sec-WebSocket-Protocol`; credentials are never placed in its URL. No
  unauthenticated route exists except
  `GET /` and `/api/v1` (discovery) and the first-run endpoints marked PUBLIC.
- **Errors**: `{"error": {"code", "message", "detail"}}` with the HTTP status.
  `401` unauthenticated; `403` carries the MISSING PERMISSION'S NAME in
  `detail.permission`; `400` invalid payload; `404` unknown id;
  `409` hierarchy conflict; `503` engine not running.
- **Permissions** are the existing vocabulary (`cvti/security/permissions.py`):
  each endpoint enforces exactly the permission its backing ConsoleBackend
  method already enforces — the API adds transport, never a second permission
  model. The permission column below names it.
- **Status**: `shipped` = implemented today; `pending` = frozen here, not yet built.
  (W4 shipped the full write-side on 10 Sep 2026 — every row below is live.) Shapes of pending endpoints follow the backing
  method's current return value unless a Shape note says otherwise.

## Endpoints

### Auth & first run

| Bridge method | Endpoint | Permission | Status |
|---|---|---|---|
| `sign_in` | `POST /auth/session` → `{token, expires_at, user}` | PUBLIC | shipped |
| `sign_out` | `DELETE /auth/session` | any | shipped |
| — | `GET /auth/me` | any | shipped |
| `auth_state` | `GET /auth/state` (setup phase, first-owner needed?) | PUBLIC | shipped |
| `create_first_owner` | `POST /auth/first-owner` | PUBLIC (only while no owner exists) | shipped |
| `role_table` | `GET /roles` | any | shipped |
| `setup_state` | `GET /setup/state` | any | shipped |
| `setup_check` | `GET /setup/check` | configure_cameras | shipped |
| `mark_configured` | `POST /setup/configured` | configure_cameras | shipped |

### Site & templates

| Bridge method | Endpoint | Permission | Status |
|---|---|---|---|
| `get_site` | `GET /site` | view_live | shipped |
| `set_site` | `PUT /site` | configure_site | shipped |
| `use_case_templates` | `GET /site/templates` | configure_cameras | shipped |
| `apply_template` | `POST /site/templates/{name}/apply` | configure_cameras | shipped |
| `approve_site_context` | `POST /site/context/approve` | configure_cameras | shipped |
| `send_test_notification` | `POST /site/notifications/test` | configure_site | shipped |

### Organization hierarchy

| Bridge method | Endpoint | Permission | Status |
|---|---|---|---|
| `organization` | `GET /organization` | view_live | shipped |
| `update_organization` | `PUT /organization` (body: `{organization: {id, name}}`) | configure_site | shipped |
| `list_branches` | `GET /branches` | view_live | shipped |
| `create_branch` | `POST /branches` (body: `{branch: {id, name}}`) | configure_cameras | shipped |
| `update_branch` | `PUT /branches/{id}` (body: `{branch: {name}}`; path id is authoritative) | configure_cameras | shipped |
| `remove_branch` | `DELETE /branches/{id}` | configure_cameras | shipped |
| `hierarchy` | `GET /hierarchy` | view_live | shipped |

`GET /hierarchy` returns
`{organization, branches: [{id, name, areas: [{..., branch_id, cameras}]}], unassigned_areas, unassigned_cameras}`.
Camera objects in `/hierarchy` redact credentials from both `source` and
`detect_source`; `/cameras` redacts its returned `source`. Flat camera reads
include their normalized `area_id` and area-derived `branch_id` when resolved.
The frozen `/areas` response remains flat and does not depend on branch metadata.

Legacy site files remain read-only during hierarchy reads. They receive a
stable virtual organization and main branch, legacy cameras receive derived
single-camera areas, and every camera remains visible. The first hierarchy
write materializes those defaults. Deleting an unknown branch returns `404`;
deleting a branch that still contains areas returns `409 conflict`.

### Cameras

| Bridge method | Endpoint | Permission | Status |
|---|---|---|---|
| `list_cameras` | `GET /cameras` | view_live | shipped |
| — | `GET /cameras/{id}` | view_live | shipped |
| `add_camera` | `POST /cameras` | configure_cameras | shipped |
| `remove_camera` | `DELETE /cameras/{id}` | configure_cameras | shipped |
| `test` | `POST /cameras/probe` (body: `{source}`) | configure_cameras | shipped |
| `discover_cameras` | `GET /cameras/discovery` | configure_cameras | shipped |
| `scan` | `POST /cameras/discovery/scan` | configure_cameras | shipped |
| `detect_subnet` | `GET /cameras/discovery/subnet` | configure_cameras | shipped |
| `presets` | `GET /cameras/presets` | configure_cameras | shipped |
| `camera_snapshot` | `GET /cameras/{id}/snapshot` (image/jpeg) | view_live | shipped |
| `camera_links` | `GET /cameras/{id}/links` | view_live | shipped |
| `assign_camera_area` | `PUT /cameras/{id}/area` | configure_cameras | shipped |
| `live_start` / `live_stop` | superseded by `GET /cameras/{id}/stream` — the descriptor is stateless; viewer-gating is publisher-side | view_live | shipped |

**Stream descriptor** (shipped): `{kind: "webrtc", url, ws, mjpeg_fallback}`
when the go2rtc gateway is up, else `{kind: "mjpeg", url}`. Players switch on
`kind`; every URL is loopback-only by design.

### Zones

| Bridge method | Endpoint | Permission | Status |
|---|---|---|---|
| `list_zones` | `GET /cameras/{id}/zones` | view_live | shipped |
| `add_zone` | `POST /cameras/{id}/zones` | configure_cameras | shipped |
| `remove_zone` | `DELETE /cameras/{id}/zones/{name}` | configure_cameras | shipped |
| `accept_suggested_zone` | `POST /cameras/{id}/zones/suggestions/{name}/accept` | configure_cameras | shipped |

### Rules & detectors

| Bridge method | Endpoint | Permission | Status |
|---|---|---|---|
| `set_camera_rules` | `PUT /cameras/{id}/rules` | configure_detectors | shipped |
| `add_custom_rule` | `POST /cameras/{id}/rules/custom` | configure_detectors | shipped |
| `remove_custom_rule` | `DELETE /cameras/{id}/rules/custom/{name}` | configure_detectors | shipped |
| `english_rules_status` | `GET /rules/english/status` | view_live | shipped |

### Scene understanding

| Bridge method | Endpoint | Permission | Status |
|---|---|---|---|
| `scene_context` | `GET /cameras/{id}/scene` | view_live | shipped |
| `update_scene_context` | `PUT /cameras/{id}/scene` | configure_cameras | shipped |
| `approve_scene_context` | `POST /cameras/{id}/scene/approve` | configure_cameras | shipped |
| `request_scene_remap` | `POST /cameras/{id}/scene/remap` | configure_cameras | shipped |
| `enqueue_scene_mapping` | `POST /scene-mapping/queue` | configure_cameras | shipped |
| `scene_mapping_progress` | `GET /scene-mapping/progress` | view_live | shipped |
| `scene_review_summary` | `GET /scene-mapping/review` | view_live | shipped |
| `list_areas` | `GET /areas` | view_live | shipped |
| `create_area` | `POST /areas` | configure_cameras | shipped |
| `area_context` | `GET /areas/{id}/context` | view_live | shipped |
| `approve_area_context` | `POST /areas/{id}/context/approve` | configure_cameras | shipped |

### Engine control

| Bridge method | Endpoint | Permission | Status |
|---|---|---|---|
| `start_monitoring` | `POST /engine/start` | control_engine | shipped |
| `stop_monitoring` | `POST /engine/stop` | control_engine | shipped |
| `monitoring_status` | `GET /monitor` | any | shipped |
| `gate_status` | `GET /engine/gate` | any | shipped |
| `pull_model` | `POST /engine/models/pull` | configure_site | shipped |
| `pull_progress` | `GET /engine/models/pull` | configure_site | shipped |
| `feed_sources` | `GET /engine/feeds` | view_live | shipped |
| `switch_feed` | `POST /engine/feeds/switch` | control_engine | shipped |
| `feed_switch_status` | `GET /engine/feeds/switch` | view_live | shipped |

### Events & triage

| Bridge method | Endpoint | Permission | Status |
|---|---|---|---|
| `list_events` | `GET /events?limit&cursor&camera&priority` | view_alerts | shipped |
| — | `GET /events/{id}` | view_alerts | shipped |
| `search_events` | `GET /events?q=` (extends list_events) | view_alerts | shipped |
| `event_clip` | `GET /events/{id}/clip` (video/mp4; the id scopes the evidence — never a raw path) | view_alerts | shipped |
| `acknowledge_alert` | `POST /events/{id}/acknowledge` | review_alerts | shipped |
| `resolve_alert` | `POST /events/{id}/resolve` (body: `{outcome, note?}`) | review_alerts | shipped |
| — | `GET /triage` | view_alerts | shipped |

### Admin & system

| Bridge method | Endpoint | Permission | Status |
|---|---|---|---|
| `list_users` | `GET /users` | manage_users | shipped |
| `add_user` | `POST /users` | manage_users | shipped |
| `remove_user` | `DELETE /users/{username}` | manage_users | shipped |
| `audit_entries` | `GET /audit` | view_audit | shipped |
| `retention_status` | `GET /retention` | configure_site | shipped |
| `set_retention` | `PUT /retention` | configure_site | shipped |
| `backup_now` | `POST /backups` | configure_site | shipped |
| `download_diagnostics` | `POST /diagnostics/bundle` → `{path, size_kb}` | view_diagnostics | shipped |
| `value_summary` | `GET /value/summary` | view_alerts | shipped |
| `disk_encryption` | `GET /system/disk-encryption` | any | shipped |
| — | `GET /system/health` | any | shipped |
| — | `GET /system/info` | any | shipped |

## WebSocket — `WS /api/v1/stream`

Clients offer the ordered subprotocols `argus.v1` and
`argus.token.<bearer-token>`. The server selects only `argus.v1`, so the token
is not echoed and never enters access-log request URLs. The socket requires
`view_alerts`, revalidates its in-memory token and account on every push-loop
iteration, and closes with:

- `4401` when the token expires/is revoked, the account is deleted, or its role
  changes; the client must clear the session and sign in again;
- `4403` when the authenticated role lacks `view_alerts`; the client keeps the
  REST session but must not reconnect the alert socket.

Messages are `{type, ts, data}`. On connect the server hydrates with one
`health` and one `triage` snapshot, then pushes:

| type | When | Status |
|---|---|---|
| `health` | the engine's health doc changed | shipped |
| `triage` | hydrate snapshot | shipped |
| `alert.new` | a new event row landed | shipped |
| `alert.update` | **a provisional alert settled in place** — the fast path shows criticals before the verdict, and the verdict (confirmed, or kept-and-RETRACTED) must reach the UI on the same row. Without this a WS client shows stale provisional alerts forever. `data` = the full updated event. | **pending — required before any client relies on the WS for triage** |

## Change discipline

A change to any shipped shape bumps this file's version and says so in the PR
title. Pending endpoints may adjust while being built ONLY by editing this
file in the same PR — the file is always ahead of or equal to the code, never
behind it. The consistency test fails any drift it can detect mechanically.
