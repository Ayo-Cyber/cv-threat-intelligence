# Desktop UI Migration Handoff

## Scope of this change

User approved the charcoal/off-white prototype and asked for its implementation
with Electron, React and TypeScript, retaining existing operator workflows.
Implementation is confined to `Frontend/`; the Python detection pipeline and
existing Qt/WebChannel console remain the fallback while migration proceeds.

The working tree was moved under `Desktop/Career/CV Threat Intelligence`.
Git worktree references were repaired. Existing edits to `configs/site_live.json`,
`configs/deluxe_demo.json` and `configs/rules/stage.json` were not overwritten.

## Connected in this pass

| Workflow           | Implementation                                                                  | Validation boundary                                                                  |
| ------------------ | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| Camera wall        | Per-camera WebRTC descriptors with MJPEG fallback; labelled demo videos         | Browser media checks; native authenticated descriptor and preview smoke              |
| Engine controls    | Existing `start_monitoring`, `stop_monitoring`, health                          | No full model inference benchmark performed                                          |
| Feed modes         | Existing source registry and async switch progress                              | No external EarthCam availability claim                                              |
| Camera onboarding  | Discover, canonical source probe (`url` alias), add/remove, area assignment     | Real native camera addition verified                                                 |
| Location hierarchy | Organization, branches, areas, assignments and normalized legacy reads          | Native two-branch/three-area API and wall acceptance                                 |
| Streams-only wall  | Branch/area/search filters, density, paging, focus, fullscreen and shell return | Browser flow plus native filter/return smoke; headed Escape is environment-gated     |
| Agent Mapper       | Camera review/edit/approve/remap, evidence guards                               | Draft/persistence tests; native missing-evidence guard                               |
| Site/area mapping  | Existing hierarchical summary/approval endpoints                                | Backend's evidence/conflict guards retained; needs real multi-camera acceptance test |
| Rules              | Detector flags/presets and English conditions                                   | Demo persistence tested; inference remains existing backend                          |
| Zones              | Rectangle/polygon editor, original dimensions, dwell rules                      | Geometry tests and browser drawing/save test                                         |
| Triage             | Evidence, acknowledge, three outcomes, notes                                    | Browser persistence; real backend methods reused                                     |
| Setup/settings     | Six-step flow, verifier checks, notification test, retention                    | No unsolicited model downloads/notifications                                         |
| Security boundary  | Explicit IPC allowlist; Electron-owned API token and child lifecycle            | Python/API tests and native token/shutdown smoke                                     |

Follow-up: the UI module is now `Frontend/`. Zones uses a full-screen,
aspect-preserving workspace with bottom-edge and discard-guard tests. Login has
terminal-only password recovery guidance, and owners can list/create accounts
under Settings > Users. Recovery preserves other users and evidence. Native tests
exercise new operator login and reject unauthorized account creation.

The loopback API is now the default Electron transport. Electron owns first-owner
authentication, the bearer token, WebSocket subscription, and API child shutdown;
the renderer receives only the narrow preload interface. Set
`ARGUS_TRANSPORT=bridge` for temporary JSON-lines rollback, and use
`ARGUS_API_PORT` to override the default port 8787.

Legacy flat sites remain readable through a stable virtual organization and
branch without write-on-read. The first organization/branch/area mutation
materializes normalized hierarchy fields. Operators configure organization and
branches in Settings, then areas and camera assignments in Cameras > Locations.
Both the standard wall and streams-only wall filter by branch and area. Streams
wall is entered from Overview and returns to the normal shell through its exit
control or ordered Escape handling.

## Still required before retiring the old console

1. Full end-to-end acceptance with real camera feeds and the selected VLM:
   remap -> evidence -> human approval -> detector candidate -> verified incident.
2. UI parity for floor plans, Ask/search, owner value analytics, full user/admin
   management, audit browsing, incident PDF/export/legal holds, backup/restore,
   diagnostics downloads and model installation/progress.
3. More granular role-aware navigation. Backend permissions remain authoritative;
   unauthorized native calls are rejected, not silently performed.
4. Extend draft recovery across app restarts and unfinished zone drawings. Scene
   drafts are protected from periodic refresh, and closing a camera panel with
   unsaved scene/English-rule/zone edits requires discard confirmation.
5. Full camera onboarding for vendor credential forms, discovery result variations,
   Windows webcam behavior, and keyboard-only precise zone editing.
6. Operator acceptance of large-site layouts and 100-camera performance. The wall
   now caps active visible/prefetch descriptors, but production-scale soak testing
   is still required.
7. Remove the bridge rollback after API transport soak time and operator sign-off.
8. Signed installers, auto-update, bundled Python/model runtime and macOS/Windows QA.
   Electron alone does not package the Python inference environment.

Do not advertise detector accuracy based on this UI work. Demo incident text is a
fixture, never an AI verdict. Inconclusive is not classified as a real incident.

## Publication scope

The original UI landed via PR #109. This API/hierarchy/wall pass is an incremental
migration, not retirement of the old console or a signed release. Existing local
camera configuration edits, the virtual environment, node_modules, generated
build output, screenshots, recordings, and runtime databases remain excluded.
Default runtime data stays in `runs/desktop/` to retain existing accounts.

Native acceptance uses a temporary site, account database, Electron profile,
local preview publisher, and unique API port. It covers API discovery, first-owner
auth, hierarchy mutation/readback, camera/scene reads, stream descriptor,
monitoring start/stop responses, streams wall filtering/return, token non-exposure,
and API child shutdown. Real hardware, a live go2rtc/WebRTC negotiation, detector
inference quality, and platform-owned fullscreen Escape remain explicit hardware
or headed-environment acceptance boundaries when automation cannot exercise them.

## Next work for Ayo's agent

Start with `Frontend/README.md`, inspect `src/lib/types.ts`, `electron/main.ts`,
`bridge.py`, and the tests. Keep the demo transport isolated. Do not replace the
canonical scene store with frontend state or auto-approve mappings without evidence.
Preserve site/area/camera boundaries and the downweighted temporal-witness role of
VideoMAE in any later detection changes. The UI must not invent a new threat policy.
