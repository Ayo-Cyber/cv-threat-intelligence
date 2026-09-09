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

| Workflow          | Implementation                                               | Validation boundary                                                                  |
| ----------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------------ |
| Camera wall       | Real MJPEG adapter; labelled local demo videos               | Browser media checks; native authenticated streaming and snapshot smoke              |
| Engine controls   | Existing `start_monitoring`, `stop_monitoring`, health       | No full model inference benchmark performed                                          |
| Feed modes        | Existing source registry and async switch progress           | No external EarthCam availability claim                                              |
| Camera onboarding | Discover, source test, add/remove, area assignment           | Real native camera addition verified                                                 |
| Agent Mapper      | Camera review/edit/approve/remap, evidence guards            | Draft/persistence tests; native missing-evidence guard                               |
| Site/area mapping | Existing hierarchical summary/approval endpoints             | Backend's evidence/conflict guards retained; needs real multi-camera acceptance test |
| Rules             | Detector flags/presets and English conditions                | Demo persistence tested; inference remains existing backend                          |
| Zones             | Rectangle/polygon editor, original dimensions, dwell rules   | Geometry tests and browser drawing/save test                                         |
| Triage            | Evidence, acknowledge, three outcomes, notes                 | Browser persistence; real backend methods reused                                     |
| Setup/settings    | Six-step flow, verifier checks, notification test, retention | No unsolicited model downloads/notifications                                         |
| Security boundary | Explicit IPC allowlist, auth and backend permissions         | Python dispatch tests and native IPC smoke                                           |

Follow-up: the UI module is now `Frontend/`. Zones uses a full-screen,
aspect-preserving workspace with bottom-edge and discard-guard tests. Login has
terminal-only password recovery guidance, and owners can list/create accounts
under Settings > Users. Recovery preserves other users and evidence. Native tests
exercise new operator login and reject unauthorized account creation.

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
6. Operator acceptance of large-site layouts and 100-camera performance; virtualize
   tiles and subscribe only to visible feeds before that scale.
7. Integrate Ayo's API as its write contracts mature and remove the interim worker
   transport. His latest `main` (`7900c69`, fetched before this PR) contains the
   FastAPI read/auth/WebSocket skeleton and go2rtc/WebRTC stream descriptors.
   This UI currently uses the existing backend and MJPEG path, not that new gateway.
8. Signed installers, auto-update, bundled Python/model runtime and macOS/Windows QA.
   Electron alone does not package the Python inference environment.

Do not advertise detector accuracy based on this UI work. Demo incident text is a
fixture, never an AI verdict. Inconclusive is not classified as a real incident.

## Publication scope

The original UI landed via PR #109. This follow-up branch,
`feat/frontend-module-zones-accounts`, starts from Ayo's `main` at `8888bf6`.
It moves `desktop/` to `Frontend/`, includes the zone/account fixes, and updates
the root README. Existing local camera configuration edits, the
virtual environment, node_modules, generated build outputs and recordings are
excluded. This is an incremental migration PR, not a replacement for the old
console or a signed release. The EarthCam registry is reused without changing
its URLs or overwriting site configuration during publication.

Verification on the follow-up base: build, 12 frontend unit tests, 12 Python
adapter/recovery tests, 9 browser workflow tests and native Electron smoke.
Default runtime data stays in `runs/desktop/` to retain existing accounts.

## Next work for Ayo's agent

Start with `Frontend/README.md`, inspect `src/lib/types.ts`, `electron/main.ts`,
`bridge.py`, and the tests. Keep the demo transport isolated. Do not replace the
canonical scene store with frontend state or auto-approve mappings without evidence.
Preserve site/area/camera boundaries and the downweighted temporal-witness role of
VideoMAE in any later detection changes. The UI must not invent a new threat policy.
